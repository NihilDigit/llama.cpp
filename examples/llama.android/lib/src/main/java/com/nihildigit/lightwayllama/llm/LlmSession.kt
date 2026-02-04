package com.nihildigit.lightwayllama.llm

import android.content.Context
import android.util.Log
import com.nihildigit.lightwayllama.internal.InferenceEngineImpl
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.flow.collect
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

interface ProgressListener {
    /**
     * Called for each chunk; return true to request stopping generation early.
     * When `chunk == null`, it indicates end-of-stream.
     */
    fun onProgress(chunk: String?): Boolean
}

data class GenerationStats(
    val promptLen: Long = 0,
    val decodeLen: Long = 0,
    val visionTimeUs: Long = 0,
    val audioTimeUs: Long = 0,
    val prefillTimeUs: Long = 0,
    val decodeTimeUs: Long = 0,
    val error: String? = null
)

data class StructuredToolCall(
    val id: String,
    val function: StructuredFunctionCall
)

data class StructuredFunctionCall(
    val name: String,
    val arguments: String
)

data class StructuredChatMessage(
    val role: String,
    val content: String? = null,
    val name: String? = null,
    val toolCallId: String? = null,
    val toolCalls: List<StructuredToolCall>? = null
) {
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("role", role)
            content?.let { put("content", it) }
            name?.let { put("name", it) }
            toolCallId?.let { put("tool_call_id", it) }
            toolCalls?.let { calls ->
                val tcArray = JSONArray()
                for (tc in calls) {
                    val tcObj = JSONObject().apply {
                        put("id", tc.id)
                        put("type", "function")
                        put("function", JSONObject().apply {
                            put("name", tc.function.name)
                            put("arguments", tc.function.arguments)
                        })
                    }
                    tcArray.put(tcObj)
                }
                put("tool_calls", tcArray)
            }
        }
    }
}

class LlmSession(
    context: Context,
    private val modelPath: File,
    private var runtimeConfig: LlamaRuntimeConfig = LlamaRuntimeConfig()
) : java.io.Closeable {
    companion object {
        private const val TAG = "LlamaLlmSession"
    }

    private val engine: InferenceEngineImpl =
        InferenceEngineImpl.getInstance(context.applicationContext) as InferenceEngineImpl

    @Volatile
    private var maxNewTokens: Int = runtimeConfig.maxNewTokens
    @Volatile
    private var systemPrompt: String? = null
    @Volatile
    private var assistantPrompt: String? = null
    @Volatile
    private var lastParsedMessageJson: String? = null

    @Synchronized
    fun load() {
        check(modelPath.isFile) { "modelPath must be a file: $modelPath" }
        runBlocking {
            engine.updateGpuLayers(runtimeConfig.nGpuLayers)
            engine.updateContextLength(runtimeConfig.contextLength)
            engine.loadModel(modelPath.absolutePath)
        }
        applyRuntimeConfig()
    }

    @Synchronized
    fun reset() {
        engine.resetConversation()
    }

    @Synchronized
    fun release() {
        runBlocking {
            engine.cleanUp()
        }
    }

    override fun close() {
        release()
    }

    fun updateMaxNewTokens(tokens: Int) {
        maxNewTokens = tokens.coerceAtLeast(1)
    }

    fun updateSystemPrompt(prompt: String) {
        systemPrompt = prompt
    }

    fun updateAssistantPrompt(prompt: String) {
        assistantPrompt = prompt
    }

    fun updateConfig(configJson: String) {
        if (configJson.isBlank()) return
        try {
            val obj = JSONObject(configJson)
            val jinja = obj.optJSONObject("jinja")
            val jinjaContext = jinja?.optJSONObject("context")

            val chatTemplate = jinja?.optString("chat_template")
                ?.takeIf { it.isNotBlank() }
                ?: obj.optString("chat_template").takeIf { it.isNotBlank() }

            val enableThinking = jinjaContext?.optBoolean("enable_thinking")
                ?: obj.optBoolean("enable_thinking", runtimeConfig.enableThinking)

            val temperature = obj.optDouble("temperature", runtimeConfig.temperature.toDouble()).toFloat()
            val topP = obj.optDouble("top_p", runtimeConfig.topP.toDouble()).toFloat()
            val topK = obj.optInt("top_k", runtimeConfig.topK)
            val minP = obj.optDouble("min_p", runtimeConfig.minP.toDouble()).toFloat()
            val repeatPenalty = obj.optDouble("repeat_penalty", runtimeConfig.repeatPenalty.toDouble()).toFloat()
            val contextLength = obj.optInt("context_length", runtimeConfig.contextLength)
            val maxTokens = obj.optInt("max_new_tokens", runtimeConfig.maxNewTokens)
            val nGpuLayers = obj.optInt("n_gpu_layers", runtimeConfig.nGpuLayers)

            runtimeConfig = runtimeConfig.copy(
                contextLength = contextLength,
                maxNewTokens = maxTokens,
                temperature = temperature,
                topP = topP,
                topK = topK,
                minP = minP,
                repeatPenalty = repeatPenalty,
                chatTemplate = chatTemplate ?: runtimeConfig.chatTemplate,
                enableThinking = enableThinking,
                nGpuLayers = nGpuLayers
            )
            maxNewTokens = runtimeConfig.maxNewTokens
            applyRuntimeConfig()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse runtime config, ignoring: ${e.message}")
        }
    }

    fun generateStructured(
        messages: List<StructuredChatMessage>,
        toolsJson: String? = null,
        listener: ProgressListener? = null
    ): GenerationStats {
        val messagesJson = JSONArray().apply {
            for (msg in messages) {
                put(msg.toJson())
            }
        }.toString()

        runBlocking {
            engine.sendStructuredPrompt(
                messagesJson = messagesJson,
                toolsJson = toolsJson,
                predictLength = maxNewTokens,
                enableThinking = runtimeConfig.enableThinking
            ).collect { chunk ->
                if (listener?.onProgress(chunk) == true) {
                    engine.cancelGeneration()
                }
            }
        }
        listener?.onProgress(null)

        lastParsedMessageJson = engine.lastParsedAssistantMessage

        return GenerationStats(
            promptLen = engine.lastPromptTokenCount.toLong(),
            decodeLen = engine.lastGeneratedTokenCount.toLong(),
            prefillTimeUs = engine.lastPrefillDurationUs,
            decodeTimeUs = engine.lastDecodeDurationUs,
            error = null
        )
    }

    fun applyChatTemplateForDebug(
        messages: List<StructuredChatMessage>,
        toolsJson: String? = null
    ): String {
        val messagesJson = JSONArray().apply {
            for (msg in messages) {
                put(msg.toJson())
            }
        }.toString()
        return engine.renderChatTemplatePrompt(messagesJson, toolsJson, runtimeConfig.enableThinking)
    }

    fun getLastParsedResponseJson(): String? = lastParsedMessageJson

    fun getSystemPrompt(): String? = systemPrompt

    fun clearHistory() {
        reset()
    }

    // === Stem Context compatibility stubs ===
    fun getCurrentSeqLen(): Long = 0L

    fun eraseHistory(begin: Long, end: Long = 0L) {
        reset()
    }

    fun prefillOnly(prompt: String): Long = 0L

    fun prefillOnlyStructured(messages: List<StructuredChatMessage>, toolsJson: String?): Long = 0L

    fun prepareSaveStemCache(filename: String) {}

    fun finishSaveStemCache(filename: String): Long = 0L

    fun saveStemCache(filename: String): Long = 0L

    fun loadStemCache(filename: String, expectedSeqLen: Long = 0L): Long = 0L

    fun isStemCacheValid(filename: String): Boolean = false

    private fun applyRuntimeConfig() {
        engine.updateSamplingParams(
            temperature = runtimeConfig.temperature,
            topP = runtimeConfig.topP,
            topK = runtimeConfig.topK,
            minP = runtimeConfig.minP,
            repeatPenalty = runtimeConfig.repeatPenalty
        )
        runtimeConfig.chatTemplate?.let { engine.updateChatTemplateOverride(it) }
        if (runtimeConfig.contextLength > 0) {
            engine.updateContextLength(runtimeConfig.contextLength)
        }
    }
}
