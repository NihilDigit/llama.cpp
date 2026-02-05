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

    /**
     * Reload model with different GPU layers setting.
     * This is useful for benchmarking CPU vs GPU performance.
     * @param nGpuLayers 0 = CPU only, 99/-1 = all layers on GPU
     */
    @Synchronized
    fun reloadWithGpuLayers(nGpuLayers: Int) {
        Log.i(TAG, "Reloading model with nGpuLayers=$nGpuLayers")
        runBlocking {
            engine.cleanUp()
        }
        runtimeConfig = runtimeConfig.copy(nGpuLayers = nGpuLayers)
        runBlocking {
            engine.updateGpuLayers(nGpuLayers)
            engine.updateContextLength(runtimeConfig.contextLength)
            engine.loadModel(modelPath.absolutePath)
        }
        applyRuntimeConfig()
        Log.i(TAG, "Model reloaded with nGpuLayers=$nGpuLayers")
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

    // === Stem Context Management ===

    /**
     * 预热 Stem（System Prompt + Tools），将其 KV Cache 保留在内存中。
     * 后续调用 pruneToStem() 可以快速回到这个状态。
     *
     * @param messages 包含 system prompt 的消息列表
     * @param toolsJson 工具定义 JSON（可选）
     * @return 0 表示成功，非 0 表示失败
     */
    fun warmupStem(messages: List<StructuredChatMessage>, toolsJson: String? = null): Int {
        val messagesJson = JSONArray().apply {
            for (msg in messages) {
                put(msg.toJson())
            }
        }.toString()
        return engine.warmupStemContext(messagesJson, toolsJson)
    }

    /**
     * 清除 Stem 之后的所有 KV Cache，回到 Stem 状态。
     * 用于开始新任务时快速复用 System Prompt + Tools 的 KV Cache。
     */
    fun pruneToStem() {
        engine.pruneToStemContext()
    }

    /**
     * 获取当前 Stem 的 token 位置。
     * @return Stem 结束位置（token 数量），0 表示没有预热的 Stem
     */
    fun getStemPosition(): Int {
        return engine.getStemContextPosition()
    }

    /**
     * 获取当前序列长度（token 数量）
     */
    fun getCurrentSeqLen(): Long = engine.getStemContextPosition().toLong()

    /**
     * 清除指定范围的历史（目前实现为完全重置）
     */
    fun eraseHistory(begin: Long, end: Long = 0L) {
        if (begin == 0L) {
            reset()
        } else {
            pruneToStem()
        }
    }

    fun prefillOnly(prompt: String): Long = 0L

    fun prefillOnlyStructured(messages: List<StructuredChatMessage>, toolsJson: String?): Long {
        return warmupStem(messages, toolsJson).toLong()
    }

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
