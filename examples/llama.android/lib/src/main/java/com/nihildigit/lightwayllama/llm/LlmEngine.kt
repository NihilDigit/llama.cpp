package com.nihildigit.lightwayllama.llm

import android.content.Context
import android.util.Log
import com.nihildigit.lightwayllama.InferenceEngine
import com.nihildigit.lightwayllama.internal.InferenceEngineImpl
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import java.io.File

/**
 * LlmEngine - 应用级 LLM 单例管理器
 *
 * 设计原则：
 * - 单例模式：全局唯一 LLM 实例，与底层 InferenceEngineImpl 单例对齐
 * - 无业务状态：不持有 messages/systemPrompt，由调用者管理
 * - 纯推理执行：暴露底层 API，不做业务逻辑封装
 *
 * 使用流程：
 * 1. LlmEngine.load(context, config) - 加载模型
 * 2. LlmEngine.generate(request, onChunk) - 生成响应
 * 3. LlmEngine.unload() - 卸载模型
 */
object LlmEngine {
    private const val TAG = "LlmEngine"

    // ===== 配置数据类 =====

    /**
     * 模型加载配置
     * 注意：contextLength 在 JNI 层固定为 4096，此处不暴露
     */
    data class Config(
        val modelPath: String,
        val nGpuLayers: Int = 0,
        val chatTemplate: String? = null
    )

    /**
     * 采样参数
     * 注意：repeatPenalty 在 JNI 层固定为 1.1f，此处不暴露
     */
    data class SamplingParams(
        val temperature: Float = 0.7f,
        val topP: Float = 0.8f,
        val topK: Int = 20,
        val minP: Float = 0.0f
    )

    /**
     * 生成请求
     */
    data class GenerateRequest(
        val messagesJson: String,
        val toolsJson: String? = null,
        val enableThinking: Boolean = false
    )

    /**
     * 生成结果
     */
    data class GenerateResult(
        val promptTokens: Int,
        val generatedTokens: Int,
        val prefillTimeUs: Long,
        val decodeTimeUs: Long,
        val parsedResponseJson: String?,
        val error: String? = null
    ) {
        val prefillSpeed: Double
            get() = if (prefillTimeUs > 0) promptTokens * 1_000_000.0 / prefillTimeUs else 0.0

        val decodeSpeed: Double
            get() = if (decodeTimeUs > 0) generatedTokens * 1_000_000.0 / decodeTimeUs else 0.0
    }

    // ===== 引擎状态 =====

    sealed class EngineState {
        object Uninitialized : EngineState()
        object Initializing : EngineState()
        object Initialized : EngineState()
        object LoadingModel : EngineState()
        object ModelReady : EngineState()
        object Generating : EngineState()
        data class Error(val message: String) : EngineState()
    }

    private val _state = MutableStateFlow<EngineState>(EngineState.Uninitialized)
    val state: StateFlow<EngineState> = _state.asStateFlow()

    private var engine: InferenceEngineImpl? = null
    private var currentConfig: Config? = null

    // ===== 生命周期 =====

    /**
     * 加载模型
     * @param context Android Context
     * @param config 模型配置
     */
    suspend fun load(context: Context, config: Config) {
        val modelFile = File(config.modelPath)
        require(modelFile.isFile) { "Model path must be a file: ${config.modelPath}" }

        Log.i(TAG, "Loading model: ${config.modelPath}, nGpuLayers=${config.nGpuLayers}")
        _state.value = EngineState.Initializing

        val impl = InferenceEngineImpl.getInstance(context.applicationContext) as InferenceEngineImpl
        engine = impl

        // 等待引擎初始化完成
        val initState = impl.state.first { state ->
            state is InferenceEngine.State.Initialized || state is InferenceEngine.State.Error
        }
        when (initState) {
            is InferenceEngine.State.Error -> {
                _state.value = EngineState.Error(initState.exception.message ?: "Init failed")
                throw initState.exception
            }
            else -> _state.value = EngineState.Initialized
        }

        // 配置 GPU 层数和聊天模板
        impl.updateGpuLayers(config.nGpuLayers)
        config.chatTemplate?.let { impl.updateChatTemplateOverride(it) }

        // 加载模型
        _state.value = EngineState.LoadingModel
        impl.loadModel(config.modelPath)
        _state.value = EngineState.ModelReady
        currentConfig = config

        Log.i(TAG, "Model loaded successfully")
    }

    /**
     * 卸载模型
     */
    fun unload() {
        Log.i(TAG, "Unloading model")
        engine?.cleanUp()
        _state.value = EngineState.Initialized
        currentConfig = null
    }

    /**
     * 重新加载模型（用于切换 GPU 层数等配置）
     */
    suspend fun reload(context: Context, config: Config) {
        Log.i(TAG, "Reloading model with new config: nGpuLayers=${config.nGpuLayers}")
        unload()
        load(context, config)
    }

    /**
     * 检查模型是否已加载
     */
    fun isModelLoaded(): Boolean = _state.value == EngineState.ModelReady

    // ===== 推理 =====

    /**
     * 生成响应
     * @param request 生成请求
     * @param onChunk 流式回调，返回 true 表示取消生成
     * @return 生成结果
     */
    fun generate(
        request: GenerateRequest,
        onChunk: ((String) -> Boolean)? = null
    ): GenerateResult {
        val impl = engine ?: throw IllegalStateException("Engine not initialized")
        check(_state.value == EngineState.ModelReady) {
            "Cannot generate in state: ${_state.value}"
        }

        _state.value = EngineState.Generating

        runBlocking {
            impl.sendStructuredPrompt(
                messagesJson = request.messagesJson,
                toolsJson = request.toolsJson,
                predictLength = 2048, // JNI 层固定为 DEFAULT_MAX_NEW_TOKENS
                enableThinking = request.enableThinking
            ).collect { chunk ->
                if (onChunk?.invoke(chunk) == true) {
                    impl.cancelGeneration()
                }
            }
        }

        _state.value = EngineState.ModelReady

        return GenerateResult(
            promptTokens = impl.lastPromptTokenCount,
            generatedTokens = impl.lastGeneratedTokenCount,
            prefillTimeUs = impl.lastPrefillDurationUs,
            decodeTimeUs = impl.lastDecodeDurationUs,
            parsedResponseJson = impl.lastParsedAssistantMessage
        )
    }

    /**
     * 取消当前生成
     */
    fun cancelGeneration() {
        engine?.cancelGeneration()
    }

    // ===== KV Cache 管理 =====

    /**
     * 重置上下文（清除所有 KV Cache）
     */
    fun resetContext() {
        engine?.resetConversation()
    }

    /**
     * 预热 Stem（System Prompt + Tools）
     * @param messagesJson 消息 JSON（通常只包含 system message）
     * @param toolsJson 工具定义 JSON
     * @return 0 表示成功，非 0 表示失败
     */
    fun warmupStem(messagesJson: String, toolsJson: String?): Int {
        val impl = engine ?: return -1
        return impl.warmupStemContext(messagesJson, toolsJson)
    }

    /**
     * 剪枝到 Stem 位置（清除 Stem 之后的 KV Cache）
     */
    fun pruneToStem() {
        engine?.pruneToStemContext()
    }

    /**
     * 获取当前 Stem 位置
     * @return Stem 结束位置（token 数量），0 表示没有预热的 Stem
     */
    fun getStemPosition(): Int {
        return engine?.getStemContextPosition() ?: 0
    }

    // ===== 配置 =====

    /**
     * 更新采样参数
     */
    fun updateSamplingParams(params: SamplingParams) {
        engine?.updateSamplingParams(
            temperature = params.temperature,
            topP = params.topP,
            topK = params.topK,
            minP = params.minP,
            repeatPenalty = 1.1f // JNI 层固定值
        )
    }

    /**
     * 更新聊天模板
     */
    fun updateChatTemplate(template: String?) {
        engine?.updateChatTemplateOverride(template)
    }

    /**
     * 渲染聊天模板（调试用）
     */
    fun renderChatTemplate(
        messagesJson: String,
        toolsJson: String?,
        enableThinking: Boolean
    ): String {
        return engine?.renderChatTemplatePrompt(messagesJson, toolsJson, enableThinking) ?: ""
    }

    // ===== 查询 =====

    /**
     * 获取当前配置
     */
    fun getCurrentConfig(): Config? = currentConfig

    /**
     * 获取当前 GPU 层数
     */
    fun getCurrentGpuLayers(): Int {
        return engine?.getCurrentGpuLayers() ?: 0
    }
}
