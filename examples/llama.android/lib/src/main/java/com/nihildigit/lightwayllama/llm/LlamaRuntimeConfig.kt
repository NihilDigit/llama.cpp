package com.nihildigit.lightwayllama.llm

/**
 * Runtime configuration for llama.cpp sessions.
 * Values are applied via native sampler/template controls.
 */
data class LlamaRuntimeConfig(
    val contextLength: Int = 8192,
    val maxNewTokens: Int = 2048,
    val temperature: Float = 0.7f,
    val topP: Float = 0.8f,
    val topK: Int = 20,
    val minP: Float = 0.0f,
    val repeatPenalty: Float = 1.1f,
    val chatTemplate: String? = null,
    val enableThinking: Boolean = false
)
