//
// Created by Han Yin on 10/31/25.
//

#ifndef AICHAT_LOGGING_H
#define AICHAT_LOGGING_H

#endif //AICHAT_LOGGING_H

#pragma once
#include <android/log.h>
#include <cstring>
#include <string>

#ifndef LOG_TAG
#define LOG_TAG "ai-chat"
#endif

#ifndef LOG_DIAG_TAG
#define LOG_DIAG_TAG "[LLAMA_DIAG] "
#endif

#ifndef LOG_MIN_LEVEL
#if defined(NDEBUG)
#define LOG_MIN_LEVEL ANDROID_LOG_INFO
#else
#define LOG_MIN_LEVEL ANDROID_LOG_VERBOSE
#endif
#endif

static inline int ai_should_log(int prio) {
    return __android_log_is_loggable(prio, LOG_TAG, LOG_MIN_LEVEL);
}

#if LOG_MIN_LEVEL <= ANDROID_LOG_VERBOSE
#define LOGv(...) do { if (ai_should_log(ANDROID_LOG_VERBOSE)) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__); } while (0)
#else
#define LOGv(...) ((void)0)
#endif

#if LOG_MIN_LEVEL <= ANDROID_LOG_DEBUG
#define LOGd(...) do { if (ai_should_log(ANDROID_LOG_DEBUG)) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__); } while (0)
#else
#define LOGd(...) ((void)0)
#endif

#define LOGi(...)   do { if (ai_should_log(ANDROID_LOG_INFO )) __android_log_print(ANDROID_LOG_INFO , LOG_TAG, __VA_ARGS__); } while (0)
#define LOGw(...)   do { if (ai_should_log(ANDROID_LOG_WARN )) __android_log_print(ANDROID_LOG_WARN , LOG_TAG, __VA_ARGS__); } while (0)
#define LOGe(...)   do { if (ai_should_log(ANDROID_LOG_ERROR)) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__); } while (0)

#define LOG_DIAGv(...) do { if (ai_should_log(ANDROID_LOG_VERBOSE)) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, LOG_DIAG_TAG __VA_ARGS__); } while (0)
#define LOG_DIAGd(...) do { if (ai_should_log(ANDROID_LOG_DEBUG))   __android_log_print(ANDROID_LOG_DEBUG,   LOG_TAG, LOG_DIAG_TAG __VA_ARGS__); } while (0)
#define LOG_DIAGi(...) do { if (ai_should_log(ANDROID_LOG_INFO ))   __android_log_print(ANDROID_LOG_INFO,    LOG_TAG, LOG_DIAG_TAG __VA_ARGS__); } while (0)
#define LOG_DIAGw(...) do { if (ai_should_log(ANDROID_LOG_WARN ))   __android_log_print(ANDROID_LOG_WARN,    LOG_TAG, LOG_DIAG_TAG __VA_ARGS__); } while (0)
#define LOG_DIAGe(...) do { if (ai_should_log(ANDROID_LOG_ERROR))   __android_log_print(ANDROID_LOG_ERROR,   LOG_TAG, LOG_DIAG_TAG __VA_ARGS__); } while (0)

static inline int android_log_prio_from_ggml(enum ggml_log_level level) {
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: return ANDROID_LOG_ERROR;
        case GGML_LOG_LEVEL_WARN:  return ANDROID_LOG_WARN;
        case GGML_LOG_LEVEL_INFO:  return ANDROID_LOG_INFO;
        case GGML_LOG_LEVEL_DEBUG: return ANDROID_LOG_DEBUG;
        default:                   return ANDROID_LOG_DEFAULT;
    }
}

static inline void aichat_android_log_callback(enum ggml_log_level level,
                                              const char* text,
                                              void* /*user*/) {
    const int prio = android_log_prio_from_ggml(level);
    if (!ai_should_log(prio)) return;
    if (text == nullptr) {
        return;
    }
    if (std::strncmp(text, LOG_DIAG_TAG, std::strlen(LOG_DIAG_TAG)) == 0) {
        __android_log_write(prio, LOG_TAG, text);
        return;
    }
    std::string msg = std::string(LOG_DIAG_TAG) + text;
    __android_log_write(prio, LOG_TAG, msg.c_str());
}
