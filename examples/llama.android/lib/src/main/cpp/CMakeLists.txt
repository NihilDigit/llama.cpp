cmake_minimum_required(VERSION 3.22.1)

project("ai-chat" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}"   CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)

# --------------------------------------------------------------------------
# Vulkan-Hpp for Android (NDK only provides C API, not C++ bindings)
# --------------------------------------------------------------------------
if(GGML_VULKAN AND ANDROID)
    # Use local vulkan.hpp (downloaded from Vulkan-Hpp v1.3.275)
    set(VULKAN_HPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vulkan-hpp)
    include_directories(SYSTEM ${VULKAN_HPP_DIR})
    message(STATUS "Using local Vulkan-Hpp from: ${VULKAN_HPP_DIR}")

    # Define Android Vulkan platform
    add_definitions(-DVK_USE_PLATFORM_ANDROID_KHR)
endif()

# --------------------------------------------------------------------------
# KleidiAI - use local copy to avoid network download during build
# --------------------------------------------------------------------------
if(GGML_CPU_KLEIDIAI AND ANDROID)
    # Get absolute path to local KleidiAI (lightway/third_party/kleidiai)
    get_filename_component(KLEIDIAI_LOCAL_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../../third_party/kleidiai"
        ABSOLUTE)
    # Pre-populate FetchContent cache to skip download
    set(FETCHCONTENT_SOURCE_DIR_KLEIDIAI_DOWNLOAD "${KLEIDIAI_LOCAL_PATH}" CACHE PATH "" FORCE)
    message(STATUS "Using local KleidiAI from: ${KLEIDIAI_LOCAL_PATH}")
endif()

# --------------------------------------------------------------------------
# AI Chat library
# --------------------------------------------------------------------------

if(DEFINED ANDROID_ABI)
    message(STATUS "Detected Android ABI: ${ANDROID_ABI}")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        set(GGML_CPU_KLEIDIAI ON)
        set(GGML_OPENMP ON)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    else()
        message(FATAL_ERROR "Unsupported ABI: ${ANDROID_ABI}")
    endif()
endif()

set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../)
add_subdirectory(${LLAMA_SRC} build-llama)

add_library(${CMAKE_PROJECT_NAME} SHARED
        ai_chat.cpp)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
        GGML_SYSTEM_ARCH=${GGML_SYSTEM_ARCH}
        GGML_CPU_KLEIDIAI=$<BOOL:${GGML_CPU_KLEIDIAI}>
        GGML_OPENMP=$<BOOL:${GGML_OPENMP}>
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${LLAMA_SRC}
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/ggml/include
        ${LLAMA_SRC}/ggml/src)

target_link_libraries(${CMAKE_PROJECT_NAME}
        llama
        common
        android
        log)
