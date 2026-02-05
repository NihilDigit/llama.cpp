#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <cstring>
#include <unistd.h>
#include <sampling.h>
#include <nlohmann/json.hpp>

#include "logging.h"
#include "chat.h"
#include "common.h"
#include "llama.h"

using json = nlohmann::ordered_json;

template<class T>
static std::string join(const std::vector<T> &values, const std::string &delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) { str << delim; }
    }
    return str.str();
}

/**
 * LLama resources: context, model, batch and sampler
 */
// Thread config optimized for Snapdragon 8 Gen 3 (1 Prime + 5 Performance + 2 Efficiency cores)
// Only use high-performance cores (6) to avoid slow efficiency cores dragging down performance
constexpr int   N_THREADS_MIN           = 2;
constexpr int   N_THREADS_MAX           = 6;
constexpr int   N_THREADS_HEADROOM      = 2;

constexpr int   DEFAULT_CONTEXT_SIZE    = 4096;
constexpr int   DEFAULT_MAX_NEW_TOKENS  = 2048;
constexpr float DEFAULT_REPEAT_PENALTY  = 1.1f;
constexpr int   OVERFLOW_HEADROOM       = 4;
constexpr int   BATCH_SIZE              = 512;
constexpr float DEFAULT_SAMPLER_TEMP    = 0.7f;

static llama_model                      * g_model;
static llama_context                    * g_context;
static llama_batch                        g_batch;
static common_chat_templates_ptr          g_chat_templates;
static common_sampler                   * g_sampler;
static int                                g_context_size = DEFAULT_CONTEXT_SIZE;
static int                                g_n_gpu_layers = 0;  // Default to CPU (Vulkan has issues on some Adreno GPUs)
static int                                g_n_threads = 0;
static common_params_sampling              g_sampling_params;
static bool                               g_sampling_params_initialized = false;
static common_chat_params                  g_last_chat_params;
static int                                g_last_prompt_tokens = 0;
static std::string                        g_last_assistant_raw;
static std::string                        g_last_assistant_parsed_json;
static std::string                        g_chat_template_override;

// === Incremental Inference State ===
static llama_tokens                       g_last_prompt_tokens_vec;  // 上次 tokenize 结果
static llama_pos                          g_stem_position = 0;       // Stem 结束位置
static size_t                             g_last_tools_hash = 0;     // Tools JSON 哈希
static bool                               g_stem_warmed = false;     // Stem 是否已预热
static size_t                             g_last_reused_tokens = 0;  // 最后一次推理复用的 token 数

extern "C"
JNIEXPORT void JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_init(JNIEnv *env, jobject /*unused*/, jstring nativeLibDir) {
    // Set llama log handler to Android
    llama_log_set(aichat_android_log_callback, nullptr);

    // Loading all CPU backend variants
    const auto *path_to_backend = env->GetStringUTFChars(nativeLibDir, 0);
    LOG_DIAGi("Loading backends from %s", path_to_backend);
    ggml_backend_load_all_from_path(path_to_backend);
    env->ReleaseStringUTFChars(nativeLibDir, path_to_backend);

    // Initialize backends
    llama_backend_init();

    // Log all available backends
    LOG_DIAGi("Available backends:");
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto *reg = ggml_backend_reg_get(i);
        LOG_DIAGi("  [%zu] %s", i, ggml_backend_reg_name(reg));
    }

    LOG_DIAGi("Backend initiated; Log handler set. Default n_gpu_layers=%d", g_n_gpu_layers);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_load(JNIEnv *env, jobject, jstring jmodel_path) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = g_n_gpu_layers;
    LOG_DIAGi("%s: n_gpu_layers = %d", __func__, g_n_gpu_layers);

    const auto *model_path = env->GetStringUTFChars(jmodel_path, 0);
    LOG_DIAGi("%s: Loading model from: %s", __func__, model_path);

    auto *model = llama_model_load_from_file(model_path, model_params);
    env->ReleaseStringUTFChars(jmodel_path, model_path);
    if (!model) {
        return 1;
    }
    g_model = model;
    char model_desc[128];
    llama_model_desc(g_model, model_desc, sizeof(model_desc));
    const double model_size_gib = (double) llama_model_size(g_model) / 1024.0 / 1024.0 / 1024.0;
    const double model_n_params_b = (double) llama_model_n_params(g_model) / 1e9;
    const int n_layer = llama_model_n_layer(g_model);
    const int n_head = llama_model_n_head(g_model);
    const int n_head_kv = llama_model_n_head_kv(g_model);
    const int n_embd = llama_model_n_embd(g_model);
    const int n_ctx_train = llama_model_n_ctx_train(g_model);
    LOG_DIAGi("Model info: desc=%s size=%.2fGiB params=%.2fB n_layer=%d n_head=%d n_head_kv=%d n_embd=%d n_ctx_train=%d",
              model_desc, model_size_gib, model_n_params_b, n_layer, n_head, n_head_kv, n_embd, n_ctx_train);
    return 0;
}

static llama_context *init_context(llama_model *model, const int n_ctx = DEFAULT_CONTEXT_SIZE) {
    if (!model) {
        LOGe("%s: model cannot be null", __func__);
        return nullptr;
    }

    // Multi-threading setup
    const int n_threads = std::max(N_THREADS_MIN, std::min(N_THREADS_MAX,
                                                     (int) sysconf(_SC_NPROCESSORS_ONLN) -
                                                     N_THREADS_HEADROOM));
    g_n_threads = n_threads;
    LOG_DIAGi("%s: Using %d threads", __func__, n_threads);

    // Context parameters setup
    llama_context_params ctx_params = llama_context_default_params();
    const int trained_context_size = llama_model_n_ctx_train(model);
    if (n_ctx > trained_context_size) {
        LOG_DIAGw("%s: Model was trained with only %d context size! Enforcing %d context size...",
                  __func__, trained_context_size, n_ctx);
    }
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = BATCH_SIZE;
    ctx_params.n_ubatch = BATCH_SIZE;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    // Disable flash attention for OpenCL compatibility (causes crashes on some GPUs)
    // When flash_attn is disabled, KV cache must use F16 (quantized KV requires flash_attn)
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.type_k = GGML_TYPE_F16;
    ctx_params.type_v = GGML_TYPE_F16;
    auto *context = llama_init_from_model(g_model, ctx_params);
    if (context == nullptr) {
        LOG_DIAGe("%s: llama_new_context_with_model() returned null)", __func__);
    }
    return context;
}

static void init_sampling_defaults() {
    if (g_sampling_params_initialized) {
        return;
    }
    g_sampling_params = common_params_sampling();
    g_sampling_params.temp = DEFAULT_SAMPLER_TEMP;
    g_sampling_params.top_p = 0.8f;
    g_sampling_params.top_k = 20;
    g_sampling_params.min_p = 0.0f;
    g_sampling_params.penalty_repeat = DEFAULT_REPEAT_PENALTY;
    g_sampling_params_initialized = true;
}

static void rebuild_sampler(const common_chat_params * chat_params = nullptr) {
    init_sampling_defaults();
    common_params_sampling sparams = g_sampling_params;
    if (chat_params != nullptr) {
        if (!chat_params->grammar.empty()) {
            sparams.grammar = chat_params->grammar;
            sparams.grammar_lazy = chat_params->grammar_lazy;
            sparams.grammar_triggers = chat_params->grammar_triggers;
        }
        if (!chat_params->preserved_tokens.empty() && g_context != nullptr) {
            for (const auto & token_str : chat_params->preserved_tokens) {
                const auto tokens = common_tokenize(g_context, token_str, /* add_special= */ true, /* parse_special= */ true);
                sparams.preserved_tokens.insert(tokens.begin(), tokens.end());
            }
        }
    }
    if (g_sampler != nullptr) {
        common_sampler_free(g_sampler);
    }
    g_sampler = common_sampler_init(g_model, sparams);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_prepare(JNIEnv * /*env*/, jobject /*unused*/) {
    auto *context = init_context(g_model, g_context_size);
    if (!context) { return 1; }
    g_context = context;
    g_batch = llama_batch_init(BATCH_SIZE, 0, 1);
    g_chat_templates = common_chat_templates_init(g_model, g_chat_template_override.c_str());
    rebuild_sampler();
    const int actual_ctx = llama_n_ctx(g_context);
    const size_t state_size = llama_state_get_size(g_context);
    LOG_DIAGi("Context config: n_ctx=%d (requested=%d) n_batch=%d n_ubatch=%d n_threads=%d flash_attn=%s type_k=%s type_v=%s state=%.2f MB",
              actual_ctx,
              g_context_size,
              BATCH_SIZE,
              BATCH_SIZE,
              g_n_threads,
              llama_flash_attn_type_name(LLAMA_FLASH_ATTN_TYPE_ENABLED),
              ggml_type_name(GGML_TYPE_Q8_0),
              ggml_type_name(GGML_TYPE_Q8_0),
              (double) state_size / 1024.0 / 1024.0);
    LOG_DIAGi("Memory breakdown (after context init):");
    llama_memory_breakdown_print(g_context);
    return 0;
}

static std::string get_backend() {
    std::vector<std::string> backends;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto *reg = ggml_backend_reg_get(i);
        std::string name = ggml_backend_reg_name(reg);
        if (name != "CPU") {
            backends.push_back(ggml_backend_reg_name(reg));
        }
    }
    return backends.empty() ? "CPU" : join(backends, ",");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_systemInfo(JNIEnv *env, jobject /*unused*/) {
    return env->NewStringUTF(llama_print_system_info());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_benchModel(JNIEnv *env, jobject /*unused*/, jint pp, jint tg,
                                                      jint pl, jint nr) {
    auto *context = init_context(g_model, pp);
    if (!context) {
        const auto *const err_msg = "Fail to init_context! Bench aborted.";
        LOGe(err_msg);
        return env->NewStringUTF(err_msg);
    }

    auto pp_avg = 0.0;
    auto tg_avg = 0.0;
    auto pp_std = 0.0;
    auto tg_std = 0.0;

    const uint32_t n_ctx = llama_n_ctx(context);
    LOG_DIAGi("bench: n_ctx = %d", n_ctx);

    int i, j;
    int nri;
    for (nri = 0; nri < nr; nri++) {
        LOG_DIAGi("bench: prompt processing (pp = %d)", pp);

        common_batch_clear(g_batch);

        const int n_tokens = pp;
        for (i = 0; i < n_tokens; i++) {
            common_batch_add(g_batch, 0, i, {0}, false);
        }

        g_batch.logits[g_batch.n_tokens - 1] = true;
        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp_start = ggml_time_us();
        if (llama_decode(context, g_batch) != 0) {
            LOG_DIAGe("bench: llama_decode() failed during prompt processing");
        }
        const auto t_pp_end = ggml_time_us();

        // bench text generation

        LOG_DIAGi("bench: text generation (tg = %d)", tg);

        llama_memory_clear(llama_get_memory(context), false);
        const auto t_tg_start = ggml_time_us();
        for (i = 0; i < tg; i++) {
            common_batch_clear(g_batch);
            for (j = 0; j < pl; j++) {
                common_batch_add(g_batch, 0, i, {j}, true);
            }

            if (llama_decode(context, g_batch) != 0) {
                LOG_DIAGe("bench: llama_decode() failed during text generation");
            }
        }
        const auto t_tg_end = ggml_time_us();

        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp = double(t_pp_end - t_pp_start) / 1000000.0;
        const auto t_tg = double(t_tg_end - t_tg_start) / 1000000.0;

        const auto speed_pp = double(pp) / t_pp;
        const auto speed_tg = double(pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;

        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;

        LOG_DIAGi("bench: pp %f t/s, tg %f t/s", speed_pp, speed_tg);
    }

    llama_free(context);

    pp_avg /= double(nr);
    tg_avg /= double(nr);

    if (nr > 1) {
        pp_std = sqrt(pp_std / double(nr - 1) - pp_avg * pp_avg * double(nr) / double(nr - 1));
        tg_std = sqrt(tg_std / double(nr - 1) - tg_avg * tg_avg * double(nr) / double(nr - 1));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    char model_desc[128];
    llama_model_desc(g_model, model_desc, sizeof(model_desc));

    const auto model_size = double(llama_model_size(g_model)) / 1024.0 / 1024.0 / 1024.0;
    const auto model_n_params = double(llama_model_n_params(g_model)) / 1e9;

    const auto backend = get_backend();
    std::stringstream result;
    result << std::setprecision(3);
    result << "| model | size | params | backend | test | t/s |\n";
    result << "| --- | --- | --- | --- | --- | --- |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | "
           << backend << " | pp " << pp << " | " << pp_avg << " ± " << pp_std << " |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | "
           << backend << " | tg " << tg << " | " << tg_avg << " ± " << tg_std << " |\n";
    return env->NewStringUTF(result.str().c_str());
}


/**
 * Completion loop's long-term states:
 * - chat management
 * - position tracking
 */
constexpr const char *ROLE_SYSTEM       = "system";
constexpr const char *ROLE_USER         = "user";
constexpr const char *ROLE_ASSISTANT    = "assistant";

static std::vector<common_chat_msg> chat_msgs;
static llama_pos system_prompt_position;
static llama_pos current_position;

static void reset_long_term_states(const bool clear_kv_cache = true) {
    chat_msgs.clear();
    system_prompt_position = 0;
    current_position = 0;

    // 清除增量推理状态
    g_last_prompt_tokens_vec.clear();
    g_stem_position = 0;
    g_last_tools_hash = 0;
    g_stem_warmed = false;

    if (clear_kv_cache)
        llama_memory_clear(llama_get_memory(g_context), false);
}

/**
 * TODO-hyin: implement sliding-window version as a better alternative
 *
 * Context shifting by discarding the older half of the tokens appended after system prompt:
 * - take the [system_prompt_position] first tokens from the original prompt
 * - take half of the last (system_prompt_position - system_prompt_position) tokens
 * - recompute the logits in batches
 */
static void shift_context() {
    const int n_discard = (current_position - system_prompt_position) / 2;
    LOG_DIAGi("%s: Discarding %d tokens", __func__, n_discard);
    llama_memory_seq_rm(llama_get_memory(g_context), 0, system_prompt_position, system_prompt_position + n_discard);
    llama_memory_seq_add(llama_get_memory(g_context), 0, system_prompt_position + n_discard, current_position, -n_discard);
    current_position -= n_discard;
    LOG_DIAGi("%s: Context shifting done! Current position: %d", __func__, current_position);
}

// === Incremental Inference Helpers ===

// 计算两个 token 序列的公共前缀长度
static size_t get_common_prefix(const llama_tokens& a, const llama_tokens& b) {
    const size_t max_idx = std::min(a.size(), b.size());
    for (size_t i = 0; i < max_idx; ++i) {
        if (a[i] != b[i]) {
            return i;
        }
    }
    return max_idx;
}

// 字符串哈希（用于检测 tools 变化）
static size_t hash_string(const std::string& str) {
    return std::hash<std::string>{}(str);
}

static std::string chat_add_and_format(const std::string &role, const std::string &content) {
    common_chat_msg new_msg;
    new_msg.role = role;
    new_msg.content = content;
    auto formatted = common_chat_format_single(
            g_chat_templates.get(), chat_msgs, new_msg, role == ROLE_USER, /* use_jinja */ false);
    chat_msgs.push_back(new_msg);
    LOG_DIAGi("%s: Added %s message. formatted_len=%zu", __func__, role.c_str(), formatted.size());
    return formatted;
}

/**
 * Completion loop's short-term states:
 * - stop generation position
 * - token chars caching
 * - current assistant message being generated
 */
static llama_pos stop_generation_position;
static std::string cached_token_chars;
static std::ostringstream assistant_ss;

static void reset_short_term_states() {
    stop_generation_position = 0;
    cached_token_chars.clear();
    assistant_ss.str("");
    assistant_ss.clear();
    g_last_assistant_raw.clear();
    g_last_assistant_parsed_json.clear();
    g_last_prompt_tokens = 0;
}

static int decode_tokens_in_batches(
        llama_context *context,
        llama_batch &batch,
        const llama_tokens &tokens,
        const llama_pos start_pos,
        const bool compute_last_logit = false) {
    // Process tokens in batches using the global batch
    LOGd("%s: Decode %d tokens starting at position %d", __func__, (int) tokens.size(), start_pos);
    const int max_ctx = llama_n_ctx(context);
    for (int i = 0; i < (int) tokens.size(); i += BATCH_SIZE) {
        const int cur_batch_size = std::min((int) tokens.size() - i, BATCH_SIZE);
        common_batch_clear(batch);
        LOGv("%s: Preparing a batch size of %d starting at: %d", __func__, cur_batch_size, i);

        // Shift context if current batch cannot fit into the context
        if (start_pos + i + cur_batch_size >= max_ctx - OVERFLOW_HEADROOM) {
            LOGw("%s: Current batch won't fit into context! Shifting...", __func__);
            shift_context();
        }

        // Add tokens to the batch with proper positions
        for (int j = 0; j < cur_batch_size; j++) {
            const llama_token token_id = tokens[i + j];
            const llama_pos position = start_pos + i + j;
            const bool want_logit = compute_last_logit && (i + j == tokens.size() - 1);
            common_batch_add(batch, token_id, position, {0}, want_logit);
        }

        // Decode this batch
        const int decode_result = llama_decode(context, batch);
        if (decode_result) {
            LOGe("%s: llama_decode failed w/ %d", __func__, decode_result);
            return 1;
        }
    }
    return 0;
}

static common_chat_params render_chat_params(
        const std::vector<common_chat_msg> & messages,
        const std::vector<common_chat_tool> & tools,
        bool add_generation_prompt,
        bool enable_thinking) {
    common_chat_templates_inputs inputs;
    inputs.messages = messages;
    inputs.tools = tools;
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.use_jinja = true;
    inputs.enable_thinking = enable_thinking;
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.parallel_tool_calls = false;
    return common_chat_templates_apply(g_chat_templates.get(), inputs);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_processSystemPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jsystem_prompt
) {
    // Reset long-term & short-term states
    reset_long_term_states();
    reset_short_term_states();

    // Obtain system prompt from JEnv
    const auto *system_prompt = env->GetStringUTFChars(jsystem_prompt, nullptr);
    LOG_DIAGi("%s: System prompt received. chars=%zu", __func__, std::strlen(system_prompt));
    std::string formatted_system_prompt(system_prompt);
    env->ReleaseStringUTFChars(jsystem_prompt, system_prompt);

    // Format system prompt if applicable
    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_chat_template) {
        formatted_system_prompt = chat_add_and_format(ROLE_SYSTEM, system_prompt);
    } else {
        // Keep chat history in sync even without templates
        common_chat_msg new_msg;
        new_msg.role = ROLE_SYSTEM;
        new_msg.content = system_prompt;
        chat_msgs.push_back(new_msg);
    }

    // Tokenize system prompt
    const auto system_tokens = common_tokenize(g_context, formatted_system_prompt,
                                               has_chat_template, has_chat_template);
    for (auto id: system_tokens) {
        LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
    }

    // Handle context overflow
    const int max_batch_size = llama_n_ctx(g_context) - OVERFLOW_HEADROOM;
    if ((int) system_tokens.size() > max_batch_size) {
        LOG_DIAGe("%s: System prompt too long for context! %d tokens, max: %d",
                  __func__, (int) system_tokens.size(), max_batch_size);
        return 1;
    }
    LOG_DIAGi("%s: system tokens=%d max_ctx=%d", __func__, (int) system_tokens.size(), llama_n_ctx(g_context));

    // Decode system tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, system_tokens, current_position)) {
        LOG_DIAGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update position
    system_prompt_position = current_position = (int) system_tokens.size();
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_processUserPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring juser_prompt,
        jint /*n_predict*/
) {
    // Reset short-term states
    reset_short_term_states();

    // Obtain and tokenize user prompt
    const auto *const user_prompt = env->GetStringUTFChars(juser_prompt, nullptr);
    LOG_DIAGi("%s: User prompt received. chars=%zu", __func__, std::strlen(user_prompt));
    std::string formatted_user_prompt(user_prompt);
    env->ReleaseStringUTFChars(juser_prompt, user_prompt);

    // Format user prompt if applicable
    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_chat_template) {
        formatted_user_prompt = chat_add_and_format(ROLE_USER, user_prompt);
    }

    // Decode formatted user prompts
    auto user_tokens = common_tokenize(g_context, formatted_user_prompt, has_chat_template, has_chat_template);
    for (auto id: user_tokens) {
        LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
    }

    // Ensure user prompt doesn't exceed the context size by truncating if necessary.
    const int user_prompt_size = (int) user_tokens.size();
    const int max_batch_size = llama_n_ctx(g_context) - OVERFLOW_HEADROOM;
    if (user_prompt_size > max_batch_size) {
        const int skipped_tokens = user_prompt_size - max_batch_size;
        user_tokens.resize(max_batch_size);
        LOG_DIAGw("%s: User prompt too long! Skipped %d tokens!", __func__, skipped_tokens);
    }

    // Decode user tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, user_tokens, current_position, true)) {
        LOG_DIAGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update position (use fixed max new tokens for stability)
    current_position += user_prompt_size;
    stop_generation_position = current_position + DEFAULT_MAX_NEW_TOKENS;
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_processStructuredPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jmessages_json,
        jstring jtools_json,
        jint /*n_predict*/,
        jboolean enable_thinking
) {
    reset_short_term_states();

    if (jmessages_json == nullptr) {
        LOG_DIAGe("%s: messages json is null", __func__);
        return 1;
    }

    const auto *messages_cstr = env->GetStringUTFChars(jmessages_json, nullptr);
    std::string messages_str(messages_cstr ? messages_cstr : "");
    env->ReleaseStringUTFChars(jmessages_json, messages_cstr);

    std::string tools_str;
    if (jtools_json != nullptr) {
        const auto *tools_cstr = env->GetStringUTFChars(jtools_json, nullptr);
        tools_str.assign(tools_cstr ? tools_cstr : "");
        env->ReleaseStringUTFChars(jtools_json, tools_cstr);
    }

    std::vector<common_chat_msg> messages;
    std::vector<common_chat_tool> tools;
    try {
        messages = common_chat_msgs_parse_oaicompat(json::parse(messages_str));
    } catch (const std::exception & e) {
        LOG_DIAGe("%s: failed to parse messages: %s", __func__, e.what());
        return 1;
    }

    if (!tools_str.empty()) {
        try {
            tools = common_chat_tools_parse_oaicompat(json::parse(tools_str));
        } catch (const std::exception & e) {
            LOG_DIAGe("%s: failed to parse tools: %s", __func__, e.what());
            return 1;
        }
    }

    common_chat_params params;
    try {
        params = render_chat_params(messages, tools, /* add_generation_prompt= */ true, enable_thinking);
    } catch (const std::exception & e) {
        LOG_DIAGe("%s: failed to render chat template: %s", __func__, e.what());
        return 1;
    }
    g_last_chat_params = params;

    std::string full_prompt = params.prompt;

    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());

    LOG_DIAGi("%s: messages=%zu tools=%zu full_len=%zu",
              __func__, messages.size(), tools.size(), full_prompt.size());

    auto tokenize_prompt = [&](const std::string & prompt) {
        return common_tokenize(g_context, prompt, has_chat_template, has_chat_template);
    };

    const int max_batch_size = llama_n_ctx(g_context) - OVERFLOW_HEADROOM;

    if (!full_prompt.empty()) {
        auto new_tokens = tokenize_prompt(full_prompt);
        if ((int) new_tokens.size() > max_batch_size) {
            LOG_DIAGe("%s: Prompt too long for context! %d tokens, max: %d",
                      __func__, (int) new_tokens.size(), max_batch_size);
            return 2;
        }

        // === Incremental Inference Logic ===
        // 检测 tools 是否变化（变化则需要完全重新 prefill）
        const size_t new_tools_hash = hash_string(tools_str);
        const bool tools_changed = (new_tools_hash != g_last_tools_hash);

        // 计算公共前缀长度
        size_t n_past = 0;
        if (!tools_changed && !g_last_prompt_tokens_vec.empty()) {
            n_past = get_common_prefix(g_last_prompt_tokens_vec, new_tokens);
        }

        // 如果 tools 变化或没有公共前缀，完全重新开始
        if (tools_changed || n_past == 0) {
            LOG_DIAGi("%s: Full prefill (tools_changed=%d, n_past=%zu)",
                      __func__, tools_changed, n_past);
            reset_long_term_states(true);
            n_past = 0;
        } else if (n_past < (size_t) current_position) {
            // 清除 [n_past, end) 的 KV Cache
            LOG_DIAGi("%s: Incremental prefill: clearing KV cache from %zu to %d",
                      __func__, n_past, current_position);
            llama_memory_seq_rm(llama_get_memory(g_context), 0, (llama_pos) n_past, current_position);
            current_position = (llama_pos) n_past;
        }

        // 只 Prefill 新增的 token
        const size_t n_new = new_tokens.size() - n_past;
        if (n_new > 0) {
            llama_tokens new_part(new_tokens.begin() + n_past, new_tokens.end());
            if (decode_tokens_in_batches(g_context, g_batch, new_part, current_position, true)) {
                LOG_DIAGe("%s: llama_decode() failed during prompt prefill", __func__);
                return 3;
            }
            current_position += (int) n_new;
        }

        LOG_DIAGi("%s: prefill total=%zu reused=%zu new=%zu",
                  __func__, new_tokens.size(), n_past, n_new);

        // 记录复用的 token 数
        g_last_reused_tokens = n_past;

        // 记录实际 prefill 的 token 数（不包括复用的）
        g_last_prompt_tokens = (int) n_new;

        // 更新状态
        g_last_prompt_tokens_vec = std::move(new_tokens);
        g_last_tools_hash = new_tools_hash;
    }

    stop_generation_position = current_position + DEFAULT_MAX_NEW_TOKENS;

    rebuild_sampler(&g_last_chat_params);
    common_sampler_reset(g_sampler);

    return 0;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_renderChatTemplate(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jmessages_json,
        jstring jtools_json,
        jboolean enable_thinking
) {
    if (jmessages_json == nullptr) {
        return env->NewStringUTF("");
    }

    const auto *messages_cstr = env->GetStringUTFChars(jmessages_json, nullptr);
    std::string messages_str(messages_cstr ? messages_cstr : "");
    env->ReleaseStringUTFChars(jmessages_json, messages_cstr);

    std::string tools_str;
    if (jtools_json != nullptr) {
        const auto *tools_cstr = env->GetStringUTFChars(jtools_json, nullptr);
        tools_str.assign(tools_cstr ? tools_cstr : "");
        env->ReleaseStringUTFChars(jtools_json, tools_cstr);
    }

    std::vector<common_chat_msg> messages;
    std::vector<common_chat_tool> tools;
    try {
        messages = common_chat_msgs_parse_oaicompat(json::parse(messages_str));
        if (!tools_str.empty()) {
            tools = common_chat_tools_parse_oaicompat(json::parse(tools_str));
        }
    } catch (const std::exception & e) {
        LOGe("%s: failed to parse inputs: %s", __func__, e.what());
        return env->NewStringUTF("");
    }

    try {
        auto params = render_chat_params(messages, tools, /* add_generation_prompt= */ true, enable_thinking);
        return env->NewStringUTF(params.prompt.c_str());
    } catch (const std::exception & e) {
        LOGe("%s: failed to render template: %s", __func__, e.what());
        return env->NewStringUTF("");
    }
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_getLastPromptTokenCount(
        JNIEnv * /*env*/,
        jobject /*unused*/) {
    return g_last_prompt_tokens;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_getLastParsedAssistantMessage(
        JNIEnv *env,
        jobject /*unused*/) {
    if (g_last_assistant_raw.empty()) {
        return env->NewStringUTF("");
    }

    try {
        common_chat_parser_params parser_params(g_last_chat_params);
        parser_params.parse_tool_calls = true;
        auto parsed = common_chat_parse(g_last_assistant_raw, /* is_partial= */ false, parser_params);
        parsed.role = ROLE_ASSISTANT;
        auto json_msg = parsed.to_json_oaicompat(/* concat_typed_text= */ true);
        g_last_assistant_parsed_json = json_msg.dump();
        return env->NewStringUTF(g_last_assistant_parsed_json.c_str());
    } catch (const std::exception & e) {
        LOGw("%s: failed to parse assistant message: %s", __func__, e.what());
        json fallback = json::object();
        fallback["role"] = ROLE_ASSISTANT;
        fallback["content"] = g_last_assistant_raw;
        g_last_assistant_parsed_json = fallback.dump();
        return env->NewStringUTF(g_last_assistant_parsed_json.c_str());
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_resetContext(
        JNIEnv * /*env*/,
        jobject /*unused*/) {
    reset_long_term_states(true);
    reset_short_term_states();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_updateSampling(
        JNIEnv * /*env*/,
        jobject /*unused*/,
        jfloat temperature,
        jfloat top_p,
        jint top_k,
        jfloat min_p,
        jfloat /*repeat_penalty*/) {
    init_sampling_defaults();
    if (temperature > 0.0f) {
        g_sampling_params.temp = temperature;
    }
    if (top_p >= 0.0f) {
        g_sampling_params.top_p = top_p;
    }
    if (top_k > 0) {
        g_sampling_params.top_k = top_k;
    }
    if (min_p >= 0.0f) {
        g_sampling_params.min_p = min_p;
    }
    // repeat_penalty is fixed at DEFAULT_REPEAT_PENALTY (1.1f) for stability
    rebuild_sampler();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_updateChatTemplate(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jchat_template) {
    std::string template_str;
    if (jchat_template != nullptr) {
        const auto *tmpl_cstr = env->GetStringUTFChars(jchat_template, nullptr);
        template_str.assign(tmpl_cstr ? tmpl_cstr : "");
        env->ReleaseStringUTFChars(jchat_template, tmpl_cstr);
    }
    g_chat_template_override = template_str;

    if (g_model != nullptr) {
        g_chat_templates = common_chat_templates_init(g_model, g_chat_template_override.c_str());
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_setContextLength(
        JNIEnv * /*env*/,
        jobject /*unused*/,
        jint /*n_ctx*/) {
    // Context length is fixed at DEFAULT_CONTEXT_SIZE (4096) for stability.
    // Dynamic context length was causing issues, so we ignore the parameter.
    LOG_DIAGi("%s: Context length fixed at %d (ignoring requested value)", __func__, DEFAULT_CONTEXT_SIZE);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_setGpuLayers(
        JNIEnv * /*env*/,
        jobject /*unused*/,
        jint n_gpu_layers) {
    if (g_model != nullptr) {
        LOG_DIAGw("%s: model already loaded; new n_gpu_layers will apply after reload", __func__);
    }
    g_n_gpu_layers = n_gpu_layers;
    LOG_DIAGi("%s: n_gpu_layers set to %d", __func__, n_gpu_layers);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_getGpuLayers(
        JNIEnv * /*env*/,
        jobject /*unused*/) {
    return g_n_gpu_layers;
}

static bool is_valid_utf8(const char *string) {
    if (!string) { return true; }

    const auto *bytes = (const unsigned char *) string;
    int num;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }
    return true;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_generateNextToken(
        JNIEnv *env,
        jobject /*unused*/
) {
    // Infinite text generation via context shifting
    if (current_position >= llama_n_ctx(g_context) - OVERFLOW_HEADROOM) {
        LOG_DIAGw("%s: Context full! Shifting...", __func__);
        shift_context();
    }

    // Stop if reaching the marked position
    if (current_position >= stop_generation_position) {
        LOG_DIAGw("%s: STOP: hitting stop position: %d", __func__, stop_generation_position);
        g_last_assistant_raw = assistant_ss.str();
        return nullptr;
    }

    // Sample next token
    const auto new_token_id = common_sampler_sample(g_sampler, g_context, -1);
    common_sampler_accept(g_sampler, new_token_id, true);

    // Populate the batch with new token, then decode
    common_batch_clear(g_batch);
    common_batch_add(g_batch, new_token_id, current_position, {0}, true);
    if (llama_decode(g_context, g_batch) != 0) {
        LOG_DIAGe("%s: llama_decode() failed for generated token", __func__);
        return nullptr;
    }

    // Update position
    current_position++;

    // Stop if next token is EOG
    if (llama_vocab_is_eog(llama_model_get_vocab(g_model), new_token_id)) {
        LOGd("id: %d,\tIS EOG!\nSTOP.", new_token_id);
        chat_add_and_format(ROLE_ASSISTANT, assistant_ss.str());
        g_last_assistant_raw = assistant_ss.str();
        return nullptr;
    }

    // If not EOG, convert to text
    auto new_token_chars = common_token_to_piece(g_context, new_token_id);
    cached_token_chars += new_token_chars;

    // Create and return a valid UTF-8 Java string
    jstring result = nullptr;
    if (is_valid_utf8(cached_token_chars.c_str())) {
        result = env->NewStringUTF(cached_token_chars.c_str());
        LOGv("id: %d,\tcached: `%s`,\tnew: `%s`", new_token_id, cached_token_chars.c_str(), new_token_chars.c_str());

        assistant_ss << cached_token_chars;
        cached_token_chars.clear();
    } else {
        LOGv("id: %d,\tappend to cache", new_token_id);
        result = env->NewStringUTF("");
    }
    return result;
}


extern "C"
JNIEXPORT void JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_unload(JNIEnv * /*unused*/, jobject /*unused*/) {
    // Reset long-term & short-term states
    reset_long_term_states();
    reset_short_term_states();

    // Free up resources
    common_sampler_free(g_sampler);
    g_chat_templates.reset();
    llama_batch_free(g_batch);
    llama_free(g_context);
    llama_model_free(g_model);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_shutdown(JNIEnv *, jobject /*unused*/) {
    llama_backend_free();
}

// === Stem Context Management ===

extern "C"
JNIEXPORT jint JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_warmupStem(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jmessages_json,
        jstring jtools_json
) {
    if (jmessages_json == nullptr) {
        LOG_DIAGe("%s: messages json is null", __func__);
        return 1;
    }

    const auto *messages_cstr = env->GetStringUTFChars(jmessages_json, nullptr);
    std::string messages_str(messages_cstr ? messages_cstr : "");
    env->ReleaseStringUTFChars(jmessages_json, messages_cstr);

    std::string tools_str;
    if (jtools_json != nullptr) {
        const auto *tools_cstr = env->GetStringUTFChars(jtools_json, nullptr);
        tools_str.assign(tools_cstr ? tools_cstr : "");
        env->ReleaseStringUTFChars(jtools_json, tools_cstr);
    }

    std::vector<common_chat_msg> messages;
    std::vector<common_chat_tool> tools;
    try {
        messages = common_chat_msgs_parse_oaicompat(json::parse(messages_str));
    } catch (const std::exception & e) {
        LOG_DIAGe("%s: failed to parse messages: %s", __func__, e.what());
        return 1;
    }

    if (!tools_str.empty()) {
        try {
            tools = common_chat_tools_parse_oaicompat(json::parse(tools_str));
        } catch (const std::exception & e) {
            LOG_DIAGe("%s: failed to parse tools: %s", __func__, e.what());
            return 1;
        }
    }

    common_chat_params params;
    try {
        params = render_chat_params(messages, tools, /* add_generation_prompt= */ false, /* enable_thinking= */ false);
    } catch (const std::exception & e) {
        LOG_DIAGe("%s: failed to render chat template: %s", __func__, e.what());
        return 1;
    }

    std::string stem_prompt = params.prompt;
    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());

    // 完全重置状态
    reset_long_term_states(true);
    reset_short_term_states();

    auto stem_tokens = common_tokenize(g_context, stem_prompt, has_chat_template, has_chat_template);
    const int max_batch_size = llama_n_ctx(g_context) - OVERFLOW_HEADROOM;

    if ((int) stem_tokens.size() > max_batch_size) {
        LOG_DIAGe("%s: Stem too long for context! %d tokens, max: %d",
                  __func__, (int) stem_tokens.size(), max_batch_size);
        return 2;
    }

    // Prefill stem tokens
    if (decode_tokens_in_batches(g_context, g_batch, stem_tokens, 0, false)) {
        LOG_DIAGe("%s: llama_decode() failed during stem prefill", __func__);
        return 3;
    }

    // 记录 stem 状态
    g_stem_position = (llama_pos) stem_tokens.size();
    current_position = g_stem_position;
    g_last_prompt_tokens_vec = std::move(stem_tokens);
    g_last_tools_hash = hash_string(tools_str);
    g_stem_warmed = true;

    LOG_DIAGi("%s: Stem warmed up with %d tokens", __func__, g_stem_position);
    return 0;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_pruneToStem(
        JNIEnv * /*env*/,
        jobject /*unused*/
) {
    if (!g_stem_warmed || g_stem_position == 0) {
        LOG_DIAGw("%s: No stem to prune to (stem_warmed=%d, stem_pos=%d)",
                  __func__, g_stem_warmed, g_stem_position);
        return;
    }

    if (current_position > g_stem_position) {
        LOG_DIAGi("%s: Pruning KV cache from %d to %d",
                  __func__, g_stem_position, current_position);
        llama_memory_seq_rm(llama_get_memory(g_context), 0, g_stem_position, current_position);
        current_position = g_stem_position;

        // 截断 token 历史到 stem 位置
        if (g_last_prompt_tokens_vec.size() > (size_t) g_stem_position) {
            g_last_prompt_tokens_vec.resize(g_stem_position);
        }
    }

    // 重置短期状态
    reset_short_term_states();
    LOG_DIAGi("%s: Pruned to stem position %d", __func__, g_stem_position);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_getStemPosition(
        JNIEnv * /*env*/,
        jobject /*unused*/
) {
    return g_stem_position;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_getLastReusedTokens(
        JNIEnv * /*env*/,
        jobject /*unused*/
) {
    return (jint) g_last_reused_tokens;
}
