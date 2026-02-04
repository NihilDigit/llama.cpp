#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <cmath>
#include <string>
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
constexpr int   N_THREADS_MIN           = 2;
constexpr int   N_THREADS_MAX           = 4;
constexpr int   N_THREADS_HEADROOM      = 2;

constexpr int   DEFAULT_CONTEXT_SIZE    = 8192;
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
static common_params_sampling              g_sampling_params;
static bool                               g_sampling_params_initialized = false;
static common_chat_params                  g_last_chat_params;
static int                                g_last_prompt_tokens = 0;
static std::string                        g_last_assistant_raw;
static std::string                        g_last_assistant_parsed_json;
static std::string                        g_chat_template_override;
// In-memory cache for system-prompt prefill state
static std::string                        cached_system_prompt;
static std::vector<uint8_t>               cached_system_state;
static llama_pos                          cached_system_prompt_position = 0;
static bool                               cached_system_state_valid = false;

extern "C"
JNIEXPORT void JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_init(JNIEnv *env, jobject /*unused*/, jstring nativeLibDir) {
    // Set llama log handler to Android
    llama_log_set(aichat_android_log_callback, nullptr);

    // Loading all CPU backend variants
    const auto *path_to_backend = env->GetStringUTFChars(nativeLibDir, 0);
    LOGi("Loading backends from %s", path_to_backend);
    ggml_backend_load_all_from_path(path_to_backend);
    env->ReleaseStringUTFChars(nativeLibDir, path_to_backend);

    // Initialize backends
    llama_backend_init();

    // Log all available backends
    LOGi("Available backends:");
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto *reg = ggml_backend_reg_get(i);
        LOGi("  [%zu] %s", i, ggml_backend_reg_name(reg));
    }

    LOGi("Backend initiated; Log handler set. Default n_gpu_layers=%d", g_n_gpu_layers);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_load(JNIEnv *env, jobject, jstring jmodel_path) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = g_n_gpu_layers;
    LOGi("%s: n_gpu_layers = %d", __func__, g_n_gpu_layers);

    const auto *model_path = env->GetStringUTFChars(jmodel_path, 0);
    LOGd("%s: Loading model from: \n%s\n", __func__, model_path);

    auto *model = llama_model_load_from_file(model_path, model_params);
    env->ReleaseStringUTFChars(jmodel_path, model_path);
    if (!model) {
        return 1;
    }
    g_model = model;
    cached_system_prompt.clear();
    cached_system_state.clear();
    cached_system_prompt_position = 0;
    cached_system_state_valid = false;
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
    LOGi("%s: Using %d threads", __func__, n_threads);

    // Context parameters setup
    llama_context_params ctx_params = llama_context_default_params();
    const int trained_context_size = llama_model_n_ctx_train(model);
    if (n_ctx > trained_context_size) {
        LOGw("%s: Model was trained with only %d context size! Enforcing %d context size...",
             __func__, trained_context_size, n_ctx);
    }
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = BATCH_SIZE;
    ctx_params.n_ubatch = BATCH_SIZE;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    auto *context = llama_init_from_model(g_model, ctx_params);
    if (context == nullptr) {
        LOGe("%s: llama_new_context_with_model() returned null)", __func__);
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
    g_sampling_params.penalty_repeat = 1.1f;
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
    LOGi("n_ctx = %d", n_ctx);

    int i, j;
    int nri;
    for (nri = 0; nri < nr; nri++) {
        LOGi("Benchmark prompt processing (pp = %d)", pp);

        common_batch_clear(g_batch);

        const int n_tokens = pp;
        for (i = 0; i < n_tokens; i++) {
            common_batch_add(g_batch, 0, i, {0}, false);
        }

        g_batch.logits[g_batch.n_tokens - 1] = true;
        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp_start = ggml_time_us();
        if (llama_decode(context, g_batch) != 0) {
            LOGe("llama_decode() failed during prompt processing");
        }
        const auto t_pp_end = ggml_time_us();

        // bench text generation

        LOGi("Benchmark text generation (tg = %d)", tg);

        llama_memory_clear(llama_get_memory(context), false);
        const auto t_tg_start = ggml_time_us();
        for (i = 0; i < tg; i++) {
            common_batch_clear(g_batch);
            for (j = 0; j < pl; j++) {
                common_batch_add(g_batch, 0, i, {j}, true);
            }

            if (llama_decode(context, g_batch) != 0) {
                LOGe("llama_decode() failed during text generation");
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

        LOGi("pp %f t/s, tg %f t/s", speed_pp, speed_tg);
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
    LOGi("%s: Discarding %d tokens", __func__, n_discard);
    llama_memory_seq_rm(llama_get_memory(g_context), 0, system_prompt_position, system_prompt_position + n_discard);
    llama_memory_seq_add(llama_get_memory(g_context), 0, system_prompt_position + n_discard, current_position, -n_discard);
    current_position -= n_discard;
    LOGi("%s: Context shifting done! Current position: %d", __func__, current_position);
}

static std::string chat_add_and_format(const std::string &role, const std::string &content) {
    common_chat_msg new_msg;
    new_msg.role = role;
    new_msg.content = content;
    auto formatted = common_chat_format_single(
            g_chat_templates.get(), chat_msgs, new_msg, role == ROLE_USER, /* use_jinja */ false);
    chat_msgs.push_back(new_msg);
    LOGi("%s: Formatted and added %s message: \n%s\n", __func__, role.c_str(), formatted.c_str());
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
    LOGd("%s: System prompt received: \n%s", __func__, system_prompt);
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

    // Fast path: reuse cached system prompt state
    if (cached_system_state_valid && cached_system_prompt == formatted_system_prompt) {
        const size_t restored = llama_state_set_data(
                g_context, cached_system_state.data(), cached_system_state.size());
        if (restored == cached_system_state.size()) {
            system_prompt_position = current_position = cached_system_prompt_position;
            LOGi("%s: Restored cached system prompt state (%d tokens)",
                 __func__, (int) cached_system_prompt_position);
            return 0;
        } else {
            LOGw("%s: Failed to restore cached state (expected %zu, got %zu). Recomputing.",
                 __func__, cached_system_state.size(), restored);
            cached_system_state_valid = false;
        }
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
        LOGe("%s: System prompt too long for context! %d tokens, max: %d",
             __func__, (int) system_tokens.size(), max_batch_size);
        return 1;
    }

    // Decode system tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, system_tokens, current_position)) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update position
    system_prompt_position = current_position = (int) system_tokens.size();

    // Cache system prompt state in memory for reuse
    cached_system_prompt = formatted_system_prompt;
    cached_system_prompt_position = system_prompt_position;
    cached_system_state.resize(llama_state_get_size(g_context));
    const size_t written = llama_state_get_data(
            g_context, cached_system_state.data(), cached_system_state.size());
    if (written != cached_system_state.size()) {
        LOGw("%s: Prompt cache write size mismatch (expected %zu, got %zu). Disabling cache.",
             __func__, cached_system_state.size(), written);
        cached_system_state_valid = false;
    } else {
        cached_system_state_valid = true;
    }
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_processUserPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring juser_prompt,
        jint n_predict
) {
    // Reset short-term states
    reset_short_term_states();

    // Obtain and tokenize user prompt
    const auto *const user_prompt = env->GetStringUTFChars(juser_prompt, nullptr);
    LOGd("%s: User prompt received: \n%s", __func__, user_prompt);
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
        LOGw("%s: User prompt too long! Skipped %d tokens!", __func__, skipped_tokens);
    }

    // Decode user tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, user_tokens, current_position, true)) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update position
    current_position += user_prompt_size;
    stop_generation_position = current_position + user_prompt_size + n_predict;
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_processStructuredPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jmessages_json,
        jstring jtools_json,
        jint n_predict,
        jboolean enable_thinking
) {
    reset_short_term_states();

    if (jmessages_json == nullptr) {
        LOGe("%s: messages json is null", __func__);
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
        LOGe("%s: failed to parse messages: %s", __func__, e.what());
        return 1;
    }

    if (!tools_str.empty()) {
        try {
            tools = common_chat_tools_parse_oaicompat(json::parse(tools_str));
        } catch (const std::exception & e) {
            LOGe("%s: failed to parse tools: %s", __func__, e.what());
            return 1;
        }
    }

    common_chat_params params;
    try {
        params = render_chat_params(messages, tools, /* add_generation_prompt= */ true, enable_thinking);
    } catch (const std::exception & e) {
        LOGe("%s: failed to render chat template: %s", __func__, e.what());
        return 1;
    }
    g_last_chat_params = params;

    std::string full_prompt = params.prompt;
    std::string stem_prompt;

    // Build stem prompt from leading system messages + tools
    std::vector<common_chat_msg> stem_messages;
    for (const auto & msg : messages) {
        if (msg.role == ROLE_SYSTEM) {
            stem_messages.push_back(msg);
        } else {
            break;
        }
    }
    if (!stem_messages.empty() || !tools.empty()) {
        try {
            auto stem_params = render_chat_params(stem_messages, tools, /* add_generation_prompt= */ false, enable_thinking);
            stem_prompt = stem_params.prompt;
        } catch (const std::exception & e) {
            LOGw("%s: failed to render stem prompt: %s", __func__, e.what());
            stem_prompt.clear();
        }
    }

    reset_long_term_states(true);

    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    const bool prefix_match = !stem_prompt.empty() && full_prompt.rfind(stem_prompt, 0) == 0;
    bool restored_cache = false;

    if (prefix_match && cached_system_state_valid && cached_system_prompt == stem_prompt) {
        const size_t restored = llama_state_set_data(
                g_context, cached_system_state.data(), cached_system_state.size());
        if (restored == cached_system_state.size()) {
            system_prompt_position = current_position = cached_system_prompt_position;
            restored_cache = true;
            LOGi("%s: Restored cached stem (%d tokens)", __func__, (int) cached_system_prompt_position);
        } else {
            LOGw("%s: Cached stem restore mismatch (expected %zu, got %zu)",
                 __func__, cached_system_state.size(), restored);
            cached_system_state_valid = false;
        }
    }

    auto tokenize_prompt = [&](const std::string & prompt) {
        return common_tokenize(g_context, prompt, has_chat_template, has_chat_template);
    };

    const int max_batch_size = llama_n_ctx(g_context) - OVERFLOW_HEADROOM;

    if (!restored_cache) {
        const std::string & prompt_to_decode = prefix_match ? stem_prompt : full_prompt;
        if (!prompt_to_decode.empty()) {
            auto tokens = tokenize_prompt(prompt_to_decode);
            if ((int) tokens.size() > max_batch_size) {
                LOGe("%s: Prompt too long for context! %d tokens, max: %d",
                     __func__, (int) tokens.size(), max_batch_size);
                return 2;
            }
            const bool want_logits = !prefix_match;
            if (decode_tokens_in_batches(g_context, g_batch, tokens, current_position, want_logits)) {
                LOGe("%s: llama_decode() failed during prompt prefill", __func__);
                return 3;
            }
            current_position += (int) tokens.size();
            system_prompt_position = prefix_match ? current_position : 0;

            if (prefix_match) {
                cached_system_prompt = stem_prompt;
                cached_system_prompt_position = system_prompt_position;
                cached_system_state.resize(llama_state_get_size(g_context));
                const size_t written = llama_state_get_data(
                        g_context, cached_system_state.data(), cached_system_state.size());
                cached_system_state_valid = (written == cached_system_state.size());
                if (!cached_system_state_valid) {
                    LOGw("%s: Prompt cache write size mismatch (expected %zu, got %zu)",
                         __func__, cached_system_state.size(), written);
                }
            }
        }
    }

    if (prefix_match) {
        std::string suffix = full_prompt.substr(stem_prompt.size());
        if (!suffix.empty()) {
            auto tokens = tokenize_prompt(suffix);
            if ((int) tokens.size() > max_batch_size) {
                LOGe("%s: Prompt suffix too long for context! %d tokens, max: %d",
                     __func__, (int) tokens.size(), max_batch_size);
                return 2;
            }
            if (decode_tokens_in_batches(g_context, g_batch, tokens, current_position, true)) {
                LOGe("%s: llama_decode() failed during prompt suffix", __func__);
                return 3;
            }
            current_position += (int) tokens.size();
        }
    }

    g_last_prompt_tokens = current_position;
    stop_generation_position = current_position + n_predict;

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
        jfloat repeat_penalty) {
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
    if (repeat_penalty > 0.0f) {
        g_sampling_params.penalty_repeat = repeat_penalty;
    }
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
        jint n_ctx) {
    if (n_ctx > 0) {
        if (g_context != nullptr) {
            LOGw("%s: context already initialized; new n_ctx will apply after reload", __func__);
        }
        g_context_size = n_ctx;
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_nihildigit_lightwayllama_internal_InferenceEngineImpl_setGpuLayers(
        JNIEnv * /*env*/,
        jobject /*unused*/,
        jint n_gpu_layers) {
    if (g_model != nullptr) {
        LOGw("%s: model already loaded; new n_gpu_layers will apply after reload", __func__);
    }
    g_n_gpu_layers = n_gpu_layers;
    LOGi("%s: n_gpu_layers set to %d", __func__, n_gpu_layers);
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
        LOGw("%s: Context full! Shifting...", __func__);
        shift_context();
    }

    // Stop if reaching the marked position
    if (current_position >= stop_generation_position) {
        LOGw("%s: STOP: hitting stop position: %d", __func__, stop_generation_position);
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
        LOGe("%s: llama_decode() failed for generated token", __func__);
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
    cached_system_prompt.clear();
    cached_system_state.clear();
    cached_system_prompt_position = 0;
    cached_system_state_valid = false;

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
