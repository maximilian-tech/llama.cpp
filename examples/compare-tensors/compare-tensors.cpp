#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "llama-impl.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

constexpr size_t HISTOGRAM_BUCKETS = 150;
constexpr double HISTOGRAM_RANGE = 0.03;

struct quantize_stats_params {
    std::string model_a = DEFAULT_MODEL_PATH;
    std::string model_b;
    bool print_histogram = false;
    bool per_layer_stats = false;
    std::vector<std::string> include_layers;
};

struct error_stats {
    size_t num_samples = 0;
    double total_error = 0;
    double max_error = 0;
    uint64_t error_histogram[HISTOGRAM_BUCKETS] = {0};
};

static void quantize_stats_print_usage(char **argv) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "  options:\n");
    fprintf(stderr, "  --model-a FNAME       path to reference model (default: %s)\n", DEFAULT_MODEL_PATH);
    fprintf(stderr, "  --model-b FNAME       path to model to compare (required)\n");
    fprintf(stderr, "  --histogram           print error histogram\n");
    fprintf(stderr, "  --per-layer-stats     print stats per layer\n");
    fprintf(stderr, "  -l LAYER              only compare layers matching regex\n\n");
}

static bool layer_included(const quantize_stats_params &params, const std::string &layer) {
    if (params.include_layers.empty()) return true;
    for (const auto &pattern : params.include_layers)
        if (std::regex_search(layer, std::regex(pattern))) return true;
    return false;
}

void update_error_stats(const float *input, const float *output, size_t size, error_stats &stats, error_stats &stats_global) {
    for (size_t i = 0; i < size; i++) {
        double diff = input[i] - output[i];
        stats.total_error += diff * diff;
        stats.max_error = std::max(stats.max_error, std::abs(diff));
        size_t bucket = std::min(static_cast<size_t>(std::floor(std::abs(diff) / HISTOGRAM_RANGE * HISTOGRAM_BUCKETS)), HISTOGRAM_BUCKETS - 1);
        stats.error_histogram[bucket]++;
        
        stats_global.total_error += diff * diff;
        stats_global.max_error = std::max(stats_global.max_error, std::abs(diff));
        stats_global.error_histogram[bucket]++;
    }
    stats.num_samples += size;
    stats_global.num_samples += size;
}

double find_quantile(const error_stats &stats, double quantile) {
    double sum = std::accumulate(std::begin(stats.error_histogram), std::end(stats.error_histogram), 0.0);
    double accum = 0;
    for (size_t i = 0; i < HISTOGRAM_BUCKETS; i++) {
        accum += stats.error_histogram[i];
        if (accum >= sum * quantile)
            return (i + 1) * HISTOGRAM_RANGE / HISTOGRAM_BUCKETS;
    }
    return INFINITY;
}

void print_error_stats(const char *name, const error_stats &stats, bool print_histogram) {
    double rmse = std::sqrt(stats.total_error / (double)stats.num_samples);
    double median = find_quantile(stats, 0.5);
    double pct95 = find_quantile(stats, 0.95);
    printf("%-50s: rmse %.8f, maxerr %.8f, 95pct<%.4f, median<%.4f\n",
           name, rmse, stats.max_error, pct95, median);
    if (print_histogram) {
        printf("Error distribution:\n");
        for (size_t i = 0; i < HISTOGRAM_BUCKETS; i++) {
            double lower = i * HISTOGRAM_RANGE / HISTOGRAM_BUCKETS;
            double upper = (i + 1) * HISTOGRAM_RANGE / HISTOGRAM_BUCKETS;
            if (i == HISTOGRAM_BUCKETS - 1) upper = INFINITY;
            printf("[%3.4f, %3.4f): %11" PRIu64 "\n", lower, upper, stats.error_histogram[i]);
        }
    }
}

int main(int argc, char **argv) {
    quantize_stats_params params;

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--model-a") params.model_a = argv[++i];
        else if (arg == "--model-b") params.model_b = argv[++i];
        else if (arg == "--histogram") params.print_histogram = true;
        else if (arg == "--per-layer-stats") params.per_layer_stats = true;
        else if (arg == "-l") params.include_layers.emplace_back(argv[++i]);
        else { quantize_stats_print_usage(argv); return 1; }
    }
    if (params.model_b.empty()) { quantize_stats_print_usage(argv); return 1; }

    llama_model *model_a = llama_load_model_from_file(params.model_a.c_str(), llama_model_default_params());
    llama_model *model_b = llama_load_model_from_file(params.model_b.c_str(), llama_model_default_params());
    llama_context *ctx_a = llama_new_context_with_model(model_a, llama_context_default_params());
    llama_context *ctx_b = llama_new_context_with_model(model_b, llama_context_default_params());

    auto tensors_a = llama_internal_get_tensor_map(ctx_a);
    auto tensors_b = llama_internal_get_tensor_map(ctx_b);
    error_stats stats_global;
    for (const auto &[name, ta] : tensors_a) {
        if (!layer_included(params, name)) continue;
        auto it_b = std::find_if(tensors_b.begin(), tensors_b.end(),
                         [&name](const auto &pair) { return pair.first == name; });

        if (it_b == tensors_b.end()) {
            fprintf(stderr, "Missing %s in model B\n", name.c_str());
            continue;
        }

        auto tb = it_b->second;
        
        assert(ggml_nelements(ta) == ggml_nelements(tb));
        
        std::vector<float> ta_f32_buf(ggml_nelements(ta));
        std::vector<float> tb_f32_buf(ggml_nelements(tb));
        
        if(ta->type == GGML_TYPE_F16)
        {
            GGML_ASSERT(ta->type == GGML_TYPE_F16);
            GGML_ASSERT(tb->type == GGML_TYPE_F16);
            ggml_fp16_t *ta_data = (ggml_fp16_t *)ta->data;
            ggml_fp16_t *tb_data = (ggml_fp16_t *)tb->data;

            for (int i = 0; i < ggml_nelements(ta); ++i)
            {
                ta_f32_buf[i] = ggml_fp16_to_fp32(ta_data[i]);
                tb_f32_buf[i] = ggml_fp16_to_fp32(tb_data[i]);
            }
        }
        else if(ta->type == GGML_TYPE_F32)
        {
            float *ta_data = (float *)ta->data;
            float *tb_data = (float *)tb->data;
            for (int i = 0; i < ggml_nelements(ta); ++i)
            {
                ta_f32_buf[i] = ta_data[i];
                tb_f32_buf[i] = tb_data[i];
            }
        }
        else{
            assert(false && "Unknown type");
        }
        
        error_stats stats;
        update_error_stats(ta_f32_buf.data(), tb_f32_buf.data(), ggml_nelements(ta), stats, stats_global);
        if (params.per_layer_stats)
        {
            print_error_stats(name.c_str(), stats, params.print_histogram);
        }
        
    }
    print_error_stats("global", stats_global, params.print_histogram);

    llama_free(ctx_a);
    llama_free(ctx_b);
    llama_free_model(model_a);
    llama_free_model(model_b);
    return 0;
}
