/* Plain C exports over the motion-bricks core for the threaded (wasm32-wasip1-threads) build. The page calls
 * these through a JS host that provides one shared memory and wasi:thread-spawn; no component model is involved. */
#include <motionbricks/inference.h>
#include <motionbricks/motionbricks.h>

#include "ggml-backend.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define JOINTS 34U
#define FRAME_VALUES (3U + JOINTS * 4U)
#define GLOBAL_ROOT_VALUES 40U
#define LOCAL_ROOT_VALUES 32U
#define POSE_VALUES 2424U

int mb_memory_file_begin(const char * name, uint64_t size);
int mb_memory_file_write(const char * name, uint64_t offset, const void * data, uint64_t count);
void mb_memory_files_clear(void);

static mb_model * g_model = NULL;
static char g_error[512] = {0};

/* Per-graph profile, recorded through the link wrap in build-threads.sh. */
#define PROFILE_GRAPHS 32U
typedef struct { uint32_t nodes; double ms; } graph_sample;
static graph_sample g_graphs[PROFILE_GRAPHS];
static uint32_t g_graph_count = 0;
static double g_infer_ms = 0.0;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

enum ggml_status __real_ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * graph);
enum ggml_status __wrap_ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * graph) {
    const double start = now_ms();
    const enum ggml_status status = __real_ggml_backend_graph_compute(backend, graph);
    if (g_graph_count < PROFILE_GRAPHS) {
        g_graphs[g_graph_count].nodes = (uint32_t)ggml_graph_n_nodes(graph);
        g_graphs[g_graph_count].ms = now_ms() - start;
        g_graph_count++;
    }
    return status;
}

__attribute__((export_name("mb_alloc"))) void * mb_alloc(uint32_t bytes) { return malloc(bytes ? bytes : 1U); }
__attribute__((export_name("mb_free"))) void mb_free(void * p) { free(p); }
__attribute__((export_name("mb_error"))) const char * mb_error(void) { return g_error; }

/* Stages one GGUF file of the model bundle; the bytes are copied in by the host. */
__attribute__((export_name("mb_stage_begin")))
int32_t mb_stage_begin(const char * name, uint64_t size) { return mb_memory_file_begin(name, size) == 0 ? 0 : -1; }
__attribute__((export_name("mb_stage_write")))
int32_t mb_stage_write(const char * name, uint64_t offset, const void * data, uint64_t count) {
    return mb_memory_file_write(name, offset, data, count) == 0 ? 0 : -1;
}

/* The host sets the inference thread count: WASI cannot report hardware concurrency, so the core would use one. */
__attribute__((export_name("mb_load")))
int32_t mb_load(uint32_t threads) {
    mb_runtime_options * options = NULL;
    if (mb_runtime_options_create(&options, g_error, sizeof g_error) != MB_OK) return -1;
    if (mb_runtime_options_set_threads(options, threads, g_error, sizeof g_error) != MB_OK) {
        mb_runtime_options_free(options);
        return -1;
    }
    mb_status status = mb_model_load("/model", options, &g_model, g_error, sizeof g_error);
    mb_runtime_options_free(options);
    mb_memory_files_clear();
    return status == MB_OK ? 0 : -1;
}

__attribute__((export_name("mb_joint_parents")))
int32_t mb_joint_parents(int32_t * out) {
    for (uint32_t joint = 0; joint < JOINTS; ++joint) {
        if (mb_model_get_joint_parent(g_model, joint, &out[joint], g_error, sizeof g_error) != MB_OK) return -1;
    }
    return 0;
}

__attribute__((export_name("mb_neutral_positions")))
int32_t mb_neutral_positions(float * out) {
    for (uint32_t joint = 0; joint < JOINTS; ++joint) {
        if (mb_model_get_neutral_joint_position(g_model, joint, &out[3 * joint], &out[3 * joint + 1],
                                                &out[3 * joint + 2], g_error, sizeof g_error) != MB_OK) return -1;
    }
    return 0;
}

/* Standing source (same layout and height lift as the component glue). */
__attribute__((export_name("mb_standing")))
int32_t mb_standing(float * global_root, float * local_root, float * pose) {
    float x[JOINTS], y[JOINTS], z[JOINTS];
    for (uint32_t joint = 0; joint < JOINTS; ++joint) {
        if (mb_model_get_neutral_joint_position(g_model, joint, &x[joint], &y[joint], &z[joint], g_error,
                                                sizeof g_error) != MB_OK) return -1;
    }
    float lowest = y[0];
    for (uint32_t joint = 1; joint < JOINTS; ++joint) if (y[joint] < lowest) lowest = y[joint];
    const float lift = -lowest;
    memset(local_root, 0, LOCAL_ROOT_VALUES * sizeof(float));
    for (uint32_t slot = 0; slot < 8U; ++slot) {
        float * root = global_root + 5U * slot;
        root[0] = 0.0f; root[1] = y[0] + lift; root[2] = 0.0f; root[3] = 1.0f; root[4] = 0.0f;
        float * row = pose + 303U * slot;
        for (uint32_t joint = 1; joint < JOINTS; ++joint) {
            row[3U * (joint - 1U)] = x[joint] - x[0];
            row[3U * (joint - 1U) + 1U] = y[joint] + lift;
            row[3U * (joint - 1U) + 2U] = z[joint] - z[0];
        }
        for (uint32_t joint = 0; joint < JOINTS; ++joint) {
            float * o = row + 99U + 6U * joint;
            o[0] = 1; o[1] = 0; o[2] = 0; o[3] = 0; o[4] = 1; o[5] = 0;
        }
    }
    return 0;
}

static int32_t set_field(mb_inference_request * request, uint32_t field, const float * data, uint64_t count) {
    return mb_inference_request_set_features(request, field, data, count, g_error, sizeof g_error) == MB_OK ? 0 : -1;
}

/* One inference. Returns 0 and writes the frame count and a malloc'd frame-major buffer (139 floats per frame). */
__attribute__((export_name("mb_infer")))
int32_t mb_infer(const float * global_root, const float * local_root, const float * pose, uint64_t seed,
                 int32_t free_target, uint32_t * frames_out, float ** values_out) {
    mb_inference_request * request = NULL;
    if (mb_inference_request_create(&request, g_error, sizeof g_error) != MB_OK) return -1;
    if (set_field(request, MB_INFERENCE_GLOBAL_ROOT, global_root, GLOBAL_ROOT_VALUES) ||
        set_field(request, MB_INFERENCE_LOCAL_ROOT, local_root, LOCAL_ROOT_VALUES) ||
        set_field(request, MB_INFERENCE_POSE, pose, POSE_VALUES)) {
        mb_inference_request_free(request);
        return -1;
    }
    if (free_target) {
        static const uint32_t global[8] = {1, 1, 1, 1, 0, 0, 0, 0};
        static const uint32_t local[8] = {1, 1, 1, 0, 0, 0, 0, 0};
        static const uint32_t pose_mask[8] = {1, 1, 1, 1, 0, 0, 0, 0};
        if (mb_inference_request_set_mask(request, MB_INFERENCE_GLOBAL_ROOT, global, 8, g_error, sizeof g_error) != MB_OK ||
            mb_inference_request_set_mask(request, MB_INFERENCE_LOCAL_ROOT, local, 8, g_error, sizeof g_error) != MB_OK ||
            mb_inference_request_set_mask(request, MB_INFERENCE_POSE, pose_mask, 8, g_error, sizeof g_error) != MB_OK) {
            mb_inference_request_free(request);
            return -1;
        }
    }
    if (mb_inference_request_set_seed(request, seed, g_error, sizeof g_error) != MB_OK) {
        mb_inference_request_free(request);
        return -1;
    }
    g_graph_count = 0;
    const double start = now_ms();
    mb_motion * motion = NULL;
    mb_status status = mb_model_infer(g_model, request, &motion, g_error, sizeof g_error);
    g_infer_ms = now_ms() - start;
    mb_inference_request_free(request);
    if (status != MB_OK) return -1;

    uint64_t frames = 0, joints = 0, root_count = 0, rotation_count = 0;
    const float * roots = NULL;
    const float * rotations = NULL;
    if (mb_motion_get_frame_count(motion, &frames, g_error, sizeof g_error) != MB_OK ||
        mb_motion_get_joint_count(motion, &joints, g_error, sizeof g_error) != MB_OK ||
        mb_motion_get_root_translations(motion, &roots, &root_count, g_error, sizeof g_error) != MB_OK ||
        mb_motion_get_local_rotations_xyzw(motion, &rotations, &rotation_count, g_error, sizeof g_error) != MB_OK ||
        joints != JOINTS || root_count != frames * 3U || rotation_count != frames * JOINTS * 4U) {
        mb_motion_free(motion);
        return -1;
    }
    float * values = malloc((size_t)frames * FRAME_VALUES * sizeof(float));
    if (values == NULL) { mb_motion_free(motion); return -1; }
    for (uint64_t frame = 0; frame < frames; ++frame) {
        float * out = values + frame * FRAME_VALUES;
        memcpy(out, roots + frame * 3U, 3U * sizeof(float));
        memcpy(out + 3U, rotations + frame * JOINTS * 4U, JOINTS * 4U * sizeof(float));
    }
    mb_motion_free(motion);
    *frames_out = (uint32_t)frames;
    *values_out = values;
    return 0;
}

__attribute__((export_name("mb_profile")))
int32_t mb_profile(char * out, uint32_t capacity) {
    double graph_total = 0.0;
    for (uint32_t i = 0; i < g_graph_count; ++i) graph_total += g_graphs[i].ms;
    int at = snprintf(out, capacity, "infer %.1f ms; graphs %.1f ms; host %.1f ms; %u graphs:", g_infer_ms,
                      graph_total, g_infer_ms - graph_total, (unsigned)g_graph_count);
    for (uint32_t i = 0; i < g_graph_count && at > 0 && (uint32_t)at < capacity; ++i) {
        const int wrote = snprintf(out + at, capacity - (uint32_t)at, " [%u nodes %.1f ms]",
                                   (unsigned)g_graphs[i].nodes, g_graphs[i].ms);
        if (wrote < 0) break;
        at += wrote;
    }
    return at;
}
