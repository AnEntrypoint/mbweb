/* WIT export implementation over the motion-bricks C ABI (PRD-03).
 *
 * Bindings in component/gen/ come from wit/motionbricks.wit (wit-bindgen c 0.63.0).
 * Lists returned to the host are malloc'd: the generated post-return hook frees them. */

#include "gen/motionbricks.h"

#include <motionbricks/inference.h>
#include <motionbricks/motionbricks.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GLOBAL_ROOT_VALUES 40U   /* 8 slots x 5 */
#define LOCAL_ROOT_VALUES 32U    /* 8 slots x 4 */
#define POSE_VALUES 2424U        /* 8 slots x 303 */
#define JOINTS 34U
#define FRAME_VALUES (3U + JOINTS * 4U) /* root xyz + local XYZW per joint */

static mb_model * g_model = NULL;

static void set_error(motionbricks_string_t * err, const char * message) {
    motionbricks_string_dup(err, message);
}

static void set_status_error(motionbricks_string_t * err, const char * what, const char * detail,
                             mb_status status) {
    char text[512];
    snprintf(text, sizeof text, "%s: %s (%s)", what, mb_status_string(status), detail);
    motionbricks_string_dup(err, text);
}

/* The bundle directory arrives as a counted string, so copy it to a C string first. */
static char * copy_string(const motionbricks_string_t * text) {
    char * out = malloc(text->len + 1U);
    if (out == NULL) return NULL;
    memcpy(out, text->ptr, text->len);
    out[text->len] = '\0';
    return out;
}

static int32_t * alloc_s32(size_t count) {
    return malloc((count == 0U ? 1U : count) * sizeof(int32_t));
}

static float * alloc_f32(size_t count) {
    return malloc((count == 0U ? 1U : count) * sizeof(float));
}

/* Standing defaults from the neutral skeleton. Enabled global-root slots need unit cos/sin,
 * so the root is a stand-still yaw of zero at the neutral height. Orientations are identity. */
static mb_status standing_from_model(const mb_model * model, float * global_root, float * local_root,
                                     float * pose, char * error, uint64_t error_capacity) {
    float x[JOINTS], y[JOINTS], z[JOINTS];
    for (uint32_t joint = 0; joint < JOINTS; ++joint) {
        mb_status status = mb_model_get_neutral_joint_position(model, joint, &x[joint], &y[joint],
                                                               &z[joint], error, error_capacity);
        if (status != MB_OK) return status;
    }
    /* The neutral skeleton is pelvis-centred, so its feet sit below y = 0. Lift the whole figure so its lowest
     * joint stands on the floor; a source below the floor makes the model fold the figure into a crouch. */
    float lowest = y[0];
    for (uint32_t joint = 1; joint < JOINTS; ++joint) {
        if (y[joint] < lowest) lowest = y[joint];
    }
    const float lift = -lowest;
    memset(local_root, 0, LOCAL_ROOT_VALUES * sizeof(float));
    for (uint32_t slot = 0; slot < 8U; ++slot) {
        float * root = global_root + 5U * slot;
        root[0] = 0.0f;
        root[1] = y[0] + lift;
        root[2] = 0.0f;
        root[3] = 1.0f;
        root[4] = 0.0f;

        float * row = pose + 303U * slot;
        for (uint32_t joint = 1; joint < JOINTS; ++joint) {
            row[3U * (joint - 1U) + 0U] = x[joint] - x[0];
            row[3U * (joint - 1U) + 1U] = y[joint] + lift;
            row[3U * (joint - 1U) + 2U] = z[joint] - z[0];
        }
        for (uint32_t joint = 0; joint < JOINTS; ++joint) {
            float * orientation = row + 99U + 6U * joint;
            /* First two columns of the identity rotation: [R00,R10,R20,R01,R11,R21]. */
            orientation[0] = 1.0f; orientation[1] = 0.0f; orientation[2] = 0.0f;
            orientation[3] = 0.0f; orientation[4] = 1.0f; orientation[5] = 0.0f;
        }
    }
    return MB_OK;
}

/* Staged bundle files (src/memory_bundle.cpp in the core). The host streams GGUF bytes in;
 * load() then reads them through the loader as the bundle directory /model. */
int mb_memory_file_begin(const char * name, uint64_t size);
int mb_memory_file_write(const char * name, uint64_t offset, const void * data, uint64_t count);
void mb_memory_files_clear(void);

bool exports_mb_motionbricks_inference_begin_file(motionbricks_string_t * name, uint64_t size,
                                                  motionbricks_string_t * err) {
    char * file = copy_string(name);
    if (file == NULL) {
        set_error(err, "out of memory copying file name");
        return false;
    }
    int status = mb_memory_file_begin(file, size);
    free(file);
    if (status != 0) {
        set_error(err, "cannot stage bundle file (size must be positive)");
        return false;
    }
    return true;
}

bool exports_mb_motionbricks_inference_write_chunk(motionbricks_string_t * name, uint64_t offset,
                                                   motionbricks_list_u8_t * data,
                                                   motionbricks_string_t * err) {
    char * file = copy_string(name);
    if (file == NULL) {
        set_error(err, "out of memory copying file name");
        return false;
    }
    int status = mb_memory_file_write(file, offset, data->ptr, data->len);
    free(file);
    if (status != 0) {
        set_error(err, "chunk is outside the staged file or the file was not begun");
        return false;
    }
    return true;
}

bool exports_mb_motionbricks_inference_load(motionbricks_string_t * err) {
    char error[512] = {0};
    mb_model * next = NULL;
    mb_status status = mb_model_load("/model", NULL, &next, error, sizeof error);
    mb_memory_files_clear();
    if (status != MB_OK) {
        set_status_error(err, "mb_model_load", error, status);
        return false;
    }
    mb_model_free(g_model);
    g_model = next;
    return true;
}

bool exports_mb_motionbricks_inference_joint_parents(motionbricks_list_s32_t * ret,
                                                     motionbricks_string_t * err) {
    char error[512] = {0};
    if (g_model == NULL) {
        set_error(err, "model is not loaded");
        return false;
    }
    int32_t * parents = alloc_s32(JOINTS);
    if (parents == NULL) {
        set_error(err, "out of memory");
        return false;
    }
    for (uint32_t joint = 0; joint < JOINTS; ++joint) {
        mb_status status = mb_model_get_joint_parent(g_model, joint, &parents[joint], error, sizeof error);
        if (status != MB_OK) {
            free(parents);
            set_status_error(err, "mb_model_get_joint_parent", error, status);
            return false;
        }
    }
    ret->ptr = parents;
    ret->len = JOINTS;
    return true;
}

bool exports_mb_motionbricks_inference_neutral_positions(motionbricks_list_f32_t * ret,
                                                         motionbricks_string_t * err) {
    char error[512] = {0};
    if (g_model == NULL) {
        set_error(err, "model is not loaded");
        return false;
    }
    float * positions = alloc_f32(JOINTS * 3U);
    if (positions == NULL) {
        set_error(err, "out of memory");
        return false;
    }
    for (uint32_t joint = 0; joint < JOINTS; ++joint) {
        mb_status status = mb_model_get_neutral_joint_position(g_model, joint, &positions[3 * joint],
                                                               &positions[3 * joint + 1],
                                                               &positions[3 * joint + 2], error, sizeof error);
        if (status != MB_OK) {
            free(positions);
            set_status_error(err, "mb_model_get_neutral_joint_position", error, status);
            return false;
        }
    }
    ret->ptr = positions;
    ret->len = JOINTS * 3U;
    return true;
}

bool exports_mb_motionbricks_inference_standing_features(
    motionbricks_tuple3_list_f32_list_f32_list_f32_t * ret, motionbricks_string_t * err) {
    char error[512] = {0};
    if (g_model == NULL) {
        set_error(err, "model is not loaded");
        return false;
    }
    float * global_root = alloc_f32(GLOBAL_ROOT_VALUES);
    float * local_root = alloc_f32(LOCAL_ROOT_VALUES);
    float * pose = alloc_f32(POSE_VALUES);
    if (global_root == NULL || local_root == NULL || pose == NULL) {
        free(global_root);
        free(local_root);
        free(pose);
        set_error(err, "out of memory");
        return false;
    }
    mb_status status = standing_from_model(g_model, global_root, local_root, pose, error, sizeof error);
    if (status != MB_OK) {
        free(global_root);
        free(local_root);
        free(pose);
        set_status_error(err, "standing features", error, status);
        return false;
    }
    ret->f0 = (motionbricks_list_f32_t){global_root, GLOBAL_ROOT_VALUES};
    ret->f1 = (motionbricks_list_f32_t){local_root, LOCAL_ROOT_VALUES};
    ret->f2 = (motionbricks_list_f32_t){pose, POSE_VALUES};
    return true;
}

/* Copies one feature field into a request and reports the failure text on error. */
static bool set_feature(mb_inference_request * request, uint32_t field, const motionbricks_list_f32_t * values,
                        uint64_t expected, motionbricks_string_t * err) {
    char error[512] = {0};
    if (values->len != expected) {
        char text[128];
        snprintf(text, sizeof text, "feature %u has %zu values, expected %llu", (unsigned)field,
                 values->len, (unsigned long long)expected);
        set_error(err, text);
        return false;
    }
    mb_status status = mb_inference_request_set_features(request, field, values->ptr, values->len,
                                                         error, sizeof error);
    if (status != MB_OK) {
        set_status_error(err, "mb_inference_request_set_features", error, status);
        return false;
    }
    return true;
}

/* Source slots 0-3 stay enabled; target slots 4-7 are masked off so the model chooses the clip's end.
 * Local-root slot 3 keeps its library default (disabled), since its outgoing velocity is unknown. */
static bool mask_target_slots(mb_inference_request * request, motionbricks_string_t * err) {
    static const struct {
        uint32_t field;
        uint32_t mask[8];
    } masks[] = {
        {MB_INFERENCE_GLOBAL_ROOT, {1, 1, 1, 1, 0, 0, 0, 0}},
        {MB_INFERENCE_LOCAL_ROOT, {1, 1, 1, 0, 0, 0, 0, 0}},
        {MB_INFERENCE_POSE, {1, 1, 1, 1, 0, 0, 0, 0}},
    };
    char error[512] = {0};
    for (size_t i = 0; i < sizeof masks / sizeof masks[0]; ++i) {
        mb_status status = mb_inference_request_set_mask(request, masks[i].field, masks[i].mask, 8,
                                                         error, sizeof error);
        if (status != MB_OK) {
            set_status_error(err, "mb_inference_request_set_mask", error, status);
            return false;
        }
    }
    return true;
}

bool exports_mb_motionbricks_inference_infer(motionbricks_list_f32_t * global_root,
                                             motionbricks_list_f32_t * local_root,
                                             motionbricks_list_f32_t * pose, uint64_t seed,
                                             bool free_target,
                                             exports_mb_motionbricks_inference_motion_t * ret,
                                             motionbricks_string_t * err) {
    char error[512] = {0};
    if (g_model == NULL) {
        set_error(err, "model is not loaded");
        return false;
    }
    mb_inference_request * request = NULL;
    mb_status status = mb_inference_request_create(&request, error, sizeof error);
    if (status != MB_OK) {
        set_status_error(err, "mb_inference_request_create", error, status);
        return false;
    }
    if (!set_feature(request, MB_INFERENCE_GLOBAL_ROOT, global_root, GLOBAL_ROOT_VALUES, err) ||
        !set_feature(request, MB_INFERENCE_LOCAL_ROOT, local_root, LOCAL_ROOT_VALUES, err) ||
        !set_feature(request, MB_INFERENCE_POSE, pose, POSE_VALUES, err) ||
        (free_target && !mask_target_slots(request, err))) {
        mb_inference_request_free(request);
        return false;
    }
    status = mb_inference_request_set_seed(request, seed, error, sizeof error);
    if (status != MB_OK) {
        mb_inference_request_free(request);
        set_status_error(err, "mb_inference_request_set_seed", error, status);
        return false;
    }

    mb_motion * motion = NULL;
    status = mb_model_infer(g_model, request, &motion, error, sizeof error);
    mb_inference_request_free(request);
    if (status != MB_OK) {
        set_status_error(err, "mb_model_infer", error, status);
        return false;
    }

    uint64_t frames = 0, joints = 0, root_count = 0, rotation_count = 0;
    const float * roots = NULL;
    const float * rotations = NULL;
    if (mb_motion_get_frame_count(motion, &frames, error, sizeof error) != MB_OK ||
        mb_motion_get_joint_count(motion, &joints, error, sizeof error) != MB_OK ||
        mb_motion_get_root_translations(motion, &roots, &root_count, error, sizeof error) != MB_OK ||
        mb_motion_get_local_rotations_xyzw(motion, &rotations, &rotation_count, error, sizeof error) != MB_OK) {
        mb_motion_free(motion);
        set_error(err, "motion getters failed");
        return false;
    }
    if (joints != JOINTS || root_count != frames * 3U || rotation_count != frames * JOINTS * 4U) {
        mb_motion_free(motion);
        set_error(err, "motion shape does not match 34 joints");
        return false;
    }

    float * values = alloc_f32((size_t)frames * FRAME_VALUES);
    if (values == NULL) {
        mb_motion_free(motion);
        set_error(err, "out of memory");
        return false;
    }
    for (uint64_t frame = 0; frame < frames; ++frame) {
        float * out = values + frame * FRAME_VALUES;
        memcpy(out, roots + frame * 3U, 3U * sizeof(float));
        memcpy(out + 3U, rotations + frame * JOINTS * 4U, JOINTS * 4U * sizeof(float));
    }
    mb_motion_free(motion);

    ret->frames = (uint32_t)frames;
    ret->values = (motionbricks_list_f32_t){values, (size_t)frames * FRAME_VALUES};
    return true;
}
