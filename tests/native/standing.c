/* Native check for PRD-02: one G1 inference call through the upstream C ABI, built from the
 * same pinned sources as the WASI core. Run by scripts/native-check.sh with the verified bundle. */
#include <motionbricks/inference.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Standing defaults: global root slots [0, root_y, 0, 1, 0]; local root zeros;
   pose slots = neutral offsets (99) then identity orientations (204). */
static int build_standing(const mb_model * model, float * global_root, float * local_root, float * pose) {
    char e[256];
    float px[34], py[34], pz[34];
    for (uint32_t j = 0; j < 34; ++j) {
        if (mb_model_get_neutral_joint_position(model, j, &px[j], &py[j], &pz[j], e, sizeof e) != MB_OK) return 0;
    }
    memset(local_root, 0, 32 * sizeof(float));
    for (int s = 0; s < 8; ++s) {
        float * g = global_root + 5 * s;
        g[0] = 0.0f; g[1] = py[0]; g[2] = 0.0f; g[3] = 1.0f; g[4] = 0.0f;
        float * p = pose + 303 * s;
        for (uint32_t j = 1; j < 34; ++j) {
            p[3 * (j - 1) + 0] = px[j] - px[0];
            p[3 * (j - 1) + 1] = py[j];
            p[3 * (j - 1) + 2] = pz[j] - pz[0];
        }
        for (uint32_t j = 0; j < 34; ++j) {
            float * o = p + 99 + 6 * j;
            o[0] = 1; o[1] = 0; o[2] = 0; o[3] = 0; o[4] = 1; o[5] = 0;
        }
    }
    return 1;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: standing <bundle-dir>\n"); return 2; }
    char e[512];
    mb_model * model = NULL;
    if (mb_model_load(argv[1], NULL, &model, e, sizeof e) != MB_OK) { fprintf(stderr, "load: %s\n", e); return 1; }

    float global_root[40], local_root[32], pose[2424];
    if (!build_standing(model, global_root, local_root, pose)) { fprintf(stderr, "standing failed\n"); return 1; }

    mb_inference_request * r = NULL;
    if (mb_inference_request_create(&r, e, sizeof e) != MB_OK) { fprintf(stderr, "create: %s\n", e); return 1; }
    if (mb_inference_request_set_features(r, MB_INFERENCE_GLOBAL_ROOT, global_root, 40, e, sizeof e) != MB_OK ||
        mb_inference_request_set_features(r, MB_INFERENCE_LOCAL_ROOT, local_root, 32, e, sizeof e) != MB_OK ||
        mb_inference_request_set_features(r, MB_INFERENCE_POSE, pose, 2424, e, sizeof e) != MB_OK ||
        mb_inference_request_set_seed(r, 0, e, sizeof e) != MB_OK) {
        fprintf(stderr, "set: %s\n", e); return 1;
    }

    mb_motion * motion = NULL;
    mb_status status = mb_model_infer(model, r, &motion, e, sizeof e);
    if (status != MB_OK) { fprintf(stderr, "infer: %s (%u)\n", e, status); return 1; }

    /* PRD-03 acceptance: a zeroed pose. Enabled global-root slots still need unit cos/sin. */
    float zero_root[40] = {0}, zero_local[32] = {0}, zero_pose[2424] = {0};
    for (int s = 0; s < 8; ++s) zero_root[5 * s + 3] = 1.0f;
    mb_inference_request * zr = NULL;
    mb_motion * zero_motion = NULL;
    if (mb_inference_request_create(&zr, e, sizeof e) != MB_OK ||
        mb_inference_request_set_features(zr, MB_INFERENCE_GLOBAL_ROOT, zero_root, 40, e, sizeof e) != MB_OK ||
        mb_inference_request_set_features(zr, MB_INFERENCE_LOCAL_ROOT, zero_local, 32, e, sizeof e) != MB_OK ||
        mb_inference_request_set_features(zr, MB_INFERENCE_POSE, zero_pose, 2424, e, sizeof e) != MB_OK ||
        mb_model_infer(model, zr, &zero_motion, e, sizeof e) != MB_OK) {
        fprintf(stderr, "zeroed pose infer: %s\n", e); return 1;
    }
    uint64_t zero_frames = 0;
    mb_motion_get_frame_count(zero_motion, &zero_frames, e, sizeof e);
    printf("zeroed pose frames=%llu\n", (unsigned long long)zero_frames);
    int zero_ok = zero_frames >= 24 && zero_frames <= 64 && zero_frames % 4 == 0;
    mb_motion_free(zero_motion);
    mb_inference_request_free(zr);

    uint64_t frames = 0, joints = 0, rv = 0, rq = 0;
    const float * roots = NULL; const float * rots = NULL;
    mb_motion_get_frame_count(motion, &frames, e, sizeof e);
    mb_motion_get_joint_count(motion, &joints, e, sizeof e);
    mb_motion_get_root_translations(motion, &roots, &rv, e, sizeof e);
    mb_motion_get_local_rotations_xyzw(motion, &rots, &rq, e, sizeof e);
    printf("frames=%llu joints=%llu roots=%llu rots=%llu\n",
           (unsigned long long)frames, (unsigned long long)joints,
           (unsigned long long)rv, (unsigned long long)rq);

    int ok = frames >= 24 && frames <= 64 && frames % 4 == 0 && joints == 34 &&
             rv == frames * 3 && rq == frames * 34 * 4;
    double worst = 0.0;
    for (uint64_t f = 0; f < frames && ok; ++f) {
        for (int k = 0; k < 3; ++k) ok = ok && isfinite(roots[f * 3 + k]);
        for (int j = 0; j < 34; ++j) {
            const float * q = rots + (f * 34 + j) * 4;
            double n = 0; for (int k = 0; k < 4; ++k) { ok = ok && isfinite(q[k]); n += (double)q[k] * q[k]; }
            if (fabs(n - 1.0) > worst) worst = fabs(n - 1.0);
        }
    }
    printf("root0=(%.4f,%.4f,%.4f) rootN=(%.4f,%.4f,%.4f) max_quat_norm_err=%.2e\n",
           roots[0], roots[1], roots[2],
           roots[(frames - 1) * 3], roots[(frames - 1) * 3 + 1], roots[(frames - 1) * 3 + 2], worst);
    ok = ok && zero_ok;
    printf("RESULT: %s\n", ok ? "PASS" : "FAIL");

    mb_motion_free(motion);
    mb_inference_request_free(r);
    mb_model_free(model);
    return ok ? 0 : 1;
}
