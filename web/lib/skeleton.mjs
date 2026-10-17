// Forward kinematics for the 34-joint G1 skeleton (Y-up, metres).
// A frame holds root xyz (3) followed by local XYZW rotations for 34 joints (136), per the WIT docs.

export const JOINTS = 34;
export const FRAME_VALUES = 3 + JOINTS * 4;

// Hamilton product of two XYZW quaternions.
function multiply(a, b) {
    const [ax, ay, az, aw] = a;
    const [bx, by, bz, bw] = b;
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ];
}

// Rotates a vector by a unit XYZW quaternion: v + w t + q x t with t = 2 (q x v).
function rotate(q, v) {
    const [x, y, z, w] = q;
    const tx = 2 * (y * v[2] - z * v[1]);
    const ty = 2 * (z * v[0] - x * v[2]);
    const tz = 2 * (x * v[1] - y * v[0]);
    return [
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    ];
}

// Writes the world position of every joint for one frame into out[34 * 3].
// Parents are resolved recursively, so the model's joint order does not matter.
export function framePositions(values, offset, parents, neutral, out, outOffset) {
    const globalRotation = new Array(JOINTS);
    const done = new Array(JOINTS).fill(false);
    const localRotation = (joint) => {
        const base = offset + 3 + joint * 4;
        return [values[base], values[base + 1], values[base + 2], values[base + 3]];
    };
    const solve = (joint) => {
        if (done[joint]) return;
        const parent = parents[joint];
        const o = outOffset + joint * 3;
        if (parent < 0) {
            globalRotation[joint] = localRotation(joint);
            out[o] = values[offset];
            out[o + 1] = values[offset + 1];
            out[o + 2] = values[offset + 2];
        } else {
            solve(parent);
            const parentRotation = globalRotation[parent];
            const bone = [
                neutral[joint * 3] - neutral[parent * 3],
                neutral[joint * 3 + 1] - neutral[parent * 3 + 1],
                neutral[joint * 3 + 2] - neutral[parent * 3 + 2],
            ];
            const rotated = rotate(parentRotation, bone);
            globalRotation[joint] = multiply(parentRotation, localRotation(joint));
            const p = outOffset + parent * 3;
            out[o] = out[p] + rotated[0];
            out[o + 1] = out[p + 1] + rotated[1];
            out[o + 2] = out[p + 2] + rotated[2];
        }
        done[joint] = true;
    };
    for (let joint = 0; joint < JOINTS; joint++) solve(joint);
}

// Positions for every frame of a motion: frames * 34 * 3 values.
export function motionPositions(motion, parents, neutral) {
    const out = new Float32Array(motion.frames * JOINTS * 3);
    for (let frame = 0; frame < motion.frames; frame++) {
        framePositions(motion.values, frame * FRAME_VALUES, parents, neutral, out, frame * JOINTS * 3);
    }
    return out;
}
