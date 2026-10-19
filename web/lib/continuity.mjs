// Clip-to-clip continuity in the upstream convention (docs/API-INFERENCE.md, "Pose-level inputs").
// The next clip's source is the previous clip's last four frames, canonicalised about the first source root:
// its X/Z becomes the origin and its yaw becomes zero. The model's output is placed back into the session
// frame with the same origin and yaw, so every clip continues the one before it without a seam.
import { FRAME_VALUES, JOINTS, framePositions, multiply, rotate } from './skeleton.mjs';

export const SOURCE_FRAMES = 4;
const FPS = 30;
const GLOBAL_ROOT_VALUES = 40;
const LOCAL_ROOT_VALUES = 32;
const POSE_VALUES = 2424;
const POSE_ROW = 303;
const ORIENTATION_OFFSET = 99;

// Unit quaternion for a rotation of angle a about +Y, XYZW.
const yawQuaternion = (a) => [0, Math.sin(a / 2), 0, Math.cos(a / 2)];

// Rotates an X/Z pair about +Y by angle a. Forward (+Z) at yaw a maps to (sin a, cos a), so yaw = atan2(x, z).
function turnXZ(a, x, z) {
    const c = Math.cos(a);
    const s = Math.sin(a);
    return [x * c + z * s, -x * s + z * c];
}

// Heading of a frame's root, in the frame the clip is stored in.
export function headingAt(values, frame) {
    const base = frame * FRAME_VALUES;
    return headingOf(values.subarray(base + 3, base + 7));
}

// Heading of a root orientation: the yaw about +Y that carries +Z forward to the root's forward axis.
function headingOf(q) {
    const forward = rotate(q, [0, 0, 1]);
    return Math.atan2(forward[0], forward[2]);
}

function wrapAngle(a) {
    return Math.atan2(Math.sin(a), Math.cos(a));
}

// First two columns of a rotation matrix, in the order the model reads them: [R00,R10,R20,R01,R11,R21].
function firstTwoColumns(q, out, at) {
    const [x, y, z, w] = q;
    out[at] = 1 - 2 * (y * y + z * z);
    out[at + 1] = 2 * (x * y + w * z);
    out[at + 2] = 2 * (x * z - w * y);
    out[at + 3] = 2 * (x * y - w * z);
    out[at + 4] = 1 - 2 * (x * x + z * z);
    out[at + 5] = 2 * (y * z + w * x);
}

// Places a motion's frames in the session frame: root X/Z turned by yaw and moved to origin, root orientation
// turned by yaw. Local joint rotations are relative to their parents, so they stay as they are.
export function placeFrames(values, frames, origin, yaw) {
    const out = new Float32Array(values.length);
    const turn = yawQuaternion(yaw);
    for (let frame = 0; frame < frames; frame++) {
        const base = frame * FRAME_VALUES;
        out.set(values.subarray(base, base + FRAME_VALUES), base);
        const [x, z] = turnXZ(yaw, values[base], values[base + 2]);
        out[base] = x + origin.x;
        out[base + 2] = z + origin.z;
        out.set(multiply(turn, values.subarray(base + 3, base + 7)), base + 3);
    }
    return out;
}

// Raw feature fields for the last SOURCE_FRAMES frames of a clip (session frame), canonicalised about the
// first source root. The four target slots repeat the last source slot: the target mask is off, so they
// only need finite values.
export function sourceFeatures(values, frames, parents, neutral) {
    const first = frames - SOURCE_FRAMES;
    const firstBase = first * FRAME_VALUES;
    const origin = { x: values[firstBase], z: values[firstBase + 2] };
    const yaw = headingOf(values.subarray(firstBase + 3, firstBase + 7));
    const back = yawQuaternion(-yaw);

    const globalRoot = new Float32Array(GLOBAL_ROOT_VALUES);
    const localRoot = new Float32Array(LOCAL_ROOT_VALUES);
    const pose = new Float32Array(POSE_VALUES);
    const positions = new Float32Array(JOINTS * 3);
    const rotations = new Float32Array(JOINTS * 4);
    const rootX = [];
    const rootY = [];
    const rootZ = [];
    const headings = [];

    for (let slot = 0; slot < SOURCE_FRAMES; slot++) {
        framePositions(values, (first + slot) * FRAME_VALUES, parents, neutral, positions, 0, rotations, 0);
        const canonical = (joint) => {
            const [cx, cz] = turnXZ(-yaw, positions[joint * 3] - origin.x, positions[joint * 3 + 2] - origin.z);
            return [cx, positions[joint * 3 + 1], cz];
        };
        const root = canonical(0);
        const rootOrientation = multiply(back, rotations.subarray(0, 4));
        const heading = headingOf(rootOrientation);
        rootX.push(root[0]);
        rootY.push(root[1]);
        rootZ.push(root[2]);
        headings.push(heading);

        globalRoot.set([root[0], root[1], root[2], Math.cos(heading), Math.sin(heading)], slot * 5);

        const row = slot * POSE_ROW;
        for (let joint = 1; joint < JOINTS; joint++) {
            const p = canonical(joint);
            pose.set([p[0] - root[0], p[1], p[2] - root[2]], row + 3 * (joint - 1));
        }
        for (let joint = 0; joint < JOINTS; joint++) {
            const q = multiply(back, rotations.subarray(joint * 4, joint * 4 + 4));
            firstTwoColumns(q, pose, row + ORIENTATION_OFFSET + 6 * joint);
        }
    }

    // Forward differences at 30 FPS. Slot 3 has no outgoing frame, so it repeats the velocity from slot 2 to 3.
    for (let slot = 0; slot < SOURCE_FRAMES; slot++) {
        const from = Math.min(slot, SOURCE_FRAMES - 2);
        const next = from + 1;
        localRoot.set([
            wrapAngle(headings[next] - headings[from]) * FPS,
            (rootX[next] - rootX[from]) * FPS,
            (rootZ[next] - rootZ[from]) * FPS,
            rootY[slot],
        ], slot * 4);
    }

    // Target slots are masked off in the request; copying the last source slot keeps every value finite.
    for (let slot = SOURCE_FRAMES; slot < 8; slot++) {
        globalRoot.copyWithin(slot * 5, (SOURCE_FRAMES - 1) * 5, SOURCE_FRAMES * 5);
        localRoot.copyWithin(slot * 4, (SOURCE_FRAMES - 1) * 4, SOURCE_FRAMES * 4);
        pose.copyWithin(slot * POSE_ROW, (SOURCE_FRAMES - 1) * POSE_ROW, SOURCE_FRAMES * POSE_ROW);
    }

    return { globalRoot, localRoot, pose, origin, yaw };
}

// Steers the next clip with a destination, as the upstream controller does: the four target slots move the root
// to a point `distance` metres ahead of the source's last frame, turned by `turn` radians, and the model walks there.
// The target pose stays the source's last pose, so the figure keeps its stride.
export function aimTargets(source, { distance, turn }) {
    const g = source.globalRoot;
    const last = (SOURCE_FRAMES - 1) * 5;
    const heading = Math.atan2(g[last + 4], g[last + 3]) + turn;
    const x = g[last] + distance * Math.sin(heading);
    const z = g[last + 2] + distance * Math.cos(heading);
    for (let slot = SOURCE_FRAMES; slot < 8; slot++) {
        const at = slot * 5;
        g.set([x, g[last + 1], z, Math.cos(heading), Math.sin(heading)], at);
    }
    return source;
}
