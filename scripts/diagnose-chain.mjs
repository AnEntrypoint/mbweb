// Diagnoses clip continuity against the real component, outside the browser.
//
// Usage: node --experimental-wasm-jspi scripts/diagnose-chain.mjs <component-base-url> [model-dir]
//   component-base-url  URL of a served site/component/ directory (the transpiled core modules)
//   model-dir           directory with pose.gguf, root.gguf, support.gguf, vq-decoder.gguf (default model/ternary)
//
// Reports, for a standing clip and one chained clip: the standing root and pelvis-to-foot stability over the
// clip, and how closely the model reproduces the source frames it was given (root, heading, joint positions).
// Numbers that drift from the figures in reports/MotionBricks game integration usage.md point at a regression.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { instantiateComponent, stageBundle } from '../web/lib/component.mjs';
import { FRAME_VALUES, JOINTS, framePositions } from '../web/lib/skeleton.mjs';
import { sourceFeatures, SOURCE_FRAMES } from '../web/lib/continuity.mjs';

const [componentBase, modelArg] = process.argv.slice(2);
if (!componentBase) {
    console.error('usage: node --experimental-wasm-jspi scripts/diagnose-chain.mjs <component-base-url> [model-dir]');
    process.exit(2);
}
const modelDir = modelArg ?? join(fileURLToPath(new URL('..', import.meta.url)), 'model', 'ternary');

const inference = await instantiateComponent(componentBase);
const files = {};
for (const name of ['pose.gguf', 'root.gguf', 'support.gguf', 'vq-decoder.gguf']) {
    files[name] = new Uint8Array(readFileSync(join(modelDir, name)));
}
await stageBundle(inference, files);
await inference.load();
const parents = Array.from(await inference.jointParents());
const neutral = Float32Array.from(await inference.neutralPositions());
const [globalRoot, localRoot, pose] = await inference.standingFeatures();

const infer = async (features, seed) => {
    const motion = await inference.infer(
        Float32Array.from(features.globalRoot), Float32Array.from(features.localRoot),
        Float32Array.from(features.pose), BigInt(seed), true,
    );
    return { frames: motion.frames, values: Float32Array.from(motion.values) };
};

// Pelvis-to-lowest-joint height per frame: a standing figure keeps it near its standing height.
const standingReport = (motion) => {
    const pos = new Float32Array(JOINTS * 3);
    const rot = new Float32Array(JOINTS * 4);
    const rows = [];
    for (let f = 0; f < motion.frames; f += 8) {
        framePositions(motion.values, f * FRAME_VALUES, parents, neutral, pos, 0, rot, 0);
        let lowest = Infinity;
        for (let j = 0; j < JOINTS; j++) lowest = Math.min(lowest, pos[j * 3 + 1]);
        const pelvis = motion.values[f * FRAME_VALUES + 1];
        rows.push({ frame: f, pelvisHeight: +pelvis.toFixed(3), lowestJoint: +lowest.toFixed(3), pelvisToFoot: +(pelvis - lowest).toFixed(3) });
    }
    return rows;
};

const standing = await infer({ globalRoot, localRoot, pose }, 1);
console.log('== standing clip (free target)');
console.log(`frames ${standing.frames}`);
console.log(`profile: ${await inference.profile()}`);
console.table(standingReport(standing));

const source = sourceFeatures(standing.values, standing.frames, parents, neutral);
const chained = await infer(source, 2);
console.log(`profile: ${await inference.profile()}`);
console.log('== chained clip: model against its source frames (canonical frame)');
const pos = new Float32Array(JOINTS * 3);
const rot = new Float32Array(JOINTS * 4);
let rootXZ = 0;
let rootY = 0;
let headingDeg = 0;
let jointSq = 0;
let jointCount = 0;
for (let f = 0; f < SOURCE_FRAMES; f++) {
    framePositions(chained.values, f * FRAME_VALUES, parents, neutral, pos, 0, rot, 0);
    const base = f * FRAME_VALUES;
    const sourceBase = f * 5;
    rootXZ += Math.hypot(chained.values[base] - source.globalRoot[sourceBase], chained.values[base + 2] - source.globalRoot[sourceBase + 2]);
    rootY += Math.abs(chained.values[base + 1] - source.globalRoot[sourceBase + 1]);
    const sourceYaw = Math.atan2(source.globalRoot[sourceBase + 4], source.globalRoot[sourceBase + 3]);
    const [x, y, z, w] = rot.subarray(0, 4);
    const forwardX = 2 * (x * z + w * y);
    const forwardZ = 1 - 2 * (x * x + y * y);
    headingDeg += Math.abs(Math.atan2(forwardX, forwardZ) - sourceYaw) * 180 / Math.PI;
    for (let j = 1; j < JOINTS; j++) {
        const row = f * 303 + 3 * (j - 1);
        const dx = source.pose[row] - (pos[j * 3] - pos[0]);
        const dy = source.pose[row + 1] - pos[j * 3 + 1];
        const dz = source.pose[row + 2] - (pos[j * 3 + 2] - pos[2]);
        jointSq += dx * dx + dy * dy + dz * dz;
        jointCount++;
    }
}
console.log(JSON.stringify({
    rootXZ_m: +(rootXZ / SOURCE_FRAMES).toFixed(4),
    rootY_m: +(rootY / SOURCE_FRAMES).toFixed(4),
    heading_deg: +(headingDeg / SOURCE_FRAMES).toFixed(2),
    jointPosRms_m: +Math.sqrt(jointSq / jointCount).toFixed(4),
    chainedFrames: chained.frames,
}, null, 2));
process.exit(0);
