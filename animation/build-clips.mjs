// Generates the text-prompted clips in animation/clips.json with Kimodo and writes them as web/model/clips/*.clip.
//
// One kmd-generate server holds the motion and text models for the whole run, so adding a clip costs only its
// sampling time. A clip whose prompt, seed, steps and frame count match the manifest is kept as it is; --force
// regenerates everything, and naming clips on the command line limits the run to those names.
//
// Usage: node animation/build-clips.mjs [--force] [name ...]
// Needs animation/fetch.sh to have run first.
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const KIMODO = join(ROOT, '.kimodo');
const GENERATE = join(KIMODO, 'src/build/release/kmd-generate');
const MOTION = join(KIMODO, 'weights/models/kimodo-g1-rp-v1-f32.gguf');
const TEXT = join(KIMODO, 'weights/Llama-3-Kimodo-Q4_K_M.gguf');
const CLIPS_DIR = join(ROOT, 'web/model/clips');
const MANIFEST = join(CLIPS_DIR, 'manifest.json');
const FORMAT = 'mbweb-clip-v1';
const FPS = 30;
const JOINTS = 34;
const FRAME_VALUES = 3 + JOINTS * 4;
const DEFAULT_STEPS = 100;
const TRANSITION_FRAMES = 8; // only used between segments; a single prompt ignores it, but the server requires >= 1.

const args = process.argv.slice(2);
const force = args.includes('--force');
const onlyNames = args.filter((arg) => !arg.startsWith('--'));
const pin = Object.fromEntries((await readFile(join(ROOT, 'animation/pin.env'), 'utf8'))
    .split('\n').filter((line) => /^[A-Z_]+=/.test(line)).map((line) => line.split('=')));

const catalogue = JSON.parse(await readFile(join(ROOT, 'animation/clips.json'), 'utf8'));
for (const clip of catalogue) {
    if (!/^[a-z][a-z0-9_]*$/.test(clip.name)) throw new Error(`clip name must be snake_case: ${clip.name}`);
    if (!Number.isInteger(clip.frames) || clip.frames < 2) throw new Error(`${clip.name}: frames must be an integer >= 2`);
}
const wanted = catalogue.filter((clip) => onlyNames.length === 0 || onlyNames.includes(clip.name));
if (wanted.length === 0) throw new Error('no clips selected');

let manifest = { format: FORMAT, fps: FPS, joints: JOINTS, frame_values: FRAME_VALUES, clips: [] };
try {
    manifest = JSON.parse(await readFile(MANIFEST, 'utf8'));
} catch {
    // First run: no manifest yet.
}
const previous = new Map(manifest.clips.map((entry) => [entry.name, entry]));

const sha256 = (buffer) => createHash('sha256').update(buffer).digest('hex');

// Reads kmd-generate's protocol lines: "OK\tframes\tjoints" or "ERR\tmessage".
function startServer() {
    const child = spawn(GENERATE, ['--server', MOTION, TEXT], { stdio: ['pipe', 'pipe', 'inherit'] });
    const waiting = [];
    let buffered = '';
    child.stdout.on('data', (chunk) => {
        buffered += chunk;
        let newline;
        while ((newline = buffered.indexOf('\n')) >= 0) {
            const line = buffered.slice(0, newline);
            buffered = buffered.slice(newline + 1);
            if (/^(OK|ERR)\t/.test(line) && waiting.length > 0) waiting.shift()(line);
        }
    });
    child.on('exit', (code) => {
        for (const resolve of waiting.splice(0)) resolve(`ERR\tserver exited with ${code}`);
    });
    const request = (line) => new Promise((resolve) => {
        waiting.push(resolve);
        child.stdin.write(`${line}\n`);
    });
    return { request, close: () => child.stdin.end() };
}

// Packs root translations [T,3] and local XYZW rotations [T,34,4] into one frame per row: 3 + 136 values.
function pack(rootBuffer, rotationBuffer, frames) {
    const root = new Float32Array(rootBuffer.buffer, rootBuffer.byteOffset, rootBuffer.byteLength / 4);
    const rotations = new Float32Array(rotationBuffer.buffer, rotationBuffer.byteOffset, rotationBuffer.byteLength / 4);
    if (root.length !== frames * 3 || rotations.length !== frames * JOINTS * 4) {
        throw new Error(`unexpected output sizes: root ${root.length}, rotations ${rotations.length}`);
    }
    const packed = new Float32Array(frames * FRAME_VALUES);
    let maxRootStep = 0;
    let maxRotationNormError = 0;
    for (let frame = 0; frame < frames; frame++) {
        const out = frame * FRAME_VALUES;
        packed.set(root.subarray(frame * 3, frame * 3 + 3), out);
        for (let joint = 0; joint < JOINTS; joint++) {
            const q = rotations.subarray((frame * JOINTS + joint) * 4, (frame * JOINTS + joint) * 4 + 4);
            const norm = Math.hypot(q[0], q[1], q[2], q[3]);
            maxRotationNormError = Math.max(maxRotationNormError, Math.abs(norm - 1));
            for (let k = 0; k < 4; k++) packed[out + 3 + joint * 4 + k] = q[k] / norm;
        }
        if (frame > 0) {
            const step = Math.hypot(
                root[frame * 3] - root[(frame - 1) * 3],
                root[frame * 3 + 1] - root[(frame - 1) * 3 + 1],
                root[frame * 3 + 2] - root[(frame - 1) * 3 + 2],
            );
            maxRootStep = Math.max(maxRootStep, step);
        }
    }
    for (const value of packed) if (!Number.isFinite(value)) throw new Error('non-finite value in generated motion');
    return { packed, maxRootStep, maxRotationNormError };
}

await mkdir(CLIPS_DIR, { recursive: true });
const promptDir = join(KIMODO, 'prompts');
await mkdir(promptDir, { recursive: true });

const unchanged = async (clip) => {
    const entry = previous.get(clip.name);
    if (force || !entry || entry.prompt !== clip.prompt || entry.seed !== (clip.seed ?? 1)
        || entry.frames !== clip.frames || entry.steps !== (clip.steps ?? DEFAULT_STEPS)) return false;
    try {
        return (await stat(join(CLIPS_DIR, entry.file))).size === entry.bytes;
    } catch {
        return false;
    }
};

const todo = [];
for (const clip of wanted) {
    if (await unchanged(clip)) console.log(`keep   ${clip.name} (manifest matches)`);
    else todo.push(clip);
}

if (todo.length > 0) {
    const server = startServer();
    try {
        for (const clip of todo) {
            const started = Date.now();
            const prompt = join(promptDir, `${clip.name}.txt`);
            await writeFile(prompt, `${clip.prompt}\n`);
            const raw = join(KIMODO, 'raw', clip.name);
            await mkdir(raw, { recursive: true });
            const steps = clip.steps ?? DEFAULT_STEPS;
            const seed = clip.seed ?? 1;
            // Protocol: transition, steps, seed, output dir, then (frames, prompt file) per segment.
            const reply = await server.request([TRANSITION_FRAMES, steps, seed, raw, clip.frames, prompt].join('\t'));
            if (!reply.startsWith('OK\t')) throw new Error(`${clip.name}: ${reply}`);
            const [, frames] = reply.split('\t').map(Number);
            if (frames !== clip.frames) throw new Error(`${clip.name}: kimodo returned ${frames} frames`);

            const rootBuffer = await readFile(join(raw, 'root_positions.f32'));
            const rotationBuffer = await readFile(join(raw, 'local_rotations_xyzw.f32'));
            const { packed, maxRootStep, maxRotationNormError } = pack(rootBuffer, rotationBuffer, frames);
            const bytes = Buffer.from(packed.buffer);
            const file = `${clip.name}.clip`;
            await writeFile(join(CLIPS_DIR, file), bytes);

            const entry = {
                name: clip.name,
                prompt: clip.prompt,
                seed,
                steps,
                frames,
                seconds: Number((frames / FPS).toFixed(3)),
                file,
                bytes: bytes.length,
                sha256: sha256(bytes),
                max_root_step_m: Number(maxRootStep.toFixed(4)),
                max_rotation_norm_error: Number(maxRotationNormError.toFixed(6)),
            };
            manifest.clips = manifest.clips.filter((other) => other.name !== clip.name).concat(entry);
            console.log(`built  ${clip.name}: ${frames} frames, max root step ${entry.max_root_step_m} m, ${((Date.now() - started) / 1000).toFixed(0)} s`);
        }
    } finally {
        server.close();
    }
}

manifest.source = {
    kimodo_repo: pin.KIMODO_REPO,
    kimodo_commit: pin.KIMODO_COMMIT,
    motion_model: pin.KIMODO_MOTION_MODEL,
    text_quantization: pin.KIMODO_TEXT_QUANTIZATION,
};
manifest.clips.sort((a, b) => a.name.localeCompare(b.name));
await writeFile(MANIFEST, `${JSON.stringify(manifest, null, 4)}\n`);
console.log(`manifest: ${manifest.clips.length} clips`);
