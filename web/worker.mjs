// Page worker: fetches and verifies the model, runs the motion core, and answers inference requests.
// Inference runs here so the page stays responsive during a clip. On a cross-origin isolated page the threaded
// core runs, with its matrix work spread over workers; otherwise the single-thread component runs.
import { fetchModel } from './lib/model.mjs';
import { instantiateComponent, stageBundle } from './lib/component.mjs';
import { loadThreadedCore } from './lib/threads-host.mjs';

// Uncaught errors and rejections are reported with their text, so a failure in one engine can be read from the page.
self.addEventListener('error', (event) => {
    self.postMessage({ type: 'error', message: `worker error: ${event.message || event.error?.message || 'unknown'} (${event.filename}:${event.lineno})` });
});
self.addEventListener('unhandledrejection', (event) => {
    self.postMessage({ type: 'error', message: `worker rejected: ${event.reason?.message ?? event.reason}` });
});

const componentBase = new URL('./component/', self.location.href);
const threadedModule = new URL('./threads/motionbricks-threads.wasm', self.location.href);
const threadWorker = new URL('./threads-worker.js', self.location.href);
let inference = null;
let threaded = null;

function describe(error) {
    // jco reports a result::err payload as a ComponentError; the payload is the readable text.
    if (typeof error?.payload === 'string') return error.payload;
    return error?.message ?? String(error);
}

function onProgressFor() {
    return (path, done, total, source) => {
        self.postMessage({ type: 'progress', path, done, total, source });
    };
}

async function loadComponent(files) {
    if (typeof WebAssembly.Suspending !== 'function') {
        throw new Error('This browser cannot run the motion model: it needs cross-origin isolation (SharedArrayBuffer, Safari 15.2 or later with the service worker) or WebAssembly JS Promise Integration (Chrome or Edge 137 or later).');
    }
    self.postMessage({ type: 'stage', text: 'Loading the model into the component' });
    inference = await instantiateComponent(componentBase);
    await stageBundle(inference, files);
    await inference.load();
}

// Threads: each new thread runs in a worker of a pool created while the page loads. A worker created during a
// blocked inference call does not start in time, so the pool is created first and each worker is confirmed alive.
const THREAD_POOL = 16;
const threadPool = [];
let nextThreadWorker = 0;
// One busy flag per pool worker, in shared memory: a worker sets its flag while it runs a thread and clears it when the thread
// returns, so a launch never waits behind a thread that is still running.
let threadBusy = null; // created with the pool, on isolated pages only

async function startThreadPool() {
    threadBusy = new Int32Array(new SharedArrayBuffer(4 * THREAD_POOL));
    const started = [];
    for (let i = 0; i < THREAD_POOL; i++) {
        const worker = new Worker(threadWorker);
        started.push(new Promise((resolve) => {
            worker.onmessage = (event) => {
                if (event.data?.ready) resolve();
            };
        }));
        worker.postMessage({ ping: true, slot: i, busy: threadBusy });
        threadPool.push(worker);
    }
    await Promise.all(started);
}

function launchThread(message) {
    let slot = -1;
    for (let k = 0; k < THREAD_POOL && slot < 0; k++) {
        const candidate = (nextThreadWorker + k) % THREAD_POOL;
        if (Atomics.load(threadBusy, candidate) === 0) slot = candidate;
    }
    if (slot < 0) slot = nextThreadWorker % THREAD_POOL;
    nextThreadWorker = slot + 1;
    Atomics.store(threadBusy, slot, 1);
    threadPool[slot].postMessage({ ...message, slot, busy: threadBusy });
}

// Styles and agents: the game path. Every style is a verified .mbstyle asset staged into the same bundle as the weights.
const styles = new Map();
let standingHeight = 0.787;
let agentReady = false;

function stageFiles(x, files) {
    const encoder = new TextEncoder();
    for (const name of Object.keys(files)) {
        const nameBytes = encoder.encode(`${name}\0`);
        const namePtr = x.mb_alloc(nameBytes.length);
        new Uint8Array(threaded.memory.buffer).set(nameBytes, namePtr);
        if (x.mb_stage_begin(namePtr, BigInt(files[name].byteLength)) !== 0) throw threadedError('stage');
        const bytes = files[name];
        const chunk = 4 * 1024 * 1024;
        for (let offset = 0; offset < bytes.byteLength; offset += chunk) {
            const end = Math.min(bytes.byteLength, offset + chunk);
            const dataPtr = x.mb_alloc(end - offset);
            new Uint8Array(threaded.memory.buffer).set(bytes.subarray(offset, end), dataPtr);
            const status = x.mb_stage_write(namePtr, BigInt(offset), dataPtr, BigInt(end - offset));
            x.mb_free(dataPtr);
            if (status !== 0) throw threadedError(`write ${name}`);
        }
        x.mb_free(namePtr);
        delete files[name];
    }
}

async function loadAgent(styleFiles) {
    const x = threaded.exports;
    const names = Object.keys(styleFiles);
    stageFiles(x, styleFiles);
    for (const name of names) {
        const namePtr = x.mb_alloc(name.length + 1);
        new Uint8Array(threaded.memory.buffer).set(new TextEncoder().encode(`${name}\0`), namePtr);
        const handle = x.mbx_style_load(namePtr);
        x.mb_free(namePtr);
        if (handle === 0) throw threadedError(`style ${name}`);
        styles.set(name.replace(/\.mbstyle$/, ''), handle);
    }
    // Standing root height: the context a plan starts from when there is no motion yet.
    const [globalRoot] = await standingThreaded();
    standingHeight = globalRoot[1];
}

// One planned motion, as the upstream stream server plans it (demo/stream.go): a fresh agent per plan, seeded with the
// four reference frames at the seam. Without context the agent starts from the style's reset pose.
let session = 0;

function seedSession(context, handle) {
    const x = threaded.exports;
    if (session !== 0) x.mb_agent_free(session);
    session = x.mb_agent_new();
    if (session === 0) throw threadedError('agent');
    if (x.mb_agent_reset(session, handle) !== 0) throw threadedError('agent reset');
    const roots = new Float32Array(12);
    const rotations = new Float32Array(4 * 34 * 4);
    if (context) {
        roots.set(context.roots);
        rotations.set(context.rotations);
    } else {
        for (let frame = 0; frame < 4; frame++) roots[frame * 3 + 1] = standingHeight;
        for (let joint = 0; joint < 4 * 34; joint++) rotations[joint * 4 + 3] = 1;
    }
    const rootPtr = x.mb_alloc(roots.byteLength);
    const rotationPtr = x.mb_alloc(rotations.byteLength);
    new Float32Array(threaded.memory.buffer, rootPtr, roots.length).set(roots);
    new Float32Array(threaded.memory.buffer, rotationPtr, rotations.length).set(rotations);
    const status = x.mb_agent_context(session, rootPtr, rotationPtr, 4);
    x.mb_free(rootPtr);
    x.mb_free(rotationPtr);
    if (status !== 0) throw threadedError('agent context');
}

function planThreaded({ style, direction, speed, target, seed, context }) {
    const x = threaded.exports;
    const handle = styles.get(style);
    const heading = direction > 0 ? 0 : Math.PI;
    seedSession(context, handle);
    const command = x.mb_command_new(handle, 0, 0, direction, 0, 0, direction, speed, target ? 1 : 0,
                                     0, 0.787, target ? target.z : 0, heading, BigInt(seed));
    const framesPtr = x.mb_alloc(4);
    const outRootsPtr = x.mb_alloc(4);
    const outRotationsPtr = x.mb_alloc(4);
    const status = x.mbx_agent_plan(session, command, framesPtr, outRootsPtr, outRotationsPtr);
    x.mb_command_free_handle(command);
    if (status !== 0) throw threadedError('plan');
    const view = new DataView(threaded.memory.buffer);
    const frames = view.getUint32(framesPtr, true);
    const rootsAt = view.getUint32(outRootsPtr, true);
    const rotationsAt = view.getUint32(outRotationsPtr, true);
    const values = new Float32Array(frames * 139);
    for (let frame = 0; frame < frames; frame++) {
        values.set(new Float32Array(threaded.memory.buffer, rootsAt + frame * 12, 3), frame * 139);
        values.set(new Float32Array(threaded.memory.buffer, rotationsAt + frame * 136 * 4, 136), frame * 139 + 3);
    }
    x.mb_free(rootsAt);
    x.mb_free(rotationsAt);
    for (const pointer of [framesPtr, outRootsPtr, outRotationsPtr]) x.mb_free(pointer);
    return { frames, values };
}

async function loadThreaded(files, styleFiles) {
    self.postMessage({ type: 'stage', text: 'Loading the model into the threaded core' });
    const threads = Math.max(1, Math.min(8, navigator.hardwareConcurrency || 4));
    await startThreadPool();
    const bytes = await (await fetch(threadedModule)).arrayBuffer();
    threaded = await loadThreadedCore(bytes, { launch: launchThread, log: (fd, text) => console.log(text) });
    const x = threaded.exports;
    stageFiles(x, files);
    if (x.mb_load(threads) !== 0) throw threadedError('load');
    if (styleFiles) {
        self.postMessage({ type: 'stage', text: 'Loading the styles' });
        await loadAgent(styleFiles);
    }
}

function threadedError(what) {
    const pointer = threaded.exports.mb_error();
    const memory = new Uint8Array(threaded.memory.buffer);
    let end = pointer;
    while (memory[end] !== 0) end++;
    const text = new TextDecoder().decode(memory.subarray(pointer, end));
    return new Error(`${what}: ${text}`);
}

async function loadModel(manifest, styleManifest) {
    const files = await fetchModel(manifest, onProgressFor());
    const isolated = self.crossOriginIsolated === true && typeof SharedArrayBuffer !== 'undefined';
    if (isolated) {
        const styleFiles = styleManifest ? await fetchModel(styleManifest, onProgressFor()) : null;
        await loadThreaded(files, styleFiles);
    } else {
        await loadComponent(files);
    }
}

function copy(array) {
    return array.slice();
}

// Threaded core: inputs go into the shared memory, the result is copied out before it is freed.
function inferThreaded({ globalRoot, localRoot, pose, seed, freeTarget }) {
    const x = threaded.exports;
    const put = (array) => {
        const pointer = x.mb_alloc(array.length * 4);
        new Float32Array(threaded.memory.buffer, pointer, array.length).set(array);
        return pointer;
    };
    const globalPtr = put(globalRoot);
    const localPtr = put(localRoot);
    const posePtr = put(pose);
    const framesPtr = x.mb_alloc(4);
    const valuesPtr = x.mb_alloc(4);
    const status = x.mb_infer(globalPtr, localPtr, posePtr, BigInt(seed), freeTarget ? 1 : 0, framesPtr, valuesPtr);
    const view = new DataView(threaded.memory.buffer);
    const frames = view.getUint32(framesPtr, true);
    const valuesAt = view.getUint32(valuesPtr, true);
    let result = null;
    if (status === 0) {
        result = { frames, values: new Float32Array(threaded.memory.buffer, valuesAt, frames * 139).slice() };
        x.mb_free(valuesAt);
    }
    for (const pointer of [globalPtr, localPtr, posePtr, framesPtr, valuesPtr]) x.mb_free(pointer);
    if (!result) throw threadedError('infer');
    return result;
}

async function standingThreaded() {
    const x = threaded.exports;
    const globalPtr = x.mb_alloc(40 * 4);
    const localPtr = x.mb_alloc(32 * 4);
    const posePtr = x.mb_alloc(2424 * 4);
    if (x.mb_standing(globalPtr, localPtr, posePtr) !== 0) throw threadedError('standing');
    const memory = threaded.memory.buffer;
    const standing = [
        new Float32Array(memory, globalPtr, 40).slice(),
        new Float32Array(memory, localPtr, 32).slice(),
        new Float32Array(memory, posePtr, 2424).slice(),
    ];
    for (const pointer of [globalPtr, localPtr, posePtr]) x.mb_free(pointer);
    return standing;
}

async function jointGeometry() {
    const x = threaded.exports;
    const parentsPtr = x.mb_alloc(34 * 4);
    const neutralPtr = x.mb_alloc(34 * 3 * 4);
    if (x.mb_joint_parents(parentsPtr) !== 0) throw threadedError('joint parents');
    if (x.mb_neutral_positions(neutralPtr) !== 0) throw threadedError('neutral positions');
    const memory = threaded.memory.buffer;
    const parents = Array.from(new Int32Array(memory, parentsPtr, 34));
    const neutral = Array.from(new Float32Array(memory, neutralPtr, 34 * 3));
    x.mb_free(parentsPtr);
    x.mb_free(neutralPtr);
    return { parents, neutral };
}

self.onmessage = async ({ data }) => {
    try {
        if (data.type === 'init') {
            await loadModel(data.manifest, data.styleManifest);
            let parents, neutral, standing;
            if (threaded) {
                ({ parents, neutral } = await jointGeometry());
                const [globalRoot, localRoot, pose] = await standingThreaded();
                standing = { globalRoot, localRoot, pose };
            } else {
                parents = copy(await inference.jointParents());
                neutral = copy(await inference.neutralPositions());
                const [globalRoot, localRoot, pose] = await inference.standingFeatures();
                standing = { globalRoot: copy(globalRoot), localRoot: copy(localRoot), pose: copy(pose) };
            }
            agentReady = styles.size > 0;
            self.postMessage({ type: 'ready', parents, neutral, standing, agent: agentReady });
        } else if (data.type === 'plan') {
            const { frames, values } = planThreaded(data);
            self.postMessage({ type: 'motion', id: data.id, frames, values }, [values.buffer]);
        } else if (data.type === 'infer') {
            const request = {
                globalRoot: new Float32Array(data.globalRoot),
                localRoot: new Float32Array(data.localRoot),
                pose: new Float32Array(data.pose),
                seed: data.seed,
                freeTarget: Boolean(data.freeTarget),
            };
            const motion = threaded
                ? inferThreaded(request)
                : await inference.infer(request.globalRoot, request.localRoot, request.pose, BigInt(request.seed), request.freeTarget);
            const values = copy(motion.values);
            self.postMessage({ type: 'motion', id: data.id, frames: motion.frames, values }, [values.buffer]);
        }
    } catch (error) {
        self.postMessage({ type: 'error', id: data.id, message: describe(error) });
    }
};
