// Page worker: fetches and verifies the model, runs the motion core, and answers inference requests.
// Inference runs here so the page stays responsive during a clip. On a cross-origin isolated page the threaded
// core runs, with its matrix work spread over workers; otherwise the single-thread component runs.
import { fetchModel } from './lib/model.mjs';
import { instantiateComponent, stageBundle } from './lib/component.mjs';
import { loadThreadedCore } from './lib/threads-host.mjs';

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

async function startThreadPool() {
    const started = [];
    for (let i = 0; i < THREAD_POOL; i++) {
        const worker = new Worker(threadWorker, { type: 'module' });
        started.push(new Promise((resolve) => {
            worker.onmessage = (event) => {
                if (event.data?.ready) resolve();
            };
        }));
        worker.postMessage({ ping: true });
        threadPool.push(worker);
    }
    await Promise.all(started);
}

function launchThread(message) {
    threadPool[nextThreadWorker % threadPool.length].postMessage(message);
    nextThreadWorker++;
}

async function loadThreaded(files) {
    self.postMessage({ type: 'stage', text: 'Loading the model into the threaded core' });
    const threads = Math.max(1, Math.min(8, navigator.hardwareConcurrency || 4));
    await startThreadPool();
    const bytes = await (await fetch(threadedModule)).arrayBuffer();
    threaded = await loadThreadedCore(bytes, { launch: launchThread, log: (fd, text) => console.log(text) });
    const x = threaded.exports;
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
    if (x.mb_load(threads) !== 0) throw threadedError('load');
}

function threadedError(what) {
    const pointer = threaded.exports.mb_error();
    const memory = new Uint8Array(threaded.memory.buffer);
    let end = pointer;
    while (memory[end] !== 0) end++;
    const text = new TextDecoder().decode(memory.subarray(pointer, end));
    return new Error(`${what}: ${text}`);
}

async function loadModel(manifest) {
    const files = await fetchModel(manifest, onProgressFor());
    const isolated = self.crossOriginIsolated === true && typeof SharedArrayBuffer !== 'undefined';
    if (isolated) {
        await loadThreaded(files);
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
            await loadModel(data.manifest);
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
            self.postMessage({ type: 'ready', parents, neutral, standing });
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
