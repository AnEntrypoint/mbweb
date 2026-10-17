// Page worker: fetches and verifies the model, runs the component, and answers inference requests.
// Inference runs here so the page stays responsive during a clip.
import { fetchModel } from './lib/model.mjs';
import { instantiateComponent, stageBundle } from './lib/component.mjs';

const componentBase = new URL('./component/', self.location.href);
let inference = null;

function describe(error) {
    // jco reports a result::err payload as a ComponentError; the payload is the readable text.
    if (typeof error?.payload === 'string') return error.payload;
    return error?.message ?? String(error);
}

async function loadComponent(manifest) {
    if (typeof WebAssembly.Suspending !== 'function') {
        throw new Error('This page needs WebAssembly JS Promise Integration (Chrome or Edge 137 or later).');
    }
    const onProgress = (path, done, total, source) => {
        self.postMessage({ type: 'progress', path, done, total, source });
    };
    self.postMessage({ type: 'stage', text: 'Fetching and verifying the model' });
    const files = await fetchModel(manifest, onProgress);
    self.postMessage({ type: 'stage', text: 'Loading the model into the component' });
    inference = await instantiateComponent(componentBase);
    await stageBundle(inference, files);
    await inference.load();
}

function copy(array) {
    return array.slice();
}

self.onmessage = async ({ data }) => {
    try {
        if (data.type === 'init') {
            await loadComponent(data.manifest);
            const [globalRoot, localRoot, pose] = await inference.standingFeatures();
            const message = {
                type: 'ready',
                parents: copy(await inference.jointParents()),
                neutral: copy(await inference.neutralPositions()),
                standing: { globalRoot: copy(globalRoot), localRoot: copy(localRoot), pose: copy(pose) },
            };
            self.postMessage(message);
        } else if (data.type === 'infer') {
            const motion = await inference.infer(
                new Float32Array(data.globalRoot),
                new Float32Array(data.localRoot),
                new Float32Array(data.pose),
                BigInt(data.seed),
            );
            const values = copy(motion.values);
            self.postMessage({ type: 'motion', id: data.id, frames: motion.frames, values }, [values.buffer]);
        }
    } catch (error) {
        self.postMessage({ type: 'error', id: data.id, message: describe(error) });
    }
};
