// Page logic: starts the worker, draws the generated clip as a stick figure, and wires the controls.
import { motionPositions, JOINTS } from './lib/skeleton.mjs';

const FPS = 30;
const els = {
    status: document.getElementById('status'),
    progress: document.getElementById('progress'),
    generate: document.getElementById('generate'),
    play: document.getElementById('play'),
    scrub: document.getElementById('scrub'),
    frameLabel: document.getElementById('frame-label'),
    canvas: document.getElementById('stage'),
};
const ctx = els.canvas.getContext('2d');

const state = {
    parents: null,
    neutral: null,
    standing: null,
    clip: null, // { frames, positions, scale, minY, maxY, centreX }
    playing: false,
    frame: 0,
    seed: 0,
    nextId: 1,
    pending: new Map(),
    startedAt: 0,
    startFrame: 0,
};

function setStatus(text, stateName) {
    els.status.textContent = text;
    document.body.dataset.state = stateName;
}

function fail(message) {
    setStatus(message, 'error');
    els.generate.disabled = true;
    els.play.disabled = true;
    els.scrub.disabled = true;
}

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

worker.onmessage = ({ data }) => {
    if (data.type === 'progress') {
        const share = data.total ? data.done / data.total : 1;
        const sourceText = data.source === 'cache' ? 'from cache' : `${Math.round(share * 100)}%`;
        els.progress.value = share;
        setStatus(`Model: ${data.path} ${sourceText}`, 'loading');
    } else if (data.type === 'stage') {
        setStatus(data.text, 'loading');
    } else if (data.type === 'ready') {
        state.parents = data.parents;
        state.neutral = data.neutral;
        state.standing = data.standing;
        els.progress.hidden = true;
        els.generate.disabled = false;
        generate();
    } else if (data.type === 'motion') {
        const pending = state.pending.get(data.id);
        state.pending.delete(data.id);
        pending?.resolve({ frames: data.frames, values: data.values });
    } else if (data.type === 'error') {
        const pending = state.pending.get(data.id);
        if (pending) {
            state.pending.delete(data.id);
            pending.reject(new Error(data.message));
        } else {
            fail(`Cannot run the motion model: ${data.message}`);
        }
    }
};

worker.onerror = (event) => fail(`Cannot run the motion model: ${event.message || 'worker failed'}`);

function request(message, transfer = []) {
    const id = state.nextId++;
    return new Promise((resolve, reject) => {
        state.pending.set(id, { resolve, reject });
        worker.postMessage({ ...message, id }, transfer);
    });
}

// Projects the clip once: pelvis-centred, side view (Z horizontal, Y up), one scale for all frames.
function prepareClip(motion) {
    const positions = motionPositions(motion, state.parents, state.neutral);
    let minY = Infinity;
    let maxY = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;
    for (let frame = 0; frame < motion.frames; frame++) {
        const base = frame * JOINTS * 3;
        const pelvisZ = positions[base + 2];
        for (let joint = 0; joint < JOINTS; joint++) {
            const y = positions[base + joint * 3 + 1];
            const z = positions[base + joint * 3 + 2] - pelvisZ;
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
            minZ = Math.min(minZ, z);
            maxZ = Math.max(maxZ, z);
        }
    }
    const span = Math.max(maxY - minY, maxZ - minZ, 1e-3);
    return { frames: motion.frames, positions, minY, span };
}

function resizeCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const width = els.canvas.clientWidth;
    const height = els.canvas.clientHeight;
    els.canvas.width = Math.round(width * ratio);
    els.canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawFrame(state.frame);
}

function drawFrame(frame) {
    const width = els.canvas.clientWidth;
    const height = els.canvas.clientHeight;
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--stage-bg').trim() || '#f7f5f0';
    ctx.fillRect(0, 0, width, height);
    if (!state.clip) return;

    const { positions, minY, span } = state.clip;
    const scale = (Math.min(width, height) * 0.8) / span;
    const groundY = height * 0.88;
    const base = frame * JOINTS * 3;
    const pelvisZ = positions[base + 2];
    const toScreen = (joint) => {
        const x = width / 2 + (positions[base + joint * 3 + 2] - pelvisZ) * scale;
        const y = groundY - (positions[base + joint * 3 + 1] - minY) * scale;
        return [x, y];
    };

    ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--ground').trim() || '#b9b3a6';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, groundY);
    ctx.lineTo(width, groundY);
    ctx.stroke();

    ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--ink').trim() || '#1d1b18';
    ctx.lineWidth = Math.max(2, scale * 0.04);
    ctx.lineCap = 'round';
    ctx.beginPath();
    for (let joint = 1; joint < JOINTS; joint++) {
        const [px, py] = toScreen(state.parents[joint]);
        const [cx, cy] = toScreen(joint);
        ctx.moveTo(px, py);
        ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    ctx.fillStyle = ctx.strokeStyle;
    for (let joint = 0; joint < JOINTS; joint++) {
        const [x, y] = toScreen(joint);
        ctx.beginPath();
        ctx.arc(x, y, Math.max(2.5, scale * 0.045), 0, Math.PI * 2);
        ctx.fill();
    }

    state.frame = frame;
    els.scrub.value = String(frame);
    els.frameLabel.textContent = `Frame ${frame + 1} of ${state.clip.frames} at ${FPS} FPS`;
}

async function generate() {
    if (!state.standing) return;
    els.generate.disabled = true;
    setStatus('Generating a motion clip', 'generating');
    try {
        const { globalRoot, localRoot, pose } = state.standing;
        const motion = await request(
            {
                type: 'infer',
                globalRoot: copyOf(globalRoot),
                localRoot: copyOf(localRoot),
                pose: copyOf(pose),
                seed: state.seed,
            },
        );
        state.seed += 1;
        state.clip = prepareClip(motion);
        els.scrub.max = String(motion.frames - 1);
        els.scrub.disabled = false;
        els.play.disabled = false;
        els.generate.disabled = false;
        state.frame = 0;
        drawFrame(0);
        startPlayback();
        document.body.dataset.frames = String(motion.frames);
        setStatus(`Generated ${motion.frames} frames`, 'ready');
    } catch (error) {
        fail(`Cannot generate a motion: ${error.message}`);
    }
}

function copyOf(array) {
    return new Float32Array(array);
}

function startPlayback() {
    state.playing = true;
    state.startedAt = performance.now();
    state.startFrame = state.frame;
    els.play.textContent = 'Pause';
    els.play.setAttribute('aria-pressed', 'true');
    requestAnimationFrame(tick);
}

function stopPlayback() {
    state.playing = false;
    els.play.textContent = 'Play';
    els.play.setAttribute('aria-pressed', 'false');
}

function tick(now) {
    if (!state.playing || !state.clip) return;
    const elapsed = (now - state.startedAt) / 1000;
    const frame = (state.startFrame + Math.floor(elapsed * FPS)) % state.clip.frames;
    if (frame !== state.frame) drawFrame(frame);
    requestAnimationFrame(tick);
}

els.generate.addEventListener('click', () => generate());
els.play.addEventListener('click', () => {
    if (!state.clip) return;
    if (state.playing) {
        stopPlayback();
    } else {
        startPlayback();
    }
});
els.scrub.addEventListener('input', () => {
    stopPlayback();
    drawFrame(Number(els.scrub.value));
});
window.addEventListener('resize', resizeCanvas);

resizeCanvas();
setStatus('Starting', 'loading');
// The worker has no document location, so the manifest's relative base is resolved here.
const manifestUrl = new URL('./model/ternary/manifest.json', location.href);
const manifest = await (await fetch(manifestUrl)).json();
manifest.base_url = new URL(manifest.base_url, manifestUrl).href;
worker.postMessage({ type: 'init', manifest });
