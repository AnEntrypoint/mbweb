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
    generating: false,
    steer: 0, // pixels of sideways avoidance, relaxing back to zero
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
    // Root travel along the view axis, relative to frame 0 (the clip starts where the figure is).
    const rootZ0 = positions[2];
    const rootDisp = new Float32Array(motion.frames);
    let maxDisp = 0;
    for (let frame = 0; frame < motion.frames; frame++) {
        rootDisp[frame] = positions[frame * JOINTS * 3 + 2] - rootZ0;
        maxDisp = Math.max(maxDisp, Math.abs(rootDisp[frame]));
    }
    // The figure's screen position is a fraction of the width; each clip continues where the last ended.
    const previous = state.clip;
    const carry = previous ? wrapUnit(previous.carry + WALK_WIDTH * previous.rootDisp[previous.frames - 1] / (previous.maxDisp || 1e-3)) : 0.5;
    return { frames: motion.frames, positions, minY, span, rootDisp, maxDisp, carry };
}

const WALK_WIDTH = 0.35; // a clip's full root travel covers this fraction of the stage width
const wrapUnit = (value) => ((value % 1) + 1) % 1;

// Fixed obstacles as fractions of the stage: x centre, width and height (both fractions).
const OBSTACLES = [
    { x: 0.28, w: 0.05, h: 0.42 },
    { x: 0.58, w: 0.07, h: 0.30 },
    { x: 0.84, w: 0.05, h: 0.50 },
];
// Steering: pixels pushed sideways per draw, and the margin kept from obstacles and the mouse.
const STEER_RATE = 0.18;
const STEER_RELAX = 0.985;
const MOUSE_RADIUS = 70;
const mouse = { x: null, y: null };

// Shortest signed distance from a to b on a wrapping stage.
const wrapDelta = (a, b, width) => {
    let d = (b - a) % width;
    if (d > width / 2) d -= width;
    if (d < -width / 2) d += width;
    return d;
};

function resizeCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const width = els.canvas.clientWidth;
    const height = els.canvas.clientHeight;
    els.canvas.width = Math.round(width * ratio);
    els.canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawFrame(state.frame);
}

// Moves the figure sideways, out of obstacles and away from the mouse; relaxes back when clear.
function steerAround(centreX, width, height, groundY, bodyHeight, scale) {
    const halfBody = scale * 0.35;
    let push = 0;
    for (const obstacle of OBSTACLES) {
        const ox = obstacle.x * width;
        const halfObstacle = (obstacle.w * width) / 2;
        const obstacleTop = groundY - obstacle.h * height;
        if (obstacleTop > groundY - bodyHeight) continue; // obstacle is below the figure's torso band
        const gap = wrapDelta(centreX, ox, width);
        const clearance = halfBody + halfObstacle + 6;
        if (Math.abs(gap) < clearance) {
            push += (gap > 0 ? -1 : 1) * (clearance - Math.abs(gap)) * STEER_RATE;
        }
    }
    if (mouse.x !== null && mouse.y !== null && mouse.y > groundY - bodyHeight && mouse.y <= groundY) {
        const gap = wrapDelta(centreX, mouse.x, width);
        if (Math.abs(gap) < MOUSE_RADIUS) {
            push += (gap > 0 ? -1 : 1) * (MOUSE_RADIUS - Math.abs(gap)) * STEER_RATE;
        }
    }
    state.steer = push !== 0 ? state.steer + push : state.steer * STEER_RELAX;
}

function drawFrame(frame) {
    const width = els.canvas.clientWidth;
    const height = els.canvas.clientHeight;
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--stage-bg').trim() || '#f7f5f0';
    ctx.fillRect(0, 0, width, height);
    if (!state.clip) return;

    const { positions, minY, span, rootDisp, maxDisp, carry } = state.clip;
    const scale = (Math.min(width, height) * 0.8) / span;
    const groundY = height * 0.88;
    const base = frame * JOINTS * 3;
    const pelvisZ = positions[base + 2];
    const walkX = wrapUnit(carry + WALK_WIDTH * rootDisp[frame] / (maxDisp || 1e-3)) * width;
    const bodyHeight = (groundY - 0) * 0.62; // figure height in pixels for the steering bands
    const centreX = ((walkX + state.steer) % width + width) % width;
    steerAround(centreX, width, height, groundY, bodyHeight, scale);
    const toScreen = (joint, offset) => {
        const x = centreX + offset + (positions[base + joint * 3 + 2] - pelvisZ) * scale;
        const y = groundY - (positions[base + joint * 3 + 1] - minY) * scale;
        return [x, y];
    };

    ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--ground').trim() || '#b9b3a6';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, groundY);
    ctx.lineTo(width, groundY);
    ctx.stroke();

    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--ground').trim() || '#b9b3a6';
    for (const obstacle of OBSTACLES) {
        const w = obstacle.w * width;
        const h = obstacle.h * height;
        ctx.fillRect(obstacle.x * width - w / 2, groundY - h, w, h);
    }

    ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--ink').trim() || '#1d1b18';
    ctx.lineWidth = Math.max(2, scale * 0.04);
    ctx.lineCap = 'round';
    for (const offset of [-width, 0, width]) {
        ctx.beginPath();
        for (let joint = 1; joint < JOINTS; joint++) {
            const [px, py] = toScreen(state.parents[joint], offset);
            const [cx, cy] = toScreen(joint, offset);
            ctx.moveTo(px, py);
            ctx.lineTo(cx, cy);
        }
        ctx.stroke();

        ctx.fillStyle = ctx.strokeStyle;
        for (let joint = 0; joint < JOINTS; joint++) {
            const [x, y] = toScreen(joint, offset);
            ctx.beginPath();
            ctx.arc(x, y, Math.max(2.5, scale * 0.045), 0, Math.PI * 2);
            ctx.fill();
        }
    }

    state.frame = frame;
    els.scrub.value = String(frame);
    els.frameLabel.textContent = `Frame ${frame + 1} of ${state.clip.frames} at ${FPS} FPS`;
}

async function generate() {
    if (!state.standing || state.generating) return;
    state.generating = true;
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
    } finally {
        state.generating = false;
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
    const position = state.startFrame + Math.floor(elapsed * FPS);
    if (position >= state.clip.frames && !state.generating) {
        // The clip has played through: the figure keeps walking with the next generated clip.
        generate();
    }
    const frame = position % state.clip.frames;
    if (frame !== state.frame) drawFrame(frame);
    requestAnimationFrame(tick);
}

els.generate.addEventListener('click', () => generate());
els.canvas.addEventListener('pointermove', (event) => {
    const rect = els.canvas.getBoundingClientRect();
    mouse.x = event.clientX - rect.left;
    mouse.y = event.clientY - rect.top;
});
els.canvas.addEventListener('pointerleave', () => {
    mouse.x = null;
    mouse.y = null;
});
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
