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
    offset: { x: 0, y: 0 }, // pixels of avoidance, relaxing back to zero
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

// Projects the clip once: pelvis-centred side view (Z horizontal, Y up), one scale for all frames,
// and the pelvis path on the ground plane (world X and Z) relative to frame 0.
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
    const pathX = new Float32Array(motion.frames);
    const pathZ = new Float32Array(motion.frames);
    let maxPath = 0;
    for (let frame = 0; frame < motion.frames; frame++) {
        pathX[frame] = positions[frame * JOINTS * 3] - positions[0];
        pathZ[frame] = positions[frame * JOINTS * 3 + 2] - positions[2];
        maxPath = Math.max(maxPath, Math.abs(pathX[frame]), Math.abs(pathZ[frame]));
    }
    // Each clip starts where the previous one ended, so the figure keeps walking across the page.
    const previous = state.clip;
    const last = previous ? previous.frames - 1 : 0;
    const carry = previous
        ? {
            x: wrapUnit(previous.carry.x + WALK_SPAN * previous.pathX[last] / previous.maxPath),
            y: wrapUnit(previous.carry.y + WALK_SPAN * previous.pathZ[last] / previous.maxPath),
        }
        : { x: 0.5, y: 0.5 };
    return { frames: motion.frames, positions, minY, span, pathX, pathZ, maxPath: maxPath || 1e-3, carry };
}

// A clip's full pelvis travel covers this fraction of the stage's smaller side.
const WALK_SPAN = 0.30;
const wrapUnit = (value) => ((value % 1) + 1) % 1;

// Obstacles as page fractions: top-left corner and size.
const OBSTACLES = [
    { x: 0.20, y: 0.52, w: 0.08, h: 0.14 },
    { x: 0.55, y: 0.66, w: 0.10, h: 0.12 },
    { x: 0.78, y: 0.30, w: 0.07, h: 0.22 },
];
// Steering: pixels pushed per draw, the figure's clearance radius, and the mouse avoidance radius.
const STEER_RATE = 0.25;
const STEER_RELAX = 0.98;
const FIGURE_RADIUS = 36;
const MOUSE_RADIUS = 80;
const mouse = { x: null, y: null };

function resizeCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const width = els.canvas.clientWidth;
    const height = els.canvas.clientHeight;
    els.canvas.width = Math.round(width * ratio);
    els.canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawFrame(state.frame);
}

// Moves the figure away from obstacles and the mouse; the offset relaxes back when clear.
function steerAround(cx, cy, width, height) {
    let px = 0;
    let py = 0;
    for (const obstacle of OBSTACLES) {
        const left = obstacle.x * width - FIGURE_RADIUS;
        const top = obstacle.y * height - FIGURE_RADIUS;
        const right = (obstacle.x + obstacle.w) * width + FIGURE_RADIUS;
        const bottom = (obstacle.y + obstacle.h) * height + FIGURE_RADIUS;
        if (cx > left && cx < right && cy > top && cy < bottom) {
            // Leave by the nearest edge.
            const exits = [
                { d: cx - left, dx: -1, dy: 0 },
                { d: right - cx, dx: 1, dy: 0 },
                { d: cy - top, dx: 0, dy: -1 },
                { d: bottom - cy, dx: 0, dy: 1 },
            ];
            const exit = exits.reduce((best, option) => (option.d < best.d ? option : best));
            px += exit.dx * (exit.d + 1) * STEER_RATE;
            py += exit.dy * (exit.d + 1) * STEER_RATE;
        }
    }
    if (mouse.x !== null) {
        const dx = cx - mouse.x;
        const dy = cy - mouse.y;
        const distance = Math.hypot(dx, dy);
        if (distance < MOUSE_RADIUS && distance > 0) {
            const push = (MOUSE_RADIUS - distance) * STEER_RATE;
            px += (dx / distance) * push;
            py += (dy / distance) * push;
        }
    }
    if (px !== 0 || py !== 0) {
        state.offset.x += px;
        state.offset.y += py;
    } else {
        state.offset.x *= STEER_RELAX;
        state.offset.y *= STEER_RELAX;
    }
}

function drawFrame(frame) {
    const width = els.canvas.clientWidth;
    const height = els.canvas.clientHeight;
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--stage-bg').trim() || '#f7f5f0';
    ctx.fillRect(0, 0, width, height);
    if (!state.clip) return;

    const { positions, minY, span, pathX, pathZ, maxPath, carry } = state.clip;
    const scale = (Math.min(width, height) * 0.3) / span;
    const base = frame * JOINTS * 3;
    const pelvisZ = positions[base + 2];
    const pelvisY = positions[base + 1];

    // Pelvis position on the page: the clip's ground-plane path, wrapped around the stage.
    const fx = carry.x + WALK_SPAN * pathX[frame] / maxPath;
    const fy = carry.y + WALK_SPAN * pathZ[frame] / maxPath;
    const cx = wrapUnit(fx) * width + state.offset.x;
    const cy = wrapUnit(fy) * height + state.offset.y;
    steerAround(cx, cy, width, height);

    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--ground').trim() || '#b9b3a6';
    for (const obstacle of OBSTACLES) {
        ctx.fillRect(obstacle.x * width, obstacle.y * height, obstacle.w * width, obstacle.h * height);
    }

    ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--ink').trim() || '#1d1b18';
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = Math.max(2, scale * 0.04);
    ctx.lineCap = 'round';
    // Side view of the skeleton at the pelvis position; copies one stage away keep it whole at the edges.
    const toScreen = (joint, dx, dy) => [
        cx + dx + (positions[base + joint * 3 + 2] - pelvisZ) * scale,
        cy + dy - (positions[base + joint * 3 + 1] - pelvisY) * scale,
    ];
    for (const dx of [-width, 0, width]) {
        for (const dy of [-height, 0, height]) {
            ctx.beginPath();
            for (let joint = 1; joint < JOINTS; joint++) {
                const [px, py] = toScreen(state.parents[joint], dx, dy);
                const [qx, qy] = toScreen(joint, dx, dy);
                ctx.moveTo(px, py);
                ctx.lineTo(qx, qy);
            }
            ctx.stroke();
            for (let joint = 0; joint < JOINTS; joint++) {
                const [x, y] = toScreen(joint, dx, dy);
                ctx.beginPath();
                ctx.arc(x, y, Math.max(2.5, scale * 0.045), 0, Math.PI * 2);
                ctx.fill();
            }
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
