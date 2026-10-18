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

// Projects the clip once: pelvis-centred side view (Z horizontal, Y up), one scale for all frames.
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

function insideObstacle(x, y, width, height) {
    return OBSTACLES.some((o) => (
        x > o.x * width - FIGURE_RADIUS && x < (o.x + o.w) * width + FIGURE_RADIUS &&
        y > o.y * height - FIGURE_RADIUS && y < (o.y + o.h) * height + FIGURE_RADIUS
    ));
}

function turnToward(current, goal, maxStep) {
    const difference = Math.atan2(Math.sin(goal - current), Math.cos(goal - current));
    return current + Math.max(-maxStep, Math.min(maxStep, difference));
}

// Walks the figure: wanders, heads for a clicked destination, and turns away from obstacles and the mouse.
function updateBody(now, width, height) {
    const dt = body.last ? Math.min(0.1, (now - body.last) / 1000) : 0;
    body.last = now;
    if (body.x === null) {
        body.x = width / 2;
        body.y = height / 2;
        body.heading = Math.random() * Math.PI * 2;
    }
    if (body.target) {
        const toTarget = Math.atan2(body.target.y - body.y, body.target.x - body.x);
        body.heading = turnToward(body.heading, toTarget, TURN_RATE * dt);
        if (Math.hypot(body.target.x - body.x, body.target.y - body.y) < 24) body.target = null;
    } else {
        body.heading += (Math.random() - 0.5) * 0.8 * dt;
    }
    // Look ahead: if the way is blocked, turn toward whichever side is clear.
    const ahead = (angle) => [body.x + Math.cos(angle) * LOOK_AHEAD, body.y + Math.sin(angle) * LOOK_AHEAD];
    const blocked = (angle) => {
        const [ax, ay] = ahead(angle);
        return insideObstacle(ax, ay, width, height) ||
            (mouse.x !== null && Math.hypot(ax - mouse.x, ay - mouse.y) < MOUSE_RADIUS);
    };
    if (blocked(body.heading)) {
        const left = blocked(body.heading - 0.7) ? 1 : 0;
        const right = blocked(body.heading + 0.7) ? 1 : 0;
        body.heading += (left <= right ? -1 : 1) * TURN_RATE * 2 * dt;
    }
    const speed = WALK_SPEED * Math.min(width, height);
    const nx = body.x + Math.cos(body.heading) * speed * dt;
    const ny = body.y + Math.sin(body.heading) * speed * dt;
    // Walk only into free space; otherwise stay and turn next frame.
    if (!insideObstacle(nx, ny, width, height)) {
        body.x = ((nx % width) + width) % width;
        body.y = ((ny % height) + height) % height;
    }
}

function drawFrame(frame) {
    const width = els.canvas.clientWidth;
    const height = els.canvas.clientHeight;
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--stage-bg').trim() || '#f7f5f0';
    ctx.fillRect(0, 0, width, height);
    if (!state.clip) return;

    const { positions, minY, span } = state.clip;
    const scale = (Math.min(width, height) * 0.3) / span;
    const base = frame * JOINTS * 3;
    const pelvisZ = positions[base + 2];
    const pelvisY = positions[base + 1];
    updateBody(performance.now(), width, height);
    const cx = body.x;
    const cy = body.y;
    // Face the direction of travel: mirror the side view when walking left.
    const facing = Math.cos(body.heading) < 0 ? -1 : 1;

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
        cx + dx + facing * (positions[base + joint * 3 + 2] - pelvisZ) * scale,
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
els.canvas.addEventListener('pointerdown', (event) => {
    const rect = els.canvas.getBoundingClientRect();
    body.target = { x: event.clientX - rect.left, y: event.clientY - rect.top };
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
