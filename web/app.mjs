// Page logic: starts the worker, draws the generated clip as a stick figure, and wires the controls.
import { motionPositions, JOINTS, FRAME_VALUES } from './lib/skeleton.mjs';
import { SOURCE_FRAMES, aimTargets, placeFrames, sourceFeatures } from './lib/continuity.mjs';

const FPS = 30;
const els = {
    status: document.getElementById('status'),
    progress: document.getElementById('progress'),
    play: document.getElementById('play'),
    scrub: document.getElementById('scrub'),
    frameLabel: document.getElementById('frame-label'),
    canvas: document.getElementById('stage'),
};
const ctx = els.canvas.getContext('2d');

// Threads need a cross-origin isolated page. A static host cannot send the headers, so a service worker adds them;
// the first load registers it and reloads once. If that fails, the page still runs on the single-thread core.
if (!self.crossOriginIsolated && 'serviceWorker' in navigator && !sessionStorage.getItem('coi-reloaded')) {
    sessionStorage.setItem('coi-reloaded', '1');
    try {
        const registration = await navigator.serviceWorker.register('./coi-sw.js');
        // "ready" never settles if registration is blocked, so the wait is bounded and the page carries on without it.
        const active = await Promise.race([
            navigator.serviceWorker.ready.then(() => true),
            new Promise((resolve) => setTimeout(() => resolve(false), 5000)),
        ]);
        if (active && registration.active) location.reload();
    } catch (error) {
        console.warn('Cross-origin isolation unavailable; using the single-thread core', error);
    }
}

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
    tail: null, // last generated clip: the source for the next one
    queue: [], // generated clips waiting to play
    clipCount: 0, // clips started so far; mirrored to body[data-clip] for the smoke test
    span: null, // one scale for the whole session, set from the first clip
    extent: { maxUp: 0, maxDown: 0, maxSide: 0 }, // largest reach over all clips so far
    blend: null, // the outgoing clip's pelvis-relative pose, eased into the next clip
    looping: false,
};

function setStatus(text, stateName) {
    els.status.textContent = text;
    document.body.dataset.state = stateName;
}

function fail(message) {
    setStatus(message, 'error');
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
        fillQueue();
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
    // Pelvis travel per frame on the ground plane: the gait's own speed, so the body moves with its feet.
    const speed = new Float32Array(motion.frames);
    for (let frame = 1; frame < motion.frames; frame++) {
        const base = frame * JOINTS * 3;
        const prev = base - JOINTS * 3;
        speed[frame] = Math.hypot(positions[base] - positions[prev], positions[base + 2] - positions[prev + 2]);
    }
    speed[0] = speed[1] ?? 0; // the loop point has no real previous frame
    // How far the figure reaches above, below and to each side of the pelvis, over all frames.
    let maxUp = 0;
    let maxDown = 0;
    let maxSide = 0;
    for (let frame = 0; frame < motion.frames; frame++) {
        const base = frame * JOINTS * 3;
        for (let joint = 0; joint < JOINTS; joint++) {
            const dy = positions[base + joint * 3 + 1] - positions[base + 1];
            const dz = positions[base + joint * 3 + 2] - positions[base + 2];
            maxUp = Math.max(maxUp, dy);
            maxDown = Math.max(maxDown, -dy);
            maxSide = Math.max(maxSide, Math.abs(dz));
        }
    }
    return { frames: motion.frames, values: motion.values, positions, minY, span, speed, maxUp, maxDown, maxSide };
}

const wrapUnit = (value) => ((value % 1) + 1) % 1;

// Obstacles as page fractions: top-left corner and size.
const OBSTACLES = [
    { x: 0.20, y: 0.52, w: 0.08, h: 0.14 },
    { x: 0.55, y: 0.66, w: 0.10, h: 0.12 },
    { x: 0.78, y: 0.30, w: 0.07, h: 0.22 },
];
const FIGURE_RADIUS = 36; // clearance kept around obstacles, in pixels
const LOOK_AHEAD = 70; // how far ahead the figure checks for obstacles
const MOUSE_RADIUS = 90; // the mouse pushes the figure only when this close to its body
const TURN_RATE = 1.8; // radians per second when turning toward a destination or away from a block
const WANDER = 1.1; // how strongly the heading drifts while wandering
const BLEND_FRAMES = 8; // frames over which a new clip eases in from the outgoing pose
const FACE_BAND = 0.25; // the figure turns around only once the heading is this far past vertical
const mouse = { x: null, y: null };
const body = { x: null, y: null, heading: 0, turnRate: 0, last: 0, target: null, facing: 1 };

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

// Walks the figure. Distance per drawn frame is the generated clip's own pelvis travel, scaled to the
// page; the heading wanders with inertia, heads for a clicked destination, and turns away from obstacles
// and from the mouse when it is close to the body. The body only moves while the clip plays.
function updateBody(now, width, height, speedPx, playing, bounds) {
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
        body.turnRate = 0;
        if (Math.hypot(body.target.x - body.x, body.target.y - body.y) < 24) body.target = null;
    } else {
        body.turnRate += (Math.random() - 0.5) * WANDER * dt;
        body.turnRate *= Math.exp(-1.2 * dt);
        body.heading += body.turnRate * dt;
    }
    // The mouse turns the figure away only when the pointer is close to its body.
    if (mouse.x !== null && Math.hypot(mouse.x - body.x, mouse.y - body.y) < MOUSE_RADIUS) {
        const away = Math.atan2(body.y - mouse.y, body.x - mouse.x);
        body.heading = turnToward(body.heading, away, TURN_RATE * dt);
    }
    // Obstacles ahead: turn toward whichever side is clear, and do not walk into them.
    const ahead = (angle) => [body.x + Math.cos(angle) * LOOK_AHEAD, body.y + Math.sin(angle) * LOOK_AHEAD];
    if (insideObstacle(...ahead(body.heading), width, height)) {
        const left = insideObstacle(...ahead(body.heading - 0.7), width, height) ? 1 : 0;
        const right = insideObstacle(...ahead(body.heading + 0.7), width, height) ? 1 : 0;
        body.heading += (left <= right ? -1 : 1) * TURN_RATE * 2 * dt;
    }
    if (!playing) return;
    const step = speedPx; // the clip's own pelvis travel since the last drawn frame
    const nx = body.x + Math.cos(body.heading) * step;
    const ny = body.y + Math.sin(body.heading) * step;
    // Stay on the stage: bounce off the edges (kept clear of the figure's height) and turn around.
    // Walls keep the whole figure on the stage: the pelvis may go no closer than its extent allows.
    const { left, right, top, bottom } = bounds;
    if (left >= right || top >= bottom) return;
    let bounced = false;
    if (nx < left || nx > right) { body.heading = Math.PI - body.heading; bounced = true; }
    if (ny < top || ny > bottom) { body.heading = -body.heading; bounced = true; }
    if (bounced) body.turnRate = 0;
    if (!insideObstacle(nx, ny, width, height)) {
        body.x = Math.min(right, Math.max(left, nx));
        body.y = Math.min(bottom, Math.max(top, ny));
    }
}

// Pelvis-relative side-view offsets for one frame, as [depth, height] pairs in model units.
function pelvisOffsets(clip, frame) {
    const { positions } = clip;
    const base = frame * JOINTS * 3;
    const out = new Float32Array(JOINTS * 2);
    for (let joint = 0; joint < JOINTS; joint++) {
        out[joint * 2] = positions[base + joint * 3 + 2] - positions[base + 2];
        out[joint * 2 + 1] = positions[base + joint * 3 + 1] - positions[base + 1];
    }
    return out;
}

function drawFrame(frame) {
    const width = els.canvas.clientWidth;
    const height = els.canvas.clientHeight;
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--stage-bg').trim() || '#f7f5f0';
    ctx.fillRect(0, 0, width, height);
    if (!state.clip) return;

    // One scale and one set of extents for the whole session, so the figure keeps its size across clips.
    const { span, extent } = state;
    const scale = (Math.min(width, height) * 0.3) / span;
    // Frames stepped since the last draw; the loop point (wrap-around) is not travel.
    const steps = (frame - state.frame + state.clip.frames) % state.clip.frames;
    const speedPx = steps > 0 && steps < state.clip.frames / 2
        ? state.clip.speed[frame] * steps * (Math.min(width, height) * 0.3) / span
        : 0;
    const pad = 6;
    const bounds = {
        left: extent.maxSide * scale + pad,
        right: width - extent.maxSide * scale - pad,
        top: extent.maxUp * scale + pad,
        bottom: height - extent.maxDown * scale - pad,
    };
    updateBody(performance.now(), width, height, speedPx, state.playing, bounds);
    const cx = body.x;
    const cy = body.y;
    // Face the direction of travel: mirror the side view when walking left. The flip waits until the heading is
    // clearly past vertical, so a heading that hovers near it does not flicker the figure.
    if (body.facing === 1 && Math.cos(body.heading) < -FACE_BAND) body.facing = -1;
    else if (body.facing === -1 && Math.cos(body.heading) > FACE_BAND) body.facing = 1;
    const facing = body.facing;
    const offsets = pelvisOffsets(state.clip, frame);
    if (state.blend) {
        // Ease the pose from the outgoing clip's last frame into this clip over its first frames.
        const weight = Math.max(0, 1 - frame / BLEND_FRAMES);
        const eased = weight * weight * (3 - 2 * weight);
        for (let i = 0; i < offsets.length; i++) offsets[i] += (state.blend[i] - offsets[i]) * eased;
        if (frame >= BLEND_FRAMES) state.blend = null;
    }

    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--ground').trim() || '#b9b3a6';
    for (const obstacle of OBSTACLES) {
        ctx.fillRect(obstacle.x * width, obstacle.y * height, obstacle.w * width, obstacle.h * height);
    }
    // A shadow under the feet keeps the figure reading as standing on the stage while it walks.
    ctx.beginPath();
    ctx.ellipse(cx, cy + extent.maxDown * scale, extent.maxSide * scale * 0.8 + 4, 3, 0, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--ink').trim() || '#1d1b18';
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = Math.max(2, scale * 0.04);
    ctx.lineCap = 'round';
    // Side view of the skeleton at the pelvis position; copies one stage away keep it whole at the edges.
    // Pin this frame's lowest joint to the ground line the shadow sits on, so a frame that rises off its floor
    // or sinks into it neither floats nor cuts through the shadow. Maximum drop over the session sets the line.
    let lowest = 0;
    for (let joint = 0; joint < JOINTS; joint++) lowest = Math.min(lowest, offsets[joint * 2 + 1]);
    const lift = (extent.maxDown + lowest) * scale;
    const toScreen = (joint, dx, dy) => [
        cx + dx + facing * offsets[joint * 2] * scale,
        cy + dy + lift - offsets[joint * 2 + 1] * scale,
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

// Clips are generated ahead of playback: each new clip chains from the last generated one, not the one on screen,
// so the queue can run PREFETCH clips deep and the figure never stops at a boundary waiting for the model.
const WALK_DISTANCE = 1.5; // metres each clip walks ahead
const WANDER_TURN = 0.8; // radians of random turn available per clip
const PREFETCH = 1; // one clip ahead: the next clip generates while the current one plays

function fillQueue() {
    if (!state.standing || state.generating || state.queue.length >= PREFETCH) return;
    produce();
}

async function produce() {
    if (!state.standing || state.generating) return;
    state.generating = true;
    setStatus('Generating a motion clip', 'generating');
    try {
        // The source is the last generated clip's last frames, placed where that clip ended; the first clip starts from standing.
        const previous = state.tail;
        const source = previous
            ? sourceFeatures(previous.values, previous.frames, state.parents, state.neutral)
            : {
                globalRoot: copyOf(state.standing.globalRoot),
                localRoot: copyOf(state.standing.localRoot),
                pose: copyOf(state.standing.pose),
                origin: { x: 0, z: 0 },
                yaw: 0,
            };
        // Each clip walks to a point ahead of the last one, with a small random turn, so the figure keeps moving.
        aimTargets(source, { distance: WALK_DISTANCE, turn: (Math.random() - 0.5) * WANDER_TURN });
        const motion = await request({
            type: 'infer',
            globalRoot: source.globalRoot,
            localRoot: source.localRoot,
            pose: source.pose,
            seed: state.seed,
            freeTarget: false,
        });
        state.seed += 1;
        // The model's clip starts with the source frames, which repeat the previous clip's last frames: play it after them.
        const skip = previous ? SOURCE_FRAMES : 0;
        const placed = placeFrames(motion.values, motion.frames, source.origin, source.yaw);
        const clip = prepareClip({ frames: motion.frames - skip, values: placed.subarray(skip * FRAME_VALUES) });
        if (state.span === null) state.span = clip.span;
        state.extent = {
            maxUp: Math.max(state.extent.maxUp, clip.maxUp),
            maxDown: Math.max(state.extent.maxDown, clip.maxDown),
            maxSide: Math.max(state.extent.maxSide, clip.maxSide),
        };
        state.tail = clip;
        state.queue.push(clip);
        if (!state.clip) {
            advanceClip();
            drawFrame(0);
            startPlayback();
        }
        setStatus(`Generated ${clip.frames} frames`, 'ready');
    } catch (error) {
        fail(`Cannot generate a motion: ${error.message}`);
    } finally {
        state.generating = false;
        fillQueue();
    }
}

// Starts the next queued clip. The outgoing pose is captured so the new clip eases in from it. False when none is ready.
function advanceClip() {
    const next = state.queue.shift();
    if (!next) return false;
    state.blend = state.clip ? pelvisOffsets(state.clip, state.clip.frames - 1) : null;
    state.clip = next;
    state.frame = 0;
    state.clipCount += 1;
    document.body.dataset.clip = String(state.clipCount);
    els.scrub.max = String(next.frames - 1);
    els.scrub.disabled = false;
    els.play.disabled = false;
    document.body.dataset.frames = String(next.frames);
    return true;
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
    // One loop at a time: a second would draw each tick twice, and one is added per generated clip.
    if (!state.looping) {
        state.looping = true;
        requestAnimationFrame(tick);
    }
}

function stopPlayback() {
    state.playing = false;
    els.play.textContent = 'Play';
    els.play.setAttribute('aria-pressed', 'false');
}

function tick(now) {
    if (!state.playing || !state.clip) {
        state.looping = false;
        return;
    }
    let position = state.startFrame + Math.floor(((now - state.startedAt) / 1000) * FPS);
    if (position >= state.clip.frames) {
        const overrun = position - state.clip.frames;
        if (advanceClip()) {
            // The next clip plays from its first frame, keeping any time the previous one overran.
            state.startedAt = now - (overrun * 1000) / FPS;
            state.startFrame = 0;
            position = overrun;
        } else {
            // Nothing is ready: the last frame holds until the next clip is queued.
            state.startedAt = now;
            state.startFrame = state.clip.frames - 1;
            position = state.clip.frames - 1;
        }
    }
    fillQueue();
    const frame = Math.min(position, state.clip.frames - 1);
    if (frame !== state.frame) drawFrame(frame);
    requestAnimationFrame(tick);
}

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
