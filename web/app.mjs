// Page logic: starts the worker, draws the generated clip as a stick figure, and wires the controls.
import { motionPositions, JOINTS, FRAME_VALUES } from './lib/skeleton.mjs';
import { SOURCE_FRAMES, aimTargets, headingAt, placeFrames, sourceFeatures } from './lib/continuity.mjs';

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
    failed: false,
    agent: false, // the threaded core's agent is ready: plan motion with commands instead of inference
    started: false,
    clipCount: 0, // clips started so far; mirrored to body[data-clip] for the smoke test
    span: null, // one scale for the whole session, set from the first clip
    extent: { maxUp: 0, maxDown: 0, maxSide: 0 }, // largest reach over all clips so far
    blend: null, // the outgoing clip's pelvis-relative pose, eased into the next clip
    looping: false,
    ref: null, // the reference: one continuous timeline of the motion being played ({ values, frames })
    playhead: 0, // position on the reference, in frames
    planning: false, // a plan is in flight
    lookahead: 48, // frames ahead of the playhead where the next plan's seam is placed
    lastTick: 0,
    lastPlanAt: 0, // playhead when the last plan was requested
    agentBase: 0, // the reference frame where the worker's agent plan starts
    walkDir: 1, // direction of the walk along the session's forward axis: +1 or -1
    legEnd: null, // where the current leg of the walk ends, in metres along the session's forward axis
    goalPx: null, // a clicked point on the stage in pixels, until the figure reaches it
};

function setStatus(text, stateName) {
    els.status.textContent = text;
    document.body.dataset.state = stateName;
}

function fail(message) {
    state.failed = true;
    setStatus(message, 'error');
    els.play.disabled = true;
    els.scrub.disabled = true;
}

const worker = new Worker(new URL('./worker.js', import.meta.url));

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
        state.agent = data.agent === true;
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

worker.onerror = (event) => fail(`Cannot run the motion model: ${event.message || 'worker failed'} (${event.filename || 'worker script did not load'})`);

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

const wrapAngle = (angle) => Math.atan2(Math.sin(angle), Math.cos(angle));

// The stage is a floor the camera follows the figure across. The figure's place along the floor is the model's own
// root trajectory in the session frame, so the floor and blocks move past it as it walks. Blocks are placed in that
// frame, so the model's walk is what decides whether it reaches one.
const FLOOR_FRACTION = 0.8; // floor height as a share of the stage height
const FIGURE_FRACTION = 0.4; // the figure is held at this share of the stage width
const WORLD_BLOCKS = [ // metres along the walk in the session frame: start, width, height
    { z: 8.0, w: 0.5, h: 0.45 },
    { z: 16.0, w: 0.5, h: 0.35 },
    { z: 24.0, w: 0.5, h: 0.5 },
];
const WORLD_LIMITS = { min: -2.0, max: 30.0 }; // metres: the walk turns round at these
const LEG_METRES = 40; // a leg runs until a block or a limit: the figure changes direction only there
const FIGURE_HALF = 0.3; // metres from the pelvis to the figure's front or back
const BLEND_FRAMES = 8; // frames over which a new clip eases in from the outgoing pose

function resizeCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const width = els.canvas.clientWidth;
    const height = els.canvas.clientHeight;
    els.canvas.width = Math.round(width * ratio);
    els.canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawFrame(state.frame);
}

// Pixels per metre of the session's figure: one scale for the whole session.
function stageScale() {
    if (state.span === null) return 1;
    return (Math.min(els.canvas.clientWidth, els.canvas.clientHeight) * 0.3) / state.span;
}

// Metres the figure can walk from z in a direction before a block or a limit of the walk stops it.
function runFrom(z, direction) {
    let run = direction > 0 ? WORLD_LIMITS.max - z : z - WORLD_LIMITS.min;
    for (const block of WORLD_BLOCKS) {
        if (direction > 0 && block.z >= z) run = Math.min(run, block.z - FIGURE_HALF - z);
        if (direction < 0 && block.z + block.w <= z) run = Math.min(run, z - (block.z + block.w) - FIGURE_HALF);
    }
    return run;
}

// Where the next clip walks: its direction and the distance it aims for. A clicked point sets the direction until the
// figure reaches it. A block or a limit closer than a clip's stride shortens the clip, so the model stops short of it;
// when the way ahead is shorter than a stride and the other way has more room, the figure turns round.
function chooseWalk() {
    const tail = state.tail;
    const z = tail ? tail.values[(tail.frames - 1) * FRAME_VALUES + 2] : 0;
    let dir = state.walkDir;
    if (state.goalPx !== null) {
        const goal = z + (state.goalPx - els.canvas.clientWidth * FIGURE_FRACTION) / stageScale();
        if (Math.abs(goal - z) < 0.1) state.goalPx = null;
        else {
            dir = Math.sign(goal - z);
            state.legEnd = goal;
        }
    }
    // A leg has one end. The direction changes only when the leg ends (at its far point, a block or a limit), so the
    // plan does not flip from one request to the next. The end is set once per leg and planning only reads it.
    const reached = state.legEnd !== null && (dir > 0 ? z >= state.legEnd - 0.05 : z <= state.legEnd + 0.05);
    if (state.legEnd === null || reached) {
        if (state.legEnd !== null && state.goalPx === null) dir = -dir;
        if (runFrom(z, dir) < 0.2 && runFrom(z, -dir) > runFrom(z, dir)) dir = -dir;
        state.legEnd = z + dir * Math.min(LEG_METRES, runFrom(z, dir));
    }
    state.walkDir = dir;
    const remaining = Math.abs(state.legEnd - z);
    // nearStop: the leg's end is within one stride, so the planner gets it as a world target to stop at.
    return { direction: dir, distance: Math.max(0.2, Math.min(WALK_DISTANCE, remaining)), nearStop: remaining < WALK_DISTANCE };
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
    const floorY = height * FLOOR_FRACTION;
    // The camera holds the figure: its pelvis is the model's root at this frame, and the floor moves past it.
    const rootZ = state.clip.values[frame * FRAME_VALUES + 2];
    const toStage = (z) => width * FIGURE_FRACTION + (z - rootZ) * scale;
    // The pelvis is placed at the model's own root height: the floor is the model's ground plane, as in the upstream
    // demo, which writes the model's root into the robot's pose without adjusting it.
    const cx = width * FIGURE_FRACTION;
    const cy = floorY - state.clip.values[frame * FRAME_VALUES + 1] * scale;
    // The side view always faces the way the model walks: the model is aimed along the floor.
    const facing = 1;
    const offsets = pelvisOffsets(state.clip, frame);
    if (state.blend) {
        // Ease the pose from the outgoing clip's last frame into this clip over its first frames.
        const weight = Math.max(0, 1 - frame / BLEND_FRAMES);
        const eased = weight * weight * (3 - 2 * weight);
        for (let i = 0; i < offsets.length; i++) offsets[i] += (state.blend[i] - offsets[i]) * eased;
        if (frame >= BLEND_FRAMES) state.blend = null;
    }

    const ground = getComputedStyle(document.body).getPropertyValue('--ground').trim() || '#b9b3a6';
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--ink').trim() || '#1d1b18';
    ctx.fillRect(0, floorY, width, 2);
    ctx.fillStyle = ground;
    for (const block of WORLD_BLOCKS) {
        const left = toStage(block.z);
        const right = toStage(block.z + block.w);
        ctx.fillRect(left, floorY - block.h * scale, right - left, block.h * scale);
    }
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--ink').trim() || '#1d1b18';

    ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--ink').trim() || '#1d1b18';
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = Math.max(2, scale * 0.04);
    ctx.lineCap = 'round';
    // Side view of the skeleton at the pelvis position; copies one stage away keep it whole at the edges.
    const toScreen = (joint, dx, dy) => [
        cx + dx + facing * offsets[joint * 2] * scale,
        cy + dy - offsets[joint * 2 + 1] * scale,
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
const PREFETCH = 1; // one clip ahead: the next clip generates while the current one plays

function fillQueue() {
    if (state.agent) {
        planAhead();
        return;
    }
    if (!state.standing || state.generating || state.queue.length >= PREFETCH) return;
    produce();
}

// Queues a generated clip; the first clip starts playback.
function enqueue(clip) {
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
}

async function produce() {
    if (!state.standing || state.generating) return;
    state.generating = true;
    // Only the first clip is awaited on screen; later clips are prefetched while the current one plays.
    if (!state.clip) setStatus('Generating a motion clip', 'generating');
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
        // Aim the clip along the floor: the model walks toward the target, so its heading turns to the chosen direction.
        const walk = chooseWalk();
        const heading = previous ? headingAt(previous.values, previous.frames - 1) : 0;
        aimTargets(source, { distance: walk.distance, turn: wrapAngle((walk.direction > 0 ? 0 : Math.PI) - heading) });
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
        enqueue(clip);
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
    state.lastTick = 0;
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

// The reference is the motion being played, one continuous timeline. Playback never holds: the playhead moves on the
// reference, and a plan is requested whenever fewer than PLAN_BELOW frames remain ahead of it. As the upstream
// streaming server does, a plan is placed at a seam ahead of the playhead, seeded with the four reference frames at
// the seam, and its first frames are blended into the reference.
const PLAN_BELOW = 40; // frames of reference left ahead of the playhead that trigger the next plan
const PLAN_CADENCE = 16; // frames of playback between plan requests, the demo's walk cadence
const TRANSITION_FRAMES = 8; // frames over which a plan is blended into the reference at its seam
const START_FRAMES = 80; // playback starts once the reference holds this many frames, so the next plan has room to arrive

function contextAt(seam) {
    const { values } = state.ref;
    const roots = new Float32Array(12);
    const rotations = new Float32Array(4 * 136);
    for (let i = 0; i < 4; i++) {
        const base = (seam + i) * FRAME_VALUES;
        roots.set(values.subarray(base, base + 3), i * 3);
        rotations.set(values.subarray(base + 3, base + FRAME_VALUES), i * 136);
    }
    return { roots, rotations };
}

async function planAhead() {
    if (state.planning) return;
    state.planning = true;
    try {
        const ref = state.ref;
        // The plan takes over from the frame on screen, as the demo's replans do: its first frame is the reference at the
        // seam, so it continues the motion the figure is showing.
        // Before playback starts a plan extends the reference from its end; after that it takes over at the playhead.
        // The plan starts at the frame on screen, as the demo's web client does. Before playback starts it extends the
        // reference from its end.
        const seam = ref ? (state.started ? Math.floor(state.playhead) : Math.max(0, ref.frames - 4)) : 0;
        const contextSeam = ref ? Math.min(seam, ref.frames - 4) : 0;
        state.lastPlanAt = state.playhead;
        const walk = chooseWalk();
        const motion = await request({
            type: 'plan',
            style: 'walk',
            direction: walk.direction,
            speed: -1,
            target: walk.nearStop ? { z: state.legEnd } : null,
            seed: state.seed,
            context: ref ? contextAt(contextSeam) : null,
            advance: ref ? seam - state.agentBase : null,
        });
        state.seed += 1;
        mergePlan(seam, motion);
        state.agentBase = seam;
    } catch (error) {
        fail(`Cannot generate a motion: ${error.message}`);
    } finally {
        state.planning = false;
    }
    // Before playback starts nothing ticks, so keep planning until the reference is long enough.
    if (!state.started && state.ref && !state.failed) planAhead();
}

// Places a plan on the reference. The plan replaces the reference from the frame on screen onward, blended in over
// TRANSITION_FRAMES so the switch is at the playhead, not at a seam far ahead of it.
function mergePlan(seam, motion) {
    const old = state.ref;
    const n = motion.frames;
    const total = seam + n;
    const cut = old ? Math.max(seam, Math.floor(state.playhead)) : seam;
    if (cut >= total) return; // the playhead has passed the whole plan: nothing to show from it
    const values = new Float32Array(total * FRAME_VALUES);
    if (old) values.set(old.values.subarray(0, Math.min(cut, old.frames) * FRAME_VALUES));
    values.set(motion.values.subarray((cut - seam) * FRAME_VALUES), cut * FRAME_VALUES);
    if (old) {
        const blend = Math.min(TRANSITION_FRAMES, total - cut, Math.max(0, old.frames - cut));
        for (let i = 0; i < blend; i++) {
            const at = (cut + i) * FRAME_VALUES;
            const u = i / (TRANSITION_FRAMES - 1);
            const eased = u * u * (3 - 2 * u);
            for (let k = 0; k < 3; k++) values[at + k] = old.values[at + k] + eased * (values[at + k] - old.values[at + k]);
            for (let joint = 0; joint < 34; joint++) {
                const q = at + 3 + joint * 4;
                let dot = 0;
                for (let k = 0; k < 4; k++) dot += old.values[q + k] * values[q + k];
                const sign = dot < 0 ? -1 : 1;
                const out = [0, 0, 0, 0];
                let norm = 0;
                for (let k = 0; k < 4; k++) {
                    out[k] = (1 - eased) * old.values[q + k] + eased * sign * values[q + k];
                    norm += out[k] * out[k];
                }
                norm = Math.sqrt(norm) || 1;
                for (let k = 0; k < 4; k++) values[q + k] = out[k] / norm;
            }
        }
    }
    state.ref = { values, frames: total };
    state.tail = state.ref;
    state.clip = prepareClip({ frames: total, values });
    if (state.span === null) state.span = state.clip.span;
    state.extent = {
        maxUp: Math.max(state.extent.maxUp, state.clip.maxUp),
        maxDown: Math.max(state.extent.maxDown, state.clip.maxDown),
        maxSide: Math.max(state.extent.maxSide, state.clip.maxSide),
    };
    els.scrub.max = String(total - 1);
    els.scrub.disabled = false;
    els.play.disabled = false;
    state.clipCount += 1;
    document.body.dataset.clip = String(state.clipCount);
    document.body.dataset.frames = String(n);
    if (!state.started && total >= START_FRAMES) {
        state.started = true;
        // Start at the newest plan's first frame: the agent's current plan begins there, so the next plan continues it.
        state.playhead = seam;
        drawFrame(0);
        startPlayback();
    }
    setStatus(`Generated ${n} frames`, 'ready');
}

function tickAgent(now) {
    if (!state.playing) {
        state.looping = false;
        state.lastTick = 0;
        return;
    }
    const dt = state.lastTick ? Math.min(0.1, (now - state.lastTick) / 1000) : 0;
    state.lastTick = now;
    state.playhead = Math.min(state.playhead + dt * FPS, state.ref.frames - 1);
    if (state.ref.frames - state.playhead < PLAN_BELOW || state.playhead - state.lastPlanAt >= PLAN_CADENCE) planAhead();
    const frame = Math.floor(state.playhead);
    if (frame !== state.frame) drawFrame(frame);
    requestAnimationFrame(tick);
}

function tick(now) {
    if (state.agent && state.ref) {
        tickAgent(now);
        return;
    }
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

els.canvas.addEventListener('pointerdown', (event) => {
    const rect = els.canvas.getBoundingClientRect();
    state.goalPx = event.clientX - rect.left;
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
const styleManifestUrl = new URL('./model/styles/manifest.json', location.href);
const styleManifest = await (await fetch(styleManifestUrl)).json();
worker.postMessage({ type: 'init', manifest, styleManifest });
