# Game integration failure modes for a generative per-frame motion model (MotionBricks-style)

Scope: real-time humanoid characters. Symptoms under study: floating or drifting off the ground, per-frame spazzing or snapping. Research budget: about 22 tool calls, so several primary sources could not be read in full (see Gaps in each section). Years are given per source.

## Q1. What causes floating or drift, and how is each cause diagnosed?

### Takeaway
MotionBricks-style models emit global root values (projected position plus cos/sin of pelvis heading) and global joint rotations, with contact labels as part of the state. Floating and drift come from a root or pelvis that is not reconciled with ground height and gameplay position, and from coordinate-frame mistakes between the model output and the engine. The paper does not document a foot-locking or IK step, so the game must supply one.

### Cited Findings
- MotionBricks state representation: "global root values: projected global positions and the cosine and sine of the heading angle of the pelvis joint"; "local and global root values are interconvertible"; joint rotations are represented "in global coordinates to avoid canonicalization with ill-defined root headings" — [MotionBricks arXiv 2604.24833 (2026), HTML v1, Sec. 4](https://arxiv.org/html/2604.24833v1)
- The state includes "joint velocities v and contact labels c"; foot contact accuracy is measured against ground-truth labels — [MotionBricks arXiv 2604.24833, Sec. 4 and 7.2](https://arxiv.org/html/2604.24833v1)
- MotionBricks itself lists penetration (m) as a metric ("measures ground penetration depth") and says "Kinematic planners like MotionBricks may generate physically implausible motions (self-collisions)" — [MotionBricks arXiv 2604.24833, Sec. 7.2 and 8.1](https://arxiv.org/html/2604.24833v1)
- The MotionBricks UE5 demo uses "UE5's built-in real-time retargeter" for on-the-fly retargeting, and the authors call runtime retargeting "fast but low-quality" — [MotionBricks arXiv 2604.24833, Sec. 7.4 and 8.1](https://arxiv.org/html/2604.24833v1)
- Root interpolation is described as "enabling precise global keyframe matching without introducing foot sliding artifacts"; the decoder can "refine the predicted root trajectory, producing cleaner footsteps" — [MotionBricks arXiv 2604.24833, Sec. 4.2 and 7.3](https://arxiv.org/html/2604.24833v1)
- Motion Matching in Unreal: locomotion animations "must contain root motion"; foot bones (`foot_l`, `foot_r`) are matched through Pose Channel flags; the "Crashing Legs" channel is experimental and "should not be relied upon in production" — [Unreal Engine docs, Motion Matching in Unreal Engine (current docs)](https://dev.epicgames.com/documentation/en-us/unreal-engine/motion-matching-in-unreal-engine)
- Foot skate is a standard metric defined as "undesirable sliding of feet during ground contact" (m/frame) — [MotionBricks arXiv 2604.24833, Sec. 7.2](https://arxiv.org/html/2604.24833v1)
- Footskate cleanup as a post-process: the SCA 2002 method "exactly satisfies footplant constraints while introducing only smooth changes" and allows small leg-length changes to avoid artifacts — [Kovar, Schreiner, Gleicher, Footskate Cleanup for Motion Capture Editing (SCA 2002)](https://research.cs.wisc.edu/graphics/Gallery/kovar.vol/Cleanup)
- Robust Motion In-betweening (2020 venue, arXiv 2102.04942): the network predicts foot contacts, and the MotionBuilder plugin exposes an "Enable IK" option guided by those predictions — [Harvey et al., Robust Motion In-betweening, arXiv 2102.04942 (2020/2021)](https://arxiv.org/pdf/2102.04942)

### Inferences
- A model that outputs root heading as cos/sin and a projected global position is stable only if the engine re-projects the root onto the ground each frame. Drift grows when the game integrates the model's root velocity in addition to the model's global position (double integration), or when the model's global frame and the engine's world frame differ by a rotation or axis swap. Both produce a steady lateral drift or heading error that looks like "drifting off the ground".
- Floating (feet above the ground) most often comes from a pelvis or root height that is taken from the model but never corrected against the ground sample under the feet. Because the model does not guarantee zero foot height at contact (only a contact label), a constant offset of a few centimetres is expected and needs correction.
- Diagnosis: log, per frame, (a) min foot height over frames where the model contact label is 1, (b) the root's world Y minus the ground sample under the root, and (c) the difference between model-integrated root position and the game's root position. A nonzero, slowly growing (c) is drift; a constant nonzero (b) with contacts active is floating.

### Gaps
- The MotionBricks HTML was truncated at about 100,000 of 141,000 characters; the remainder (possibly including the coordinate convention, joint rotation format, and any post-processing) was not read. The axis convention (Y-up or Z-up) is not stated in the portion read.
- No primary source retrieved describes the exact root-to-ground reconciliation used by MotionBricks in the UE5 demo.

## Q2. What causes per-frame spazzing or snapping, and what are the recommended fixes?

### Takeaway
Per-frame jitter comes from three sources: state-dependent replanning (MotionBricks replans at 10 Hz or when commands change), a discontinuous jump in the pose or root when a new plan is applied, and frame-rate-dependent smoothing. The fixes are exact dt-independent smoothing, fixed-step simulation with interpolation, and blending or inertialization across plan switches.

### Cited Findings
- MotionBricks replans "at 10 Hz or whenever commands change" and "replanning is triggered whenever the control signals are changed" — [MotionBricks arXiv 2604.24833, Sec. 3 and 7.4](https://arxiv.org/html/2604.24833v1). This is a direct source for a periodic seam every 100 ms and a snap on every command change.
- The paper's Joint Jitter and Root Jitter are measured as acceleration in m/s^2 — [MotionBricks arXiv 2604.24833, Sec. 7.2](https://arxiv.org/html/2604.24833v1). Use the same metric in game-side tests.
- The smart locomotion root estimate is produced "by linearly extrapolating the current state using the user's commanded velocity and heading" — [MotionBricks arXiv 2604.24833, Sec. 6.1](https://arxiv.org/html/2604.24833v1). A command change therefore moves the root by a new extrapolation, which is a likely snap source if the game also smooths input.
- Frame-rate-dependent damping: a fixed per-frame blend "moves objects more slowly at lower framerates", and multiplying by dt is unstable once damping*dt > 1. The fix is to fix the half-life: x(t+dt) = lerp(x, g, 1 − 2^(−dt/halflife)) — [Holden, "Spring-It-On: The Game Developer's Spring-Roll-Call" (2021)](https://www.theorangeduck.com/page/spring-roll-call)
- Fixed-timestep accumulator: "the renderer produces time and the simulation consumes it in discrete dt sized steps"; cap steps per frame to avoid the "spiral of death"; interpolate between previous and current state with alpha = accumulator/dt — [Fiedler, "Fix Your Timestep!" (2004)](https://www.gafferongames.com/post/fix_your_timestep/)
- Learned Motion Matching (SIGGRAPH 2020) replaces Motion Matching's database with a neural model and keeps the "quality, control, and quick iteration time" of Motion Matching — [Holden et al., Learned Motion Matching, SIGGRAPH 2020](https://www.theorangeduck.com/page/learned-motion-matching). The page abstract does not describe its blending method.
- Quaternion double cover: q and -q encode the same rotation; a sign flip between consecutive frames produces a 360-degree-looking flip if interpolated naively. SciPy's `Rotation.as_quat(canonical=True)` picks the sign with positive w, and its default is `canonical=False` — [SciPy docs, Rotation.as_quat](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html); the double cover is also described in [arXiv 1409.1559](https://arxiv.org/pdf/1409.1559).

### Inferences
- The most likely spazz is the 10 Hz replan seam combined with a sudden change in root extrapolation. Recommended fix: keep the generated root and pose, but apply a dt-independent critical spring or inertialization to the displayed pose and root offset when a new plan arrives. The spring form from Holden (2021) is directly applicable.
- Quaternion sign flips between consecutive frames will look like a snap in the spine or hips. Canonicalize each frame's quaternion relative to the previous frame (negate if the dot product is negative) before any smoothing.
- Input hysteresis (a dead zone and a rate limit on commanded velocity or heading) should be applied before the model input, not after the output, so the model sees a smooth command.
- Heading wrap-around: integrate heading as an angle and wrap to (-pi, pi]; never smooth raw heading angles across the ±pi seam. Smooth (cos, sin) pairs or the angular difference instead.

### Gaps
- The MotionBricks paper does not describe its own post-plan blending; the game must supply it.
- The blending or inertialization method in Learned Motion Matching was not retrieved (the PDF exceeded the fetch size limit).
- Robust Motion In-betweening's IK solver details were not retrieved.

## Q3. How do shipped engines handle root and foot consistency in motion matching?

### Takeaway
Unreal's motion matching expects root-motion locomotion data and handles foot consistency through pose-matching of foot bones and optional IK post-processing. Unity's Animation Rigging exposes Two Bone IK with a target, hint, and weights, but does not document its update order in the page read. Godot's TwoBoneIK3D requires a pole target and constrains the hierarchy. The engine docs read did not give enough detail on ground-contact placement; treat the engine-specific claims below as partly unverified.

### Cited Findings
- Unreal Motion Matching: "Animations used in a Motion Matching setup for locomotion must contain root motion" — [Unreal Engine docs, Motion Matching in Unreal Engine](https://dev.epicgames.com/documentation/en-us/unreal-engine/motion-matching-in-unreal-engine)
- Unreal Motion Matching: the Trajectory Channel samples the movement path and "is designed to be used in conjunction with the Character Trajectory blueprint component"; Pose Channel foot bones are configurable — [same page](https://dev.epicgames.com/documentation/en-us/unreal-engine/motion-matching-in-unreal-engine)
- Unreal Motion Matching: "For poses that coverage doesn't handle, ... Animation Warping can be used alongside Motion Matching" — [same page](https://dev.epicgames.com/documentation/en-us/unreal-engine/motion-matching-in-unreal-engine)
- Unity Two Bone IK: Target is "the desired position of the Tip"; Hint "specifies the direction the limb should be oriented when it bends"; Weight 0 means no influence; Target Position Weight 1 lets Root and Mid "extend as far as necessary" — [Unity Animation Rigging 1.2, Two Bone IK constraint](https://docs.unity3d.com/Packages/com.unity.animation.rigging@1.2/manual/constraints/TwoBoneIKConstraint.html)
- Godot TwoBoneIK3D: `set_target_node` "Sets the target node that the end bone is trying to reach"; a pole target is required; "The pole direction and the forward vector shouldn't be colinear to avoid unintended rotation"; "The middle bone must be a child of the root bone" — [Godot docs, TwoBoneIK3D](https://docs.godotengine.org/en/stable/classes/class_twoboneik3d.html)
- Clavet's motion matching (GDC 2016, For Honor), the predecessor that Learned Motion Matching replaces, was described as finding "the frame in the mocap database that simultaneously matches the current pose and the desired future plan" with "a small blend time" — [GDC Vault, Clavet, Motion Matching and The Road to Next-Gen Animation (GDC 2016)](https://gdcvault.com/play/1022985/Motion-Matching-and-The-Road), via a vault summary; full slides not read.

### Inferences
- In the shipped motion-matching model, root and foot consistency come from data (root motion baked into clips, foot-bone features in the pose search) rather than from a runtime ground solve. A learned generative model like MotionBricks lacks that data-side guarantee, so the game needs a runtime foot or pelvis correction that the shipped engines apply as a post-process (Two Bone IK on feet, or a Full Body IK-style pelvis drop).
- Unity and Godot two-bone IK are per-limb and need the foot target placed on the ground sample under each foot; the pelvis must be lowered first, or the leg IK will overextend and pop.

### Gaps
- The Unreal Foot Placement IK and Full Body IK pages returned empty tables of contents; their ground trace, interpolation, and update-order details were not read.
- Unity's update-order page (ScriptExecutionOrder) returned 404; the order of Animator vs. IK constraints was not verified.
- Godot import-axis conventions and Unreal's coordinate-system reference page were not readable through the fetch tool.

## Q4. Concrete diagnostic checks that can run in code

### Takeaway
A minimal per-frame assertion set catches most floating, drift, and snap bugs. Checks are listed in order of diagnostic power. They are engineering recommendations derived from the sources above, not claims from a single cited source.

### Cited Findings
- Quaternion unit norm and sign: q and -q are the same rotation; canonicalize sign before comparison — [arXiv 1409.1559](https://arxiv.org/pdf/1409.1559); [SciPy as_quat](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html)
- Foot skate defined as foot sliding during contact (m/frame) and penetration as ground depth (m) — [MotionBricks arXiv 2604.24833, Sec. 7.2](https://arxiv.org/html/2604.24833v1)
- Jitter as acceleration (m/s^2) for joints and root — [MotionBricks arXiv 2604.24833, Sec. 7.2](https://arxiv.org/html/2604.24833v1)
- dt-independence test: run the same command sequence at two frame rates and compare trajectories — implied by the half-life smoothing argument — [Holden 2021](https://www.theorangeduck.com/page/spring-roll-call)

### Inferences (engineering checks to implement)
1. Unit quaternion: abs(|q| - 1) < 1e-4 for every bone every frame, after the model output and after any blend. Failure means a non-normalized blend or a bad conversion.
2. Quaternion continuity: dot(q_t, q_{t-1}) >= 0 for every bone. Negative values mean a sign flip; canonicalize.
3. Heading continuity: wrapped angular difference |wrap(h_t - h_{t-1})| < threshold (for example 0.2 rad at 60 Hz). Larger values are a snap or a wrap bug.
4. Foot height at contact: during frames with contact label 1, the foot's world Y minus the ground sample under it should be within about 0.01 m. Larger is floating (positive) or penetration (negative).
5. Root drift: model-integrated root minus game root, per frame, must stay bounded; a monotonic growth indicates a double integration or frame mismatch.
6. Axis and unit sanity: confirm the model's up axis and units (meters vs. centimeters) by measuring a known standing pose; the pelvis height at rest should match the rig's expected height. A factor of 100 or a 90-degree rotation is immediately visible here.
7. Teleport or command-jump reset: when the commanded velocity changes by more than a threshold, reset smoothing state rather than letting a spring carry the old pose across the jump.
8. Timestep stress test: run the update at 30, 60, and 144 Hz with the same input and compare the foot-height and root metrics.

### Gaps
- No primary source gives recommended numeric thresholds for these checks; the thresholds above are starting values to be tuned on the target rig.
- The MotionBricks paper's own metric thresholds (for example, what counts as acceptable foot skate) were not extracted.

## Source list (primary unless noted)

- MotionBricks, arXiv 2604.24833 (2026): https://arxiv.org/html/2604.24833v1 (read to 100,000 of 141,000 characters)
- Holden et al., Learned Motion Matching, SIGGRAPH 2020: https://www.theorangeduck.com/page/learned-motion-matching (project page; PDF not read)
- Harvey et al., Robust Motion In-betweening, 2020 venue / arXiv 2102.04942 (2021): https://arxiv.org/pdf/2102.04942 (search summary only)
- Kovar, Schreiner, Gleicher, Footskate Cleanup for Motion Capture Editing, SCA 2002: https://research.cs.wisc.edu/graphics/Gallery/kovar.vol/Cleanup (search summary only)
- Clavet, Motion Matching and The Road to Next-Gen Animation, GDC 2016: https://gdcvault.com/play/1022985/Motion-Matching-and-The-Road (vault summary only)
- Grassia, Practical Parameterization of Rotations Using the Exponential Map, Journal of Graphics Tools, 1998: referenced via a TUM-hosted reprint (search summary only)
- Fiedler, Fix Your Timestep!, 2004: https://www.gafferongames.com/post/fix_your_timestep/
- Holden, Spring-It-On: The Game Developer's Spring-Roll-Call, 2021: https://www.theorangeduck.com/page/spring-roll-call
- Unreal Engine docs, Motion Matching in Unreal Engine (current): https://dev.epicgames.com/documentation/en-us/unreal-engine/motion-matching-in-unreal-engine
- Unity Animation Rigging 1.2, Two Bone IK constraint: https://docs.unity3d.com/Packages/com.unity.animation.rigging@1.2/manual/constraints/TwoBoneIKConstraint.html
- Godot docs, TwoBoneIK3D: https://docs.godotengine.org/en/stable/classes/class_twoboneik3d.html
- SciPy docs, Rotation.as_quat: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html

## Unresolved items for the report writer
- The MotionBricks coordinate convention (up axis, handedness, units) and joint rotation format are unverified; the rest of the paper must be read before fixing them.
- The Unreal Foot Placement and Full Body IK pages, Unity script execution order, and Godot axis-import docs were not retrieved. Do not cite them as verified.
- Several sources were read only through search summaries (Clavet GDC 2016, Robust Motion In-betweening, Kovar 2002, Grassia 1998). Confirm their specifics against the primary PDFs before quoting numbers.
