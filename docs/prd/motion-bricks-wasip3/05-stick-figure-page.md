# PRD-05: Stick-figure page rendering motion output

## Problem
The project's demo renders a full skinned/physics scene with Go and WebSocket streaming. The
requested result is a self-contained page that moves a stick figure from inference output.

## Requirements
1. Single static `index.html` plus the transpiled component from PRD-03 and model loader from
   PRD-04. No server, no WebSocket.
2. Use the 34-joint parent topology from the model (`parents` in the bundle, validated by the
   loader) to draw bones as line segments on a 2D canvas, projecting X/Z or X/Y.
3. Controls: a button to generate a motion (one inference call per clip), a play/pause, and a
   frame scrubber. Playback at 30 FPS as stated by the API.
4. Default inputs: a canonical standing root and a neutral pose, so the page works without any
   user input.
5. Degrade to an error message, not a blank page, when WebAssembly or the component fails.

## Acceptance
- In a headless browser, the page loads, generates one clip, and the canvas shows changing
  bone positions across frames (a pixel diff between frame 0 and frame N is non-zero).
- No console errors on a clean run.

## Risks
- Coordinate convention: the API expects Y-up metres at 30 FPS; the projection must match.
