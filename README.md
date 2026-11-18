# Virtual Background Removal — real-time background removal for video calls

MediaPipe's Selfie Segmentation (multiclass model, Tasks API) cuts the
person out of the frame live and composites them over a **blurred**
version of the real background, a **solid colour**, or a **chosen
image** — the Zoom / Meet-style virtual background, with an edge-feather
control and a **before/after split view**: ORIGINAL on the left,
VIRTUAL BACKGROUND on the right. A live person-coverage readout shows
how much of the frame is person at any moment.

## What it does

- **Segmentation:** MediaPipe ImageSegmenter with the multiclass
  selfie model (`selfie_multiclass_256x256.tflite`, 16 MB, shipped in
  this folder). The model has 6 classes — background, hair, body skin,
  face skin, clothes, accessories — and every class except background
  counts as person. It runs every 2nd frame (15 fps) and the mask is
  reused in between.
- **Backgrounds:** blurred real background (default), solid colour
  (deep teal), or an image — a built-in gradient scene, or any image
  file via `--background-image`.
- **Edge feather:** the person mask is Gaussian-blurred (default 9 px)
  so hair does not look cut out; `[` / `]` adjust it live.
- **Split view:** original vs composited side by side (S toggles to
  composited-only).
- **Coverage + event log:** live person-coverage percentage and a log
  line for every mode switch and feather change.

## Demo (real footage)

`virtual-background-removal-demo.mp4` — free-license Pexels footage
of a man talking to the camera at a desk in a glass office (Pexels
video 5941016): a busy real background — glass walls, ceiling lights,
red pillars, a desk plant — which is exactly what the blur has to
separate from. The clip is 16.5 s, resampled to 1280×720 at 30 fps,
looped to 30 s, blur mode. Verified 18 Nov 2026 — see
`virtual-background-removal-demo_summary.txt` for the exact counters:

- 30.0 s, 900 frames, **450 segmentations** (every 2nd frame)
- Person coverage: **mean 23.7%, min 23.1%, max 24.9%** — the seated
  man holds a steady share of the frame
- Background mode: blur, 30.0 s

H.264 copy in `../project-demos/linkedin-h264/` (codec-verified h264,
yuv420p, faststart); frames inspected at 5 s, 15 s and 25 s — the man
stays sharp, hands included, while the office blurs smoothly behind
him. The desk plant blurs too: it is background, not person, which is
correct.

Two other talking-head candidates were probed first: Pexels 7643836
(a man in front of flat window blinds — too plain a background to show
the effect) and 8135871 (a man in a plant-filled office, kept as a
spare). The glass-office clip was picked for the busiest background.

## Simulation (internal logic check only — NOT the demo)

`--simulate` renders a room (window, shelves, plant, rug) with a
face-like figure in front. The segmenter reads the rendered pixels —
the pipeline never sees the renderer's ground truth. Verified
18 Nov 2026 (30 s): 900 frames, 450 segmentations, but person
coverage **mean 0.1%** — MediaPipe's selfie model does not recognise
the flat drawn figure as a person (it was trained on photographs), so
the composited view is fully blurred. The simulation verifies the
plumbing only (segmentation calls, mask handling, compositing, modes,
outputs); the segmentation itself is verified on the real-footage demo
above.

## Honest limitations

- It is a segmentation model, not a detector: a person half out of
  frame is only half segmented, and a second person walking behind you
  becomes part of the "person" too.
- Segmentation costs ~0.4 s per call on the shared CPU used for
  verification, so a 30 s demo takes ~3 minutes wall time; on a
  faster machine, or with `--fps` lowered, it runs closer to real time.
- Fast hand motion can leave a faint edge halo for a frame, because
  the mask is refreshed every 2nd frame.
- The colour and image modes share the verified segmentation code but
  the shipped demo runs blur mode only; switch modes live with B / C /
  I in the display window.
- On Linux the MediaPipe Tasks API needs the stock OpenGL libraries
  (`libegl1`, `libgles2`); without them it fails at import with a
  `libEGL.so.1` error. On the Linux build machine the stock Ubuntu
  mesa libraries were extracted to a throwaway folder and pointed at
  with `LD_LIBRARY_PATH` — nothing installed system-wide.

## Weights

`selfie_multiclass_256x256.tflite` (16 MB, Google MediaPipe models)
ships in this folder. If it is missing, the script downloads it once
from the Google storage URL in the script header and caches it.

## Usage

```bash
pip install -r requirements.txt
python virtual_background_removal.py                  # file picker
python virtual_background_removal.py --source talk.mp4
python virtual_background_removal.py --source 0       # webcam
python virtual_background_removal.py --simulate --max-seconds 30 --no-display
python virtual_background_removal.py --source clip.mp4 --loop --max-seconds 30
python virtual_background_removal.py --source clip.mp4 --mode colour
python virtual_background_removal.py --source clip.mp4 --background-image room.jpg --mode image
```

Keys in the display window: **B** blurred background, **C** solid
colour, **I** image background, **S** split view on/off, **[** / **]**
less/more feather, **Q** quit. Outputs next to `--out` (default
`output.mp4`): `output.mp4`, `output_summary.txt`, `output_events.log`.

## Files

- `virtual_background_removal.py` — segmenter + compositor + renderer + simulation
- `selfie_multiclass_256x256.tflite` — MediaPipe selfie segmentation model (16 MB)
- `requirements.txt` — opencv-python-headless, numpy, mediapipe
- `README.md` — this file
