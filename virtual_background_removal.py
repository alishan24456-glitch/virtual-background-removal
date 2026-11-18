#!/usr/bin/env python3
"""
virtual_background_removal.py — real-time background removal / virtual
background for video calls.

MediaPipe's Selfie Segmentation (multiclass model, Tasks API) cuts the
person out of the frame live and composites them over a blurred version
of the real background, a solid colour, or a chosen image — the Zoom /
Meet-style virtual background, with an edge-feather control and a
before/after split view. The person-coverage readout shows how much of
the frame is person at any moment.

The segmenter is a 6-class model (background, hair, body skin, face skin,
clothes, accessories); every class except background counts as person.
It is a segmentation model, not a detector — it reads the pixels, so a
person half out of frame is only half segmented, and a second person
walking behind you becomes part of the "person" too.

Real mode:  --source video.mp4 | 0 (webcam) | pick (file chooser)
Simulation: --simulate renders a room with a face-like figure against a
busy background; the segmenter reads the rendered pixels — the pipeline
never sees the renderer's ground truth.

Usage:
    python virtual_background_removal.py                  # file picker
    python virtual_background_removal.py --source talk.mp4
    python virtual_background_removal.py --source 0       # webcam
    python virtual_background_removal.py --simulate --max-seconds 30 --no-display
    python virtual_background_removal.py --source clip.mp4 --loop --max-seconds 30
    python virtual_background_removal.py --source clip.mp4 --mode colour

Keys in the display window: B = blurred background, C = solid colour,
I = image background, S = split view on/off, [ / ] = less/more feather,
Q = quit. Outputs next to --out (default output.mp4): output.mp4,
output_summary.txt, output_events.log

On Linux the MediaPipe Tasks API needs the stock OpenGL libraries
(libegl1 / libgles2); without them it fails at import with a
`libEGL.so.1` error — install them or point LD_LIBRARY_PATH at an
extracted copy. Simulation mode (--simulate) needs no model at all.
"""

import argparse
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

# ---------------------------------------------------------------- constants

MODEL_DEFAULT = "selfie_multiclass_256x256.tflite"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "image_segmenter/selfie_multiclass_256x256/float32/latest/"
             "selfie_multiclass_256x256.tflite")
SEGMENT_EVERY = 2       # re-run the segmenter every 2nd frame (15 fps)
CANVAS_W, CANVAS_H = 1280, 720
PANEL_W = 640           # split view: 640 px per side
MODES = ("blur", "colour", "image")
COLOUR_BG = (110, 90, 40)   # BGR deep teal for the solid-colour mode


# --------------------------------------------------------------- simulation

def _draw_figure(img, cx, cy, s, t):
    """A rendered person: skin-tone head with hair, shirt, gesturing arms.

    Drawn with soft gradients and realistic tones so the segmenter has
    something photo-like to read; the pipeline only ever sees the pixels.
    """
    skin = (120, 165, 215)
    shirt = (90, 110, 170)
    hair = (45, 45, 55)
    # torso
    cv2.ellipse(img, (cx, cy + int(150 * s)), (int(150 * s), int(170 * s)),
                0, 0, 360, shirt, -1)
    # arms — one raised, waving slowly
    wave = int(18 * s * np.sin(t * 1.5))
    cv2.line(img, (cx - int(130 * s), cy + int(120 * s)),
             (cx - int(210 * s), cy + int(20 * s) + wave), shirt,
             int(46 * s))
    cv2.circle(img, (cx - int(210 * s), cy + int(20 * s) + wave),
               int(26 * s), skin, -1)
    cv2.line(img, (cx + int(130 * s), cy + int(120 * s)),
             (cx + int(190 * s), cy + int(180 * s)), shirt, int(46 * s))
    cv2.circle(img, (cx + int(190 * s), cy + int(180 * s)),
               int(26 * s), skin, -1)
    # neck + head
    cv2.rectangle(img, (cx - int(28 * s), cy - int(10 * s)),
                  (cx + int(28 * s), cy + int(55 * s)), skin, -1)
    cv2.circle(img, (cx, cy - int(55 * s)), int(62 * s), skin, -1)
    # hair cap
    cv2.ellipse(img, (cx, cy - int(75 * s)), (int(64 * s), int(42 * s)),
                0, 180, 360, hair, -1)
    # simple facial features (eyes, brows, smile)
    for ex in (cx - int(24 * s), cx + int(24 * s)):
        cv2.circle(img, (ex, cy - int(58 * s)), int(5 * s), (30, 30, 30), -1)
    cv2.ellipse(img, (cx, cy - int(28 * s)), (int(20 * s), int(10 * s)),
                0, 0, 180, (30, 30, 30), max(2, int(3 * s)))


def render_sim_scene(t, w=CANVAS_W, h=CANVAS_H):
    """Rendered room: window, shelves, plant, rug — with a figure in
    front. The segmenter reads the rendered pixels only."""
    img = np.full((h, w, 3), (210, 215, 222), dtype=np.uint8)
    # back wall + floor
    cv2.rectangle(img, (0, 0), (w, 480), (218, 222, 228), -1)
    cv2.rectangle(img, (0, 480), (w, h), (140, 135, 128), -1)
    # window with sky + sun (left)
    cv2.rectangle(img, (70, 60), (350, 330), (190, 215, 235), -1)
    cv2.rectangle(img, (70, 60), (350, 330), (90, 100, 115), 4)
    cv2.line(img, (210, 60), (210, 330), (90, 100, 115), 4)
    cv2.circle(img, (300, 120), 34, (120, 200, 240), -1)
    # shelves with books (right)
    cv2.rectangle(img, (920, 70), (1210, 420), (150, 130, 105), -1)
    for sy in (150, 240, 330):
        cv2.rectangle(img, (930, sy), (1200, sy + 12), (110, 95, 75), -1)
        for bx in range(935, 1195, 26):
            col = (60 + (bx * 7) % 120, 90 + (bx * 13) % 90, 140)
            cv2.rectangle(img, (bx, sy - 52), (bx + 20, sy), col, -1)
    # plant (left floor) + rug
    cv2.ellipse(img, (430, 470), (90, 110), 0, 180, 360, (60, 140, 80), -1)
    cv2.rectangle(img, (415, 470), (445, 560), (120, 90, 70), -1)
    cv2.ellipse(img, (640, 640), (330, 46), 0, 0, 360, (170, 120, 100), -1)
    _draw_figure(img, 640, 330, 1.0, t)
    return img


def render_background_image(w=CANVAS_W, h=CANVAS_H):
    """Built-in 'image' background: a soft gradient office scene drawn
    in code (used when no --background-image file is given)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        f = y / h
        img[y, :] = (int(200 - 60 * f), int(190 - 40 * f),
                     int(120 + 80 * f))
    cv2.circle(img, (1060, 130), 70, (235, 235, 220), -1)
    cv2.rectangle(img, (80, 420), (500, 700), (90, 110, 140), -1)
    cv2.rectangle(img, (80, 420), (500, 470), (70, 90, 120), -1)
    cv2.rectangle(img, (820, 300), (1200, 700), (110, 130, 90), -1)
    cv2.putText(img, "VIRTUAL BACKGROUND", (90, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return img


# ------------------------------------------------------------------ helpers

def pick_file():
    """tkinter file chooser (no arguments = ask the user)."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choose a video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                   ("All files", "*.*")])
    root.destroy()
    return path or None


class Segmenter:
    """MediaPipe Tasks ImageSegmenter (selfie multiclass), lazily loaded.

    Works with both mediapipe layouts: BaseOptions under vision (1.1.x)
    or under tasks.python.core (1.2.x). The model is shipped in the repo
    (16 MB); if it is missing it is downloaded once from Google storage.
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.segmenter = None

    def ensure(self):
        if self.segmenter is not None:
            return
        if not os.path.exists(self.model_path):
            print("Selfie model not found locally — downloading it once "
                  "(16 MB, Google MediaPipe models)...")
            try:
                urllib.request.urlretrieve(MODEL_URL, self.model_path)
            except Exception as exc:  # noqa: BLE001 - report and exit cleanly
                raise RuntimeError(
                    "Could not download the selfie model from %s\n%s"
                    % (MODEL_URL, exc))
            print("Downloaded %s" % self.model_path)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except OSError as exc:
            raise RuntimeError(
                "MediaPipe could not load its native library (%s).\n"
                "On Linux this is almost always the missing OpenGL "
                "libraries: install libegl1 / libgles2, or point "
                "LD_LIBRARY_PATH at an extracted mesa copy.\n"
                "(Simulation mode --simulate does not need mediapipe.)"
                % exc)
        base_options_cls = getattr(mp_vision, "BaseOptions", None)
        if base_options_cls is None:
            base_options_cls = mp_python.BaseOptions
        options = mp_vision.ImageSegmenterOptions(
            base_options=base_options_cls(model_asset_path=self.model_path),
            output_category_mask=True,
            running_mode=mp_vision.RunningMode.IMAGE)
        self.segmenter = mp_vision.ImageSegmenter.create_from_options(options)
        self._mp = mp

    def segment(self, frame_bgr):
        """Return a float32 person mask in [0, 1] at frame resolution."""
        self.ensure()
        mp = self._mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.segmenter.segment(mp_image)
        mask = result.category_mask.numpy_view().astype(np.float32)
        person = (mask > 0).astype(np.float32)
        h, w = frame_bgr.shape[:2]
        if person.shape[:2] != (h, w):
            person = cv2.resize(person, (w, h),
                                interpolation=cv2.INTER_LINEAR)
        return person

    def close(self):
        if self.segmenter is not None:
            try:
                self.segmenter.close()
            except Exception:  # noqa: BLE001 - shutdown noise, ignore
                pass
            self.segmenter = None


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--source", default=None,
                    help="video file, webcam index, or 'pick'")
    ap.add_argument("--simulate", action="store_true",
                    help="rendered scene instead of a video file")
    ap.add_argument("--max-seconds", type=float, default=30)
    ap.add_argument("--loop", action="store_true",
                    help="loop the video until --max-seconds")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--model", default=MODEL_DEFAULT,
                    help="selfie segmentation model (shipped)")
    ap.add_argument("--mode", default="blur", choices=MODES,
                    help="starting virtual background")
    ap.add_argument("--feather", type=int, default=9,
                    help="edge feather radius in px (odd kernel)")
    ap.add_argument("--background-image", default=None,
                    help="image file to use as the virtual background")
    ap.add_argument("--no-split", action="store_true",
                    help="show only the composited view (no original)")
    args = ap.parse_args()

    segmenter = Segmenter(args.model)
    bg_image = None
    if args.background_image:
        bg_image = cv2.imread(args.background_image)
        if bg_image is None:
            print("Could not read background image: %s"
                  % args.background_image)
            return 1
        bg_image = cv2.resize(bg_image, (CANVAS_W, CANVAS_H))
    else:
        bg_image = render_background_image()

    cap = None
    source_desc = "simulation"
    if not args.simulate:
        src = args.source
        if src is None or src == "pick":
            src = pick_file()
            if not src:
                print("No video chosen.")
                return 1
        try:
            src = int(src)
            source_desc = "webcam %s" % src
        except ValueError:
            source_desc = src
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print("Could not open video source: %s" % src)
            return 1

    fps = args.fps
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (CANVAS_W, CANVAS_H))
    base = os.path.splitext(args.out)[0]
    summary_path = base + "_summary.txt"
    events_path = base + "_events.log"
    events = []

    def log_event(t, text):
        line = "[%6.1fs] %s" % (t, text)
        events.append(line)
        print(line)

    mode = args.mode
    feather = max(1, args.feather)
    if feather % 2 == 0:
        feather += 1
    split = not args.no_split
    mask = None
    frame_no = 0
    seg_calls = 0
    coverages = []
    mode_seconds = {m: 0.0 for m in MODES}
    last_mode_change = 0.0
    start = time.time()

    while True:
        t = frame_no / fps
        if t >= args.max_seconds:
            break
        if args.simulate:
            frame = render_sim_scene(t)
        else:
            ok, frame = cap.read()
            if not ok:
                if args.loop and cap is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    log_event(t, "LOOP video restarted")
                    ok, frame = cap.read()
                if not ok:
                    break
        frame = cv2.resize(frame, (CANVAS_W, CANVAS_H))

        if mask is None or frame_no % SEGMENT_EVERY == 0:
            mask = segmenter.segment(frame)
            seg_calls += 1
        coverage = float(mask.mean())
        coverages.append(coverage)
        mode_seconds[mode] += 1.0 / fps

        # feather the mask edges so hair does not look cut out
        soft = cv2.GaussianBlur(mask, (feather, feather), 0)
        soft = np.clip(soft, 0.0, 1.0)[..., None]

        if mode == "blur":
            background = cv2.GaussianBlur(frame, (0, 0), 18)
        elif mode == "colour":
            background = np.full_like(frame, COLOUR_BG)
        else:
            background = bg_image
        composited = (frame.astype(np.float32) * soft
                      + background.astype(np.float32) * (1.0 - soft))
        composited = np.clip(composited, 0, 255).astype(np.uint8)

        # ---- draw ------------------------------------------------------
        if split:
            left = cv2.resize(frame, (PANEL_W, CANVAS_H))
            right = cv2.resize(composited, (PANEL_W, CANVAS_H))
            canvas = np.hstack([left, right])
            cv2.putText(canvas, "ORIGINAL", (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, "VIRTUAL BACKGROUND", (PANEL_W + 16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.line(canvas, (PANEL_W, 0), (PANEL_W, CANVAS_H),
                     (255, 255, 255), 2)
        else:
            canvas = composited
        cv2.rectangle(canvas, (0, CANVAS_H - 64), (CANVAS_W, CANVAS_H),
                      (0, 0, 0), -1)
        cv2.putText(canvas,
                    "Mode: %-6s   Person: %5.1f%%   Feather: %d px"
                    % (mode.upper(), 100 * coverage, feather),
                    (16, CANVAS_H - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        cv2.putText(canvas,
                    "B blur | C colour | I image | S split | [ ] feather | Q quit",
                    (16, CANVAS_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)

        writer.write(canvas)
        if not args.no_display:
            cv2.imshow("Virtual Background Removal — Q to quit", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key == ord("b") and mode != "blur":
                mode_seconds[mode] += 0  # seconds already counted per frame
                log_event(t, "Background mode -> BLUR")
                mode = "blur"
            elif key == ord("c") and mode != "colour":
                log_event(t, "Background mode -> COLOUR")
                mode = "colour"
            elif key == ord("i") and mode != "image":
                log_event(t, "Background mode -> IMAGE")
                mode = "image"
            elif key == ord("s"):
                split = not split
                log_event(t, "Split view %s" % ("ON" if split else "OFF"))
            elif key == ord("["):
                feather = max(1, feather - 2)
                log_event(t, "Feather -> %d px" % feather)
            elif key == ord("]"):
                feather = min(31, feather + 2)
                log_event(t, "Feather -> %d px" % feather)
        frame_no += 1

    if cap is not None:
        cap.release()
    writer.release()
    segmenter.close()
    cv2.destroyAllWindows()
    elapsed = time.time() - start

    cov = np.array(coverages) if coverages else np.array([0.0])
    lines = [
        "Virtual Background Removal — run summary",
        "Source: %s" % source_desc,
        "Duration: %.1f s (%d frames at %d fps, %.1f s wall time)"
        % (frame_no / fps, frame_no, fps, elapsed),
        "Segmentations run: %d (every %d frames)" % (seg_calls, SEGMENT_EVERY),
        "Person coverage: mean %.1f%%, min %.1f%%, max %.1f%%"
        % (100 * cov.mean(), 100 * cov.min(), 100 * cov.max()),
        "",
        "Seconds per background mode:",
    ]
    for m in MODES:
        lines.append("  %-7s %.1f s" % (m, mode_seconds[m]))
    with open(summary_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(events_path, "w") as fh:
        fh.write("\n".join(events) + "\n")
    print("\n".join(lines))
    print("Wrote %s, %s, %s" % (args.out, summary_path, events_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
