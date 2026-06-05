# Codebase overview for newcomers

This document explains how Avatarify Python is organized, how the main runtime flows work, and what to learn next when you are new to the project.

## Project in one paragraph

Avatarify Python turns a still face image into a real-time animated avatar for video calls. The application reads frames from a webcam, uses the First Order Motion Model (FOMM) to transfer motion from the webcam frame to a selected avatar image, previews the result, and can stream that result to a virtual camera. The project can run inference locally on a CUDA-capable machine or split the work between a lightweight client and a remote GPU worker.

## Top-level layout

| Path | Purpose |
| --- | --- |
| `afy/` | Main Python application package: camera loop, predictors, networking helpers, argument parsing, and image utilities. |
| `avatars/` | Example source avatar images that can be selected from the UI. |
| `docs/` | Installation, usage, video-call setup, and this codebase overview. |
| `scripts/` | Platform setup helpers for installation, virtual camera setup, tunnels, and environment settings. |
| `run.sh`, `run_mac.sh`, `run_windows.bat` | Platform launchers that prepare the environment and start the app. |
| `requirements.txt`, `requirements_client.txt` | Python dependencies for full/local and client-only setups. |
| `config.yaml` | Small runtime configuration file for camera selection. |
| `Dockerfile` | Container image definition used by Docker launch mode. |
| `avatarify.ipynb` | Notebook-oriented remote/Colab workflow. |

## Runtime architecture

At a high level, the app has five stages:

1. **Launch and environment setup**: a platform script configures conda, virtual camera support, Docker flags, and command-line arguments.
2. **Predictor selection**: `afy/cam_fomm.py` decides whether this process is a local predictor, a remote client, or a remote GPU worker.
3. **Camera and avatar setup**: the app selects a webcam, opens it asynchronously, and loads static avatar images from the avatars directory.
4. **Real-time loop**: each webcam frame is cropped, resized, optionally used for calibration/keyframe search, passed through the predictor, displayed, and optionally streamed to a virtual camera.
5. **Shutdown**: camera capture, windows, and remote predictor processes are stopped cleanly.

The important runtime paths are:

```text
Local mode
---------
run.sh
  -> afy/cam_fomm.py
    -> afy/predictor_local.py
      -> FOMM generator + keypoint detector

Remote mode
-----------
Client machine:
run.sh --is-client --in-addr ... --out-addr ...
  -> afy/cam_fomm.py
    -> afy/predictor_remote.py
      -> ZeroMQ send/recv processes

GPU worker machine:
run.sh --is-worker
  -> afy/cam_fomm.py
    -> afy/predictor_worker.py
      -> afy/predictor_local.py
      -> FOMM generator + keypoint detector
```

## Key files and responsibilities

### `run.sh`

`run.sh` is the main Linux launcher. It parses convenience flags such as `--docker`, `--no-vcam`, `--is-worker`, and `--is-client`, sources `scripts/settings.sh`, optionally creates the Linux virtual camera, activates conda, extends `PYTHONPATH`, and finally invokes `afy/cam_fomm.py` with the FOMM config/checkpoint defaults.

Start here when you want to understand how users actually launch the application.

### `afy/arguments.py`

This file defines the command-line interface consumed by the app. The most important groups are:

- model/config flags: `--config`, `--checkpoint`, `--relative`, `--adapt_scale`, `--enc_downscale`
- stream/UI flags: `--virt-cam`, `--no-stream`, `--hide-rect`, `--avatars`
- remote execution flags: `--is-worker`, `--is-client`, `--in-port`, `--out-port`, `--in-addr`, `--out-addr`, `--jpg_quality`

If you add a new runtime knob, it usually starts here and then gets consumed by `cam_fomm.py`, a predictor, or a launcher script.

### `afy/cam_fomm.py`

This is the main application orchestrator. It handles:

- loading `config.yaml`
- choosing local, client, or worker predictor mode
- selecting/opening the camera
- loading avatar images
- creating the virtual webcam stream on Linux
- displaying the `cam` and `avatarify` windows
- handling keyboard controls
- running the per-frame preprocess → predict → postprocess loop

Most user-facing behavior changes happen in this file.

### `afy/predictor_local.py`

This is the local inference implementation. It loads the FOMM generator and keypoint detector from a YAML config and checkpoint, prepares the selected source avatar, computes driving keypoints for webcam frames, normalizes motion, and returns the generated avatar frame as a NumPy image.

Read this file when you want to understand the model-facing data flow.

### `afy/predictor_remote.py`

This is the client-side proxy for remote GPU inference. It keeps the same high-level predictor interface as the local predictor, but forwards method calls to a remote worker over ZeroMQ. It compresses predicted frames as JPEG payloads, uses multiprocessing queues, and treats `predict` as non-critical so stale frames can be skipped instead of increasing latency.

### `afy/predictor_worker.py`

This is the server-side remote GPU worker. It receives method calls, initializes or reuses a `PredictorLocal`, executes predictor methods, compresses results, and sends responses back to the client. It deliberately drops older non-critical requests so real-time performance favors freshness over processing every frame.

### `afy/networking.py`

This file wraps ZeroMQ sockets with helper methods for sending metadata plus binary data. The remote predictor and worker use these helpers for request and response payloads.

### `afy/utils.py`

This file contains small shared helpers: timestamped logging, tee logging to files, periodic logging, timing accumulation, image cropping, image padding, and resizing.

### `afy/videocaptureasync.py` and `afy/camera_selector.py`

These files support webcam selection and asynchronous capture. They keep the main loop from blocking unnecessarily on camera reads and help users choose the correct camera on first launch.

## The main frame loop

The core loop in `afy/cam_fomm.py` does roughly this:

1. Read the newest camera frame.
2. Convert channel order for OpenCV/model compatibility.
3. Crop and resize the frame to the model input size.
4. If keyframe search is enabled, compare landmarks and reset the driving reference when a better frame is found.
5. If calibrated, call `predictor.predict(frame)`.
6. Process keyboard controls.
7. Build the camera preview, including optional overlay, landmarks, FPS, and calibration text.
8. Pad/resize/flip the output if needed.
9. Send the output frame to the virtual camera if streaming is enabled.
10. Show the preview and output windows.

The calibration step is important: pressing `X` resets the reference frame used to drive motion. Poor calibration usually looks like poor avatar control, even if the model is working correctly.

## Local predictor data flow

`PredictorLocal` follows this model-facing flow:

1. Load model config and checkpoint.
2. Build `OcclusionAwareGenerator` and `KPDetector`.
3. On avatar selection, call `set_source_image()` to encode the source avatar and compute source keypoints.
4. On the first driving frame after reset, save initial driving keypoints.
5. For each driving frame, compute current driving keypoints.
6. Normalize driving keypoints relative to the source and initial driving keypoints.
7. Run the generator and convert the prediction tensor back to an unsigned 8-bit image.

The helper `normalize_kp()` is a compact but important function: it controls whether movement is transferred as absolute coordinates or relative motion, and whether movement scale is adapted between the source avatar and driving face.

## Remote predictor data flow

Remote mode is an RPC-like split of the same predictor interface:

1. The client wraps predictor method calls in metadata containing method name, critical/non-critical status, and an incrementing message id.
2. `predict` frames are JPEG-encoded to reduce bandwidth.
3. Non-`predict` calls are serialized with MessagePack.
4. A client send process pushes requests to the worker.
5. A client receive process pulls responses from the worker.
6. The worker receives requests, decodes payloads, calls `PredictorLocal`, encodes the result, and sends it back.
7. Non-critical work can be skipped when queues are backed up.

This means remote mode is optimized for live responsiveness, not for exact one-request/one-frame processing.

## User-facing controls to know

The docs list the full control table, but these are the controls most relevant to understanding the code:

- `X`: calibrate/reset the reference frame
- `A` / `D`: previous/next avatar
- number keys `1`-`9`: direct avatar selection
- `W` / `S`: zoom crop in/out
- `U` / `H` / `J` / `K`: translate camera crop
- `Z` / `C`: adjust source-avatar overlay opacity
- `F`: toggle reference-frame search
- `O`: toggle landmark overlay
- `I`: toggle FPS/timing display
- `0`: passthrough mode
- `ESC`: quit

When changing the UI loop, check both the code path in `cam_fomm.py` and the user-facing docs so they stay consistent.

## Common change locations

| Change you want to make | Start here |
| --- | --- |
| Add a new command-line flag | `afy/arguments.py`, then consume it in `afy/cam_fomm.py` or predictor code. |
| Add or change a keyboard shortcut | `afy/cam_fomm.py`, plus update `docs/README.md`. |
| Change model loading or inference behavior | `afy/predictor_local.py`. |
| Improve remote GPU behavior | `afy/predictor_remote.py`, `afy/predictor_worker.py`, and `afy/networking.py`. |
| Change launch behavior | `run.sh` and platform-specific scripts. |
| Change camera selection | `afy/camera_selector.py`, `afy/videocaptureasync.py`, and `config.yaml`. |
| Change preview/virtual camera output | `afy/cam_fomm.py` and `scripts/create_virtual_camera.sh`. |

## Things to watch out for

- **Import paths are launcher-dependent.** `run.sh` extends `PYTHONPATH` with the repo root and `fomm`, which is why FOMM imports work from the predictor code.
- **Some remote files use direct imports.** Remote predictor/worker modules import names like `arguments` and `networking`; test remote-mode changes with the same `PYTHONPATH` assumptions as the launcher.
- **Frame freshness matters.** Do not accidentally force every non-critical frame to be processed in remote mode unless you want higher latency.
- **OpenCV channel order matters.** The app frequently converts between BGR and RGB conventions.
- **Model weights are not committed.** The checkpoint is expected to be downloaded separately and placed at the path used by the launcher.
- **Virtual camera behavior is platform-specific.** Linux uses v4l2loopback/pyfakewebcam; macOS and Windows use different external tools/workflows.

## Suggested onboarding path

1. Read `README.md` to understand the project goal.
2. Read `docs/README.md` sections for requirements, run modes, controls, and driving tips.
3. Read `run.sh` and `afy/arguments.py` to understand startup and configuration.
4. Trace `afy/cam_fomm.py` from `if __name__ == "__main__"` through the main loop.
5. Read `afy/predictor_local.py` to understand how FOMM is invoked.
6. If you need remote mode, read `afy/predictor_remote.py`, `afy/predictor_worker.py`, and `afy/networking.py` together.
7. Make a small first change, such as adding a log line, documenting a control, or exposing a simple flag.

## Glossary

- **Avatar/source image**: the static face image that will be animated.
- **Driving frame**: the current webcam frame used to drive motion.
- **Keypoints**: face/motion landmarks used by FOMM to describe movement.
- **Reference/start frame**: the driving frame captured during calibration; subsequent motion can be interpreted relative to it.
- **Virtual camera**: an OS-level video device that conferencing apps can select as if it were a webcam.
- **Worker**: remote GPU process that runs local inference on behalf of a client.
- **Client**: process that owns the camera/UI but delegates inference to a worker.
