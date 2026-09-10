# LARRI Studio

Run the local ComfyUI application with:

```sh
go run ./cmd/larri comfy
```

Open the launch URL printed in the terminal.
The default ports are 8180 for Studio and 8181 for the native ComfyUI editor.
Use `--port`, `--editor-port`, and `--outputs` to change these locations.
Set `VASTAI_API_KEY` in the launching process environment before renting a GPU.
Credentials never enter the browser or rented instance.

Choose Image for Stable Diffusion 1.5 or Video for Wan 2.1 1.3B text-to-video.
The initial video is 17 frames at 512 by 320, with a 16 FPS MP4 output.
Use Check price first to resolve model metadata and search without renting anything.
Find GPU & create automatically rents the compatible offer with the lowest estimated cost for the specified session duration.
The estimate includes cold-start download time and follows LARRI's existing reliability and pricing filters.
VRAM floors are estimates, and readiness requires execution of the actual submitted workflow.

Models are downloaded directly from Hugging Face on the rented host using aria2 JSON-RPC.
LARRI resolves the model repository commit, size, and SHA-256 before rental.
The host verifies model files before loading them and retains them for the session.
No model weights are uploaded from this computer.
ComfyUI is pinned to commit `40c4fcdf513a4523e39d54a9d391908af8df8171` and runs in the PyTorch 2.6.0 CUDA 12.4 image.

Run again to reuse the same instance and downloaded models.
Import accepts ComfyUI API-format JSON or a request containing `workflow`, `models`, `min_vram_gb`, and `session_hours`.
Built-in image and video nodes are supported; arbitrary custom nodes are not installed automatically.
Unknown model profiles require a declared VRAM floor and remote model mappings.
A workflow that exceeds the current session plan or needs additional model files is rejected before submission.
Finish that session and launch the new workflow to select its hardware.

The native editor runs on a separate local origin through LARRI's SSH proxy.
The provider only exposes SSH, and ComfyUI binds remote loopback.
Both browser surfaces require a per-launch session credential and validate Host and Origin.

Quit & release GPU, Ctrl-C, or SIGTERM ends the application session.
Closing a browser tab does not stop the foreground application.
Shutdown waits for active generation, saves completed media locally, destroys the owned instance through the provider API, and confirms absence.
Retrieval or teardown errors remain visible and are not reported as successful cleanup.
Abrupt process or machine failure can leave an instance billing; inspect `larri status` and `larri orphans` before starting another session.

Outputs default to `~/Pictures/LARRI/<prompt-id>/`.
Downloads are streamed, validated, synced, and published atomically without overwriting existing output files.
An interrupted output download can be retried.
Each output currently has a 2 GiB limit.

## Validation status

Local tests cover repeated execution on one simulated instance, image retrieval and PNG decoding, video retrieval with a valid WebM fixture, interrupted downloads, refusal to destroy before retrieval, workflow sizing, and browser authentication.
The actual browser UI was exercised twice with a valid one-second VP9/WebM fixture, including playback, reuse of one simulated instance, gallery updates, and exactly one verified simulated teardown.
The pinned ComfyUI source was inspected to match the MP4 encoder's nested API inputs and video output metadata.
Race tests cover quitting immediately after a run is accepted and retaining artifacts across failed startup.
These tests do not prove GPU compatibility or model generation quality.
Live image and video generation, dependency bootstrap, and paid provider teardown must be verified before this feature is considered complete.
The current execution environment does not expose the required Vast.ai credential.
