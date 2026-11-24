# Chessmate cloud deployment

The app is live at **https://chessmate-duke-robotics.onrender.com**.

Private source repository: `https://github.com/Jog-sama/chessmate`. Service: `srv-dbgivs6c6qcc739cclig`. Blueprint: `exs-dbgivmon74is73dr0340`. The first deployed source commit is `9d625f02b978fc6dab740f39695532f01640fabe`.

## Included configuration

- `Dockerfile`: Python 3.12, OpenCV, NumPy, Flask, Linux Stockfish, and Gunicorn, running as an unprivileged user.
- `render.yaml`: one free-plan Docker web service, HTTPS session cookies, a generated signing secret, and `/healthz` monitoring.
- `gunicorn.conf.py`: one worker process with four request threads. Keep one worker and one service instance while boards use the in-memory visitor registry.
- `.dockerignore`: allows only runtime source, static assets, model files, and requirements into the image. Local environments, macOS engine binaries, screenshots, and secrets are excluded.

The Linux image installs Debian's packaged Stockfish. Its version can differ from the project's local Apple Silicon Stockfish 18. The vision model is the same `models/templates.npz` on both platforms.

## Deploy

1. Push the **contents of `chess_vision_poc/`** as the root of a private Git repository accessible to your Render account. Include the fitted template model; do not commit `.venv`, `.env`, or `bin`.
2. In Render, create a Blueprint from that repository and select `render.yaml`.
3. Verify the proposed plan is free before creating the service. No paid plan is required by this configuration. Availability and limits are controlled by Render.
4. Wait for the Docker build and `/healthz` check to succeed. Use the actual URL returned by Render; the service name alone does not establish its public URL.
5. Test detection, moves, and Stockfish on the public URL. Open a second browser or private window to verify independent boards.

If the whole workspace is used as the repository instead, set the service root directory to `chess_vision_poc` and use its Blueprint path accordingly.

## Local production-server check

```bash
python -m pip install -r requirements-hosting.txt
gunicorn --config gunicorn.conf.py --bind 127.0.0.1:8766 'chess_vision.app:create_app()'
```

To build and test the container when Docker is running:

```bash
docker build -t chessmate .
docker run --rm -p 127.0.0.1:8766:8765 chessmate
```

The container can generate an ephemeral signing key for local testing. Render supplies `CHESSMATE_SECRET_KEY` automatically. Do not put the secret in source control or chat. `CHESSMATE_HTTPS=1` is for HTTPS hosting; leave it unset for local HTTP testing.

## Session behavior and capacity

Each page load generates a random board key held in page memory. Refreshing or opening a new tab starts a fresh board and capture history. API clients that omit the board key use a signed HTTP-only session cookie. No account is needed. Different friends do not share a board. This is an independent tinkering demo, not multiplayer chess.

Sessions are intentionally ephemeral: restart/redeploy, an idle service shutdown, two hours of inactivity, or eviction beyond the 256-session limit can reset a board. A free hosted service may sleep when idle; a later visit may take time to start. No database or paid persistence is configured.

Rendering and detection admit at most two simultaneous requests. Additional attempts return a clear busy response and can be retried. A single session retains 16 capture snapshots. Image dimensions and scene settings are checked before expensive processing. The template model is shared read-only; game state and captures are isolated per visitor.

Rendering reuses bounded, read-only RGB/depth layers for unchanged pieces and board planes. Docker bakes all 768 default overhead piece/square combinations during the build, so a fresh visitor or a new move does not have to rasterize those meshes. Other camera/rotation/lighting settings populate a bounded cache on demand; their first frame can still take longer. Shadows and noise are applied to each fresh frame. Recognition continues to operate on the final captured PNG, with no access to these rendering caches.

Browser moves use `/api/action-frame` to validate the scene, apply the action, and return the new image in one request. The older action and frame endpoints remain available. Measure public response times with `.venv/bin/python tests/hosted_performance.py https://chessmate-duke-robotics.onrender.com`; it checks pixel recognition after four moves as well. Render's idle wake-up delay is separate from rendering and still applies on the free plan.

## Verification status

The local test suite and local Gunicorn checks are recorded in `artifacts/`. Render built the Docker image and reported the deployment live. The public smoke test passed all six checks; see `artifacts/hosted_report.json`. Run it again with `.venv/bin/python tests/hosted_smoke.py https://chessmate-duke-robotics.onrender.com`. This machine's Docker daemon was not used; the container was built and run successfully on Render.

Performance update verified October 27, 2026: commit `10479d9`, deploy `dep-dbgj720ae00c73bn0nvg`. Before the change, warm public frame requests took 16.335–17.528 seconds. Afterward, the first frame for a new visitor took 0.768 seconds and four combined move/frame requests took 0.380–0.419 seconds (median 0.393). These include network time but exclude waking an idle service. Every test move retained 64/64 recognition; the full public smoke test, including Stockfish and visitor isolation, passed in 6.49 seconds. All 25 local tests passed. See `artifacts/performance_before.json`, `artifacts/performance_after.json`, and `artifacts/hosted_performance_verified.png`.

References: [Render Docker deployment](https://render.com/docs/docker), [Render Blueprint configuration](https://render.com/docs/blueprint-spec), [Render free-instance limitations](https://render.com/docs/free).
