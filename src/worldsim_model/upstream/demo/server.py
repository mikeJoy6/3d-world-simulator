"""The interactive demo of the website, served locally: open the printed address, pick a task, drive the arms.

    python -m demo.server --release release                 # reference: fp32, plain PyTorch (about 12 to 25 FPS)
    python -m demo.server_fast --release release            # optimised: about 40 to 60 FPS, same picture (see demo/server_fast.py)

The browser receives JPEG frames over a WebSocket and sends keys, gizmo drags and camera moves; everything is rendered here.
A task is loaded the first time it is opened. The robot arms are drawn by the kinematic robot model (models/arm_rig.py) as in
the public demo; ``--no-arms`` shows the world model's own arms (also what happens without release/arms).
"""
import argparse
import asyncio
import json
import threading
import time
from pathlib import Path
import torch

HERE = Path(__file__).parent


def create_app(release, fast=False, session_seconds=0.0, arms=True):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse
    from demo.session import Session
    from models.world_model import WorldModel

    release = Path(release)
    tasks = {p.parent.name: json.load(open(p)) for p in sorted(release.glob("*/task.json")) if "arm_base" in json.load(open(p))}
    first = ("cloth", "box", "pusht", "assembly")
    order = [t for t in first if t in tasks] + [t for t in tasks if t not in first]
    worlds, gpu, loading = {}, threading.Lock(), threading.Lock()
    app = FastAPI()

    def world(task):
        with loading:
            if task not in worlds:
                t0 = time.time()
                wm = WorldModel(release / task, fast=fast, arms=arms)
                s = Session(wm)
                with gpu:  # first frames: compile and warm every view
                    s.reseed()
                    for disp in ("splats", "particles", "latents", "voxels", "splats"):
                        s.ctl["display"] = disp
                        for _ in range(8 if fast else 2):
                            s.z_last = wm.step(s.st, s.build_action())
                            s.publish()
                worlds[task] = wm
                print(f"[demo] {task} ready in {time.time() - t0:.1f} s", flush=True)
            return worlds[task]

    @app.get("/")
    def page(task: str = ""):
        t = task if task in tasks else order[0]
        c = tasks[t]
        info = dict(
            task=t, title=c.get("title", t), move=sorted(c["control"]["keys"]), tap=sorted((c["control"].get("gripper") or {}).get("keys", {})) + ["r"],
            view=c["view"], tasks=[dict(key=k, name=tasks[k].get("title", k)) for k in order],
        )
        return HTMLResponse((HERE / "page.html").read_text().replace("%INFO%", json.dumps(info)))

    @app.get("/thumb/{task}/{name}")
    def thumb(task: str, name: str):
        p = release / task / "thumbs" / name
        return FileResponse(p) if task in tasks and p.is_file() and p.suffix == ".jpg" else HTMLResponse("", 404)

    @app.websocket("/ws")
    async def socket(ws: WebSocket):
        await ws.accept()
        t = ws.query_params.get("task", order[0])
        t = t if t in tasks else order[0]
        loop, latest = asyncio.get_running_loop(), asyncio.Queue(maxsize=1)
        s = Session(await asyncio.to_thread(world, t))

        def on_frame(jpeg):
            def put():
                if latest.full():
                    latest.get_nowait()
                latest.put_nowait(jpeg)

            loop.call_soon_threadsafe(put)

        th = threading.Thread(target=s.run, args=(on_frame, gpu), daemon=True)
        th.start()

        async def sender():
            t_stat = 0.0
            while True:
                try:
                    jpeg = await asyncio.wait_for(latest.get(), 0.5)
                except asyncio.TimeoutError:
                    jpeg = None
                if jpeg:
                    await ws.send_bytes(jpeg)
                    if s.gizmo is not None:
                        await ws.send_text(json.dumps({"gz": s.gizmo}))
                if time.time() - t_stat > 0.5:
                    t_stat = time.time()
                    await ws.send_text(json.dumps({"fps": s.fps, "elapsed": time.time() - s.t_seed}))

        async def time_limit():
            await asyncio.sleep(session_seconds)
            await ws.send_text(json.dumps({"timeout": session_seconds}))
            await ws.close()

        jobs = [asyncio.create_task(sender())] + ([asyncio.create_task(time_limit())] if session_seconds > 0 else [])
        try:
            while True:
                m = json.loads(await ws.receive_text())
                (s.on_keys if m.get("u") == "/keys" else s.on_control)(m.get("d") or {})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            for j in jobs:
                j.cancel()
            s.stop = True
            await asyncio.to_thread(th.join, 10)

    return app


def main(fast=False):
    p = argparse.ArgumentParser()
    p.add_argument("--release", default="release", help="folder with one sub-folder per task (task.json, vae.pth, dynamics.pth, seed episode)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--session-seconds", type=float, default=0.0, help="end a session after this long (0 = never; the public demo uses 300)")
    p.add_argument("--no-arms", action="store_true", help="draw the world model's own arms instead of the kinematic robot model (also the fallback without release/arms)")
    a = p.parse_args()
    import uvicorn

    torch.set_grad_enabled(False)
    print(f"[demo] open http://localhost:{a.port}  ({'optimised' if fast else 'reference'} inference)", flush=True)
    uvicorn.run(create_app(a.release, fast, a.session_seconds, arms=not a.no_arms), host=a.host, port=a.port, ws_ping_interval=20, ws_max_size=2**24, log_level="warning")


if __name__ == "__main__":
    main()
