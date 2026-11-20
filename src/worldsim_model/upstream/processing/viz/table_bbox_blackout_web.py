"""Tune a TABLE-WORKSPACE bbox that blacks out true background (plain web UI).

Uses full-frame FoundationStereo depth (fs_disp/) to lift every IR pixel to a
WORLD point (packed E_cam2world — authoritative, fs meta E is pre-tweak), then
blacks every RGB pixel whose world point falls OUTSIDE the adjustable box.
Unlike ray/crop heuristics this kills real background (legs, rig, room) even
when it appears "inside" the table region of the image.

    python viz/table_bbox_blackout_web.py --dataset bimanual_cloth_v3 --port 8091

(what `./pw tablebbox` runs; needs packed episodes).  Open http://localhost:8091,
adjust x/y/z ranges, scrub episodes/frames, save -> <root>/table_bbox.json,
then re-run `./pw pack` so add_render_to_packed bakes the blackout.
"""
import argparse
import glob
import json
import os
import re
import sys
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import h5video  # noqa: E402
from processing import paths  # noqa: E402

_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>table bbox blackout</title>
<style>
 body{margin:0;background:#111;color:#ddd;font-family:system-ui,Arial,sans-serif;font-size:13px}
 #bar{padding:10px 14px;background:#1c1c1c;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 #cams{display:flex;flex-wrap:wrap;gap:10px;padding:10px}
 .cam img{display:block;width:44vw}
 input[type=number]{width:64px;background:#222;color:#ddd;border:1px solid #444}
 input[type=range]{width:260px;vertical-align:middle}
 button{padding:4px 12px} select{background:#222;color:#ddd}
 #msg{color:#8f8}
 .grp{border-left:2px solid #333;padding-left:10px}
</style></head><body>
<div id=bar>
 <label>episode <select id=ep></select></label>
 <label>frame <input type=range id=fr min=0 max=0 value=0><span id=frv>0</span></label>
 <span class=grp>x <input type=number id=x0 step=0.02> to <input type=number id=x1 step=0.02></span>
 <span class=grp>y <input type=number id=y0 step=0.02> to <input type=number id=y1 step=0.02></span>
 <span class=grp>z <input type=number id=z0 step=0.02> to <input type=number id=z1 step=0.02></span>
 <button id=save>save bbox</button><span id=msg></span>
</div>
<div id=cams></div>
<script>
let META=null, BOX=null, timer=null;
function q(s){return document.querySelector(s)}
async function j(u,opt){const r=await fetch(u,opt);return r.json()}
function imgUrl(c){return `/img/${c}?ep=${q('#ep').value}&fr=${q('#fr').value}&t=${Date.now()}`}
function refresh(){META.cams.forEach(c=>{q('#img'+c).src=imgUrl(c)})}
function pushBox(){
  ['x0','x1','y0','y1','z0','z1'].forEach((k,i)=>BOX[i]=+q('#'+k).value);
  clearTimeout(timer);
  timer=setTimeout(async()=>{await j('/box',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(BOX)});refresh();},150);
}
async function loadEp(){
  const m=await j('/meta?ep='+q('#ep').value);
  q('#fr').max=m.T-1; if(+q('#fr').value>m.T-1)q('#fr').value=0;
  q('#frv').textContent=q('#fr').value; refresh();
}
async function init(){
  META=await j('/meta'); BOX=META.box;
  META.eps.forEach(e=>{const o=document.createElement('option');o.value=e;o.textContent=e;
    q('#ep').appendChild(o)});
  ['x0','x1','y0','y1','z0','z1'].forEach((k,i)=>{q('#'+k).value=BOX[i];
    q('#'+k).addEventListener('input',pushBox)});
  META.cams.forEach(c=>{
    const d=document.createElement('div');d.className='cam';
    d.innerHTML=`<div>cam ${c}</div><img id=img${c}>`;q('#cams').appendChild(d);});
  q('#ep').addEventListener('change',loadEp);
  q('#fr').addEventListener('input',()=>{q('#frv').textContent=q('#fr').value;refresh();});
  q('#save').addEventListener('click',async()=>{
    const r=await j('/save',{method:'POST',headers:{'Content-Type':'application/json'},
                             body:JSON.stringify(BOX)});
    q('#msg').textContent='saved -> '+r.path;setTimeout(()=>q('#msg').textContent='',4000);});
  await loadEp();
}
init();
</script></body></html>"""


class App:
    def __init__(self, a):
        self.a = a
        self.files = {}
        for fp in sorted(glob.glob(os.path.join(a.packed_dir, "episode_*.hdf5"))):
            m = re.search(r"episode_(\d+)\.hdf5", os.path.basename(fp))
            if m:
                self.files[int(m.group(1))] = fp
        assert self.files, f"no episodes in {a.packed_dir}"
        self.lock = threading.Lock()
        self._h = OrderedDict()          # ep -> (packed, raw, disp, T, cams)
        self._frame = OrderedDict()      # (ep,fr,cam) -> (rgb, world (H,W,3), src_idx)
        self.box = list(a.init_box)
        ep0 = min(self.files)
        *_, cams = self._handles(ep0)
        self.cams = cams
        rgb, _, _ = self._get_frame(ep0, 0, cams[0])
        self.H, self.W = rgb.shape[:2]

    def _handles(self, ep):
        with self.lock:
            if ep in self._h:
                self._h.move_to_end(ep); return self._h[ep]
        p = h5video.File(self.files[ep], "r")
        src = str(p.attrs.get("src_episode", os.path.basename(self.files[ep])))
        r = h5video.File(os.path.join(self.a.raw_dir, src), "r")
        dsp = h5video.File(os.path.join(self.a.disp_dir, f"disp_{src}"), "r")
        cams = [int(x) for x in p.attrs["cam_ids"]]
        T = r[f"observations/images/camera_{cams[0]}_color"].shape[0]
        with self.lock:
            self._h[ep] = (p, r, dsp, T, cams)
            if len(self._h) > 3:
                _, (op, orr, od, _, _) = self._h.popitem(last=False)
                op.close(); orr.close(); od.close()
            return self._h[ep]

    def _get_frame(self, ep, fr, cam):
        """(rgb, world (H,W,3) float32, src_idx (H,W) int32): per-RGB-pixel world
        point via FS depth lifted with the PACKED extrinsics, z-buffer-warped
        IR->RGB. Cached (LRU)."""
        key = (ep, fr, cam)
        with self.lock:
            if key in self._frame:
                self._frame.move_to_end(key); return self._frame[key]
        p, r, dsp, T, _ = self._handles(ep)
        fr = min(fr, T - 1)
        rgb = r[f"observations/images/camera_{cam}_color"][fr]
        H, W = rgb.shape[:2]
        disp = dsp[f"disp/camera_{cam}"][fr].astype(np.float32)
        Kir = dsp[f"meta/camera_{cam}/K_ir"][fr].astype(np.float64)
        Krgb = dsp[f"meta/camera_{cam}/K_rgb"][fr].astype(np.float64)
        Trgb2ir = dsp[f"meta/camera_{cam}/T_rgb_to_ir"][fr].astype(np.float64)
        b = float(dsp[f"meta/camera_{cam}/baseline"][fr])
        E = p[f"cam/camera_{cam}/E_cam2world"][fr].astype(np.float64)   # packed = authoritative

        z = Kir[0, 0] * b / np.clip(disp, 1e-3, None)
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        x = (u - Kir[0, 2]) / Kir[0, 0] * z
        y = (v - Kir[1, 2]) / Kir[1, 1] * z
        pts = np.stack([x, y, z], -1).reshape(-1, 3)
        world = (pts @ E[:3, :3].T + E[:3, 3]).astype(np.float32)

        # z-buffer warp IR->RGB: src_idx[v,u] = flat IR index whose point lands there
        Vr = np.linalg.inv(E @ Trgb2ir)
        pc = world @ Vr[:3, :3].T + Vr[:3, 3]
        zz = pc[:, 2]
        ok = zz > 0.05
        uu = np.round(Krgb[0, 0] * pc[:, 0] / np.clip(zz, 1e-6, None) + Krgb[0, 2]).astype(np.int32)
        vv = np.round(Krgb[1, 1] * pc[:, 1] / np.clip(zz, 1e-6, None) + Krgb[1, 2]).astype(np.int32)
        ok &= (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        idx = np.nonzero(ok)[0]
        order = idx[np.argsort(-zz[idx])]                    # far first, near overwrites
        src = np.full((H, W), -1, np.int64)
        src[vv[order], uu[order]] = order
        out = (rgb, world.reshape(H, W, 3), src)
        with self.lock:
            self._frame[key] = out
            if len(self._frame) > 24:
                self._frame.popitem(last=False)
        return out

    def jpeg(self, ep, fr, cam):
        rgb, world, src = self._get_frame(ep, fr, cam)
        x0, x1, y0, y1, z0, z1 = self.box
        w = world.reshape(-1, 3)
        inside = ((w[:, 0] > x0) & (w[:, 0] < x1) & (w[:, 1] > y0) & (w[:, 1] < y1)
                  & (w[:, 2] > z0) & (w[:, 2] < z1))
        keep = np.where(src >= 0, inside[np.clip(src, 0, None)], False)
        keep = cv2.medianBlur(keep.astype(np.uint8) * 255, 5) > 127   # seal warp seams
        img = rgb.copy(); img[~keep] = 0
        d = self.a.display_downscale
        if d > 1:
            img = cv2.resize(img, (img.shape[1] // d, img.shape[0] // d),
                             interpolation=cv2.INTER_AREA)
        okc, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes()


def make_handler(app, save_path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *_): pass

        def _send(self, body, ctype="application/json", code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path); qs = parse_qs(u.query)
            if u.path == "/":
                self._send(_PAGE.encode(), "text/html")
            elif u.path == "/meta":
                if "ep" in qs:
                    *_, T, _ = app._handles(int(qs["ep"][0]))
                    self._send(json.dumps({"T": T}).encode())
                else:
                    self._send(json.dumps(dict(eps=sorted(app.files), cams=app.cams,
                                               box=app.box)).encode())
            elif u.path.startswith("/img/"):
                cam = int(u.path.rsplit("/", 1)[1])
                ep = int(qs.get("ep", [min(app.files)])[0])
                fr = int(qs.get("fr", [0])[0])
                self._send(app.jpeg(ep, fr, cam), "image/jpeg")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"null")
            if urlparse(self.path).path == "/box":
                app.box = [float(v) for v in data]
                self._send(b"{}")
            elif urlparse(self.path).path == "/save":
                app.box = [float(v) for v in data]
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                json.dump(dict(x=[app.box[0], app.box[1]], y=[app.box[2], app.box[3]],
                               z=[app.box[4], app.box[5]]), open(save_path, "w"), indent=2)
                print(f"saved table bbox -> {save_path}: {app.box}", flush=True)
                self._send(json.dumps({"path": save_path}).encode())
            else:
                self._send(b"not found", "text/plain", 404)
    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None,
                   help="resolve packed/raw/disp dirs + save path via processing.paths")
    p.add_argument("--packed-dir", default=None)
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--disp-dir", default=None, help="default: <root>/fs_disp")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--display-downscale", type=int, default=2)
    p.add_argument("--init-box", type=float, nargs=6,
                   default=[-0.70, 0.70, -0.55, 0.55, -0.12, 0.45],
                   metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"))
    p.add_argument("--save-path", default=None,
                   help="default: <parent of packed-dir>/table_bbox.json (dataset-specific)")
    a = p.parse_args()
    if a.dataset:
        a.packed_dir = a.packed_dir or str(paths.packed_dir(a.dataset))
        a.raw_dir = a.raw_dir or str(paths.raw_dir(a.dataset))
        a.disp_dir = a.disp_dir or str(paths.disp_dir(a.dataset))
        a.save_path = a.save_path or str(paths.table_bbox_path(a.dataset))
    if not (a.packed_dir and a.raw_dir):
        p.error("--dataset or --packed-dir + --raw-dir required")
    a.disp_dir = a.disp_dir or os.path.join(os.path.dirname(os.path.abspath(a.packed_dir)), "fs_disp")
    if a.save_path is None:
        a.save_path = os.path.join(os.path.dirname(os.path.abspath(a.packed_dir)),
                                   "table_bbox.json")

    app = App(a)
    server = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(app, a.save_path))
    server.daemon_threads = True
    print(f"table-bbox blackout tuner at http://localhost:{a.port}  "
          f"({len(app.files)} eps, cams {app.cams})\nsaves to: {a.save_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
