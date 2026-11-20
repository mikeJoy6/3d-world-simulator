"""Joint bbox + crop tweaker (plain local web page, no viser).

For every camera it shows TWO panels side by side:
  left  = original full-res RGB with the crop rectangle (drag to move it,
          everything outside the crop is dimmed)
  right = the SAME image dimmed to greyscale, with the bbox-cropped point
          cloud (full FS-stereo depth + LRC, colored from the raw RGB)
          projected back into the view. Colored pixels = "what the pack would
          keep with this bbox"; grey things = NOT inside the bbox.

The world bbox is editable live (number inputs, debounced) so you can grow /
shrink it and immediately see the projected coverage change in all cameras,
while dragging the crop rectangles on the same page. Wireframe overlay of the
live bbox is optional (cyan).

Saving:
  * "save crops" -> <root>/cam_crops.json   (consumed by add_render_to_packed.py)
  * "save bbox"  -> <root>/bbox_tweak.json + prints the config.json-style
    "bbox": [...] line. With --write-config (what `./pw crop` passes) it also
    patches config.json in place (previous file kept as config.json.bak).
    NOTE: neither save re-packs anything — re-run `./pw pack` to apply.

    python viz/bbox_crop_tweak_web.py --dataset bimanual_cloth_v3 --port 8093 --write-config

then open http://localhost:8093  (<root> = <PW_DATA_ROOT>/<ds>; raw/ and
fs_disp/ are read from there).

Needs only raw + fs_disp + config.json — NO packed episodes: extrinsics are
fs_disp meta E + config.json camera calibration applied live (identical math
to the packer), so run this right after `./pw camera` and BEFORE packing.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5video  # noqa: E402
from processing import paths                                      # noqa: E402
from processing.config import DatasetConfig                       # noqa: E402
from processing.pc.geometry import apply_tweak, _tweaks_from_config  # noqa: E402

_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>bbox+crop tweak</title>
<style>
 body{margin:0;background:#111;color:#ddd;font-family:system-ui,Arial,sans-serif;font-size:13px}
 #bar{padding:8px 14px;background:#1c1c1c;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 #bbx{padding:6px 14px;background:#181818;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 #cams{display:flex;flex-direction:column;gap:12px;padding:10px}
 .row{display:flex;gap:8px}
 .cam{position:relative;user-select:none}
 .cam img{display:block;width:46vw;pointer-events:none}
 .cam .wrap{position:relative;overflow:hidden;cursor:crosshair}
 .rect{position:absolute;border:2px solid #fff;box-shadow:0 0 0 9999px rgba(0,0,0,.55);pointer-events:none}
 .lbl{margin:3px 0}
 input[type=number]{width:66px;background:#222;color:#ddd;border:1px solid #444}
 input[type=range]{width:300px;vertical-align:middle}
 button{padding:3px 10px}
 select{background:#222;color:#ddd}
 #msg{color:#8f8} .bl{color:#7df}
</style></head><body>
<div id=bar>
 <label>episode <select id=ep></select></label>
 <label>frame <input type=range id=fr min=0 max=0 value=0><span id=frv>0</span></label>
 <label><input type=checkbox id=wire checked> bbox wireframe</label>
 <button id=save>save crops</button>
 <button id=savebb>save bbox</button><span id=msg></span>
 <span style="color:#888">drag on any image to set that camera's crop · right panel: colored = inside bbox</span>
</div>
<div id=bbx>
 <span class=bl>world bbox</span>
 <label>x <input type=number id=b0 step=0.01> .. <input type=number id=b1 step=0.01></label>
 <label>y <input type=number id=b2 step=0.01> .. <input type=number id=b3 step=0.01></label>
 <label>z <input type=number id=b4 step=0.01> .. <input type=number id=b5 step=0.01></label>
 <button id=bbrst>reset to config</button>
 <span id=bbtxt style="color:#888"></span>
</div>
<div id=cams></div>
<script>
let META=null, CROPS={}, BB=null, SCALE={}, tmr=null;
function q(s){return document.querySelector(s)}
async function j(u,opt){const r=await fetch(u,opt);return r.json()}
function bbs(){return BB.map(v=>v.toFixed(3)).join(',')}
function imgUrl(c,mode){return `/img/${c}?ep=${q('#ep').value}&fr=${q('#fr').value}`+
  `&mode=${mode}&bbox=${bbs()}&wire=${q('#wire').checked?1:0}&t=${Date.now()}`}
function refreshImgs(projOnly){META.cams.forEach(c=>{
  if(!projOnly||q('#wire').checked)q('#img'+c+'r').src=imgUrl(c,'rgb');
  q('#img'+c+'p').src=imgUrl(c,'proj');})}
function refreshDebounced(projOnly){clearTimeout(tmr);tmr=setTimeout(()=>refreshImgs(projOnly),250)}
function drawRect(c){
  const [x0,y0,x1,y1]=CROPS['camera_'+c]; const sc=SCALE[c]||1;
  for(const s of ['r','p']){const r=q(`#rect${c}${s}`);
    r.style.left=(x0*sc)+'px'; r.style.top=(y0*sc)+'px';
    r.style.width=((x1-x0)*sc-4)+'px'; r.style.height=((y1-y0)*sc-4)+'px';}
  ['x0','y0','x1','y1'].forEach((k,i)=>q(`#${k}_${c}`).value=CROPS['camera_'+c][i]);
}
function setCrop(c,vals){
  const [W,H]=META.sizes[c];
  vals[0]=Math.max(0,Math.min(vals[0],W-8)); vals[1]=Math.max(0,Math.min(vals[1],H-8));
  vals[2]=Math.max(vals[0]+8,Math.min(vals[2],W)); vals[3]=Math.max(vals[1]+8,Math.min(vals[3],H));
  CROPS['camera_'+c]=vals.map(v=>Math.round(v)); drawRect(c);
}
function setBB(vals){BB=vals.map(Number);
  BB.forEach((v,i)=>q('#b'+i).value=+v.toFixed(3));
  q('#bbtxt').textContent='"bbox": ['+BB.map(v=>+v.toFixed(3)).join(', ')+']';}
async function loadEp(){
  const m=await j('/meta?ep='+q('#ep').value);
  q('#fr').max=m.T-1; if(+q('#fr').value>m.T-1)q('#fr').value=0;
  q('#frv').textContent=q('#fr').value; refreshImgs();
}
async function init(){
  META=await j('/meta'); CROPS=await j('/load'); setBB(META.bbox);
  META.eps.forEach(e=>{const o=document.createElement('option');o.value=e;o.textContent=e;q('#ep').appendChild(o)});
  META.cams.forEach(c=>{
    if(!CROPS['camera_'+c])CROPS['camera_'+c]=[0,0,...META.sizes[c]];
    const d=document.createElement('div');d.className='cam';
    d.innerHTML=`<div class=lbl>cam ${c} &nbsp;
      x0 <input type=number id=x0_${c} step=8> y0 <input type=number id=y0_${c} step=8>
      x1 <input type=number id=x1_${c} step=8> y1 <input type=number id=y1_${c} step=8>
      <button id=rst_${c}>full</button></div>
      <div class=row>
       <div class=wrap id=wrap${c}r><img id=img${c}r><div class=rect id=rect${c}r></div></div>
       <div class=wrap id=wrap${c}p><img id=img${c}p><div class=rect id=rect${c}p></div></div>
      </div>`;
    q('#cams').appendChild(d);
    ['x0','y0','x1','y1'].forEach((k,i)=>q(`#${k}_${c}`).addEventListener('change',()=>{
      const v=CROPS['camera_'+c].slice(); v[i]=+q(`#${k}_${c}`).value; setCrop(c,v);}));
    q('#rst_'+c).addEventListener('click',()=>setCrop(c,[0,0,...META.sizes[c]]));
    setCrop(c,CROPS['camera_'+c]);   // clamp a stale rect to this camera's image
    for(const s of ['r','p']){
      const w=q('#wrap'+c+s); let drag=null;
      w.addEventListener('pointerdown',e=>{const b=w.getBoundingClientRect();
        const sc=SCALE[c]||1; drag=[(e.clientX-b.left)/sc,(e.clientY-b.top)/sc]; w.setPointerCapture(e.pointerId);});
      w.addEventListener('pointermove',e=>{if(!drag)return;const b=w.getBoundingClientRect();
        const sc=SCALE[c]||1; const x=(e.clientX-b.left)/sc, y=(e.clientY-b.top)/sc;
        setCrop(c,[Math.min(drag[0],x),Math.min(drag[1],y),Math.max(drag[0],x),Math.max(drag[1],y)]);});
      w.addEventListener('pointerup',()=>{drag=null;});
    }
    q('#img'+c+'r').addEventListener('load',()=>{SCALE[c]=q('#img'+c+'r').clientWidth/META.sizes[c][0]; drawRect(c);});
  });
  for(let i=0;i<6;i++)q('#b'+i).addEventListener('change',()=>{
    const v=BB.slice(); v[i]=+q('#b'+i).value; setBB(v); refreshDebounced(true);});
  q('#bbrst').addEventListener('click',()=>{setBB(META.bbox); refreshDebounced(true);});
  q('#ep').addEventListener('change',loadEp);
  q('#wire').addEventListener('change',()=>refreshImgs());
  q('#fr').addEventListener('input',()=>{q('#frv').textContent=q('#fr').value;refreshDebounced(false);});
  q('#save').addEventListener('click',async()=>{
    const r=await j('/save',{method:'POST',headers:{'Content-Type':'application/json'},
                             body:JSON.stringify(CROPS)});
    q('#msg').textContent='crops -> '+r.path; setTimeout(()=>q('#msg').textContent='',5000);});
  q('#savebb').addEventListener('click',async()=>{
    const r=await j('/save_bbox',{method:'POST',headers:{'Content-Type':'application/json'},
                                  body:JSON.stringify(BB)});
    q('#msg').textContent='bbox -> '+r.path+(r.config?' (+config.json)':'');
    setTimeout(()=>q('#msg').textContent='',5000);});
  window.addEventListener('resize',()=>META.cams.forEach(c=>{
    SCALE[c]=q('#img'+c+'r').clientWidth/META.sizes[c][0]; drawRect(c);}));
  await loadEp();
}
init();
</script></body></html>"""

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def bbox_corners(b):
    x0, x1, y0, y1, z0, z1 = b
    return np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                     [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float64)


class App:
    def __init__(self, a):
        self.a = a
        self.files = {}                       # ep -> (raw path, fs_disp path)
        for fp in sorted(glob.glob(os.path.join(a.disp_dir, "disp_episode_*.hdf5"))):
            m = re.search(r"disp_episode_(\d+)\.hdf5", os.path.basename(fp))
            if not m:
                continue
            raw = os.path.join(a.raw_dir, f"episode_{m.group(1)}.hdf5")
            if os.path.exists(raw):
                self.files[int(m.group(1))] = (raw, fp)
        assert self.files, f"no episodes with both raw/{a.raw_dir} and disp/{a.disp_dir}"
        # E = fs_disp meta extrinsics + config.json camera calibration applied
        # LIVE (same math the packer uses) — no packed episodes needed, and
        # fresh tweak_extrinsics edits show up on restart without repacking.
        cfg = DatasetConfig.load(os.path.join(a.proc_dir, "config.json"))
        self.tweaks = _tweaks_from_config(cfg)
        self.bbox0 = [float(x) for x in
                      (a.bbox or json.load(open(os.path.join(a.proc_dir, "config.json")))["bbox"])]
        self._h = OrderedDict()               # ep -> (raw h5, fs h5, T)
        self._cl = OrderedDict()              # (ep,fr) -> (pts Nx3 f32 world, rgb Nx3 u8)
        self.lock = threading.Lock()
        ep0 = min(self.files)
        r, fs, _ = self._handle(ep0)
        self.cams = sorted(int(k.split("_")[1]) for k in fs["meta"].keys())
        # crop rects are adjustable for EVERY cam (incl. wrist) — but the
        # world point cloud / bbox wire use FIXED cams only: the wrist cam's
        # meta E is identity (FK-dependent pose), it can't be lifted here.
        cfg_cams = {int(k.split("_")[1]) for k in cfg.cameras.keys()}
        self.pc_cams = ([c for c in self.cams if c in cfg_cams]
                        if cfg_cams else list(self.cams))
        if set(self.pc_cams) != set(self.cams):
            print(f"[crop] cam(s) {sorted(set(self.cams)-set(self.pc_cams))} are "
                  f"crop-only (not in config.cameras): excluded from PC/bbox overlay")
        self.sizes = {c: tuple(int(v) for v in r[f"observations/images/camera_{c}_color"].shape[1:3][::-1])
                      for c in self.cams}                                   # cam -> (W, H), per camera
        self.H, self.W = self.sizes[self.cams[0]][1], self.sizes[self.cams[0]][0]
        if len(set(self.sizes.values())) > 1:
            print(f"[crop] per-camera image sizes: {self.sizes}")

    def _handle(self, ep):
        with self.lock:
            if ep in self._h:
                self._h.move_to_end(ep)
                return self._h[ep]
            r = h5video.File(self.files[ep][0], "r")
            fs = h5video.File(self.files[ep][1], "r")
            T = fs["disp/camera_0"].shape[0]
            self._h[ep] = (r, fs, T)
            if len(self._h) > 3:
                _, old = self._h.popitem(last=False)
                old[0].close(); old[1].close()
            return self._h[ep]

    def _cam_calib(self, fs, c, fr):
        m = fs[f"meta/camera_{c}"]
        E = apply_tweak(m["E_cam2world"][fr].astype(np.float64),
                        self.tweaks.get(c, {})).astype(np.float64)
        return (m["K_ir"][fr].astype(np.float64), m["K_rgb"][fr].astype(np.float64),
                m["T_rgb_to_ir"][fr].astype(np.float64), E,
                float(m["baseline"][fr]))

    def _cloud(self, ep, fr):
        """Full-scene world cloud from FS disp (+LRC, z-range, RGB-in-frame) of ALL
        cams, colored by sampling the raw color image — the same validity rules the
        pack applies MINUS the bbox (so any live bbox can be applied on top)."""
        key = (ep, fr)
        with self.lock:
            if key in self._cl:
                self._cl.move_to_end(key)
                return self._cl[key]
        r, fs, T = self._handle(ep)
        fr = min(fr, T - 1)
        s = self.a.point_stride
        P, C = [], []
        for c in self.pc_cams:
            K_ir, K_rgb, T_r2i, E, base = self._cam_calib(fs, c, fr)
            if not np.isfinite(E).all() or abs(np.linalg.det(E[:3, :3])) < 1e-6:
                continue
            disp = fs[f"disp/camera_{c}"][fr].astype(np.float32)
            H, W = disp.shape
            yy, xx = np.mgrid[0:H:s, 0:W:s]
            d = disp[::s, ::s]
            ok = d > 0.5
            if self.a.lrc_thresh > 0 and f"disp_right/camera_{c}" in fs:
                dr = fs[f"disp_right/camera_{c}"][fr].astype(np.float32)
                xR = np.round(xx - d).astype(np.int32)
                inR = (xR >= 0) & (xR < W)
                drm = dr[yy, np.clip(xR, 0, W - 1)]
                ok &= inR & (np.abs(d - drm) <= self.a.lrc_thresh)
            z = np.where(d > 0.5, K_ir[0, 0] * base / np.maximum(d, 0.5), 0.0)
            ok &= (z >= self.a.min_z) & (z <= self.a.max_z)
            x_ir = (xx - K_ir[0, 2]) / K_ir[0, 0] * z
            y_ir = (yy - K_ir[1, 2]) / K_ir[1, 1] * z
            pts_ir = np.stack([x_ir[ok], y_ir[ok], z[ok]], 1)
            # sample color in the RGB cam (IR -> RGB via inv(T_rgb_to_ir)), drop out-of-frame
            R_i2r = T_r2i[:3, :3].T
            t_i2r = -R_i2r @ T_r2i[:3, 3]
            pr = pts_ir @ R_i2r.T + t_i2r
            zr = pr[:, 2]
            good = zr > 1e-3
            u = K_rgb[0, 0] * pr[:, 0] / np.where(good, zr, 1.0) + K_rgb[0, 2]
            v = K_rgb[1, 1] * pr[:, 1] / np.where(good, zr, 1.0) + K_rgb[1, 2]
            Wc_, Hc_ = self.sizes[c]
            good &= (u >= 0) & (u < Wc_ - 1) & (v >= 0) & (v < Hc_ - 1)
            pts_ir = pts_ir[good]
            color = r[f"observations/images/camera_{c}_color"][fr]      # RGB u8
            cc = json.load(open(os.path.join(self.a.proc_dir, "config.json"))).get("color_calib", {}).get(f"camera_{c}")
            if cc:
                color = np.clip(color.astype(np.float32) * np.asarray(cc["gain"])
                                + np.asarray(cc.get("offset", [0, 0, 0])), 0, 255).astype(np.uint8)
            C.append(color[np.round(v[good]).astype(np.int32),
                           np.round(u[good]).astype(np.int32)])
            P.append(pts_ir @ E[:3, :3].T + E[:3, 3])
        pts = np.concatenate(P, 0).astype(np.float32) if P else np.zeros((0, 3), np.float32)
        rgb = np.concatenate(C, 0) if C else np.zeros((0, 3), np.uint8)
        with self.lock:
            self._cl[key] = (pts, rgb)
            if len(self._cl) > 6:
                self._cl.popitem(last=False)
        return pts, rgb

    def _draw_wire(self, img, bbox, K, V, scale=1.0):
        C = bbox_corners(bbox)
        pc = C @ V[:3, :3].T + V[:3, 3]
        Z = pc[:, 2]
        u = (K[0, 0] * pc[:, 0] / np.where(np.abs(Z) < 1e-6, 1e-6, Z) + K[0, 2]) * scale
        v = (K[1, 1] * pc[:, 1] / np.where(np.abs(Z) < 1e-6, 1e-6, Z) + K[1, 2]) * scale
        for i, jj in EDGES:
            if Z[i] <= 0.02 or Z[jj] <= 0.02:
                continue
            pa = (int(round(u[i])), int(round(v[i])))
            pb = (int(round(u[jj])), int(round(v[jj])))
            cv2.line(img, pa, pb, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.line(img, pa, pb, (255, 255, 0), 1, cv2.LINE_AA)   # cyan (imgs are BGR here)

    def jpeg(self, ep, fr, cam, mode, bbox, wire):
        r, fs, T = self._handle(ep)
        fr = min(fr, T - 1)
        img = np.ascontiguousarray(r[f"observations/images/camera_{cam}_color"][fr][..., ::-1])
        # Match the brightness/color calibration used by the packed RGB targets.
        cc = json.load(open(os.path.join(self.a.proc_dir, "config.json"))).get("color_calib", {}).get(f"camera_{cam}")
        if cc:
            img = np.clip(img.astype(np.float32) * np.asarray(cc["gain"])[::-1]
                          + np.asarray(cc.get("offset", [0, 0, 0]))[::-1], 0, 255).astype(np.uint8)
        K_ir, K_rgb, T_r2i, E, _ = self._cam_calib(fs, cam, fr)
        V = None
        if (cam in self.pc_cams and np.isfinite(E).all()
                and abs(np.linalg.det(E[:3, :3])) > 1e-6):
            V = np.linalg.inv(E @ T_r2i)                            # world -> this RGB cam
        d = self.a.display_downscale
        Wd, Hd = self.sizes[cam][0] // d, self.sizes[cam][1] // d
        if mode == "proj":
            base = cv2.resize(img, (Wd, Hd), interpolation=cv2.INTER_AREA)
            grey = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
            out = (np.repeat(grey[..., None], 3, 2) * 0.35).astype(np.uint8)
            if V is not None:
                pts, rgb = self._cloud(ep, fr)
                m = ((pts[:, 0] >= bbox[0]) & (pts[:, 0] <= bbox[1]) &
                     (pts[:, 1] >= bbox[2]) & (pts[:, 1] <= bbox[3]) &
                     (pts[:, 2] >= bbox[4]) & (pts[:, 2] <= bbox[5]))
                p = pts[m] @ V[:3, :3].T.astype(np.float32) + V[:3, 3].astype(np.float32)
                cc = rgb[m]
                z = p[:, 2]
                f = z > 0.02
                u = (K_rgb[0, 0] * p[f, 0] / z[f] + K_rgb[0, 2]) / d
                v = (K_rgb[1, 1] * p[f, 1] / z[f] + K_rgb[1, 2]) / d
                ui = np.round(u).astype(np.int32); vi = np.round(v).astype(np.int32)
                inb = (ui >= 0) & (ui < Wd) & (vi >= 0) & (vi < Hd)
                order = np.argsort(-z[f][inb])                       # far -> near; near wins
                ui, vi = ui[inb][order], vi[inb][order]
                col = cc[f][inb][order][:, ::-1]                     # RGB -> BGR
                splat = np.zeros((Hd, Wd, 3), np.uint8)
                hit = np.zeros((Hd, Wd), bool)
                splat[vi, ui] = col
                hit[vi, ui] = True
                ker = np.ones((3, 3), np.uint8)
                fill = cv2.dilate(splat, ker)
                hitd = cv2.dilate(hit.astype(np.uint8), ker) > 0
                splat[~hit & hitd] = fill[~hit & hitd]               # 1px hole fill
                out[hitd] = splat[hitd]
            if wire and V is not None:
                self._draw_wire(out, bbox, K_rgb, V, scale=1.0 / d)
            img = out
        else:
            if wire and V is not None:
                self._draw_wire(img, bbox, K_rgb, V)
            if d > 1:
                img = cv2.resize(img, (Wd, Hd), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes()


def make_handler(app, crops_path, bbox_path, config_path, write_config):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, body, ctype="application/json", code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if u.path == "/":
                self._send(_PAGE.encode(), "text/html")
            elif u.path == "/meta":
                if "ep" in qs:
                    _, _, T = app._handle(int(qs["ep"][0]))
                    self._send(json.dumps({"T": T}).encode())
                else:
                    self._send(json.dumps(dict(eps=sorted(app.files), cams=app.cams,
                                               W=app.W, H=app.H, bbox=app.bbox0,
                                               sizes={c: list(app.sizes[c]) for c in app.cams})).encode())
            elif u.path == "/load":
                crops = {}
                if os.path.exists(crops_path):
                    crops = json.load(open(crops_path))
                self._send(json.dumps(crops).encode())
            elif u.path.startswith("/img/"):
                cam = int(u.path.rsplit("/", 1)[1])
                ep = int(qs.get("ep", [min(app.files)])[0])
                fr = int(qs.get("fr", [0])[0])
                mode = qs.get("mode", ["rgb"])[0]
                bbox = [float(x) for x in qs.get("bbox", [",".join(map(str, app.bbox0))])[0].split(",")]
                wire = qs.get("wire", ["1"])[0] != "0"
                self._send(app.jpeg(ep, fr, cam, mode, bbox, wire), "image/jpeg")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            path = urlparse(self.path).path
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n)) if n else None
            if path == "/save":
                json.dump(body, open(crops_path, "w"), indent=2)
                print(f"saved crops -> {crops_path}: {body}", flush=True)
                self._send(json.dumps({"path": crops_path}).encode())
            elif path == "/save_bbox":
                bb = [round(float(x), 4) for x in body]
                json.dump({"bbox": bb}, open(bbox_path, "w"), indent=2)
                print(f"saved bbox -> {bbox_path}\n  \"bbox\": {bb}", flush=True)
                wrote_cfg = False
                if write_config and os.path.exists(config_path):
                    cfg = json.load(open(config_path))
                    cfg["bbox"] = bb
                    shutil.copy2(config_path, config_path + ".bak")
                    json.dump(cfg, open(config_path, "w"), indent=2)
                    print(f"patched {config_path} (backup: config.json.bak)", flush=True)
                    wrote_cfg = True
                self._send(json.dumps({"path": bbox_path, "config": wrote_cfg}).encode())
            else:
                self._send(b"not found", "text/plain", 404)
    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", help="shortcut: sets raw/disp/proc dirs via processing.paths")
    p.add_argument("--raw-dir", help="original episode h5s (camera_X_color); default <root>/raw")
    p.add_argument("--disp-dir", help="FS disparity dir (disp_episode_N.hdf5); default <root>/fs_disp")
    p.add_argument("--proc-dir", help="dataset root (config.json, cam_crops.json)")
    p.add_argument("--port", type=int, default=8093)
    p.add_argument("--display-downscale", type=int, default=2)
    p.add_argument("--point-stride", type=int, default=2, help="pixel stride when unprojecting")
    p.add_argument("--lrc-thresh", type=float, default=2.0, help="pw pack default; 0 disables LRC")
    p.add_argument("--min-z", type=float, default=0.1)
    p.add_argument("--max-z", type=float, default=2.0)
    p.add_argument("--bbox", type=float, nargs=6, default=None,
                   metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                   help="initial bbox; default: config.json 'bbox'")
    p.add_argument("--write-config", action="store_true",
                   help="'save bbox' also patches config.json in place (with .bak)")
    a = p.parse_args()
    if a.dataset:
        a.proc_dir = a.proc_dir or str(paths.root(a.dataset))
        a.raw_dir = a.raw_dir or str(paths.raw_dir(a.dataset))
        a.disp_dir = a.disp_dir or str(paths.disp_dir(a.dataset))
    assert a.raw_dir and a.proc_dir, "--dataset or explicit --raw-dir/--proc-dir required"
    a.disp_dir = a.disp_dir or os.path.join(a.proc_dir, "fs_disp")

    app = App(a)
    crops_path = os.path.join(a.proc_dir, "cam_crops.json")
    bbox_path = os.path.join(a.proc_dir, "bbox_tweak.json")
    config_path = os.path.join(a.proc_dir, "config.json")
    server = ThreadingHTTPServer(("0.0.0.0", a.port),
                                 make_handler(app, crops_path, bbox_path, config_path,
                                              a.write_config))
    server.daemon_threads = True
    print(f"bbox+crop tweaker up at http://localhost:{a.port}  "
          f"({len(app.files)} episodes, cams {app.cams}, {app.W}x{app.H})\n"
          f"init bbox: {app.bbox0}\ncrops -> {crops_path}\nbbox  -> {bbox_path}"
          f"{'  (+config.json)' if a.write_config else ''}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
