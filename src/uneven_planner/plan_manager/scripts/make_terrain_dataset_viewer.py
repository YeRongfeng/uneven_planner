#!/usr/bin/env python3
"""Build a standalone interactive HTML viewer for sampled 20 m terrain scenes."""

import argparse
import base64
import json
import math
import pickle
from pathlib import Path

import numpy as np

from terrain_map_quality import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="Scene NPZ files or directories")
    parser.add_argument("--output", required=True, type=Path, help="Output HTML file")
    parser.add_argument("--title", default="20 m Terrain Dataset Viewer")
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search input directories recursively instead of only their top level")
    parser.add_argument("--max-scenes", type=int, default=0, help="0 keeps every scene")
    parser.add_argument(
        "--trajectory-root", type=Path,
        help="Dataset or domain root containing train/val/env*/map.p and path_*.p")
    parser.add_argument(
        "--max-paths-per-scene", type=int, default=0,
        help="Limit embedded trajectories per scene; 0 keeps every path")
    return parser.parse_args()


def collect_paths(inputs, recursive):
    paths = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            iterator = path.rglob("scene_*.npz") if recursive else path.glob("scene_*.npz")
            paths.extend(iterator)
        elif path.is_file() and path.suffix == ".npz":
            paths.append(path)
        else:
            raise FileNotFoundError(f"No NPZ file or directory found: {value}")
    return sorted(dict.fromkeys(path.resolve() for path in paths))


def encode_bytes(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def quantize(array, valid, bits, minimum=None, maximum=None):
    values = np.asarray(array, dtype=np.float64)
    finite_valid = valid & np.isfinite(values)
    if minimum is None:
        minimum = float(np.min(values[finite_valid]))
    if maximum is None:
        maximum = float(np.max(values[finite_valid]))
    if maximum <= minimum:
        maximum = minimum + 1e-9
    levels = (1 << bits) - 1
    normalized = np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)
    if bits == 16:
        encoded = np.rint(normalized * levels).astype("<u2")
    elif bits == 8:
        encoded = np.rint(normalized * levels).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported quantization width: {bits}")
    return encode_bytes(encoded), minimum, maximum


def scene_record(path):
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        metadata["canonical_map_index"] = int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        pass
    quality = evaluate(str(path))
    with np.load(path) as data:
        elevation = data["elevation"].astype(np.float64)
        valid = data["valid_mask"].astype(bool)
        normals = np.stack(
            (data["normal_x"], data["normal_y"], data["normal_z"]), axis=-1
        ).astype(np.float64)
        observed = data["observed_mask"].astype(bool)
        obstacle_mask = np.asarray(
            data["obstacle_mask"] if "obstacle_mask" in data.files
            else np.zeros(elevation.shape, dtype=bool), dtype=bool)
        obstacle_height = np.asarray(
            data["obstacle_height"] if "obstacle_height" in data.files
            else np.zeros(elevation.shape, dtype=np.float32), dtype=np.float64)

    return make_record(
        scene_id=path.stem,
        relative_source=str(path),
        elevation=elevation,
        normals=normals,
        valid=valid,
        observed=observed,
        obstacle_mask=obstacle_mask,
        obstacle_height=obstacle_height,
        resolution=float(metadata.get("resolution_m", 0.2)),
        metadata=metadata,
        quality=quality,
        trajectories=[],
        split="sampled",
    )


def make_record(scene_id, relative_source, elevation, normals, valid, observed,
                obstacle_mask, obstacle_height, resolution, metadata, quality,
                trajectories, split):
    normal_length = np.linalg.norm(normals, axis=-1)
    slope = np.degrees(np.arccos(np.clip(
        normals[:, :, 2] / np.maximum(normal_length, 1e-12), -1.0, 1.0)))
    height, width = elevation.shape
    elevation_b64, elevation_min, elevation_max = quantize(
        elevation, valid, 16)
    slope_max = max(
        30.0,
        min(90.0, math.ceil(float(np.max(slope[valid])) / 5.0) * 5.0),
    )
    slope_b64, slope_min, slope_max = quantize(
        slope, valid, 8, minimum=0.0, maximum=slope_max)
    valid_b64 = encode_bytes(np.packbits(valid.ravel(), bitorder="big"))
    observed_b64 = encode_bytes(np.packbits(observed.ravel(), bitorder="big"))
    obstacle_b64 = encode_bytes(
        np.packbits(obstacle_mask.ravel(), bitorder="big"))
    obstacle_height_b64, _, obstacle_height_max = quantize(
        obstacle_height, valid, 8, minimum=0.0,
        maximum=max(1e-9, float(np.max(obstacle_height))))
    return {
        "id": scene_id,
        "display_id": scene_id,
        "relative_source": relative_source,
        "split": split,
        "domain": metadata.get("domain", ""),
        "width": int(width),
        "height": int(height),
        "resolution": float(resolution),
        "elevation": elevation_b64,
        "elevation_min": elevation_min,
        "elevation_max": elevation_max,
        "slope": slope_b64,
        "slope_min": slope_min,
        "slope_max": slope_max,
        "valid": valid_b64,
        "observed": observed_b64,
        "obstacle": obstacle_b64,
        "obstacle_height": obstacle_height_b64,
        "obstacle_height_max": obstacle_height_max,
        "grade": quality.get("grade") or "reject",
        "score": quality.get("geometry_score"),
        "quality": quality.get("quality"),
        "quality_metrics": quality.get("metrics", {}),
        "needs_return": bool(
            metadata.get("needs_return")
            or (metadata.get("needs_return_reason"))),
        "metadata": metadata,
        "trajectories": trajectories,
    }


def trajectory_scene_record(map_path, trajectory_root, max_paths):
    with map_path.open("rb") as stream:
        map_data = pickle.load(stream)
    tensor = np.asarray(map_data["tensor"], dtype=np.float64)
    if tensor.ndim != 3 or tensor.shape[2] < 4:
        raise ValueError(f"Unexpected map tensor shape: {tensor.shape}")
    elevation = tensor[:, :, 0]
    normals = tensor[:, :, 1:4]
    obstacle_height = (
        tensor[:, :, 4] if tensor.shape[2] >= 5
        else np.zeros(elevation.shape, dtype=np.float32))
    valid = np.asarray(
        map_data.get("valid_mask", np.ones(elevation.shape, dtype=bool)),
        dtype=bool,
    )
    crop = map_data.get("crop") or {}
    metadata = dict(crop.get("mother_map_sample") or {})
    metadata["environment_id"] = map_path.parent.name
    metadata["dataset_phase"] = map_data.get("dataset_phase")
    metadata["planner_map_version"] = map_data.get("planner_map_version")
    metadata["original_map_index"] = map_data.get("original_map_index")
    metadata["source_map"] = map_data.get("source_map")
    metadata["original_map_path"] = map_data.get("original_map_path")

    observed = np.asarray(
        map_data.get("observed_mask", valid), dtype=bool)
    if observed.shape != elevation.shape:
        observed = valid.copy()
    obstacle_mask = np.asarray(
        map_data.get("obstacle_mask", obstacle_height > 0.0), dtype=bool)
    if obstacle_mask.shape != elevation.shape:
        obstacle_mask = np.zeros(elevation.shape, dtype=bool)
    obstacle_height = np.asarray(
        map_data.get("obstacle_height", obstacle_height), dtype=np.float64)
    if obstacle_height.shape != elevation.shape:
        obstacle_height = np.zeros(elevation.shape, dtype=np.float64)
    quality = map_data.get("quality")
    source_value = map_data.get("original_map_path")
    source_path = Path(source_value) if source_value else None
    sidecar = source_path.with_suffix(".npz") if source_path else None
    if not isinstance(quality, dict) or not quality.get("quality"):
        quality = {
            "quality": "pass",
            "grade": "unknown",
            "geometry_score": None,
            "metrics": {"valid_fraction": float(valid.mean())},
        }
        if sidecar and sidecar.is_file():
            quality = evaluate(str(sidecar))

    path_files = sorted(map_path.parent.glob("path_*.p"))
    if max_paths > 0:
        path_files = path_files[:max_paths]
    trajectories = []
    for path_file in path_files:
        with path_file.open("rb") as stream:
            path_data = pickle.load(stream)
        points = np.asarray(path_data["path"], dtype="<f4")
        if points.ndim != 2 or points.shape[1] < 2 or not np.isfinite(points).all():
            raise ValueError(f"Invalid trajectory array in {path_file}")
        trajectories.append({
            "name": path_file.stem,
            "points": int(len(points)),
            "data": encode_bytes(points[:, :3]),
        })

    if (map_path.parent / "needs_return.json").is_file():
        metadata["needs_return"] = True
    relative = map_path.relative_to(trajectory_root)
    return make_record(
        scene_id=f"{relative.parts[-3]}/{map_path.parent.name}",
        relative_source=str(map_path),
        elevation=elevation,
        normals=normals,
        valid=valid,
        observed=observed,
        obstacle_mask=obstacle_mask,
        obstacle_height=obstacle_height,
        resolution=float(map_data.get("resolution", 0.2)),
        metadata=metadata,
        quality=quality,
        trajectories=trajectories,
        split=str(map_data.get("dataset_phase") or relative.parts[-3]),
    )


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#101417;--panel:#171d21;--panel2:#20272c;--line:#303a40;--text:#e8eef2;--muted:#99a8b2;--accent:#56b4d3;--ok:#4fc38a;--needs_return:#ef7d65}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;overflow:hidden}
button,input,select{font:inherit;color:inherit}button,select,input[type=search]{background:var(--panel2);border:1px solid var(--line);border-radius:7px;padding:7px 10px}button{cursor:pointer}button:hover,.mode.active{border-color:var(--accent);background:#24333a}.app{height:100vh;display:grid;grid-template-columns:310px minmax(0,1fr) 330px;grid-template-rows:auto 1fr}
header{grid-column:1/-1;display:flex;align-items:center;gap:16px;padding:11px 16px;background:var(--panel);border-bottom:1px solid var(--line);flex-wrap:wrap}header h1{font-size:17px;margin:0;white-space:nowrap}.summary{color:var(--muted)}.dataset-controls{display:flex;align-items:center;gap:7px;min-width:0;max-width:100%;flex-wrap:wrap}.dataset-label{color:var(--muted);white-space:nowrap}.dataset-root{min-width:0;max-width:27vw;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px}.quick-actions{display:flex;gap:7px;flex-wrap:wrap}.header-actions{margin-left:auto;display:flex;gap:7px}
.sidebar,.details{min-height:0;background:var(--panel);display:flex;flex-direction:column}.sidebar{border-right:1px solid var(--line)}.details{border-left:1px solid var(--line);overflow:auto}.filters{padding:12px;border-bottom:1px solid var(--line);display:grid;gap:9px}.filters input[type=search]{width:100%}.filter-row{display:flex;gap:11px;flex-wrap:wrap;color:var(--muted)}.filter-row label{display:flex;gap:4px;align-items:center}.scene-list{overflow:auto;padding:7px}.scene-card{display:grid;grid-template-columns:18px 1fr auto;gap:3px 8px;padding:9px 10px;margin:4px 0;border:1px solid transparent;border-radius:8px;cursor:pointer;align-items:center}.scene-card input[type=checkbox]{margin:0}.scene-card:hover{background:var(--panel2)}.scene-card.active{border-color:var(--accent);background:#1d2b31}.scene-name{font-weight:650}.scene-sub{font-size:12px;color:var(--muted)}.badge{align-self:start;border-radius:999px;padding:2px 7px;font-size:11px;font-weight:700;color:#102018}.badge.ok{background:var(--ok)}.badge.needs_return{background:var(--needs_return);color:#2a0f0c}
main{min-width:0;min-height:0;display:grid;grid-template-rows:auto 1fr;background:#0d1114}.toolbar{display:flex;align-items:center;gap:8px;padding:9px 12px;border-bottom:1px solid var(--line);flex-wrap:wrap}.toolbar .spacer{flex:1}.toolbar label{display:flex;align-items:center;gap:7px;color:var(--muted)}input[type=range]{accent-color:var(--accent)}.canvas-wrap{position:relative;min-height:0}canvas{width:100%;height:100%;display:block;touch-action:none}.hint{position:absolute;left:12px;bottom:10px;background:#101417d9;border:1px solid var(--line);color:var(--muted);padding:5px 8px;border-radius:6px;pointer-events:none}.tooltip{position:fixed;z-index:10;display:none;pointer-events:none;background:#0b0f12ed;border:1px solid #51616b;border-radius:7px;padding:7px 9px;white-space:pre;font-size:12px;box-shadow:0 5px 18px #0008}
.detail-head{padding:14px;border-bottom:1px solid var(--line)}.detail-head h2{font-size:18px;margin:0 0 4px}.detail-head .source{color:var(--muted);font-size:12px;overflow-wrap:anywhere}.detail-actions{display:flex;align-items:center;gap:8px;padding:10px 12px;border-bottom:1px solid var(--line)}.detail-actions button{flex:1}.detail-actions span{color:var(--muted);font-size:12px}.stats{padding:12px;display:grid;grid-template-columns:1fr 1fr;gap:8px}.stat{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:9px}.stat .k{font-size:11px;color:var(--muted)}.stat .v{font-size:17px;font-weight:650;margin-top:2px}.metadata{padding:0 12px 16px}.metadata table{width:100%;border-collapse:collapse}.metadata td{padding:6px 3px;border-bottom:1px solid var(--line);vertical-align:top}.metadata td:first-child{color:var(--muted);width:43%}.metadata pre{font:11px/1.4 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#0d1114;padding:9px;border-radius:7px;max-height:320px;overflow:auto}details summary{cursor:pointer;color:var(--accent);padding:10px 0}
 .picker-backdrop{position:fixed;inset:0;z-index:20;display:grid;place-items:center;background:#000b;padding:20px}.picker-backdrop[hidden]{display:none!important}.picker{width:min(760px,calc(100vw - 40px));max-height:min(680px,calc(100vh - 40px));display:grid;grid-template-rows:auto auto auto minmax(180px,1fr) auto;gap:10px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;box-shadow:0 16px 50px #000b}.picker-head,.picker-bar,.picker-foot{display:flex;align-items:center;gap:8px}.picker-head h2{margin:0;font-size:15px;flex:1}.picker-bar input{min-width:0;flex:1}.picker-current{color:var(--muted);font-size:12px;overflow-wrap:anywhere}.picker-list{min-height:180px;overflow:auto;display:grid;align-content:start;gap:6px;padding:2px}.picker-item{width:100%;text-align:left}.picker-foot{border-top:1px solid var(--line);padding-top:9px}.picker-foot .picker-hint{flex:1;color:var(--muted);font-size:12px}
@media(max-width:1050px){.app{grid-template-columns:260px minmax(0,1fr)}.details{position:absolute;right:0;top:53px;bottom:0;width:330px;z-index:5;box-shadow:-8px 0 22px #0008}.details.hidden{display:none}}
</style>
</head>
<body>
<div class="app">
<header><h1>__TITLE__</h1><span class="summary" id="summary"></span><div class="dataset-controls" id="datasetControls" hidden><span class="dataset-label">当前数据目录</span><span class="dataset-root" id="datasetRoot"></span><button id="chooseDatasetRoot">选择目录</button><button id="reloadDataset">重新读取</button><div class="quick-actions"><button id="quickCanonical">快速加载中间地图</button><button id="quickFinal">快速加载最终结果</button></div></div><div class="header-actions"><button id="prev">← 上一张</button><button id="next">下一张 →</button><button id="toggleDetails">参数</button></div></header>
<aside class="sidebar">
  <div class="filters"><select id="domainFilter"><option value="all">全部地形</option></select><input id="search" type="search" placeholder="搜索场景或站点">
    <div class="filter-row" id="returnFilters"><label><input type="checkbox" value="needs_return" checked>需要打回</label><label><input type="checkbox" value="ok" checked>正常</label></div>
    <select id="sort"><option value="name">按编号</option><option value="returnFirst">需要打回优先</option><option value="yaw">按裁剪角度</option></select>
    <div class="filter-row" id="batchReturnRow"><button id="selectNeedsReturn">全选需要打回</button><button id="clearReturnSelection">清除选择</button><button id="returnSelected">批量打回所选</button></div>
  </div><div class="scene-list" id="sceneList"></div>
</aside>
<main>
  <div class="toolbar">
    <button class="mode active" data-mode="2d">2D 地图</button><button class="mode" data-mode="3d">3D 地形</button><button class="mode" data-mode="overview">母图分布</button>
    <select id="layer"><option value="elevation">高程</option><option value="slope">坡度</option><option value="obstacle">物理障碍物</option><option value="observed">直接点云支撑</option><option value="valid">最终有效区域</option></select>
    <label><input id="showTrajectories" type="checkbox" checked>轨迹</label><button id="prevTrajectory">← 上一条</button><select id="trajectory"><option value="0">轨迹 0</option></select><button id="nextTrajectory">下一条 →</button><label><input id="allTrajectories" type="checkbox">全部轨迹</label>
    <label id="exagLabel">垂直夸张 <input id="exag" type="range" min="1" max="30" value="1"><span id="exagValue">1×</span></label>
    <label><input id="grid" type="checkbox" checked>网格</label><span class="spacer"></span><button id="resetView">重置视角</button>
  </div>
  <div class="canvas-wrap"><canvas id="canvas"></canvas><div class="hint" id="hint">滚轮缩放，拖拽平移，悬停读取数值</div></div>
</main>
<aside class="details" id="details"><div class="detail-head"><h2 id="detailName"></h2><div class="source" id="detailSource"></div></div><div class="detail-actions"><button id="replaceScene" disabled>打回，去8767补位</button><span id="replaceStatus">选择地图后可用</span></div><div class="stats" id="stats"></div><div class="metadata" id="metadata"></div></aside>
</div><div class="picker-backdrop" id="datasetPicker" hidden><section class="picker" role="dialog" aria-modal="true" aria-labelledby="pickerTitle"><div class="picker-head"><h2 id="pickerTitle">选择数据目录</h2><button id="closeDatasetPicker">关闭</button></div><div class="picker-bar"><input id="pickerPath" spellcheck="false"><button id="pickerGo">打开</button><button id="pickerUp">上一级</button></div><div class="picker-current" id="pickerCurrent"></div><div class="picker-list" id="pickerList"></div><div class="picker-foot"><span class="picker-hint">选择包含 train/val 或其上层目录的数据集文件夹。</span><button id="useDatasetDirectory">加载此目录</button></div></section></div><div class="tooltip" id="tooltip"></div>
<script>
let SCENES=__SCENES__;const LIVE=__LIVE__;let DATASET_ROOT=__DATASET_ROOT__;const QUICK_ROOTS=__QUICK_ROOTS__;
const state={selected:0,filtered:[],mode:'2d',layer:'elevation',trajectory:'0',view2d:{zoom:1,px:0,py:0},view3d:{az:-0.72,el:0.68,zoom:1},drag:null,cache:new Map(),replacementJob:null,selectedIds:new Set(),returning:false};
const $=id=>document.getElementById(id),canvas=$('canvas'),ctx=canvas.getContext('2d'),tooltip=$('tooltip');
const returnColor={ok:'#4fc38a',needs_return:'#ef7d65'};
function returnStatus(s){return s.needs_return?'needs_return':'ok'}
function returnLabel(s){return s.needs_return?'需要打回':'正常'}
function bytes(b64){const s=atob(b64),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return a}
function decodeScene(s){if(state.cache.has(s.id))return state.cache.get(s.id);const e8=bytes(s.elevation),e=new Uint16Array(e8.length/2);for(let i=0;i<e.length;i++)e[i]=e8[i*2]|(e8[i*2+1]<<8);const q8=bytes(s.slope),hb=bytes(s.obstacle_height||''),vb=bytes(s.valid),ob=bytes(s.observed),bb=bytes(s.obstacle||''),n=s.width*s.height,v=new Uint8Array(n),o=new Uint8Array(n),b=new Uint8Array(n),h=new Uint8Array(n);for(let i=0;i<n;i++){v[i]=(vb[i>>3]>>(7-(i&7)))&1;o[i]=(ob[i>>3]>>(7-(i&7)))&1;b[i]=(bb.length?(bb[i>>3]>>(7-(i&7))):0);h[i]=hb.length?hb[i]:0}const trajectories=(s.trajectories||[]).map(t=>({...t,values:new Float32Array(bytes(t.data).buffer)})),d={e,s:q8,v,o,b,h,trajectories};state.cache.set(s.id,d);return d}
function valueAt(scene,data,layer,i){if(layer==='elevation')return scene.elevation_min+(data.e[i]/65535)*(scene.elevation_max-scene.elevation_min);if(layer==='slope')return scene.slope_min+(data.s[i]/255)*(scene.slope_max-scene.slope_min);if(layer==='obstacle')return data.b[i];if(layer==='observed')return data.o[i];return data.v[i]}
function fmt(v,n=2){return Number.isFinite(v)?Number(v).toFixed(n):'—'}
function palette(t,layer,valid=true){if(!valid)return [38,41,44,255];t=Math.max(0,Math.min(1,t));if(layer==='obstacle')return t>.5?[241,73,63,255]:[37,48,55,255];if(layer==='valid'||layer==='observed')return t>.5?[68,185,128,255]:[218,78,70,255];let stops=layer==='elevation'?[[24,49,83],[36,127,146],[104,181,116],[231,190,92],[244,238,194]]:[[34,111,94],[111,184,116],[241,204,91],[222,92,71],[132,44,88]];let x=t*(stops.length-1),i=Math.min(stops.length-2,Math.floor(x)),f=x-i,a=stops[i],b=stops[i+1];return [Math.round(a[0]+(b[0]-a[0])*f),Math.round(a[1]+(b[1]-a[1])*f),Math.round(a[2]+(b[2]-a[2])*f),255]}
function current(){return SCENES[state.selected]}
function setupCanvas(){const dpr=Math.min(devicePixelRatio||1,2),r=canvas.getBoundingClientRect(),w=Math.max(1,r.width),h=Math.max(1,r.height);if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}ctx.setTransform(dpr,0,0,dpr,0,0);return {w,h,dpr}}
function layerRange(s){if(state.layer==='elevation')return [s.elevation_min,s.elevation_max,'m'];if(state.layer==='slope')return [s.slope_min,s.slope_max,'°'];if(state.layer==='obstacle')return [0,1,'占用'];return [0,1,'']}
function draw(){if(!SCENES.length){const {w,h}=setupCanvas();state.mapRect=null;state.overviewPanels=[];tooltip.style.display='none';ctx.fillStyle='#0d1114';ctx.fillRect(0,0,w,h);ctx.fillStyle='#9fb0ba';ctx.font='14px system-ui';ctx.textAlign='center';ctx.fillText('当前目录没有可查看的场景',w/2,h/2);ctx.textAlign='start';return}tooltip.style.display='none';if(state.mode==='2d')draw2d();else if(state.mode==='3d')draw3d();else drawOverview()}
function visibleTrajectories(d){if(!$('showTrajectories').checked||!d.trajectories.length)return[];if($('allTrajectories').checked)return d.trajectories.map((t,i)=>({t,i}));const i=Math.max(0,Math.min(d.trajectories.length-1,+state.trajectory||0));return [{t:d.trajectories[i],i}]}
function elevationAt(s,d,x,y){const half=s.width*s.resolution/2,c=Math.max(0,Math.min(s.width-1,Math.round((x+half)/s.resolution-.5))),r=Math.max(0,Math.min(s.height-1,Math.round((y+half)/s.resolution-.5)));return valueAt(s,d,'elevation',r*s.width+c)}
function pathColor(i,alpha=1){const hue=(i*47+195)%360;return `hsla(${hue},90%,65%,${alpha})`}
function drawTrajectories2d(s,d,x,y,size){const paths=visibleTrajectories(d),half=s.width*s.resolution/2,all=$('allTrajectories').checked;for(const {t,i} of paths){const a=t.values;if(a.length<6)continue;ctx.beginPath();for(let k=0;k<t.points;k++){const px=x+(a[k*3]+half)/(2*half)*size,py=y+(half-a[k*3+1])/(2*half)*size;if(k)ctx.lineTo(px,py);else ctx.moveTo(px,py)}ctx.strokeStyle=pathColor(i,all?.38:1);ctx.lineWidth=all?1.4:3;ctx.stroke();if(!all){const sx=x+(a[0]+half)/(2*half)*size,sy=y+(half-a[1])/(2*half)*size,k=(t.points-1)*3,ex=x+(a[k]+half)/(2*half)*size,ey=y+(half-a[k+1])/(2*half)*size;ctx.fillStyle='#4fe38a';ctx.beginPath();ctx.arc(sx,sy,5,0,Math.PI*2);ctx.fill();ctx.fillStyle='#ff675f';ctx.beginPath();ctx.arc(ex,ey,5,0,Math.PI*2);ctx.fill()}}}
function draw2d(){const {w,h}=setupCanvas(),s=current(),d=decodeScene(s),off=document.createElement('canvas');off.width=s.width;off.height=s.height;const oc=off.getContext('2d'),im=oc.createImageData(s.width,s.height),[lo,hi]=layerRange(s);for(let i=0;i<d.v.length;i++){const row=Math.floor(i/s.width),col=i%s.width,src=(s.height-1-row)*s.width+col,value=valueAt(s,d,state.layer,src),t=(value-lo)/(hi-lo||1),c=palette(t,state.layer,d.v[src]);im.data.set(c,i*4)}oc.putImageData(im,0,0);ctx.fillStyle='#0d1114';ctx.fillRect(0,0,w,h);const base=Math.max(20,Math.min(w,h)-72),size=base*state.view2d.zoom,x=(w-size)/2+state.view2d.px,y=(h-size)/2+state.view2d.py;state.mapRect={x,y,size};ctx.imageSmoothingEnabled=false;ctx.drawImage(off,x,y,size,size);ctx.strokeStyle='#72828c';ctx.strokeRect(x-.5,y-.5,size+1,size+1);if($('grid').checked){ctx.strokeStyle='#ffffff28';ctx.lineWidth=1;for(let i=0;i<=s.width;i++){const p=x+i/s.width*size;ctx.beginPath();ctx.moveTo(p,y);ctx.lineTo(p,y+size);ctx.stroke()}for(let i=0;i<=s.height;i++){const q=y+i/s.height*size;ctx.beginPath();ctx.moveTo(x,q);ctx.lineTo(x+size,q);ctx.stroke()}}drawTrajectories2d(s,d,x,y,size);drawLegend(w,h,s);ctx.fillStyle='#9fb0ba';ctx.font='12px system-ui';ctx.fillText(`-10 m · ${s.width}×${s.height}`,x,y+size+19);ctx.fillText('0',x+size/2-3,y+size+19);ctx.fillText('+10 m',x+size-31,y+size+19)}
function drawLegend(w,h,s){const [lo,hi,unit]=layerRange(s),x=w-42,y=28,gh=Math.min(220,h-80),steps=80;for(let i=0;i<steps;i++){const c=palette(1-i/(steps-1),state.layer,true);ctx.fillStyle=`rgb(${c[0]},${c[1]},${c[2]})`;ctx.fillRect(x,y+i*gh/steps,15,gh/steps+1)}ctx.strokeStyle='#91a0a9';ctx.strokeRect(x,y,15,gh);ctx.fillStyle='#bdc8cf';ctx.font='11px system-ui';ctx.fillText(`${fmt(hi,2)} ${unit}`,x-3,y-7);ctx.fillText(`${fmt(lo,2)} ${unit}`,x-3,y+gh+15)}
function project(x,y,z,w,h){const a=state.view3d.az,e=state.view3d.el,ca=Math.cos(a),sa=Math.sin(a),ce=Math.cos(e),se=Math.sin(e),x1=ca*x-sa*y,y1=sa*x+ca*y,up=ce*z+se*y1,depth=ce*y1-se*z,scale=Math.min(w,h)*.34*state.view3d.zoom;return [w*.5+x1*scale,h*.53-up*scale,depth]}
function drawTrajectories3d(s,d,w,h,zmid,exag){const half=s.width*s.resolution/2,all=$('allTrajectories').checked;for(const {t,i} of visibleTrajectories(d)){const a=t.values;if(a.length<6)continue;ctx.beginPath();let first,last;for(let k=0;k<t.points;k++){const x=a[k*3],y=a[k*3+1],z=(elevationAt(s,d,x,y)-zmid)/20*exag*2+.018,p=project(x/half,y/half,z,w,h);if(k)ctx.lineTo(p[0],p[1]);else{ctx.moveTo(p[0],p[1]);first=p}last=p}ctx.strokeStyle=pathColor(i,all?.42:1);ctx.lineWidth=all?1.2:3;ctx.stroke();if(!all){ctx.fillStyle='#4fe38a';ctx.beginPath();ctx.arc(first[0],first[1],5,0,Math.PI*2);ctx.fill();ctx.fillStyle='#ff675f';ctx.beginPath();ctx.arc(last[0],last[1],5,0,Math.PI*2);ctx.fill()}}}
function draw3d(){const {w,h}=setupCanvas(),s=current(),d=decodeScene(s),faces=[],zmid=(s.elevation_min+s.elevation_max)/2,exag=+$('exag').value,[lo,hi]=layerRange(s);ctx.fillStyle='#0d1114';ctx.fillRect(0,0,w,h);function pt(r,c){const i=r*s.width+c,x=s.width<=1?0:(c/(s.width-1)-.5)*2,y=s.height<=1?0:(r/(s.height-1)-.5)*2,z=(valueAt(s,d,'elevation',i)-zmid)/20*exag*2;return {p:project(x,y,z,w,h),i}}for(let r=0;r<s.height-1;r++)for(let c=0;c<s.width-1;c++){const ids=[r*s.width+c,r*s.width+c+1,(r+1)*s.width+c+1,(r+1)*s.width+c];if(ids.some(i=>!d.v[i]))continue;const q=[pt(r,c),pt(r,c+1),pt(r+1,c+1),pt(r+1,c)],val=ids.reduce((a,i)=>a+valueAt(s,d,state.layer,i),0)/4;faces.push({q,depth:q.reduce((a,v)=>a+v.p[2],0)/4,t:(val-lo)/(hi-lo||1)})}faces.sort((a,b)=>b.depth-a.depth);for(const f of faces){const color=palette(f.t,state.layer,true);ctx.beginPath();ctx.moveTo(f.q[0].p[0],f.q[0].p[1]);for(let i=1;i<4;i++)ctx.lineTo(f.q[i].p[0],f.q[i].p[1]);ctx.closePath();ctx.fillStyle=`rgb(${color[0]},${color[1]},${color[2]})`;ctx.fill();if($('grid').checked){ctx.strokeStyle='#07101555';ctx.lineWidth=.5;ctx.stroke()}}drawTrajectories3d(s,d,w,h,zmid,exag);ctx.fillStyle='#b5c0c7';ctx.font='12px system-ui';ctx.fillText(`20 m × 20 m · ${s.width}×${s.height} · 高差 ${fmt(s.elevation_max-s.elevation_min)} m · 垂直夸张 ${exag}×`,14,22);drawLegend(w,h,s)}
function filteredScenes(){return state.filtered.map(i=>SCENES[i])}
function overviewTransform(rect,items){const xs=items.map(({s})=>s.metadata.source_center_xy?.[0]).filter(Number.isFinite),ys=items.map(({s})=>s.metadata.source_center_xy?.[1]).filter(Number.isFinite);let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);if(xmax===xmin){xmin-=1;xmax+=1}if(ymax===ymin){ymin-=1;ymax+=1}const pad=34,sx=(rect.w-2*pad)/(xmax-xmin),sy=(rect.h-2*pad)/(ymax-ymin),scale=Math.min(sx,sy),cx=rect.x+rect.w/2,cy=rect.y+rect.h/2;return {xmin,xmax,ymin,ymax,rect,pad,scale,x:x=>cx+(x-(xmin+xmax)/2)*scale,y:y=>cy-(y-(ymin+ymax)/2)*scale}}
function drawOverview(){const {w,h}=setupCanvas(),groups=new Map();ctx.fillStyle='#0d1114';ctx.fillRect(0,0,w,h);for(const i of state.filtered){const s=SCENES[i],key=`${s.metadata.site_id||'unknown'}|${s.metadata.source_crs||'unknown CRS'}`;if(!groups.has(key))groups.set(key,[]);groups.get(key).push({s,i})}if(!groups.size)return;const entries=[...groups.entries()],cols=Math.ceil(Math.sqrt(entries.length)),rows=Math.ceil(entries.length/cols),gap=18,top=38,bottom=30,pw=(w-gap*(cols+1))/cols,ph=(h-top-bottom-gap*(rows-1))/rows;state.overviewPanels=[];entries.forEach(([key,items],index)=>{const col=index%cols,row=Math.floor(index/cols),rect={x:gap+col*(pw+gap),y:top+row*(ph+gap),w:pw,h:ph},tr=overviewTransform(rect,items),[site,crs]=key.split('|');state.overviewPanels.push({tr,items});ctx.fillStyle='#151b1f';ctx.fillRect(rect.x,rect.y,rect.w,rect.h);ctx.strokeStyle='#3b4850';ctx.strokeRect(rect.x+.5,rect.y+.5,rect.w-1,rect.h-1);ctx.fillStyle='#dce5ea';ctx.font='600 13px system-ui';ctx.fillText(`${site} · ${crs} · ${items.length} 个裁剪`,rect.x+8,rect.y-12);ctx.fillStyle='#82939d';ctx.font='11px system-ui';ctx.fillText(`X ${fmt(tr.xmin,0)}–${fmt(tr.xmax,0)} m`,rect.x+8,rect.y+17);ctx.fillText(`Y ${fmt(tr.ymin,0)}–${fmt(tr.ymax,0)} m`,rect.x+8,rect.y+32);for(const {s} of items){const p=s.metadata.source_center_xy;if(!p)continue;const x=tr.x(p[0]),y=tr.y(p[1]),yaw=(s.metadata.crop_yaw_deg||0)*Math.PI/180,colr=returnColor[returnStatus(s)]||'#aaa';ctx.strokeStyle=colr;ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+Math.cos(yaw)*8,y-Math.sin(yaw)*8);ctx.stroke();ctx.fillStyle=colr;ctx.beginPath();ctx.arc(x,y,s===current()?6:3.5,0,Math.PI*2);ctx.fill();if(s===current()){ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke()}}})}
function updateDomainOptions(){const select=$('domainFilter'),current=select.value||'all',domains=[...new Set(SCENES.map(s=>s.domain||s.metadata.domain).filter(Boolean))].sort();select.textContent='';select.add(new Option('全部地形','all'));for(const domain of domains)select.add(new Option(domain,domain));select.value=domains.includes(current)?current:'all'}
function updateFilters(){const text=$('search').value.trim().toLowerCase(),domain=$('domainFilter').value,boxes=[...$('returnFilters').querySelectorAll('input')],allowed=new Set(boxes.filter(x=>x.checked).map(x=>x.value));state.filtered=SCENES.map((s,i)=>[s,i]).filter(([s])=>(domain==='all'||(s.domain||s.metadata.domain)===domain)&&allowed.has(returnStatus(s))&&(!text||`${s.display_id||s.id} ${s.metadata.site_id||''} ${returnLabel(s)}`.toLowerCase().includes(text))).sort((a,b)=>{const mode=$('sort').value;if(mode==='returnFirst')return (b[0].needs_return?1:0)-(a[0].needs_return?1:0)|| (a[0].display_id||a[0].id).localeCompare(b[0].display_id||b[0].id);if(mode==='yaw')return (a[0].metadata.crop_yaw_deg||0)-(b[0].metadata.crop_yaw_deg||0);return (a[0].display_id||a[0].id).localeCompare(b[0].display_id||b[0].id)}).map(x=>x[1]);renderList();$('summary').textContent=`${state.filtered.length}/${SCENES.length} 个场景`;draw()}
function changeDomain(){updateFilters();if(state.filtered.length&&!state.filtered.includes(state.selected))select(state.filtered[0])}
function artifactLabel(s){return s.artifact==='canonical'?'canonical 中间地图':'最终轨迹数据'}
function trajectoryCount(s){return s.path_count??s.trajectories?.length??0}
function renderList(){const root=$('sceneList');root.textContent='';for(const i of state.filtered){const s=SCENES[i],el=document.createElement('div');el.className='scene-card'+(i===state.selected?' active':'');const box=document.createElement('input');box.type='checkbox';box.checked=state.selectedIds.has(s.id);box.addEventListener('click',e=>{e.stopPropagation();if(box.checked)state.selectedIds.add(s.id);else state.selectedIds.delete(s.id)});el.appendChild(box);const name=document.createElement('div');name.className='scene-name';name.textContent=s.display_id||s.id;el.appendChild(name);const badge=document.createElement('span');badge.className='badge '+returnStatus(s);badge.textContent=returnLabel(s);el.appendChild(badge);const sub=document.createElement('div');sub.className='scene-sub';sub.style.gridColumn='2 / -1';sub.textContent=`${artifactLabel(s)} · ${trajectoryCount(s)} 条轨迹`;el.appendChild(sub);el.addEventListener('click',()=>select(i));root.appendChild(el)}root.querySelector('.active')?.scrollIntoView({block:'nearest'})}
function renderTrajectoryControl(preserve=false){const select=$('trajectory');if(!SCENES.length){select.textContent='';select.add(new Option('暂无场景','0'));select.disabled=true;$('allTrajectories').checked=false;$('allTrajectories').disabled=true;$('prevTrajectory').disabled=true;$('nextTrajectory').disabled=true;state.trajectory='0';return}const s=current(),available=s.trajectories.length>0,oldIndex=preserve?(+state.trajectory||0):0,oldAll=preserve&&$('allTrajectories').checked;select.textContent='';$('allTrajectories').checked=oldAll;$('allTrajectories').disabled=!available;$('prevTrajectory').disabled=!available||oldAll;$('nextTrajectory').disabled=!available||oldAll;if(!available){select.add(new Option('暂无轨迹','0'));select.disabled=true;state.trajectory='0';return}for(let i=0;i<s.trajectories.length;i++)select.add(new Option(`轨迹 ${i+1}/${s.trajectories.length} · ${s.trajectories[i].name}`,String(i)));state.trajectory=String(Math.min(oldIndex,s.trajectories.length-1));select.value=state.trajectory;select.disabled=oldAll}
function stepTrajectory(delta){const s=current();if(!s.trajectories.length)return;$('allTrajectories').checked=false;const next=((+state.trajectory||0)+delta+s.trajectories.length)%s.trajectories.length;state.trajectory=String(next);$('trajectory').value=state.trajectory;draw()}
async function select(i,preserveTrajectory=false){state.selected=i;state.view2d={zoom:1,px:0,py:0};if(LIVE&&!SCENES[i].elevation){const id=SCENES[i].id,response=await fetch(`/api/scene?id=${encodeURIComponent(id)}`);if(!response.ok)throw new Error(await response.text());SCENES[i]=await response.json();state.cache.delete(id)}renderList();renderDetails();renderTrajectoryControl(preserveTrajectory);draw()}
function updateReplacementButton(){const button=$('replaceScene'),status=$('replaceStatus'),s=current();if(!button)return;if(!LIVE){button.disabled=true;status.textContent='静态查看器不可替换';return}const busy=state.returning;button.disabled=!s||busy;status.textContent=busy?'正在登记待补位…':s?(s.needs_return?'这张图出不了轨迹，请打回补位':'打回后到8767补充该空位'):'选择地图后可用'}
function renderDetails(){updateReplacementButton();const s=current(),m=s.metadata,q=s.quality_metrics;$('detailName').textContent=s.display_id||s.id;$('detailSource').textContent=`${s.domain||m.domain||''} · ${m.site_id||''} · ${s.relative_source}`;const cards=[['类型',artifactLabel(s)],['数据分割',s.split],['轨迹数',trajectoryCount(s)],['打回',returnLabel(s)],['最终有效率',`${fmt((q.valid_fraction??m.valid_fraction)*100,2)}%`],['直接支撑率',`${fmt((q.observed_fraction??m.observed_fraction)*100,2)}%`],['物理障碍格',m.obstacle_cells_in_patch??'—'],['高差',`${fmt(s.elevation_max-s.elevation_min,3)} m`],['坡度 P95',`${fmt(q.slope_p95_deg??m.slope_degrees?.p95,2)}°`]];$('stats').innerHTML=cards.map(([k,v])=>`<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');const rows=[['地形类别',s.domain||m.domain],['环境',m.environment_id],['母图站点',m.site_id],['中心 X/Y',m.source_center_xy?.map(x=>fmt(x,2)).join(', ')],['CRS',m.source_crs],['1 m 源支撑',`${fmt((m.source_support_coverage||0)*100,2)}%`],['原始点数',m.source_raw_points_in_patch],['障碍物点数',m.obstacle_points_in_patch],['拟合半径',`${m.processing?.fit_radius_m??'—'} m`],['缺口补全',m.processing?.gap_completion],['许可',m.license]];$('metadata').innerHTML=`<table>${rows.map(([k,v])=>`<tr><td>${k}</td><td>${v??'—'}</td></tr>`).join('')}</table><details><summary>完整元数据</summary><pre></pre></details>`;$('metadata').querySelector('pre').textContent=JSON.stringify({needs_return:Boolean(s.needs_return),quality:{grade:s.grade,score:s.score,metrics:q},metadata:m},null,2)}
function modeHint(){const h=$('hint');h.textContent=state.mode==='2d'?'滚轮缩放，拖拽平移，悬停读取数值':state.mode==='3d'?'拖拽旋转，滚轮缩放':'点为裁剪中心，短线为裁剪朝向；点击选择场景';$('exagLabel').style.display=state.mode==='3d'?'flex':'none'}
function eventPoint(e){const r=canvas.getBoundingClientRect();return {x:e.clientX-r.left,y:e.clientY-r.top}}
canvas.addEventListener('pointerdown',e=>{const p=eventPoint(e);state.drag={x:p.x,y:p.y,moved:false,base2:{...state.view2d},base3:{...state.view3d}};canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{const p=eventPoint(e);if(state.drag){const dx=p.x-state.drag.x,dy=p.y-state.drag.y;state.drag.moved|=Math.abs(dx)+Math.abs(dy)>3;if(state.mode==='2d'){state.view2d.px=state.drag.base2.px+dx;state.view2d.py=state.drag.base2.py+dy}else if(state.mode==='3d'){state.view3d.az=state.drag.base3.az+dx*.009;state.view3d.el=Math.max(.1,Math.min(1.45,state.drag.base3.el-dy*.007))}draw();return}if(state.mode==='2d')show2dTooltip(e,p);else if(state.mode==='overview')showOverviewTooltip(e,p)});
canvas.addEventListener('pointerup',e=>{const p=eventPoint(e),drag=state.drag;state.drag=null;if(state.mode==='overview'&&drag&&!drag.moved)selectNearestOverview(p)});canvas.addEventListener('pointerleave',()=>{tooltip.style.display='none';state.drag=null});
canvas.addEventListener('wheel',e=>{e.preventDefault();const factor=Math.exp(-e.deltaY*.001);if(state.mode==='2d')state.view2d.zoom=Math.max(.5,Math.min(12,state.view2d.zoom*factor));else if(state.mode==='3d')state.view3d.zoom=Math.max(.45,Math.min(3,state.view3d.zoom*factor));draw()},{passive:false});
function show2dTooltip(e,p){const r=state.mapRect;if(!r||p.x<r.x||p.y<r.y||p.x>=r.x+r.size||p.y>=r.y+r.size){tooltip.style.display='none';return}const s=current(),d=decodeScene(s),c=Math.min(s.width-1,Math.floor((p.x-r.x)/r.size*s.width)),screenRow=Math.min(s.height-1,Math.floor((p.y-r.y)/r.size*s.height)),row=s.height-1-screenRow,i=row*s.width+c,z=valueAt(s,d,'elevation',i),v=valueAt(s,d,state.layer,i),unit=layerRange(s)[2];tooltip.textContent=`x ${(c+.5)*s.resolution-10>=0?'+':''}${fmt((c+.5)*s.resolution-10,1)} m\ny ${(row+.5)*s.resolution-10>=0?'+':''}${fmt((row+.5)*s.resolution-10,1)} m\nz ${fmt(z,3)} m\n${state.layer} ${fmt(v,2)} ${unit}\nobstacle_height ${fmt(d.h[i]/255*(s.obstacle_height_max||0),2)} m\nobserved ${d.o[i]?1:0}\nvalid ${d.v[i]?1:0}`;tooltip.style.display='block';tooltip.style.left=`${e.clientX+14}px`;tooltip.style.top=`${e.clientY+14}px`}
function nearestOverview(p){if(!state.overviewPanels)return null;let best=null,dist=Infinity;for(const {tr,items} of state.overviewPanels){if(p.x<tr.rect.x||p.x>tr.rect.x+tr.rect.w||p.y<tr.rect.y||p.y>tr.rect.y+tr.rect.h)continue;for(const {s,i} of items){const q=s.metadata.source_center_xy;if(!q)continue;const d=Math.hypot(p.x-tr.x(q[0]),p.y-tr.y(q[1]));if(d<dist){dist=d;best={i,s,d}}}}return best&&best.d<15?best:null}
function showOverviewTooltip(e,p){const n=nearestOverview(p);if(!n){tooltip.style.display='none';return}tooltip.textContent=`${n.s.id}\n${returnLabel(n.s)} · ${trajectoryCount(n.s)} 条轨迹\nyaw ${fmt(n.s.metadata.crop_yaw_deg,1)}°`;tooltip.style.display='block';tooltip.style.left=`${e.clientX+14}px`;tooltip.style.top=`${e.clientY+14}px`}
function selectNearestOverview(p){const n=nearestOverview(p);if(n)select(n.i)}
document.querySelectorAll('.mode').forEach(b=>b.onclick=()=>{document.querySelectorAll('.mode').forEach(x=>x.classList.remove('active'));b.classList.add('active');state.mode=b.dataset.mode;modeHint();draw()});$('layer').onchange=e=>{state.layer=e.target.value;draw()};$('trajectory').onchange=e=>{state.trajectory=e.target.value;draw()};$('prevTrajectory').onclick=()=>stepTrajectory(-1);$('nextTrajectory').onclick=()=>stepTrajectory(1);$('allTrajectories').onchange=e=>{const lock=e.target.checked;$('trajectory').disabled=lock;$('prevTrajectory').disabled=lock;$('nextTrajectory').disabled=lock;draw()};$('showTrajectories').onchange=draw;$('exag').oninput=e=>{$('exagValue').textContent=`${e.target.value}×`;draw()};$('grid').onchange=draw;$('resetView').onclick=()=>{state.view2d={zoom:1,px:0,py:0};state.view3d={az:-.72,el:.68,zoom:1};$('exag').value=1;$('exagValue').textContent='1×';draw()};$('domainFilter').onchange=changeDomain;$('search').oninput=updateFilters;$('sort').onchange=updateFilters;$('returnFilters').onchange=updateFilters;$('prev').onclick=()=>{const p=state.filtered.indexOf(state.selected);if(p>0)select(state.filtered[p-1])};$('next').onclick=()=>{const p=state.filtered.indexOf(state.selected);if(p>=0&&p<state.filtered.length-1)select(state.filtered[p+1])};$('toggleDetails').onclick=()=>$('details').classList.toggle('hidden');window.addEventListener('resize',draw);window.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click();if(e.key==='[')stepTrajectory(-1);if(e.key===']')stepTrajectory(1)});
async function responseJson(response){const text=await response.text();let payload;try{payload=text?JSON.parse(text):{}}catch(error){throw new Error(text||`HTTP ${response.status}`)}if(!response.ok)throw new Error(payload.error||text||`HTTP ${response.status}`);return payload}
async function postJson(url,body){return responseJson(await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}))}
async function returnScenes(ids){if(!LIVE||state.returning||!ids.length)return;state.returning=true;updateReplacementButton();$('replaceStatus').textContent=`正在打回 ${ids.length} 张…`;try{const job=ids.length===1?await postJson('/api/replace-scene',{id:ids[0]}):await postJson('/api/replace-scenes',{ids});state.replacementJob=null;await refreshLive();$('replaceStatus').textContent=ids.length===1?'已打回，8767 已出现待补位':`已打回 ${job.count||ids.length} 张，请到 8767 补位`;}catch(error){$('replaceStatus').textContent=`打回失败：${error.message}`}finally{state.returning=false;updateReplacementButton()}}
async function replaceSelectedScene(){const s=current();if(!LIVE||!s||state.returning)return;if(!confirm(`确定打回 ${s.display_id||s.id} 吗？该地图编号会在 8767 显示为待补位。`))return;await returnScenes([s.id])}
function selectNeedsReturn(){for(const i of state.filtered){const s=SCENES[i];if(s.needs_return)state.selectedIds.add(s.id)}renderList()}
function clearReturnSelection(){state.selectedIds.clear();renderList()}
async function returnSelectedScenes(){const ids=[...state.selectedIds];if(!ids.length){$('replaceStatus').textContent='请先勾选要打回的图';return}if(!confirm(`确定打回选中的 ${ids.length} 张图吗？`))return;await returnScenes(ids);state.selectedIds.clear();renderList()}
if($('replaceScene'))$('replaceScene').addEventListener('click',()=>replaceSelectedScene());
if($('selectNeedsReturn'))$('selectNeedsReturn').addEventListener('click',selectNeedsReturn);
if($('clearReturnSelection'))$('clearReturnSelection').addEventListener('click',clearReturnSelection);
if($('returnSelected'))$('returnSelected').addEventListener('click',()=>returnSelectedScenes());
function updateDatasetRoot(){if(LIVE){$('datasetRoot').textContent=DATASET_ROOT||'未设置';$('datasetRoot').title=DATASET_ROOT||''}}
function clearViewer(){state.selected=0;state.filtered=[];state.cache.clear();state.mapRect=null;state.overviewPanels=[];state.replacementJob=null;renderList();renderTrajectoryControl();$('detailName').textContent='暂无场景';$('detailSource').textContent='';$('replaceStatus').textContent='选择地图后可用';$('replaceScene').disabled=true;$('stats').textContent='';$('metadata').textContent='';draw()}
const pickerState={path:'',parent:null};
async function browsePicker(path){const payload=await responseJson(await fetch('/api/browse?path='+encodeURIComponent(path||DATASET_ROOT||'.')));pickerState.path=payload.path;pickerState.parent=payload.parent;$('pickerPath').value=payload.path;$('pickerCurrent').textContent=`当前目录：${payload.path}`;$('pickerUp').disabled=!payload.parent;const host=$('pickerList');host.replaceChildren();for(const directory of payload.directories){const button=document.createElement('button');button.className='picker-item';button.textContent='目录 · '+directory.name;button.addEventListener('click',()=>browsePicker(directory.path).catch(error=>$('summary').textContent=`浏览目录失败：${error.message}`));host.appendChild(button)}if(!host.children.length){const empty=document.createElement('div');empty.className='picker-current';empty.textContent='当前目录没有子目录';host.appendChild(empty)}}
async function openDatasetPicker(){if(!LIVE)return;$('datasetPicker').hidden=false;await browsePicker(DATASET_ROOT||'.')}
function closeDatasetPicker(){$('datasetPicker').hidden=true}
async function loadViewerRoot(path,label){const target=String(path||'').trim();if(!target){$('summary').textContent=`未配置${label}目录`;return}$('summary').textContent=`正在加载${label}…`;try{const payload=await postJson('/api/load-root',{path:target});closeDatasetPicker();await refreshLive(payload);$('summary').textContent=SCENES.length?`${SCENES.length}/${SCENES.length} 个场景 · ${label}已加载`:'该目录还没有可查看的场景'}catch(error){$('summary').textContent=`加载${label}失败：${error.message}`}}
async function loadDatasetRoot(){const path=pickerState.path||$('pickerPath').value.trim();return loadViewerRoot(path,'数据目录')}
function setupLiveControls(){ $('datasetControls').hidden=!LIVE;if($('batchReturnRow'))$('batchReturnRow').hidden=!LIVE;if(!LIVE)return;updateDatasetRoot();$('chooseDatasetRoot').onclick=()=>openDatasetPicker().catch(error=>$('summary').textContent=`打开目录选择器失败：${error.message}`);$('reloadDataset').onclick=()=>refreshLive().catch(error=>$('summary').textContent=`重新读取失败：${error.message}`);const quickCanonical=$('quickCanonical'),quickFinal=$('quickFinal');quickCanonical.disabled=!QUICK_ROOTS.canonical;quickFinal.disabled=!QUICK_ROOTS.final;quickCanonical.title=QUICK_ROOTS.canonical||'8767 未提供中间地图目录';quickFinal.title=QUICK_ROOTS.final||'8767 未提供最终结果目录';quickCanonical.onclick=()=>loadViewerRoot(QUICK_ROOTS.canonical,'中间地图');quickFinal.onclick=()=>loadViewerRoot(QUICK_ROOTS.final,'最终结果');$('closeDatasetPicker').onclick=closeDatasetPicker;$('pickerUp').onclick=()=>{if(pickerState.parent)browsePicker(pickerState.parent).catch(error=>$('summary').textContent=`打开上一级失败：${error.message}`)};$('pickerGo').onclick=()=>browsePicker($('pickerPath').value.trim()).catch(error=>$('summary').textContent=`打开目录失败：${error.message}`);$('useDatasetDirectory').onclick=loadDatasetRoot;$('datasetPicker').onclick=event=>{if(event.target===$('datasetPicker'))closeDatasetPicker()}}
let liveRefreshRunning=false;
async function refreshLive(indexPayload=null){if(liveRefreshRunning)return;liveRefreshRunning=true;try{const selectedId=SCENES[state.selected]?.id,old=new Map(SCENES.map(s=>[s.id,s])),incoming=indexPayload||await responseJson(await fetch('/api/index')),incomingRoot=incoming.dataset_root||DATASET_ROOT,rootChanged=incomingRoot!==DATASET_ROOT;DATASET_ROOT=incomingRoot;updateDatasetRoot();if(rootChanged)state.cache.clear();SCENES=incoming.scenes.map(summary=>{const previous=old.get(summary.id);return !rootChanged&&previous?.elevation&&previous.live_token===summary.live_token?{...previous,...summary,trajectories:previous.trajectories}:summary});if(!SCENES.length){clearViewer();$('summary').textContent='还没有生成场景';return}updateDomainOptions();const selectedIndex=SCENES.findIndex(s=>s.id===selectedId);state.selected=selectedIndex>=0?selectedIndex:0;state.filtered=SCENES.map((_,i)=>i);await select(state.selected,Boolean(selectedId&&!rootChanged));updateFilters();$('summary').textContent=`${state.filtered.length}/${SCENES.length} 个场景 · 自动更新`}catch(error){$('summary').textContent=`更新失败：${error.message}`}finally{liveRefreshRunning=false}}
async function initialize(){setupLiveControls();if(LIVE){await refreshLive();setInterval(refreshLive,5000)}else if(SCENES.length){updateDomainOptions();state.filtered=SCENES.map((_,i)=>i);renderList();renderDetails();renderTrajectoryControl();modeHint();draw();$('summary').textContent=`${SCENES.length}/${SCENES.length} 个场景`}else {clearViewer();$('summary').textContent='没有找到场景'}const requestedMode=location.hash.slice(1);if(SCENES.length&&['2d','3d','overview'].includes(requestedMode))document.querySelector(`[data-mode="${requestedMode}"]`).click()}
initialize();
</script>
</body></html>'''


def main():
    args = parse_args()
    records = []
    skipped = []
    paths = collect_paths(args.inputs, args.recursive) if args.inputs else []
    trajectory_maps = []
    if args.trajectory_root:
        trajectory_root = args.trajectory_root.resolve()
        if not trajectory_root.is_dir():
            raise FileNotFoundError(
                f"Trajectory root does not exist: {trajectory_root}")
        trajectory_maps = sorted(
            path for path in trajectory_root.rglob("map.p")
            if path.parent.parent.name in {"train", "val"})
    sources = [("npz", path) for path in paths]
    sources.extend(("trajectory", path) for path in trajectory_maps)
    if not sources:
        raise RuntimeError("No scene NPZ or trajectory map.p files found")

    for index, (kind, path) in enumerate(sources, start=1):
        if args.max_scenes > 0 and len(records) >= args.max_scenes:
            break
        try:
            if kind == "npz":
                record = scene_record(path)
            else:
                record = trajectory_scene_record(
                    path, trajectory_root, args.max_paths_per_scene)
            records.append(record)
            print(
                f"[{index}/{len(sources)}] {record['id']}: "
                f"{len(record['trajectories'])} trajectories")
        except (OSError, ValueError, KeyError, EOFError,
                pickle.UnpicklingError) as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
            print(f"WARNING: skipped incomplete source {path}: {exc}")
    if not records:
        raise RuntimeError("No complete terrain scenes could be loaded")
    encoded = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("</", "<\\/")
    html = (HTML_TEMPLATE.replace("__TITLE__", args.title)
            .replace("__SCENES__", encoded)
            .replace("__LIVE__", "false")
            .replace("__DATASET_ROOT__", json.dumps(""))
            .replace("__QUICK_ROOTS__", "{}"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "scenes": len(records),
        "skipped": len(skipped),
        "skipped_scenes": skipped,
        "bytes": args.output.stat().st_size,
        "standalone": True,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
