#!/usr/bin/env python3
"""Build a standalone interactive HTML viewer for sampled 20 m terrain scenes."""

import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np

from terrain_map_quality import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Scene NPZ files or directories")
    parser.add_argument("--output", required=True, type=Path, help="Output HTML file")
    parser.add_argument("--title", default="20 m Terrain Dataset Viewer")
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search input directories recursively instead of only their top level")
    parser.add_argument("--max-scenes", type=int, default=0, help="0 keeps every scene")
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
    quality = evaluate(str(path))
    with np.load(path) as data:
        elevation = data["elevation"].astype(np.float64)
        valid = data["valid_mask"].astype(bool)
        normals = np.stack(
            (data["normal_x"], data["normal_y"], data["normal_z"]), axis=-1
        ).astype(np.float64)
        normal_length = np.linalg.norm(normals, axis=-1)
        slope = np.degrees(np.arccos(np.clip(
            normals[:, :, 2] / np.maximum(normal_length, 1e-12), -1.0, 1.0)))
        rmse = data["fit_rmse"].astype(np.float64)
        height, width = elevation.shape

    elevation_b64, elevation_min, elevation_max = quantize(
        elevation, valid, 16)
    slope_max = max(30.0, min(90.0, math.ceil(float(np.max(slope[valid])) / 5.0) * 5.0))
    slope_b64, slope_min, slope_max = quantize(
        slope, valid, 8, minimum=0.0, maximum=slope_max)
    rmse_max = max(0.01, math.ceil(float(np.nanmax(rmse)) * 1000.0) / 1000.0)
    rmse_b64, rmse_min, rmse_max = quantize(
        np.nan_to_num(rmse, nan=rmse_max), np.ones_like(valid), 8,
        minimum=0.0, maximum=rmse_max)
    valid_b64 = encode_bytes(np.packbits(valid.ravel(), bitorder="big"))

    return {
        "id": path.stem,
        "relative_source": str(path),
        "width": int(width),
        "height": int(height),
        "resolution": float(metadata.get("resolution_m", 0.2)),
        "elevation": elevation_b64,
        "elevation_min": elevation_min,
        "elevation_max": elevation_max,
        "slope": slope_b64,
        "slope_min": slope_min,
        "slope_max": slope_max,
        "rmse": rmse_b64,
        "rmse_min": rmse_min,
        "rmse_max": rmse_max,
        "valid": valid_b64,
        "grade": quality.get("grade") or "reject",
        "score": quality.get("geometry_score"),
        "quality": quality.get("quality"),
        "quality_metrics": quality.get("metrics", {}),
        "metadata": metadata,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#101417;--panel:#171d21;--panel2:#20272c;--line:#303a40;--text:#e8eef2;--muted:#99a8b2;--accent:#56b4d3;--easy:#4fc38a;--medium:#e7b75c;--hard:#ef7d65;--reject:#c66be0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;overflow:hidden}
button,input,select{font:inherit;color:inherit}button,select,input[type=search]{background:var(--panel2);border:1px solid var(--line);border-radius:7px;padding:7px 10px}button{cursor:pointer}button:hover,.mode.active{border-color:var(--accent);background:#24333a}.app{height:100vh;display:grid;grid-template-columns:310px minmax(0,1fr) 330px;grid-template-rows:auto 1fr}
header{grid-column:1/-1;display:flex;align-items:center;gap:16px;padding:11px 16px;background:var(--panel);border-bottom:1px solid var(--line)}header h1{font-size:17px;margin:0;white-space:nowrap}.summary{color:var(--muted)}.header-actions{margin-left:auto;display:flex;gap:7px}
.sidebar,.details{min-height:0;background:var(--panel);display:flex;flex-direction:column}.sidebar{border-right:1px solid var(--line)}.details{border-left:1px solid var(--line);overflow:auto}.filters{padding:12px;border-bottom:1px solid var(--line);display:grid;gap:9px}.filters input[type=search]{width:100%}.filter-row{display:flex;gap:11px;flex-wrap:wrap;color:var(--muted)}.filter-row label{display:flex;gap:4px;align-items:center}.scene-list{overflow:auto;padding:7px}.scene-card{display:grid;grid-template-columns:1fr auto;gap:3px 8px;padding:9px 10px;margin:4px 0;border:1px solid transparent;border-radius:8px;cursor:pointer}.scene-card:hover{background:var(--panel2)}.scene-card.active{border-color:var(--accent);background:#1d2b31}.scene-name{font-weight:650}.scene-sub{font-size:12px;color:var(--muted)}.badge{align-self:start;border-radius:999px;padding:2px 7px;font-size:11px;font-weight:700;color:#102018}.badge.easy{background:var(--easy)}.badge.medium{background:var(--medium)}.badge.hard{background:var(--hard)}.badge.reject{background:var(--reject)}
main{min-width:0;min-height:0;display:grid;grid-template-rows:auto 1fr;background:#0d1114}.toolbar{display:flex;align-items:center;gap:8px;padding:9px 12px;border-bottom:1px solid var(--line);flex-wrap:wrap}.toolbar .spacer{flex:1}.toolbar label{display:flex;align-items:center;gap:7px;color:var(--muted)}input[type=range]{accent-color:var(--accent)}.canvas-wrap{position:relative;min-height:0}canvas{width:100%;height:100%;display:block;touch-action:none}.hint{position:absolute;left:12px;bottom:10px;background:#101417d9;border:1px solid var(--line);color:var(--muted);padding:5px 8px;border-radius:6px;pointer-events:none}.tooltip{position:fixed;z-index:10;display:none;pointer-events:none;background:#0b0f12ed;border:1px solid #51616b;border-radius:7px;padding:7px 9px;white-space:pre;font-size:12px;box-shadow:0 5px 18px #0008}
.detail-head{padding:14px;border-bottom:1px solid var(--line)}.detail-head h2{font-size:18px;margin:0 0 4px}.detail-head .source{color:var(--muted);font-size:12px;overflow-wrap:anywhere}.stats{padding:12px;display:grid;grid-template-columns:1fr 1fr;gap:8px}.stat{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:9px}.stat .k{font-size:11px;color:var(--muted)}.stat .v{font-size:17px;font-weight:650;margin-top:2px}.metadata{padding:0 12px 16px}.metadata table{width:100%;border-collapse:collapse}.metadata td{padding:6px 3px;border-bottom:1px solid var(--line);vertical-align:top}.metadata td:first-child{color:var(--muted);width:43%}.metadata pre{font:11px/1.4 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#0d1114;padding:9px;border-radius:7px;max-height:320px;overflow:auto}details summary{cursor:pointer;color:var(--accent);padding:10px 0}
@media(max-width:1050px){.app{grid-template-columns:260px minmax(0,1fr)}.details{position:absolute;right:0;top:53px;bottom:0;width:330px;z-index:5;box-shadow:-8px 0 22px #0008}.details.hidden{display:none}}
</style>
</head>
<body>
<div class="app">
<header><h1>__TITLE__</h1><span class="summary" id="summary"></span><div class="header-actions"><button id="prev">← 上一张</button><button id="next">下一张 →</button><button id="toggleDetails">参数</button></div></header>
<aside class="sidebar">
  <div class="filters"><input id="search" type="search" placeholder="搜索场景或站点">
    <div class="filter-row" id="gradeFilters"><label><input type="checkbox" value="easy" checked>easy</label><label><input type="checkbox" value="medium" checked>medium</label><label><input type="checkbox" value="hard" checked>hard</label><label><input type="checkbox" value="reject" checked>reject</label></div>
    <select id="sort"><option value="name">按编号</option><option value="scoreAsc">难度从低到高</option><option value="scoreDesc">难度从高到低</option><option value="yaw">按裁剪角度</option></select>
  </div><div class="scene-list" id="sceneList"></div>
</aside>
<main>
  <div class="toolbar">
    <button class="mode active" data-mode="2d">2D 地图</button><button class="mode" data-mode="3d">3D 地形</button><button class="mode" data-mode="overview">母图分布</button>
    <select id="layer"><option value="elevation">高程</option><option value="slope">坡度</option><option value="rmse">拟合 RMSE</option><option value="valid">有效区域</option></select>
    <label id="exagLabel">垂直夸张 <input id="exag" type="range" min="1" max="30" value="12"><span id="exagValue">12×</span></label>
    <label><input id="grid" type="checkbox">网格</label><span class="spacer"></span><button id="resetView">重置视角</button>
  </div>
  <div class="canvas-wrap"><canvas id="canvas"></canvas><div class="hint" id="hint">滚轮缩放，拖拽平移，悬停读取数值</div></div>
</main>
<aside class="details" id="details"><div class="detail-head"><h2 id="detailName"></h2><div class="source" id="detailSource"></div></div><div class="stats" id="stats"></div><div class="metadata" id="metadata"></div></aside>
</div><div class="tooltip" id="tooltip"></div>
<script>
const SCENES=__SCENES__;
const state={selected:0,filtered:[],mode:'2d',layer:'elevation',view2d:{zoom:1,px:0,py:0},view3d:{az:-0.72,el:0.68,zoom:1},drag:null,cache:new Map()};
const $=id=>document.getElementById(id),canvas=$('canvas'),ctx=canvas.getContext('2d'),tooltip=$('tooltip');
const gradeColor={easy:'#4fc38a',medium:'#e7b75c',hard:'#ef7d65',reject:'#c66be0'};
function bytes(b64){const s=atob(b64),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return a}
function decodeScene(s){if(state.cache.has(s.id))return state.cache.get(s.id);const e8=bytes(s.elevation),e=new Uint16Array(e8.length/2);for(let i=0;i<e.length;i++)e[i]=e8[i*2]|(e8[i*2+1]<<8);const q8=bytes(s.slope),r8=bytes(s.rmse),vb=bytes(s.valid),n=s.width*s.height,v=new Uint8Array(n);for(let i=0;i<n;i++)v[i]=(vb[i>>3]>>(7-(i&7)))&1;const d={e,s:q8,r:r8,v};state.cache.set(s.id,d);return d}
function valueAt(scene,data,layer,i){if(layer==='elevation')return scene.elevation_min+(data.e[i]/65535)*(scene.elevation_max-scene.elevation_min);if(layer==='slope')return scene.slope_min+(data.s[i]/255)*(scene.slope_max-scene.slope_min);if(layer==='rmse')return scene.rmse_min+(data.r[i]/255)*(scene.rmse_max-scene.rmse_min);return data.v[i]}
function fmt(v,n=2){return Number.isFinite(v)?Number(v).toFixed(n):'—'}
function palette(t,layer,valid=true){if(!valid)return [38,41,44,255];t=Math.max(0,Math.min(1,t));if(layer==='valid')return t>.5?[68,185,128,255]:[218,78,70,255];let stops=layer==='elevation'?[[24,49,83],[36,127,146],[104,181,116],[231,190,92],[244,238,194]]:layer==='slope'?[[34,111,94],[111,184,116],[241,204,91],[222,92,71],[132,44,88]]:[[31,85,112],[56,151,142],[120,190,116],[232,193,83],[193,69,76]];let x=t*(stops.length-1),i=Math.min(stops.length-2,Math.floor(x)),f=x-i,a=stops[i],b=stops[i+1];return [Math.round(a[0]+(b[0]-a[0])*f),Math.round(a[1]+(b[1]-a[1])*f),Math.round(a[2]+(b[2]-a[2])*f),255]}
function current(){return SCENES[state.selected]}
function setupCanvas(){const dpr=Math.min(devicePixelRatio||1,2),r=canvas.getBoundingClientRect(),w=Math.max(1,r.width),h=Math.max(1,r.height);if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}ctx.setTransform(dpr,0,0,dpr,0,0);return {w,h,dpr}}
function layerRange(s){if(state.layer==='elevation')return [s.elevation_min,s.elevation_max,'m'];if(state.layer==='slope')return [s.slope_min,s.slope_max,'°'];if(state.layer==='rmse')return [s.rmse_min,s.rmse_max,'m'];return [0,1,'']}
function draw(){if(!SCENES.length)return;tooltip.style.display='none';if(state.mode==='2d')draw2d();else if(state.mode==='3d')draw3d();else drawOverview()}
function draw2d(){const {w,h}=setupCanvas(),s=current(),d=decodeScene(s),off=document.createElement('canvas');off.width=s.width;off.height=s.height;const oc=off.getContext('2d'),im=oc.createImageData(s.width,s.height),[lo,hi]=layerRange(s);for(let i=0;i<d.v.length;i++){const value=valueAt(s,d,state.layer,i),t=(value-lo)/(hi-lo||1),c=palette(t,state.layer,d.v[i]);im.data.set(c,i*4)}oc.putImageData(im,0,0);ctx.fillStyle='#0d1114';ctx.fillRect(0,0,w,h);const base=Math.max(20,Math.min(w,h)-72),size=base*state.view2d.zoom,x=(w-size)/2+state.view2d.px,y=(h-size)/2+state.view2d.py;state.mapRect={x,y,size};ctx.imageSmoothingEnabled=false;ctx.drawImage(off,x,y,size,size);ctx.strokeStyle='#72828c';ctx.strokeRect(x-.5,y-.5,size+1,size+1);if($('grid').checked&&state.view2d.zoom>=2){ctx.strokeStyle='#ffffff18';ctx.lineWidth=1;for(let i=0;i<=s.width;i+=10){const p=x+i/s.width*size;ctx.beginPath();ctx.moveTo(p,y);ctx.lineTo(p,y+size);ctx.stroke();const q=y+i/s.height*size;ctx.beginPath();ctx.moveTo(x,q);ctx.lineTo(x+size,q);ctx.stroke()}}drawLegend(w,h,s);ctx.fillStyle='#9fb0ba';ctx.font='12px system-ui';ctx.fillText('-10 m',x,y+size+19);ctx.fillText('0',x+size/2-3,y+size+19);ctx.fillText('+10 m',x+size-31,y+size+19)}
function drawLegend(w,h,s){const [lo,hi,unit]=layerRange(s),x=w-42,y=28,gh=Math.min(220,h-80),steps=80;for(let i=0;i<steps;i++){const c=palette(1-i/(steps-1),state.layer,true);ctx.fillStyle=`rgb(${c[0]},${c[1]},${c[2]})`;ctx.fillRect(x,y+i*gh/steps,15,gh/steps+1)}ctx.strokeStyle='#91a0a9';ctx.strokeRect(x,y,15,gh);ctx.fillStyle='#bdc8cf';ctx.font='11px system-ui';ctx.fillText(`${fmt(hi,state.layer==='rmse'?3:2)} ${unit}`,x-3,y-7);ctx.fillText(`${fmt(lo,state.layer==='rmse'?3:2)} ${unit}`,x-3,y+gh+15)}
function project(x,y,z,w,h){const a=state.view3d.az,e=state.view3d.el,ca=Math.cos(a),sa=Math.sin(a),ce=Math.cos(e),se=Math.sin(e),x1=ca*x-sa*y,y1=sa*x+ca*y,up=ce*z-se*y1,depth=ce*y1+se*z,scale=Math.min(w,h)*.34*state.view3d.zoom;return [w*.5+x1*scale,h*.53-up*scale,depth]}
function draw3d(){const {w,h}=setupCanvas(),s=current(),d=decodeScene(s),faces=[],step=2,zmid=(s.elevation_min+s.elevation_max)/2,exag=+$('exag').value,[lo,hi]=layerRange(s);ctx.fillStyle='#0d1114';ctx.fillRect(0,0,w,h);function pt(r,c){const i=r*s.width+c,x=(c/(s.width-1)-.5)*2,y=(r/(s.height-1)-.5)*2,z=(valueAt(s,d,'elevation',i)-zmid)/20*exag*2;return {p:project(x,y,z,w,h),i}}for(let r=0;r<s.height-1;r+=step)for(let c=0;c<s.width-1;c+=step){const r2=Math.min(s.height-1,r+step),c2=Math.min(s.width-1,c+step),ids=[r*s.width+c,r*s.width+c2,r2*s.width+c2,r2*s.width+c];if(ids.some(i=>!d.v[i]))continue;const q=[pt(r,c),pt(r,c2),pt(r2,c2),pt(r2,c)],val=ids.reduce((a,i)=>a+valueAt(s,d,state.layer,i),0)/4;faces.push({q,depth:q.reduce((a,v)=>a+v.p[2],0)/4,t:(val-lo)/(hi-lo||1)})}faces.sort((a,b)=>b.depth-a.depth);for(const f of faces){const c=palette(f.t,state.layer,true);ctx.beginPath();ctx.moveTo(f.q[0].p[0],f.q[0].p[1]);for(let i=1;i<4;i++)ctx.lineTo(f.q[i].p[0],f.q[i].p[1]);ctx.closePath();ctx.fillStyle=`rgb(${c[0]},${c[1]},${c[2]})`;ctx.fill();if($('grid').checked){ctx.strokeStyle='#07101555';ctx.lineWidth=.5;ctx.stroke()}}ctx.fillStyle='#b5c0c7';ctx.font='12px system-ui';ctx.fillText(`20 m × 20 m · 高差 ${fmt(s.elevation_max-s.elevation_min)} m · 垂直夸张 ${exag}×`,14,22);drawLegend(w,h,s)}
function filteredScenes(){return state.filtered.map(i=>SCENES[i])}
function overviewTransform(w,h,scenes){const xs=scenes.map(s=>s.metadata.source_center_xy?.[0]).filter(Number.isFinite),ys=scenes.map(s=>s.metadata.source_center_xy?.[1]).filter(Number.isFinite);let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);if(xmax===xmin){xmin-=1;xmax+=1}if(ymax===ymin){ymin-=1;ymax+=1}const pad=55,sx=(w-2*pad)/(xmax-xmin),sy=(h-2*pad)/(ymax-ymin),scale=Math.min(sx,sy);return {xmin,xmax,ymin,ymax,pad,scale,x:x=>w/2+(x-(xmin+xmax)/2)*scale,y:y=>h/2-(y-(ymin+ymax)/2)*scale}}
function drawOverview(){const {w,h}=setupCanvas(),scenes=filteredScenes();ctx.fillStyle='#0d1114';ctx.fillRect(0,0,w,h);if(!scenes.length)return;const tr=overviewTransform(w,h,scenes);state.overviewTransform=tr;ctx.strokeStyle='#344148';ctx.strokeRect(tr.x(tr.xmin),tr.y(tr.ymax),tr.x(tr.xmax)-tr.x(tr.xmin),tr.y(tr.ymin)-tr.y(tr.ymax));for(const s of scenes){const p=s.metadata.source_center_xy;if(!p)continue;const x=tr.x(p[0]),y=tr.y(p[1]),yaw=(s.metadata.crop_yaw_deg||0)*Math.PI/180,col=gradeColor[s.grade]||'#aaa';ctx.strokeStyle=col;ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+Math.cos(yaw)*8,y-Math.sin(yaw)*8);ctx.stroke();ctx.fillStyle=col;ctx.beginPath();ctx.arc(x,y,s===current()?6:3.5,0,Math.PI*2);ctx.fill();if(s===current()){ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke()}}ctx.fillStyle='#aab7bf';ctx.font='12px system-ui';ctx.fillText(`X ${fmt(tr.xmin,0)}–${fmt(tr.xmax,0)} m`,tr.pad,22);ctx.fillText(`Y ${fmt(tr.ymin,0)}–${fmt(tr.ymax,0)} m`,tr.pad,39);ctx.fillText('点为裁剪中心，短线为裁剪朝向；点击选择场景',tr.pad,h-18)}
function updateFilters(){const text=$('search').value.trim().toLowerCase(),grades=new Set([...$('gradeFilters').querySelectorAll('input:checked')].map(x=>x.value));state.filtered=SCENES.map((s,i)=>[s,i]).filter(([s])=>grades.has(s.grade)&&(!text||`${s.id} ${s.metadata.site_id||''} ${s.grade}`.toLowerCase().includes(text))).sort((a,b)=>{const mode=$('sort').value;if(mode==='scoreAsc')return a[0].score-b[0].score;if(mode==='scoreDesc')return b[0].score-a[0].score;if(mode==='yaw')return (a[0].metadata.crop_yaw_deg||0)-(b[0].metadata.crop_yaw_deg||0);return a[0].id.localeCompare(b[0].id)}).map(x=>x[1]);renderList();$('summary').textContent=`${state.filtered.length}/${SCENES.length} 个场景`;draw()}
function renderList(){const root=$('sceneList');root.textContent='';for(const i of state.filtered){const s=SCENES[i],el=document.createElement('div');el.className='scene-card'+(i===state.selected?' active':'');el.innerHTML=`<div class="scene-name">${s.id}</div><span class="badge ${s.grade}">${s.grade}</span><div class="scene-sub">score ${fmt(s.score,1)} · yaw ${fmt(s.metadata.crop_yaw_deg,1)}°</div>`;el.onclick=()=>select(i);root.appendChild(el)}root.querySelector('.active')?.scrollIntoView({block:'nearest'})}
function select(i){state.selected=i;state.view2d={zoom:1,px:0,py:0};renderList();renderDetails();draw()}
function renderDetails(){const s=current(),m=s.metadata,q=s.quality_metrics;$('detailName').textContent=s.id;$('detailSource').textContent=`${m.domain||''} · ${m.site_id||''} · ${s.relative_source}`;const cards=[['等级',s.grade],['几何分数',fmt(s.score,1)],['有效率',`${fmt((q.valid_fraction??m.valid_fraction)*100,2)}%`],['高差',`${fmt(s.elevation_max-s.elevation_min,3)} m`],['坡度 P95',`${fmt(q.slope_p95_deg??m.slope_degrees?.p95,2)}°`],['RMSE P95',`${fmt(q.fit_rmse_p95_m??m.fit_rmse_m?.p95,3)} m`],['源点密度',`${fmt(q.source_surface_density_points_per_m2??m.source_surface_density_points_per_m2,2)}/m²`],['裁剪角度',`${fmt(m.crop_yaw_deg,1)}°`]];$('stats').innerHTML=cards.map(([k,v])=>`<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');const rows=[['母图站点',m.site_id],['中心 X/Y',m.source_center_xy?.map(x=>fmt(x,2)).join(', ')],['CRS',m.source_crs],['有效支撑',`${fmt((m.source_support_coverage||0)*100,2)}%`],['拟合半径',`${m.processing?.fit_radius_m??'—'} m`],['最少邻点',m.processing?.min_neighbors],['许可',m.license]];$('metadata').innerHTML=`<table>${rows.map(([k,v])=>`<tr><td>${k}</td><td>${v??'—'}</td></tr>`).join('')}</table><details><summary>完整元数据</summary><pre></pre></details>`;$('metadata').querySelector('pre').textContent=JSON.stringify({quality:{grade:s.grade,score:s.score,metrics:q},metadata:m},null,2)}
function modeHint(){const h=$('hint');h.textContent=state.mode==='2d'?'滚轮缩放，拖拽平移，悬停读取数值':state.mode==='3d'?'拖拽旋转，滚轮缩放':'点为裁剪中心，短线为裁剪朝向；点击选择场景';$('exagLabel').style.display=state.mode==='3d'?'flex':'none'}
function eventPoint(e){const r=canvas.getBoundingClientRect();return {x:e.clientX-r.left,y:e.clientY-r.top}}
canvas.addEventListener('pointerdown',e=>{const p=eventPoint(e);state.drag={x:p.x,y:p.y,moved:false,base2:{...state.view2d},base3:{...state.view3d}};canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{const p=eventPoint(e);if(state.drag){const dx=p.x-state.drag.x,dy=p.y-state.drag.y;state.drag.moved|=Math.abs(dx)+Math.abs(dy)>3;if(state.mode==='2d'){state.view2d.px=state.drag.base2.px+dx;state.view2d.py=state.drag.base2.py+dy}else if(state.mode==='3d'){state.view3d.az=state.drag.base3.az+dx*.009;state.view3d.el=Math.max(.1,Math.min(1.45,state.drag.base3.el-dy*.007))}draw();return}if(state.mode==='2d')show2dTooltip(e,p);else if(state.mode==='overview')showOverviewTooltip(e,p)});
canvas.addEventListener('pointerup',e=>{const p=eventPoint(e),drag=state.drag;state.drag=null;if(state.mode==='overview'&&drag&&!drag.moved)selectNearestOverview(p)});canvas.addEventListener('pointerleave',()=>{tooltip.style.display='none';state.drag=null});
canvas.addEventListener('wheel',e=>{e.preventDefault();const factor=Math.exp(-e.deltaY*.001);if(state.mode==='2d')state.view2d.zoom=Math.max(.5,Math.min(12,state.view2d.zoom*factor));else if(state.mode==='3d')state.view3d.zoom=Math.max(.45,Math.min(3,state.view3d.zoom*factor));draw()},{passive:false});
function show2dTooltip(e,p){const r=state.mapRect;if(!r||p.x<r.x||p.y<r.y||p.x>=r.x+r.size||p.y>=r.y+r.size){tooltip.style.display='none';return}const s=current(),d=decodeScene(s),c=Math.min(s.width-1,Math.floor((p.x-r.x)/r.size*s.width)),row=Math.min(s.height-1,Math.floor((p.y-r.y)/r.size*s.height)),i=row*s.width+c,z=valueAt(s,d,'elevation',i),v=valueAt(s,d,state.layer,i),unit=layerRange(s)[2];tooltip.textContent=`x ${(c+.5)*s.resolution-10>=0?'+':''}${fmt((c+.5)*s.resolution-10,1)} m\ny ${10-(row+.5)*s.resolution>=0?'+':''}${fmt(10-(row+.5)*s.resolution,1)} m\nz ${fmt(z,3)} m\n${state.layer} ${fmt(v,state.layer==='rmse'?4:2)} ${unit}\nvalid ${d.v[i]?1:0}`;tooltip.style.display='block';tooltip.style.left=`${e.clientX+14}px`;tooltip.style.top=`${e.clientY+14}px`}
function nearestOverview(p){const tr=state.overviewTransform;if(!tr)return null;let best=null,dist=Infinity;for(const i of state.filtered){const s=SCENES[i],q=s.metadata.source_center_xy;if(!q)continue;const d=Math.hypot(p.x-tr.x(q[0]),p.y-tr.y(q[1]));if(d<dist){dist=d;best={i,s,d}}}return best&&best.d<15?best:null}
function showOverviewTooltip(e,p){const n=nearestOverview(p);if(!n){tooltip.style.display='none';return}tooltip.textContent=`${n.s.id}\n${n.s.grade} · score ${fmt(n.s.score,1)}\nyaw ${fmt(n.s.metadata.crop_yaw_deg,1)}°`;tooltip.style.display='block';tooltip.style.left=`${e.clientX+14}px`;tooltip.style.top=`${e.clientY+14}px`}
function selectNearestOverview(p){const n=nearestOverview(p);if(n)select(n.i)}
document.querySelectorAll('.mode').forEach(b=>b.onclick=()=>{document.querySelectorAll('.mode').forEach(x=>x.classList.remove('active'));b.classList.add('active');state.mode=b.dataset.mode;modeHint();draw()});$('layer').onchange=e=>{state.layer=e.target.value;draw()};$('exag').oninput=e=>{$('exagValue').textContent=`${e.target.value}×`;draw()};$('grid').onchange=draw;$('resetView').onclick=()=>{state.view2d={zoom:1,px:0,py:0};state.view3d={az:-.72,el:.68,zoom:1};draw()};$('search').oninput=updateFilters;$('sort').onchange=updateFilters;$('gradeFilters').onchange=updateFilters;$('prev').onclick=()=>{const p=state.filtered.indexOf(state.selected);if(p>0)select(state.filtered[p-1])};$('next').onclick=()=>{const p=state.filtered.indexOf(state.selected);if(p>=0&&p<state.filtered.length-1)select(state.filtered[p+1])};$('toggleDetails').onclick=()=>$('details').classList.toggle('hidden');window.addEventListener('resize',draw);window.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click()});
if(SCENES.length){state.filtered=SCENES.map((_,i)=>i);renderList();renderDetails();modeHint();draw();$('summary').textContent=`${SCENES.length}/${SCENES.length} 个场景`}else $('summary').textContent='没有找到场景';
</script>
</body></html>'''


def main():
    args = parse_args()
    paths = collect_paths(args.inputs, args.recursive)
    if not paths:
        raise RuntimeError("No scene_*.npz files found")
    records = []
    skipped = []
    for index, path in enumerate(paths, start=1):
        if args.max_scenes > 0 and len(records) >= args.max_scenes:
            break
        try:
            records.append(scene_record(path))
            print(f"[{index}/{len(paths)}] {path.name}")
        except (OSError, ValueError, KeyError, EOFError) as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
            print(f"WARNING: skipped incomplete scene {path.name}: {exc}")
    if not records:
        raise RuntimeError("No complete terrain scenes could be loaded")
    encoded = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__TITLE__", args.title).replace("__SCENES__", encoded)
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
