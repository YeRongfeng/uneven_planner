#!/usr/bin/env python3
"""Serve the interactive terrain viewer against any live dataset directory."""

import argparse
import json
import pickle
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from make_terrain_dataset_viewer import (
    HTML_TEMPLATE,
    scene_record,
    trajectory_scene_record,
)
from terrain_map_quality import evaluate
from review_slots import append_return_record


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]


def resolve_workspace_path(value, root=WORKSPACE_ROOT):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.absolute()


def browse_directory(value, root=WORKSPACE_ROOT):
    requested = resolve_workspace_path(value or root, root)
    path = requested
    while not path.exists() and path != path.parent:
        path = path.parent
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        raise FileNotFoundError(path)
    directories = []
    files = []
    for child in sorted(path.iterdir(), key=lambda item: item.name.casefold()):
        if child.is_dir():
            directories.append({"name": child.name, "path": str(child)})
        elif child.is_file():
            files.append({"name": child.name, "path": str(child)})
    return {
        "requested_path": str(requested),
        "path": str(path),
        "parent": None if path == path.parent else str(path.parent),
        "directories": directories,
        "files": files,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root", type=Path,
        help="Dataset or domain directory to inspect")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--title", default="Live Terrain Dataset Viewer")
    parser.add_argument(
        "--quick-canonical-root", type=Path,
        help="Canonical directory used by the quick-load button")
    parser.add_argument(
        "--quick-final-root", type=Path,
        help="Final dataset directory used by the quick-load button")
    parser.add_argument(
        "--domain", default="",
        help="Domain used to infer the standard canonical directory")
    return parser.parse_args()


def infer_domain(root):
    parts = list(Path(root).parts)
    for marker in ("canonical", "public_terrain_20m", "public_terrain_20m_obstacle"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    return ""


def quick_roots(args, catalog):
    final_root = (Path(args.quick_final_root).expanduser().absolute()
                  if args.quick_final_root else catalog.root)
    domain = args.domain.strip() or infer_domain(final_root) or infer_domain(catalog.root)
    canonical_root = (
        Path(args.quick_canonical_root).expanduser().absolute()
        if args.quick_canonical_root else
        WORKSPACE_ROOT / "dataset" / "canonical" / domain / "reviewed_regions"
        if domain else None
    )
    return {
        "canonical": str(canonical_root) if canonical_root else "",
        "final": str(final_root),
    }


def numeric_map_index(metadata, scene_id):
    for value in (
            metadata.get("canonical_map_index"),
            metadata.get("original_map_index")):
        if isinstance(value, int) and value >= 0:
            return value
    source_map = metadata.get("source_map") or {}
    scene = str(source_map.get("scene", ""))
    match = re.fullmatch(r"(?:map|scene)_(\d+)", scene)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"(?:train|val)/map_(\d+)", str(scene_id))
    return int(match.group(1)) if match else None


class DatasetCatalog:
    def __init__(self, root):
        # Keep the lexical path instead of resolving a symlink.  A long-running
        # generation can replace the stable symlink with the final directory
        # without invalidating this viewer.
        self.root = Path(root).expanduser().absolute()
        self.summary_cache = {}
        self.lock = threading.RLock()

    def set_root(self, root):
        path = resolve_workspace_path(root)
        if path.exists() and not path.is_dir():
            raise NotADirectoryError(path)
        with self.lock:
            self.root = path
            self.summary_cache.clear()

    def map_paths(self):
        trajectory_maps = sorted(
            path for path in self.root.rglob("map.p")
            if path.parent.parent.name in {"train", "val"}
        )
        canonical_maps = sorted(
            path for path in self.root.rglob("map_*.npz")
            if path.parent.name in {"train", "val"}
        )
        return trajectory_maps + canonical_maps

    def scene_id(self, map_path):
        relative = map_path.relative_to(self.root)
        if map_path.suffix == ".npz":
            return relative.with_suffix("").as_posix()
        return relative.parent.as_posix()

    @staticmethod
    def path_state(map_path):
        paths = sorted(map_path.parent.glob("path_*.p"))
        latest = max((path.stat().st_mtime_ns for path in paths), default=0)
        return paths, latest

    def base_summary(self, map_path):
        metadata_path = map_path.with_suffix(".json")
        modified = max(
            map_path.stat().st_mtime_ns,
            metadata_path.stat().st_mtime_ns if metadata_path.is_file() else 0,
        )
        cached = self.summary_cache.get(map_path)
        if cached and cached[0] == modified:
            return dict(cached[1])

        if map_path.suffix == ".npz":
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            quality = metadata.get("quality")
            if not isinstance(quality, dict) or not quality.get("quality"):
                quality = evaluate(str(map_path))
            split = map_path.parent.name
            summary = {
                "id": self.scene_id(map_path),
                "display_id": f"{split}/{map_path.stem}",
                "relative_source": str(map_path),
                "split": split,
                "domain": metadata.get("domain", ""),
                "artifact": "canonical",
                "grade": quality.get("grade") or "reject",
                "score": quality.get("geometry_score"),
                "quality": quality.get("quality"),
                "quality_metrics": quality.get("metrics", {}),
                "needs_return": bool(metadata.get("needs_return")),
                "metadata": metadata,
            }
            self.summary_cache[map_path] = (modified, summary)
            return dict(summary)

        with map_path.open("rb") as stream:
            map_data = pickle.load(stream)
        crop = map_data.get("crop") or {}
        metadata = dict(crop.get("mother_map_sample") or {})
        metadata["environment_id"] = map_path.parent.name
        metadata["dataset_phase"] = map_data.get("dataset_phase")
        metadata["planner_map_version"] = map_data.get("planner_map_version")
        metadata["original_map_index"] = map_data.get("original_map_index")
        metadata["source_map"] = map_data.get("source_map")
        metadata["original_map_path"] = map_data.get("original_map_path")
        quality = map_data.get("quality")
        if not isinstance(quality, dict) or not quality.get("quality"):
            source_value = map_data.get("original_map_path")
            source_path = Path(source_value) if source_value else None
            sidecar = source_path.with_suffix(".npz") if source_path else None
            quality = evaluate(str(sidecar)) if sidecar and sidecar.is_file() else {
                "quality": "unknown", "grade": None,
                "geometry_score": None, "metrics": {},
            }
        summary = {
            "id": self.scene_id(map_path),
            "display_id": (
                f"{map_data.get('dataset_phase') or map_path.parent.parent.name}/"
                f"{map_path.parent.name}"),
            "relative_source": str(map_path),
            "split": str(map_data.get("dataset_phase")
                         or map_path.parent.parent.name),
            "domain": metadata.get("domain", ""),
            "artifact": "dataset",
            "grade": quality.get("grade") or "reject",
            "score": quality.get("geometry_score"),
            "quality": quality.get("quality"),
            "quality_metrics": quality.get("metrics", {}),
            "needs_return": (map_path.parent / "needs_return.json").is_file(),
            "metadata": metadata,
        }
        self.summary_cache[map_path] = (modified, summary)
        return dict(summary)

    def index(self):
        with self.lock:
            scenes = []
            active_paths = set()
            for map_path in self.map_paths():
                active_paths.add(map_path)
                summary = self.base_summary(map_path)
                paths, latest = self.path_state(map_path)
                summary["path_count"] = len(paths)
                if map_path.suffix != ".npz":
                    summary["needs_return"] = (
                        map_path.parent / "needs_return.json").is_file()
                summary["live_token"] = (
                    f"{summary['artifact']}:{map_path.stat().st_mtime_ns}:"
                    f"{len(paths)}:{latest}:{int(bool(summary.get('needs_return')))}")
                scenes.append(summary)
            self.summary_cache = {
                path: value for path, value in self.summary_cache.items()
                if path in active_paths
            }
            return {"dataset_root": str(self.root), "scenes": scenes}

    def scene(self, requested_id):
        with self.lock:
            matches = {
                self.scene_id(path): path for path in self.map_paths()
            }
            map_path = matches.get(requested_id)
            if map_path is None:
                raise FileNotFoundError(f"Unknown scene: {requested_id}")
            if map_path.suffix == ".npz":
                record = scene_record(map_path)
            else:
                record = trajectory_scene_record(map_path, self.root, 0)
            summary = self.base_summary(map_path)
            paths, latest = self.path_state(map_path)
            record["id"] = requested_id
            record["display_id"] = summary["display_id"]
            record["domain"] = summary["domain"]
            record["artifact"] = summary["artifact"]
            record["path_count"] = len(paths)
            record["needs_return"] = bool(summary.get("needs_return"))
            record["live_token"] = (
                f"{summary['artifact']}:{map_path.stat().st_mtime_ns}:"
                f"{len(paths)}:{latest}:{int(record['needs_return'])}")
            record["split"] = summary["split"]
            return record

    def replacement_context(self, requested_id, canonical_root, final_root):
        with self.lock:
            matches = {self.scene_id(path): path for path in self.map_paths()}
            map_path = matches.get(requested_id)
            if map_path is None:
                raise FileNotFoundError(f"Unknown scene: {requested_id}")
            summary = self.base_summary(map_path)
            metadata = summary["metadata"]
            split = summary["split"]
            map_index = numeric_map_index(metadata, requested_id)
            if map_index is None:
                raise ValueError(
                    f"场景 {summary['display_id']} 没有可定位的 canonical 编号")
            canonical = Path(canonical_root) if canonical_root else None
            if canonical is None:
                raise ValueError("没有配置 canonical 地图目录")
            canonical_stem = canonical / split / f"map_{map_index:03d}"
            if not Path(str(canonical_stem) + ".json").is_file():
                raise FileNotFoundError(
                    f"canonical 地图不存在: {canonical_stem}.json")

            target_root = None
            environment_id = None
            path_count = 0
            if summary["artifact"] == "dataset":
                environment_match = re.fullmatch(
                    r"env(\d+)", str(metadata.get("environment_id", "")))
                if environment_match:
                    environment_id = int(environment_match.group(1))
                path_count = len(self.path_state(map_path)[0])
                # A partially generated environment may have a map but no
                # trajectories yet.  There is nothing to regenerate in that
                # case; replace the canonical slot only.
                if environment_id is not None and path_count > 0:
                    target_root = self.root
            elif final_root:
                candidate_root = Path(final_root)
                candidate_env = (
                    candidate_root / split / f"env{map_index:06d}")
                if (candidate_env / "map.p").is_file():
                    target_root = candidate_root
                    environment_id = map_index
                    path_count = len(list(candidate_env.glob("path_*.p")))
            manual_review = metadata.get("manual_review") or {}
            source_path = (
                metadata.get("source_file")
                or manual_review.get("source_path")
                or metadata.get("original_map_path", ""))
            previous_center = metadata.get("source_center_xy")
            if not isinstance(previous_center, list):
                previous_center = manual_review.get("center_xy")
            previous_yaw = metadata.get("crop_yaw_deg")
            if previous_yaw is None:
                previous_yaw = manual_review.get("yaw_deg", 0.0)
            return {
                "scene_id": requested_id,
                "artifact": summary["artifact"],
                "split": split,
                "map_index": map_index,
                "canonical_root": str(canonical),
                "final_root": str(target_root) if target_root else "",
                "environment_id": environment_id,
                "paths_per_env": path_count,
                "display_id": summary["display_id"],
                "source_id": manual_review.get("source_id", ""),
                "source_path": str(source_path),
                "previous_center_xy": previous_center,
                "previous_yaw_deg": float(previous_yaw or 0.0),
                "size_m": float(manual_review.get("size_m", 20.0)),
            }


class ReplacementJobs:
    def __init__(self, workspace, domain, canonical_root, final_root):
        self.workspace = workspace
        self.domain = domain
        self.canonical_root = canonical_root
        self.final_root = final_root
        self.review_file = (
            workspace / "dataset" / "reviews" / domain / "manual_regions.jsonl")
        self.lock = threading.RLock()
        self.jobs = {}

    @staticmethod
    def snapshot(job):
        return {key: value for key, value in job.items() if key != "process"}

    def _append_return(self, context):
        record, _created = append_return_record(self.review_file, {
            **context,
            "reason": context.get("reason") or f"8765: {context['display_id']}",
        })
        return record

    def start(self, context):
        with self.lock:
            job_id = uuid.uuid4().hex[:12]
            record = self._append_return(context)
            job = {
                "id": job_id,
                "status": "completed",
                "action": "slot_returned",
                "target": context,
                "return_record": record,
            }
            self.jobs[job_id] = job
            return self.snapshot(job)

    def start_many(self, contexts):
        results = []
        with self.lock:
            for context in contexts:
                results.append(self.start(context))
        return {
            "status": "completed",
            "count": len(results),
            "jobs": results,
        }

    def status(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return self.snapshot(job)


def make_handler(catalog, html, replacement_jobs):
    class Handler(BaseHTTPRequestHandler):
        def send_payload(self, payload, content_type, status=200):
            body = payload if isinstance(payload, bytes) else payload.encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def send_json(self, value, status=200):
            self.send_payload(
                json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                "application/json; charset=utf-8", status)

        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self.send_payload(html, "text/html; charset=utf-8")
                elif parsed.path == "/api/index":
                    self.send_json(catalog.index())
                elif parsed.path == "/api/browse":
                    self.send_json(browse_directory(
                        parse_qs(parsed.query).get("path", [""])[0]))
                elif parsed.path == "/api/scene":
                    scene_id = parse_qs(parsed.query).get("id", [""])[0]
                    self.send_json(catalog.scene(scene_id))
                elif parsed.path == "/api/replacement-status":
                    job_id = parse_qs(parsed.query).get("id", [""])[0]
                    self.send_json(replacement_jobs.status(job_id))
                else:
                    self.send_json({"error": "not found"}, 404)
            except FileNotFoundError as exc:
                self.send_json({"error": str(exc)}, 404)
            except KeyError as exc:
                self.send_json({"error": f"unknown replacement job: {exc.args[0]}"}, 404)
            except (OSError, ValueError, EOFError,
                    pickle.UnpicklingError) as exc:
                self.send_json({"error": str(exc)}, 500)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path not in {
                    "/api/load-root", "/api/replace-scene",
                    "/api/replace-scenes"}:
                self.send_json({"error": "not found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                record = json.loads(self.rfile.read(length).decode("utf-8"))
                if parsed.path == "/api/load-root":
                    path = str(record.get("path", "")).strip()
                    if not path:
                        raise ValueError("path is required")
                    catalog.set_root(path)
                    self.send_json(catalog.index())
                    return
                if parsed.path == "/api/replace-scenes":
                    ids = record.get("ids")
                    if not isinstance(ids, list) or not ids:
                        raise ValueError("ids must be a non-empty list")
                    contexts = [
                        catalog.replacement_context(
                            str(scene_id).strip(),
                            replacement_jobs.canonical_root,
                            replacement_jobs.final_root,
                        )
                        for scene_id in ids if str(scene_id).strip()
                    ]
                    if not contexts:
                        raise ValueError("ids must be a non-empty list")
                    self.send_json(replacement_jobs.start_many(contexts))
                    return
                scene_id = str(record.get("id", "")).strip()
                if not scene_id:
                    raise ValueError("scene id is required")
                context = catalog.replacement_context(
                    scene_id,
                    replacement_jobs.canonical_root,
                    replacement_jobs.final_root,
                )
                self.send_json(replacement_jobs.start(context))
            except FileNotFoundError as exc:
                self.send_json({"error": str(exc)}, 404)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                self.send_json({"error": str(exc)}, 400)

        def log_message(self, format_string, *args):
            return

    return Handler


def main():
    args = parse_args()
    if args.dataset_root.exists() and not args.dataset_root.is_dir():
        raise NotADirectoryError(args.dataset_root)
    catalog = DatasetCatalog(args.dataset_root)
    quick_root_paths = quick_roots(args, catalog)
    replacement_jobs = ReplacementJobs(
        WORKSPACE_ROOT,
        args.domain.strip() or infer_domain(catalog.root) or "terrain",
        Path(quick_root_paths["canonical"])
        if quick_root_paths["canonical"] else None,
        Path(quick_root_paths["final"])
        if quick_root_paths["final"] else None,
    )
    html = (HTML_TEMPLATE.replace("__TITLE__", args.title)
            .replace("__SCENES__", "[]")
            .replace("__LIVE__", "true")
            .replace("__DATASET_ROOT__", json.dumps(str(catalog.root)))
            .replace("__QUICK_ROOTS__", json.dumps(quick_root_paths)))
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(catalog, html, replacement_jobs))
    print(json.dumps({
        "url": f"http://{args.host}:{args.port}/",
        "dataset_root": str(catalog.root),
        "refresh_seconds": 5,
    }, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
