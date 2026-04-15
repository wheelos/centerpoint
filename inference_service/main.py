from typing import List, Optional

import os
import io
import json
import runpy
import uuid
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="centerpoint-inference")


LABEL_MAP = {
    0: "car",
    1: "pedestrian",
    2: "cyclist",
    3: "traffic_cone",
}


class BatchInferenceItem(BaseModel):
    sample_id: str
    sample_uri: str


class BatchInferenceRequest(BaseModel):
    request_id: Optional[str] = None
    model: Optional[str] = None
    items: List[BatchInferenceItem] = Field(default_factory=list)

# Config from env: default to repo reference ONNX files and example config
ROOT = os.environ.get("REPO_ROOT", ".")
PFE_ONNX = os.environ.get("PFE_ONNX", os.path.join(ROOT, "reference/cpdet_pfe.onnx"))
BACKBONE_ONNX = os.environ.get("BACKBONE_ONNX", os.path.join(ROOT, "reference/cpdet_backbone.onnx"))
CONFIG_PY = os.environ.get("MODEL_CONFIG", os.path.join(ROOT, "mmdet3d_example_configs/centerpoint_trt_nuscenes_4task_train.py"))


def _load_points_from_upload(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    name = upload.filename or ""
    # try numpy .npy
    try:
        if name.endswith(".npy"):
            arr = np.load(io.BytesIO(data))
            return arr.astype(np.float32)
    except Exception:
        pass
    # try raw float32 binary (N x 4)
    try:
        arr = np.frombuffer(data, dtype=np.float32)
        if arr.size % 4 == 0:
            arr = arr.reshape(-1, 4)
            return arr
    except Exception:
        pass
    # try json array
    try:
        arr = np.asarray(json.loads(data.decode("utf-8")), dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr
    except Exception:
        pass
    raise HTTPException(status_code=400, detail="Unsupported pointcloud format")


def _resolve_uri_to_path(uri: str) -> str:
    if uri.startswith("file://"):
        return uri[len("file://") :]

    if uri.startswith("s3://"):
        # local-mount-first mapping for deployed environments
        mount_root = os.environ.get("S3_MOUNT_ROOT", "/mnt/synology")
        body = uri[len("s3://") :]
        return os.path.join(mount_root, body)

    if os.path.isabs(uri):
        return uri

    raise ValueError(f"unsupported uri: {uri}")


def _load_points_from_path(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".pcd":
        # Custom PCD parser to preserve arbitrary channels (e.g. intensity)
        with open(path, "rb") as f:
            lines = []
            while True:
                line = f.readline().decode('utf-8', errors='ignore')
                lines.append(line)
                if line.startswith("DATA"):
                    break

        data_type = lines[-1].strip().split()[1]

        if data_type.lower() == "ascii":
            pts = np.loadtxt(path, skiprows=len(lines), dtype=np.float32)
        else:
            # Fallback to Open3D if binary, and pad intensity to 0
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(path)
            pts3 = np.asarray(pcd.points, dtype=np.float32)
            pts = np.concatenate([pts3, np.zeros((pts3.shape[0], 1), dtype=np.float32)], axis=1)

        return pts

    if suffix == ".npy":
        arr = np.load(path)
        return arr.astype(np.float32)

    if suffix == ".bin":
        arr = np.fromfile(path, dtype=np.float32)
        if arr.size % 4 == 0:
            return arr.reshape(-1, 4)
        if arr.size % 3 == 0:
            pts3 = arr.reshape(-1, 3)
            return np.concatenate([pts3, np.zeros((pts3.shape[0], 1), dtype=np.float32)], axis=1)
        raise ValueError("invalid .bin format")

    raise ValueError(f"unsupported pointcloud extension: {suffix}")


def _decode_to_detections_3d(decoded: dict) -> List[dict]:
    boxes = decoded.get("boxes") or []
    scores = decoded.get("scores") or []
    labels = decoded.get("labels") or []
    out = []
    for idx, box in enumerate(boxes):
        if len(box) < 7:
            continue
        cx, cy, cz, l, w, h, yaw = [float(v) for v in box[:7]]
        score = float(scores[idx]) if idx < len(scores) else 0.0
        label_id = int(labels[idx]) if idx < len(labels) else 0
        out.append(
            {
                "label": LABEL_MAP.get(label_id, f"class_{label_id}"),
                "score": score,
                "center_xyz": [cx, cy, cz],
                "size_lwh": [l, w, h],
                "yaw": yaw,
            }
        )
    return out


class ONNXCenterpointRunner:
    def __init__(self, pfe_path: str, backbone_path: str, cfg_py: str):
        self.pfe_path = pfe_path
        self.backbone_path = backbone_path
        self.cfg_py = cfg_py
        self._load_cfg()
        self._create_sessions()

    def _load_cfg(self):
        cfg_globals = runpy.run_path(self.cfg_py, run_name="__cfg__")
        # import plugin if present
        try:
            custom_imports = cfg_globals.get("custom_imports")
            if isinstance(custom_imports, dict):
                for mod in custom_imports.get("imports", []) or []:
                    __import__(mod)
        except Exception:
            pass
        # build minimal objects
        bev_cfg = cfg_globals.get("bev_feature_cfg", {})
        from apollo_centerpoint_trt.bev_feature import ApolloBevFeatureConfig, ApolloBevFeatureGenerator

        bev_conf = ApolloBevFeatureConfig(**bev_cfg)
        self.bev_gen = ApolloBevFeatureGenerator(bev_conf)

        pts_bbox_head_cfg = cfg_globals.get("pts_bbox_head") or cfg_globals.get("model", {}).get("centerpoint", {}).get("pts_bbox_head")
        if pts_bbox_head_cfg is None:
            # try model dict
            model_dict = cfg_globals.get("model")
            if isinstance(model_dict, dict):
                pts_bbox_head_cfg = model_dict.get("centerpoint", {}).get("pts_bbox_head")
        self.head_cfg = pts_bbox_head_cfg or {}
        # Build minimal head object for config access
        self.head = SimpleNamespace(**{k: v for k, v in (self.head_cfg.items() if isinstance(self.head_cfg, dict) else [])})
        # Ensure test_cfg is available
        if isinstance(self.head_cfg, dict):
            self.head = SimpleNamespace(test_cfg=self.head_cfg.get("test_cfg", self.head_cfg.get("test_cfg", {})), common_heads=self.head_cfg.get("common_heads"), separate_head=self.head_cfg.get("separate_head"))
        binding_cfg = cfg_globals.get("trt_binding_cfg", {})
        self.binding_cfg = binding_cfg if isinstance(binding_cfg, dict) else {}

    def _create_sessions(self):
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = [p for p in (os.environ.get("ORT_PROVIDERS", "CPUExecutionProvider").split(",")) if p]
        try:
            self.pfe_session = ort.InferenceSession(self.pfe_path, sess_options=sess_options, providers=providers)
        except Exception:
            self.pfe_session = ort.InferenceSession(self.pfe_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
        try:
            self.backbone_session = ort.InferenceSession(self.backbone_path, sess_options=sess_options, providers=providers)
        except Exception:
            self.backbone_session = ort.InferenceSession(self.backbone_path, sess_options=sess_options, providers=["CPUExecutionProvider"])

    def _resolve_output_map(self, session, outputs):
        names = [out.name for out in session.get_outputs()]
        return {name: value for name, value in zip(names, outputs)}

    def _build_canvas_feature(self, points: np.ndarray):
        import torch

        pts = torch.from_numpy(points).to(dtype=torch.float32)
        pts_xyzi = pts[:, :4]
        voxels, grid_idx, _ = self.bev_gen.build_voxel_features(pts_xyzi)
        if voxels.size(0) == 0:
            gx = int(self.bev_gen.grid_x_size)
            gy = int(self.bev_gen.grid_y_size)
            channels = int(self.bev_gen.cfg.pillar_feature_dim + self.bev_gen.cfg.cnnseg_feature_dim)
            return torch.zeros((1, channels, gy, gx), dtype=torch.float32)

        voxels_np = voxels.unsqueeze(1).unsqueeze(-1).cpu().numpy().astype(np.float32, copy=False)
        pfe_input_name = str(self.binding_cfg.get("input_voxels") or self.pfe_session.get_inputs()[0].name)
        outputs = self.pfe_session.run(None, {pfe_input_name: voxels_np})
        output_map = self._resolve_output_map(self.pfe_session, outputs)
        pfe_output_name = str(self.binding_cfg.get("pillar_feature_blob") or "pillar_feature")
        pillar_feature_np = output_map.get(pfe_output_name)
        if pillar_feature_np is None:
            pillar_feature_np = output_map.get("pillar_feature", outputs[0])
        pillar_feature = torch.from_numpy(pillar_feature_np).to(dtype=pts_xyzi.dtype, device="cpu")
        canvas = self.bev_gen(pts_xyzi, pillar_feature)
        return canvas

    # copy of rotated iou and greedy nms from tools/test_onnx.py
    def _xywhr_to_corners(self, box: np.ndarray) -> np.ndarray:
        x, y, w, h, yaw = [float(v) for v in box.tolist()]
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        half_w = w * 0.5
        half_h = h * 0.5
        corners = np.asarray(
            [
                [-half_w, -half_h],
                [half_w, -half_h],
                [half_w, half_h],
                [-half_w, half_h],
            ],
            dtype=np.float32,
        )
        rot = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
        return corners @ rot.T + np.asarray([x, y], dtype=np.float32)

    def _polygon_area(self, points: np.ndarray) -> float:
        if points.shape[0] < 3:
            return 0.0
        x = points[:, 0]
        y = points[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def _inside_edge(self, point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
        return float((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) -
                     (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])) >= -1e-6

    def _segment_intersection(self, p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        s = p2 - p1
        r = q2 - q1
        denom = float(s[0] * r[1] - s[1] * r[0])
        if abs(denom) < 1e-6:
            return p2.astype(np.float32)
        t = float(((q1[0] - p1[0]) * r[1] - (q1[1] - p1[1]) * r[0]) / denom)
        return (p1 + t * s).astype(np.float32)

    def _polygon_clip(self, subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
        output = subject.astype(np.float32)
        for i in range(clipper.shape[0]):
            clip_start = clipper[i]
            clip_end = clipper[(i + 1) % clipper.shape[0]]
            if output.shape[0] == 0:
                break
            input_list = output
            output_pts: list[np.ndarray] = []
            prev = input_list[-1]
            for cur in input_list:
                cur_inside = self._inside_edge(cur, clip_start, clip_end)
                prev_inside = self._inside_edge(prev, clip_start, clip_end)
                if cur_inside:
                    if not prev_inside:
                        output_pts.append(self._segment_intersection(prev, cur, clip_start, clip_end))
                    output_pts.append(cur.astype(np.float32))
                elif prev_inside:
                    output_pts.append(self._segment_intersection(prev, cur, clip_start, clip_end))
                prev = cur
            if output_pts:
                output = np.stack(output_pts, axis=0).astype(np.float32)
            else:
                output = np.zeros((0, 2), dtype=np.float32)
        return output

    def _rotated_iou_single(self, box1: np.ndarray, box2: np.ndarray) -> float:
        poly1 = self._xywhr_to_corners(box1)
        poly2 = self._xywhr_to_corners(box2)
        inter_poly = self._polygon_clip(poly1, poly2)
        inter_area = self._polygon_area(inter_poly)
        area1 = float(box1[2] * box1[3])
        area2 = float(box2[2] * box2[3])
        union = max(area1 + area2 - inter_area, 1e-6)
        return float(inter_area / union)

    def _greedy_nms_rotated(self, boxes_xywhr: np.ndarray, scores: np.ndarray, iou_thr: float, post_max_size: int) -> np.ndarray:
        if boxes_xywhr.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        order = np.argsort(-scores)
        keep: list[int] = []
        suppressed = np.zeros((boxes_xywhr.shape[0],), dtype=bool)
        for pos, idx in enumerate(order):
            if suppressed[pos]:
                continue
            keep.append(int(idx))
            if 0 < post_max_size <= len(keep):
                break
            for next_pos in range(pos + 1, order.shape[0]):
                if suppressed[next_pos]:
                    continue
                jdx = int(order[next_pos])
                if self._rotated_iou_single(boxes_xywhr[idx], boxes_xywhr[jdx]) > iou_thr:
                    suppressed[next_pos] = True
        return np.asarray(keep, dtype=np.int64)

    def _build_head_outputs(self, scores_np: np.ndarray, bbox_preds_np: np.ndarray, dir_scores_np: np.ndarray):
        import torch
        scores = torch.from_numpy(scores_np).to(dtype=torch.float32, device="cpu")
        bbox_preds = torch.from_numpy(bbox_preds_np).to(dtype=torch.float32, device="cpu")
        dir_scores = torch.from_numpy(dir_scores_np).to(dtype=torch.float32, device="cpu")

        num_tasks = int(scores.shape[1])
        expected_bbox_channels = num_tasks * 6
        expected_rot_channels = num_tasks * 2
        if int(bbox_preds.shape[1]) != expected_bbox_channels:
            raise ValueError("bbox_preds channels mismatch")
        if int(dir_scores.shape[1]) != expected_rot_channels:
            raise ValueError("dir_scores channels mismatch")

        task_outs = []
        bbox_offset = 0
        rot_offset = 0
        # minimal need_vel detection
        need_vel = False
        for task_idx in range(num_tasks):
            task = dict(
                heatmap=scores[:, task_idx:task_idx + 1, :, :],
                reg=bbox_preds[:, bbox_offset:bbox_offset + 2, :, :],
                height=bbox_preds[:, bbox_offset + 2:bbox_offset + 3, :, :],
                dim=bbox_preds[:, bbox_offset + 3:bbox_offset + 6, :, :],
                rot=dir_scores[:, rot_offset:rot_offset + 2, :, :],
            )
            if need_vel:
                task["vel"] = torch.zeros((scores.shape[0], 2, scores.shape[2], scores.shape[3]), dtype=scores.dtype)
            task_outs.append(task)
            bbox_offset += 6
            rot_offset += 2

        return task_outs

    def _decode_single_sample(self, task_outs):
        import torch
        cfg = getattr(self.head, "test_cfg", {})
        score_threshold = float(cfg.get("score_threshold", 0.1))
        task_score_thresholds = cfg.get("task_score_thresholds", None)
        pre_max_size = int(cfg.get("pre_max_size", 1000))
        post_max_size = int(cfg.get("post_max_size", 83))
        nms_thr = float(cfg.get("nms_thr", 0.2))
        out_size_factor = float(cfg.get("out_size_factor", 4))
        voxel_size = cfg.get("voxel_size", [0.2, 0.2])
        pc_range = cfg.get("pc_range") or cfg.get("point_cloud_range")
        if pc_range is None:
            pc_range = [-51.2, -51.2]
        min_x = float(pc_range[0])
        min_y = float(pc_range[1])

        all_boxes = []
        all_scores = []
        all_labels = []

        for task_id, task in enumerate(task_outs):
            heatmap = torch.sigmoid(task["heatmap"][0]).cpu()
            reg = task["reg"][0].cpu()
            height = task["height"][0].cpu()
            dim = task["dim"][0].cpu()
            rot = task["rot"][0].cpu()
            num_classes, feat_h, feat_w = heatmap.shape

            scores_flat = heatmap.reshape(num_classes, -1)
            cls_scores_all, cls_ids_all = torch.max(scores_flat, dim=0)
            task_score_threshold = score_threshold
            if isinstance(task_score_thresholds, (list, tuple)) and task_id < len(task_score_thresholds):
                task_score_threshold = float(task_score_thresholds[task_id])
            keep_mask = cls_scores_all > task_score_threshold
            if int(keep_mask.sum()) == 0:
                continue

            candidate_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
            # threshold first, then keep the earliest valid feature-map positions
            # up to pre_max_size instead of top-k, like apollo c++ inference
            if pre_max_size > 0:
                candidate_idx = candidate_idx[:pre_max_size]
            cls_scores = cls_scores_all[candidate_idx]
            cls_ids = cls_ids_all[candidate_idx]

            xs = candidate_idx % feat_w
            ys = torch.div(candidate_idx, feat_w, rounding_mode="floor")

            x = (reg[0, ys, xs] + xs.to(dtype=reg.dtype)) * out_size_factor * float(voxel_size[0]) + min_x
            y = (reg[1, ys, xs] + ys.to(dtype=reg.dtype)) * out_size_factor * float(voxel_size[1]) + min_y
            z = height[0, ys, xs]
            l = torch.exp(dim[0, ys, xs])
            w = torch.exp(dim[1, ys, xs])
            h = torch.exp(dim[2, ys, xs])
            yaw = torch.atan2(rot[0, ys, xs], rot[1, ys, xs])

            labels = cls_ids.to(dtype=torch.long) + task_id
            boxes = torch.stack((x, y, z, l, w, h, yaw), dim=1)

            if int(boxes.shape[0]) > 0:
                boxes_xywhr = torch.stack((boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6]), dim=1)
                keep = self._greedy_nms_rotated(
                    boxes_xywhr.numpy().astype(np.float32, copy=False),
                    cls_scores.numpy().astype(np.float32, copy=False),
                    iou_thr=nms_thr,
                    post_max_size=post_max_size,
                )
                if keep.size == 0:
                    continue
                keep_t = torch.from_numpy(keep).to(dtype=torch.long)
                boxes = boxes[keep_t]
                cls_scores = cls_scores[keep_t]
                labels = labels[keep_t]

            all_boxes.append(boxes)
            all_scores.append(cls_scores)
            all_labels.append(labels)

        if all_boxes:
            boxes = torch.cat(all_boxes, dim=0)
            scores = torch.cat(all_scores, dim=0)
            labels = torch.cat(all_labels, dim=0)
            order = torch.argsort(scores, descending=True)
            if post_max_size > 0:
                order = order[:post_max_size]
            boxes = boxes[order]
            scores = scores[order]
            labels = labels[order]
        else:
            import torch as _torch
            boxes = _torch.zeros((0, 7), dtype=_torch.float32)
            scores = _torch.zeros((0,), dtype=_torch.float32)
            labels = _torch.zeros((0,), dtype=_torch.long)

        if boxes.shape[0] > 0 and bool(self.bev_gen.cfg.enable_rotate_45degree):
            c = float(1.0 / np.sqrt(2.0))
            orig_x = c * boxes[:, 0] + c * boxes[:, 1]
            orig_y = -c * boxes[:, 0] + c * boxes[:, 1]
            boxes[:, 0] = orig_x
            boxes[:, 1] = orig_y
            boxes[:, 6] = boxes[:, 6] - float(np.pi / 4.0)

        if boxes.shape[0] > 0:
            boxes[:, 6] = ((boxes[:, 6] + float(np.pi)) % float(2 * np.pi)) - float(np.pi)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 5] * 0.5

        # return numpy arrays
        return {
            "boxes": boxes.cpu().numpy().tolist(),
            "scores": scores.cpu().numpy().tolist(),
            "labels": labels.cpu().numpy().tolist(),
        }

    def predict(self, points_list: List[np.ndarray]):
        results = []
        for pts in points_list:
            canvas = self._build_canvas_feature(pts)
            canvas_np = canvas.cpu().numpy().astype(np.float32, copy=False)
            backbone_input_name = str(self.binding_cfg.get("input_canvas_feature") or self.backbone_session.get_inputs()[0].name)
            ort_outputs = self.backbone_session.run(None, {backbone_input_name: canvas_np})
            output_map = self._resolve_output_map(self.backbone_session, ort_outputs)
            output_cls_name = str(self.binding_cfg.get("output_cls") or "scores")
            output_box_name = str(self.binding_cfg.get("output_box") or "bbox_preds")
            output_dir_name = str(self.binding_cfg.get("output_dir") or "dir_scores")
            scores_np = output_map.get(output_cls_name)
            bbox_preds_np = output_map.get(output_box_name)
            dir_scores_np = output_map.get(output_dir_name)
            if scores_np is None:
                scores_np = output_map.get("scores", ort_outputs[0])
            if bbox_preds_np is None:
                bbox_preds_np = output_map.get("bbox_preds", ort_outputs[1] if len(ort_outputs) > 1 else None)
            if dir_scores_np is None:
                dir_scores_np = output_map.get("dir_scores", ort_outputs[2] if len(ort_outputs) > 2 else None)
            if bbox_preds_np is None or dir_scores_np is None:
                raise RuntimeError("Backbone ONNX missing expected outputs")
            task_outs = self._build_head_outputs(scores_np, bbox_preds_np, dir_scores_np)
            decoded = self._decode_single_sample(task_outs)
            results.append(decoded)
        return results


RUNNER = None


def _ensure_runner():
    global RUNNER
    if RUNNER is None:
        RUNNER = ONNXCenterpointRunner(PFE_ONNX, BACKBONE_ONNX, CONFIG_PY)
    return RUNNER


@app.get("/health")
def health():
    ok = os.path.exists(PFE_ONNX) and os.path.exists(BACKBONE_ONNX) and os.path.exists(CONFIG_PY)
    return {"status": "ok" if ok else "missing_files", "pfe": PFE_ONNX, "backbone": BACKBONE_ONNX, "config": CONFIG_PY}


@app.get("/v1/health/live")
def health_live():
    return {"status": "ok"}


@app.get("/v1/health/ready")
def health_ready():
    ok = os.path.exists(PFE_ONNX) and os.path.exists(BACKBONE_ONNX) and os.path.exists(CONFIG_PY)
    return {
        "status": "ok" if ok else "missing_files",
        "pfe": PFE_ONNX,
        "backbone": BACKBONE_ONNX,
        "config": CONFIG_PY,
    }


@app.post("/v1/inference/batch")
def infer_batch_v1(req: BatchInferenceRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="items is required")

    request_id = req.request_id or uuid.uuid4().hex
    runner = _ensure_runner()

    results = []
    errors = []

    for item in req.items:
        try:
            path = _resolve_uri_to_path(item.sample_uri)
            pts = _load_points_from_path(path)
            if pts.ndim != 2 or pts.shape[1] not in (3, 4):
                raise ValueError("points must be Nx3 or Nx4")
            if pts.shape[1] == 3:
                pts = np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)], axis=1)

            decoded = runner.predict([pts])[0]
            detections_3d = _decode_to_detections_3d(decoded)
            results.append(
                {
                    "sample_id": item.sample_id,
                    "detections_3d": detections_3d,
                    "metrics": {},
                }
            )
        except Exception as e:
            errors.append(
                {
                    "sample_id": item.sample_id,
                    "sample_uri": item.sample_uri,
                    "error": str(e),
                }
            )

    return {
        "request_id": request_id,
        "results": results,
        "errors": errors,
    }


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    pts = _load_points_from_upload(file)
    if pts.ndim == 1:
        raise HTTPException(status_code=400, detail="Points must be Nx3 or Nx4")
    # Accept Nx3 or Nx4, convert to Nx4
    if pts.shape[1] == 3:
        pts = np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)], axis=1)

    runner = _ensure_runner()
    out = runner.predict([pts])
    return JSONResponse(content={"predictions": out})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
