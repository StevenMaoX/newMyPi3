import base64
import io
import math
import os
import re
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from torchvision import transforms
from plyfile import PlyData, PlyElement

from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor


def _get_device() -> torch.device:
    env_device = os.getenv("PI3_DEVICE", "").strip().lower()
    if env_device:
        return torch.device(env_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in ("1", "true", "yes", "on")


def _load_model(device: torch.device) -> torch.nn.Module:
    ckpt = os.getenv("PI3_CKPT", "").strip()
    if ckpt:
        model = Pi3().eval()
        if ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file

            weight = load_file(ckpt)
        else:
            weight = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight, strict=False)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").eval()
    return model.to(device)


AUTH_TOKEN = os.getenv("PI3_AUTH_TOKEN", "").strip()
PIXEL_LIMIT = int(os.getenv("PI3_PIXEL_LIMIT", "255000"))
DEFAULT_MAX_VELOCITY = float(os.getenv("PI3_DEFAULT_MAX_VELOCITY", "2.2"))
DEFAULT_MAX_ACCELERATION = float(os.getenv("PI3_DEFAULT_MAX_ACCELERATION", "2.5"))
DEFAULT_SAVE_CLOUD = _env_bool("PI3_SAVE_CLOUD", True)
CLOUD_DIR = Path(os.getenv("PI3_CLOUD_DIR", str(Path(__file__).resolve().parent / "pi3" / "cloud"))).resolve()
CLOUD_MAX_POINTS = int(os.getenv("PI3_CLOUD_MAX_POINTS", "120000"))
CLOUD_SAVE_PLY = _env_bool("PI3_CLOUD_SAVE_PLY", True)
CLOUD_SAVE_PNG = _env_bool("PI3_CLOUD_SAVE_PNG", True)
CLOUD_PNG_POINT_SIZE = float(os.getenv("PI3_CLOUD_PNG_POINT_SIZE", "0.4"))

DEVICE = _get_device()
MODEL = _load_model(DEVICE)
APP_DTYPE = (
    torch.bfloat16
    if DEVICE.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)

app = FastAPI(title="Pi3 Remote Decode Service", version="1.0.0")


class DecodeReq(BaseModel):
    instruction: Optional[str] = ""
    video_path: Optional[str] = ""
    frames_b64: List[str] = Field(default_factory=list)
    target_waypoints: int = 6
    return_fields: List[str] = Field(
        default_factory=lambda: ["waypoints_norm", "yaw_deg", "camera_poses"]
    )
    max_velocity: float = DEFAULT_MAX_VELOCITY
    max_acceleration: float = DEFAULT_MAX_ACCELERATION
    frame_interval: int = 1
    max_frames: int = 0
    # Cloud saving controls: by default use PI3_SAVE_CLOUD env switch.
    save_cloud: Optional[bool] = None
    cloud_tag: str = ""
    cloud_sample_stride: int = 4
    # Backward-compatible aliases for client convenience.
    save_point_cloud: Optional[bool] = None
    point_cloud_filename: str = ""


def _resize_to_pi3_shape(images: List[Image.Image], pixel_limit: int = PIXEL_LIMIT) -> torch.Tensor:
    if not images:
        return torch.empty(0)

    first_img = images[0]
    w_orig, h_orig = first_img.size
    scale = math.sqrt(pixel_limit / (w_orig * h_orig)) if w_orig * h_orig > 0 else 1.0
    w_target, h_target = w_orig * scale, h_orig * scale
    k, m = round(w_target / 14), round(h_target / 14)

    while (k * 14) * (m * 14) > pixel_limit:
        if k / max(m, 1) > w_target / max(h_target, 1):
            k -= 1
        else:
            m -= 1

    target_w, target_h = max(1, k) * 14, max(1, m) * 14
    to_tensor = transforms.ToTensor()

    out = []
    for img in images:
        resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        out.append(to_tensor(resized))

    return torch.stack(out, dim=0)


def _decode_frames(frames_b64: List[str]) -> torch.Tensor:
    images: List[Image.Image] = []
    for i, b64 in enumerate(frames_b64):
        try:
            payload = b64.split(",", 1)[1] if "base64," in b64 else b64
            arr = base64.b64decode(payload)
            img = Image.open(io.BytesIO(arr)).convert("RGB")
            images.append(img)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid frames_b64 at index {i}: {exc}") from exc

    if not images:
        raise HTTPException(status_code=400, detail="frames_b64 is empty")

    return _resize_to_pi3_shape(images)


def _load_video(video_path: str, frame_interval: int, max_frames: int) -> torch.Tensor:
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"video_path not found: {video_path}")

    interval = max(frame_interval, 1)
    imgs = load_images_as_tensor(video_path, interval=interval, PIXEL_LIMIT=PIXEL_LIMIT, verbose=False)
    if imgs.numel() == 0:
        raise HTTPException(status_code=400, detail="no frames decoded from video_path")

    if max_frames > 0:
        imgs = imgs[:max_frames]
    return imgs


def _poses_to_waypoints(camera_poses: np.ndarray, target_n: int = 6):
    # Pi3/Pi3X poses are OpenCV camera-to-world matrices:
    # camera x=right, y=down, z=forward. Express camera centers in the first
    # camera frame, then convert to NavDreamer local x=forward, y=right, z=up.
    r0 = camera_poses[0, :3, :3]
    t0 = camera_poses[0, :3, 3]
    cam0_rdf = (camera_poses[:, :3, 3] - t0) @ r0
    xyz = np.column_stack([cam0_rdf[:, 2], cam0_rdf[:, 0], -cam0_rdf[:, 1]])

    if len(xyz) > target_n:
        idx = np.linspace(0, len(xyz) - 1, target_n, dtype=int)
        xyz = xyz[idx]

    scale = float(max(np.linalg.norm(xyz[-1]), 1e-6))
    xyz_norm = xyz / scale

    yaws = []
    for i in range(len(xyz_norm) - 1):
        d = xyz_norm[i + 1, :2] - xyz_norm[i, :2]
        yaws.append(float(np.degrees(np.arctan2(d[1], max(abs(d[0]), 1e-6)))))
    yaws.append(yaws[-1] if yaws else 0.0)

    return xyz_norm.tolist(), yaws, scale


def _encode_depth_npz_b64(depth: np.ndarray) -> Optional[str]:
    """Encode one Pi3 local depth map as compressed NPZ base64."""
    try:
        arr = np.asarray(depth, dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            return None
        if not np.isfinite(arr).any():
            return None
        buffer = io.BytesIO()
        np.savez_compressed(buffer, depth=arr)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        return None


def _validate_auth(authorization: Optional[str]) -> None:
    if not AUTH_TOKEN:
        return
    if authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


def _safe_name(name: str) -> str:
    if not name:
        return ""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("_")


def _flatten_points(
    points: np.ndarray,
    sample_stride: int,
    max_points: int,
) -> np.ndarray:
    """
    points: [N, H, W, 3], return [M, 3]
    """
    stride = max(1, int(sample_stride))
    sampled = points[:, ::stride, ::stride, :]
    xyz = sampled.reshape(-1, 3)
    finite_mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite_mask]
    if xyz.shape[0] == 0:
        return xyz

    # Remove extreme outliers to make saved point cloud more robust.
    lo = np.percentile(xyz, 0.5, axis=0)
    hi = np.percentile(xyz, 99.5, axis=0)
    mask = np.logical_and(xyz >= lo, xyz <= hi).all(axis=1)
    xyz = xyz[mask]
    if xyz.shape[0] == 0:
        return xyz

    if xyz.shape[0] > max_points:
        idx = np.random.default_rng(42).choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[idx]
    return xyz.astype(np.float32, copy=False)


def _save_ply(xyz: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertex = np.empty(xyz.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]
    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False).write(str(output_path))


def _save_cloud_png(xyz: np.ndarray, camera_poses: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cam_xyz = camera_poses[:, :3, 3]

    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=CLOUD_PNG_POINT_SIZE, c="#4C78A8", alpha=0.22, linewidths=0)
    ax.plot(cam_xyz[:, 0], cam_xyz[:, 1], cam_xyz[:, 2], color="#E45756", linewidth=1.8, label="camera trajectory")
    ax.scatter([cam_xyz[0, 0]], [cam_xyz[0, 1]], [cam_xyz[0, 2]], color="green", s=50, label="start")
    ax.scatter([cam_xyz[-1, 0]], [cam_xyz[-1, 1]], [cam_xyz[-1, 2]], color="red", marker="^", s=60, label="end")
    ax.set_title(f"Pi3 Point Cloud ({xyz.shape[0]} points)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)


def _save_cloud_artifacts(
    points: np.ndarray,
    camera_poses: np.ndarray,
    video_path: str,
    cloud_tag: str,
    point_cloud_filename: str,
    sample_stride: int,
) -> Dict[str, Any]:
    xyz = _flatten_points(points, sample_stride=sample_stride, max_points=CLOUD_MAX_POINTS)
    if xyz.shape[0] == 0:
        raise RuntimeError("empty point cloud after filtering")

    CLOUD_DIR.mkdir(parents=True, exist_ok=True)
    custom = _safe_name(Path(point_cloud_filename).stem) if point_cloud_filename else ""
    if custom:
        prefix = custom
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source = _safe_name(Path(video_path).stem if video_path else "")
        tag = _safe_name(cloud_tag)
        parts = [p for p in (stamp, source, tag) if p]
        prefix = "_".join(parts) if parts else stamp

    saved: Dict[str, Any] = {
        "cloud_dir": str(CLOUD_DIR),
        "num_points": int(xyz.shape[0]),
        "sample_stride": int(max(1, sample_stride)),
    }

    if CLOUD_SAVE_PLY:
        ply_path = CLOUD_DIR / f"{prefix}.ply"
        _save_ply(xyz, ply_path)
        saved["ply_path"] = str(ply_path)

    if CLOUD_SAVE_PNG:
        png_path = CLOUD_DIR / f"{prefix}.png"
        _save_cloud_png(xyz, camera_poses, png_path)
        saved["png_path"] = str(png_path)

    return saved


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "device": str(DEVICE),
        "dtype": str(APP_DTYPE),
        "auth_enabled": bool(AUTH_TOKEN),
    }


@app.post("/decode")
def decode(req: DecodeReq, authorization: Optional[str] = Header(default=None)):
    _validate_auth(authorization)

    if req.target_waypoints <= 1:
        raise HTTPException(status_code=400, detail="target_waypoints must be > 1")

    if req.frames_b64:
        imgs = _decode_frames(req.frames_b64)
    elif req.video_path:
        imgs = _load_video(req.video_path, req.frame_interval, req.max_frames)
    else:
        raise HTTPException(status_code=400, detail="frames_b64 is empty and video_path is missing")

    if imgs.shape[0] < 2:
        raise HTTPException(status_code=400, detail="at least 2 frames are required")

    with torch.no_grad():
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=APP_DTYPE) if DEVICE.type == "cuda" else nullcontext()
        )
        with autocast_ctx:
            res = MODEL(imgs.unsqueeze(0).to(DEVICE))
        camera_poses = res["camera_poses"][0].detach().cpu().numpy()
        pi3_depth_npz_b64 = None
        if "local_points" in res:
            try:
                local_points = res["local_points"][0].detach().cpu().numpy()
                # Pi3 local pointmaps are OpenCV camera-frame xyz. z is the
                # per-pixel forward depth used for MoGe2/Pi3 scale recovery.
                if local_points.ndim == 4 and local_points.shape[-1] >= 3:
                    pi3_depth_npz_b64 = _encode_depth_npz_b64(local_points[-1, ..., 2])
            except Exception as exc:
                print(f"[Pi3] depth export failed: {exc}")
        if req.save_cloud is not None:
            should_save_cloud = bool(req.save_cloud)
        elif req.save_point_cloud is not None:
            should_save_cloud = bool(req.save_point_cloud)
        else:
            should_save_cloud = DEFAULT_SAVE_CLOUD
        cloud_saved = None
        if should_save_cloud:
            points = res["points"][0].detach().cpu().numpy()
            try:
                cloud_saved = _save_cloud_artifacts(
                    points=points,
                    camera_poses=camera_poses,
                    video_path=req.video_path or "",
                    cloud_tag=req.cloud_tag,
                    point_cloud_filename=req.point_cloud_filename,
                    sample_stride=req.cloud_sample_stride,
                )
                print(f"[Pi3] cloud saved: {cloud_saved}")
            except Exception as exc:
                cloud_saved = {"error": str(exc)}
                print(f"[Pi3] cloud save failed: {exc}")

    waypoints, yaws, scale_hint = _poses_to_waypoints(camera_poses, req.target_waypoints)

    full_resp = {
        "waypoints_norm": waypoints,
        "yaw_deg": yaws,
        "camera_poses": camera_poses.tolist(),
        "max_velocity": req.max_velocity,
        "max_acceleration": req.max_acceleration,
        "reason": "pi3_remote_waypoints",
        "scale_hint": scale_hint,
        "pi3_depth_npz_b64": pi3_depth_npz_b64,
        "meta": {
            "num_frames": int(imgs.shape[0]),
            "resize_h": int(imgs.shape[2]),
            "resize_w": int(imgs.shape[3]),
            "cloud_saved": cloud_saved,
        },
    }

    if req.return_fields:
        allowed = set(req.return_fields) | {"reason"}
        return {k: v for k, v in full_resp.items() if k in allowed}
    return full_resp
