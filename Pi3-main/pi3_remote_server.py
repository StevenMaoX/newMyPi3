import base64
import io
import math
import os
import re
from datetime import datetime
from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from torchvision import transforms

from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge


def _get_device() -> torch.device:
    env_device = os.getenv("PI3_DEVICE", "").strip().lower()
    if env_device:
        return torch.device(env_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
DEFAULT_SAVE_POINT_CLOUD = os.getenv("PI3_DEFAULT_SAVE_POINT_CLOUD", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_POINT_CLOUD_DIR = os.path.abspath(
    os.getenv(
        "PI3_POINT_CLOUD_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudpoints"),
    )
)
DEFAULT_POINT_CLOUD_CONF_THRES = float(os.getenv("PI3_POINT_CLOUD_CONF_THRES", "0.1"))
DEFAULT_POINT_CLOUD_EDGE_RTL = float(os.getenv("PI3_POINT_CLOUD_EDGE_RTL", "0.03"))

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
    save_point_cloud: bool = DEFAULT_SAVE_POINT_CLOUD
    point_cloud_filename: Optional[str] = ""
    point_cloud_conf_thres: float = DEFAULT_POINT_CLOUD_CONF_THRES
    point_cloud_edge_rtol: float = DEFAULT_POINT_CLOUD_EDGE_RTL
    point_cloud_skip_edge: bool = True


def _sanitize_point_cloud_filename(name: str) -> str:
    cleaned = os.path.basename((name or "").strip())
    if not cleaned:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        cleaned = f"cloud_{ts}.ply"
    else:
        cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
        if not cleaned.lower().endswith(".ply"):
            cleaned = f"{cleaned}.ply"
    return cleaned


def _save_point_cloud(res: dict, imgs: torch.Tensor, req: DecodeReq) -> tuple[str, int]:
    conf_thres = max(float(req.point_cloud_conf_thres), 0.0)
    edge_rtol = max(float(req.point_cloud_edge_rtol), 0.0)

    masks = torch.sigmoid(res["conf"][..., 0]) > conf_thres
    if req.point_cloud_skip_edge:
        non_edge = ~depth_edge(res["local_points"][..., 2], rtol=edge_rtol)
        masks = torch.logical_and(masks, non_edge)
    masks = masks[0]

    points_count = int(masks.sum().item())
    if points_count <= 0:
        raise HTTPException(status_code=422, detail="point cloud mask is empty after filtering")

    os.makedirs(DEFAULT_POINT_CLOUD_DIR, exist_ok=True)
    filename = _sanitize_point_cloud_filename(req.point_cloud_filename or "")
    save_path = os.path.abspath(os.path.join(DEFAULT_POINT_CLOUD_DIR, filename))
    if os.path.commonpath([DEFAULT_POINT_CLOUD_DIR, save_path]) != DEFAULT_POINT_CLOUD_DIR:
        raise HTTPException(status_code=400, detail="invalid point_cloud_filename")

    write_ply(
        res["points"][0][masks].detach().cpu(),
        imgs.permute(0, 2, 3, 1)[masks].detach().cpu(),
        save_path,
    )
    return save_path, points_count


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
    xyz = camera_poses[:, :3, 3]
    xyz = xyz - xyz[0]

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


def _validate_auth(authorization: Optional[str]) -> None:
    if not AUTH_TOKEN:
        return
    if authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "device": str(DEVICE),
        "dtype": str(APP_DTYPE),
        "auth_enabled": bool(AUTH_TOKEN),
        "point_cloud_dir": DEFAULT_POINT_CLOUD_DIR,
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

    waypoints, yaws, scale_hint = _poses_to_waypoints(camera_poses, req.target_waypoints)

    point_cloud_path = None
    point_cloud_points = None
    if req.save_point_cloud:
        point_cloud_path, point_cloud_points = _save_point_cloud(res, imgs, req)

    full_resp = {
        "waypoints_norm": waypoints,
        "yaw_deg": yaws,
        "camera_poses": camera_poses.tolist(),
        "max_velocity": req.max_velocity,
        "max_acceleration": req.max_acceleration,
        "reason": "pi3_remote_waypoints",
        "scale_hint": scale_hint,
        "meta": {
            "num_frames": int(imgs.shape[0]),
            "resize_h": int(imgs.shape[2]),
            "resize_w": int(imgs.shape[3]),
        },
        "point_cloud_path": point_cloud_path,
        "point_cloud_points": point_cloud_points,
    }

    if req.return_fields:
        allowed = set(req.return_fields) | {"reason"}
        return {k: v for k, v in full_resp.items() if k in allowed}
    return full_resp
