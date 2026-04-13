# Pi3 Remote Service Integration

This repo now includes a ready-to-run server entrypoint:

- `pi3_remote_server.py`

It exposes:

- `GET /healthz`
- `POST /decode`

and follows the API contract expected by `navdreamer.pi3.endpoint`.

## 1) Deploy on Pi3 server machine

Install dependencies:

```bash
pip install -r requirements_server.txt
```

Optional environment variables:

```bash
# optional auth token (if set, client must send Bearer token)
export PI3_AUTH_TOKEN=YOUR_OPTIONAL_TOKEN

# optional: force device (default auto: cuda if available else cpu)
export PI3_DEVICE=cuda

# optional: custom local checkpoint
export PI3_CKPT=/path/to/model.safetensors

# optional: default point cloud output directory (default: Pi3-main/cloudpoints)
export PI3_POINT_CLOUD_DIR=/path/to/Pi3-main/cloudpoints

# optional: always save point cloud for every /decode request (default: 0)
export PI3_DEFAULT_SAVE_POINT_CLOUD=1
```

Start server:

```bash
uvicorn pi3_remote_server:app --host 0.0.0.0 --port 8000
```

## 2) Client-side config

Set `src/airsim/config.json`:

```json
{
  "navdreamer": {
    "pi3": {
      "enabled": true,
      "endpoint": "http://YOUR_SERVER_IP:8000/decode",
      "api_key": "YOUR_OPTIONAL_TOKEN",
      "timeout_sec": 120.0,
      "default_waypoints": 6,
      "default_max_velocity": 2.2,
      "default_max_acceleration": 2.5,
      "remote_accept_raw_pose": true,
      "pose_stride": 1,
      "pose_min_translation": 0.02,
      "normalize_pose_waypoints": true,
      "max_waypoints_from_pose": 8
    }
  }
}
```

## 3) /decode request contract

Client sends:

```json
{
  "instruction": "fly to target",
  "video_path": "optional path on server",
  "frames_b64": ["..."],
  "target_waypoints": 6,
  "return_fields": ["waypoints_norm", "yaw_deg", "camera_poses"],
  "max_velocity": 2.2,
  "max_acceleration": 2.5,
  "frame_interval": 1,
  "max_frames": 0,
  "save_point_cloud": true,
  "point_cloud_filename": "run_001.ply",
  "point_cloud_conf_thres": 0.1,
  "point_cloud_edge_rtol": 0.03,
  "point_cloud_skip_edge": true
}
```

Notes:

- Prefer `frames_b64` for remote callers.
- `video_path` is only for files directly readable by the server machine.
- At least 2 frames are required.

## 4) /decode response contract

Preferred (direct waypoints):

```json
{
  "waypoints_norm": [[0, 0, 0], [0.3, 0.0, 0.0], [1.0, 0.1, 0.0]],
  "yaw_deg": [0, 2, 4],
  "point_cloud_path": "/abs/path/to/Pi3-main/cloudpoints/run_001.ply",
  "point_cloud_points": 248397,
  "max_velocity": 2.2,
  "max_acceleration": 2.5,
  "reason": "pi3_remote_waypoints",
  "scale_hint": 4.5
}
```

Raw camera poses mode (when `return_fields` only asks for poses):

```json
{
  "camera_poses": [[[...4x4...]], [[...4x4...]]],
  "reason": "pi3_remote_waypoints"
}
```

`reason` is always returned.

## 5) Quick validation

Health check:

```bash
curl http://YOUR_SERVER_IP:8000/healthz
```

Decode check (`frames_b64` omitted in this example, using server-local video):

```bash
curl -X POST http://YOUR_SERVER_IP:8000/decode \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_OPTIONAL_TOKEN" \
  -d '{
    "video_path": "examples/skating.mp4",
    "target_waypoints": 6,
    "return_fields": ["waypoints_norm", "yaw_deg", "camera_poses"]
  }'
```

Or call with local images from another project:

```bash
pip install requests
python example_remote_client.py \
  --endpoint http://YOUR_SERVER_IP:8000/decode \
  --api_key YOUR_OPTIONAL_TOKEN \
  --image_dir /path/to/frames \
  --save_point_cloud \
  --point_cloud_filename run_001.ply \
  --target_waypoints 6
```

## 6) Behavior details

- Input frames are resized to a Pi3-compatible shape (multiple of 14, capped by pixel budget).
- Server runs Pi3 inference and reads `camera_poses`.
- `camera_poses` are converted to normalized waypoints by translation track:
  - subtract first pose translation
  - resample to `target_waypoints`
  - normalize by final displacement norm
- If `save_point_cloud=true`, server saves `.ply` to `Pi3-main/cloudpoints` by default.
- If your client can consume raw `camera_poses`, keep `remote_accept_raw_pose=true`.

## 7) Push cloudpoints to GitHub and download locally

From `Pi3-main` on server:

```bash
mkdir -p cloudpoints
git add cloudpoints
git commit -m "Add generated point clouds"
git push origin <your-branch>
```

Then on local machine:

```bash
git pull origin <your-branch>
```
