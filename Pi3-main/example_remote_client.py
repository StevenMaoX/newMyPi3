import argparse
import base64
import io
import json
import os
from typing import List

import requests
from PIL import Image


def encode_images_to_b64(image_dir: str, max_frames: int = 0, interval: int = 1) -> List[str]:
    names = sorted(
        n for n in os.listdir(image_dir) if n.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    names = names[:: max(interval, 1)]
    if max_frames > 0:
        names = names[:max_frames]

    out = []
    for name in names:
        path = os.path.join(image_dir, name)
        with Image.open(path).convert("RGB") as img:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return out


def main():
    parser = argparse.ArgumentParser(description="Call remote Pi3 /decode API")
    parser.add_argument("--endpoint", type=str, required=True, help="http://host:8000/decode")
    parser.add_argument("--image_dir", type=str, default="", help="Directory with input frames")
    parser.add_argument("--api_key", type=str, default="", help="Optional bearer token")
    parser.add_argument("--target_waypoints", type=int, default=6)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument(
        "--return_fields",
        type=str,
        default="waypoints_norm,yaw_deg,camera_poses",
        help="Comma-separated fields",
    )
    args = parser.parse_args()

    payload = {
        "instruction": "remote_decode",
        "frames_b64": [],
        "target_waypoints": args.target_waypoints,
        "return_fields": [x.strip() for x in args.return_fields.split(",") if x.strip()],
    }

    if args.image_dir:
        payload["frames_b64"] = encode_images_to_b64(
            args.image_dir, max_frames=args.max_frames, interval=args.interval
        )

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    resp = requests.post(args.endpoint, headers=headers, json=payload, timeout=120)
    print("status:", resp.status_code)
    try:
        print(json.dumps(resp.json(), ensure_ascii=False, indent=2)[:4000])
    except Exception:
        print(resp.text[:4000])


if __name__ == "__main__":
    main()

