"""Render labeled MuJoCo views and train a one-keypoint YOLO pose model."""
import argparse
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np

from .piece_pose import RECT_SIZE, WEIGHTS, SimPieceLayout
from .simulation import Camera, Simulation, look_at
from .vision import rotation_about


def generate_dataset(output, train_count, val_count, seed):
    sim = Simulation.from_files()
    layout = SimPieceLayout(sim.model)
    plane = layout.plane
    arm = sim.data.joint("joint_1").qpos
    try:
        for split, count, split_seed in (("train", train_count, seed), ("val", val_count, seed + 1)):
            rng = np.random.default_rng(split_seed)
            images = output / "images" / split
            labels = output / "labels" / split
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            # Replace only our generated samples if the dataset size changed.
            for directory, suffix in ((images, "jpg"), (labels, "txt")):
                for old in directory.glob(f"view_*.{suffix}"):
                    old.unlink()
            for index in range(count):
                if index % 6 == 0:
                    xyz = layout.initial
                    visible = np.ones(len(layout.ids), dtype=bool)
                    layout.place(xyz, visible)
                else:
                    n = int(rng.integers(0, 33)) if index % 5 == 0 else int(rng.integers(19, 33))
                    xyz, visible = layout.randomize(rng, n)
                arm[0] = rng.uniform(-0.85, -0.30)
                if index == 0:
                    camera = Camera.initial(sim.view_center)
                else:
                    azimuth = rng.uniform(-np.pi, np.pi)
                    slope = rng.uniform(0.0, 0.82)
                    offset = rng.uniform(0.67, 1.95) * np.array(
                        [slope * np.cos(azimuth), slope * np.sin(azimuth), 1])
                    target = np.array([0.20, 0.20, 0.04]) + rng.uniform(-0.10, 0.10, 3)
                    rotation = look_at(offset, up=(0, 1, 0)) @ rotation_about(
                        [0, 0, 1], rng.uniform(-0.65, 0.65))
                    camera = Camera(target + offset, rotation)
                rgb, truth = sim.render_camera(camera)
                h = plane.image_to_rect(rgb.shape, sim.intrinsics, truth)
                if h is None:
                    continue
                sim.renderer.enable_segmentation_rendering()
                ids = sim.renderer.render()[:, :, 0]
                sim.renderer.disable_segmentation_rendering()
                rect = cv2.warpPerspective(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), h,
                                           (RECT_SIZE, RECT_SIZE))
                mask = cv2.warpPerspective(ids.astype(np.float32), h, (RECT_SIZE, RECT_SIZE),
                                           flags=cv2.INTER_NEAREST, borderValue=-1)
                keypoints = plane.board_to_rect(xyz)
                rows = []
                for geom_id, enabled, point in zip(layout.ids, visible, keypoints):
                    if not enabled or np.any(point < 0) or np.any(point >= RECT_SIZE):
                        continue
                    piece_mask = np.uint8(mask == geom_id)
                    if cv2.countNonZero(piece_mask) < 20:
                        continue
                    x, y, w, height = cv2.boundingRect(piece_mask)
                    values = np.array([x + w / 2, y + height / 2, w, height, *point]) / RECT_SIZE
                    rows.append("0 " + " ".join(f"{v:.6f}" for v in values) + " 2")
                stem = f"view_{index:05d}"
                if not cv2.imwrite(str(images / f"{stem}.jpg"), rect):
                    raise RuntimeError(f"Could not write {stem}")
                (labels / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            print(f"Rendered {split}: {len(list(images.glob('view_*.jpg')))} images", flush=True)
    finally:
        sim.renderer.close()
    config = output / "pieces.yaml"
    config.write_text(
        f"path: {json.dumps(str(output))}\ntrain: images/train\nval: images/val\n"
        "kpt_shape: [1, 3]\nflip_idx: [0]\nnames: {0: piece}\n"
    )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path(".piece-training"))
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--train-count", type=int, default=640)
    parser.add_argument("--val-count", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--reuse-dataset", action="store_true")
    args = parser.parse_args()
    output = args.work.resolve()
    config = output / "pieces.yaml"
    if not args.reuse_dataset:
        config = generate_dataset(output, args.train_count, args.val_count, args.seed)

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(min(6, torch.get_num_threads()))
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO("yolo11n-pose.pt")
    model.train(data=str(config), imgsz=RECT_SIZE, epochs=args.epochs, batch=args.batch,
                device=device, workers=0, project=str(output), name="fit", exist_ok=True,
                mosaic=0.0, translate=0.04, scale=0.12, degrees=6, flipud=0.5,
                hsv_h=0.02, hsv_s=0.25, hsv_v=0.22, plots=False, seed=args.seed,
                amp=False, lr0=0.002, optimizer="AdamW", warmup_epochs=1.0)
    checkpoint = Path(model.trainer.best)
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, args.weights)
    print(f"Saved YOLO piece weights: {args.weights}", flush=True)


if __name__ == "__main__":
    main()
