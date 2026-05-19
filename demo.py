# Copyright (c) HKUST SAIL-Lab and Horizon Robotics.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import shutil
import subprocess

import argparse
import os

import torch
from tqdm import tqdm

from eval.utils.device import to_cpu
from eval.utils.eval_utils import uniform_sample
from sailrecon.models.sail_recon import SailRecon
from sailrecon.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

def rotmat2qvec(R):
    """
    Convert rotation matrix to COLMAP Hamilton quaternion:
    qvec = [qw, qx, qy, qz]
    """
    import numpy as np

    K = np.array(
        [
            [R[0, 0] - R[1, 1] - R[2, 2], 0.0, 0.0, 0.0],
            [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0.0, 0.0],
            [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0.0],
            [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]],
        ],
        dtype=float,
    ) / 3.0

    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]

    if qvec[0] < 0:
        qvec *= -1

    return qvec


def read_ply_xyz_rgb(ply_path, max_points=None):
    """
    Read XYZ/RGB from pred.ply so points3D.txt/bin is non-empty.
    Requires:
        pip install plyfile
    """
    import numpy as np

    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `plyfile`. Install it with: pip install plyfile"
        ) from exc

    ply = PlyData.read(ply_path)
    vertices = ply["vertex"].data
    names = vertices.dtype.names

    xyz = np.stack(
        [vertices["x"], vertices["y"], vertices["z"]],
        axis=1,
    ).astype(np.float64)

    if all(k in names for k in ["red", "green", "blue"]):
        rgb = np.stack(
            [vertices["red"], vertices["green"], vertices["blue"]],
            axis=1,
        ).astype(np.uint8)
    elif all(k in names for k in ["r", "g", "b"]):
        rgb = np.stack(
            [vertices["r"], vertices["g"], vertices["b"]],
            axis=1,
        ).astype(np.uint8)
    else:
        rgb = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)

    valid = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid]
    rgb = rgb[valid]

    if max_points is not None and xyz.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    return xyz, rgb


def save_colmap_sparse_txt(predictions, image_names, images_tensor, pred_ply_path, save_dir, max_points=200000):
    """
    Write COLMAP text model:
        cameras.txt
        images.txt
        points3D.txt

    For 3DGS, PINHOLE is the safest camera model.
    """
    import os
    import numpy as np
    from PIL import Image

    os.makedirs(save_dir, exist_ok=True)

    cameras_txt = os.path.join(save_dir, "cameras.txt")
    images_txt = os.path.join(save_dir, "images.txt")
    points3d_txt = os.path.join(save_dir, "points3D.txt")

    H_model, W_model = images_tensor.shape[-2:]

    # ---------------------------------------------------------------------
    # cameras.txt
    # ---------------------------------------------------------------------
    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(predictions)}\n")

        for i, pred in enumerate(predictions, start=1):
            K = pred["intrinsic"][0].cpu().numpy().copy()

            with Image.open(image_names[i - 1]) as im:
                W_orig, H_orig = im.size

            # Scale intrinsics from model/preprocessed resolution back to original image resolution.
            sx = W_orig / W_model
            sy = H_orig / H_model

            fx = K[0, 0] * sx
            fy = K[1, 1] * sy
            cx = K[0, 2] * sx
            cy = K[1, 2] * sy

            f.write(
                f"{i} PINHOLE {W_orig} {H_orig} "
                f"{fx:.12f} {fy:.12f} {cx:.12f} {cy:.12f}\n"
            )

    # ---------------------------------------------------------------------
    # images.txt
    # ---------------------------------------------------------------------
    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(predictions)}, mean observations per image: 0\n")

        for i, pred in enumerate(predictions, start=1):
            # SAIL-Recon gives world-to-camera extrinsic.
            # COLMAP images.txt also expects world-to-camera pose.
            T_w2c = pred["extrinsic"][0].cpu().numpy()
            R_w2c = T_w2c[:3, :3]
            t_w2c = T_w2c[:3, 3]

            qw, qx, qy, qz = rotmat2qvec(R_w2c)
            tx, ty, tz = t_w2c.tolist()

            image_name = os.path.basename(image_names[i - 1])

            f.write(
                f"{i} {qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} "
                f"{tx:.12f} {ty:.12f} {tz:.12f} {i} {image_name}\n"
            )

            # Empty POINTS2D line.
            # 3DGS does not need COLMAP feature observations.
            f.write("\n")

    # ---------------------------------------------------------------------
    # points3D.txt
    # ---------------------------------------------------------------------
    xyz, rgb = read_ply_xyz_rgb(pred_ply_path, max_points=max_points)

    with open(points3d_txt, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {xyz.shape[0]}, mean track length: 0\n")

        for point_id, (p, c) in enumerate(zip(xyz, rgb), start=1):
            x, y, z = p.tolist()
            r, g, b = c.tolist()

            # Empty track is okay for 3DGS initialization.
            # GraphDeco 3DGS uses XYZ/RGB from points3D.
            f.write(
                f"{point_id} "
                f"{x:.12f} {y:.12f} {z:.12f} "
                f"{int(r)} {int(g)} {int(b)} "
                f"0.0\n"
            )


def export_colmap_sparse_model(predictions, image_names, images_tensor, pred_ply_path, scene_dir, colmap_exe="colmap"):
    """
    Writes both text and binary COLMAP sparse model to:
        scene_dir/sparse/0/
    """
    import os

    sparse_dir = os.path.join(scene_dir, "sparse", "0")
    tmp_txt_dir = os.path.join(scene_dir, "_sparse_txt_tmp")

    os.makedirs(sparse_dir, exist_ok=True)

    if os.path.exists(tmp_txt_dir):
        shutil.rmtree(tmp_txt_dir)
    os.makedirs(tmp_txt_dir, exist_ok=True)

    save_colmap_sparse_txt(
        predictions=predictions,
        image_names=image_names,
        images_tensor=images_tensor,
        pred_ply_path=pred_ply_path,
        save_dir=tmp_txt_dir,
    )

    # Copy text model into sparse/0 so you keep both txt and bin.
    for name in ["cameras.txt", "images.txt", "points3D.txt"]:
        shutil.copy2(
            os.path.join(tmp_txt_dir, name),
            os.path.join(sparse_dir, name),
        )

    # Remove stale binaries so converter cannot leave old files around.
    for name in ["cameras.bin", "images.bin", "points3D.bin"]:
        path = os.path.join(sparse_dir, name)
        if os.path.exists(path):
            os.remove(path)

    # Convert TXT -> BIN using official COLMAP converter.
    subprocess.run(
        [
            colmap_exe,
            "model_converter",
            "--input_path",
            tmp_txt_dir,
            "--output_path",
            sparse_dir,
            "--input_type",
            "TXT",
            "--output_type",
            "BIN",
        ],
        check=True,
    )

    print(f"[COLMAP] Wrote text and binary sparse model to: {sparse_dir}")

def demo(args):
    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    _URL = "https://huggingface.co/HKUST-SAIL/SAIL-Recon/resolve/main/sailrecon.pt"
    model_dir = args.ckpt
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model = SailRecon(kv_cache=True)
    if model_dir is not None:
        model.load_state_dict(torch.load(model_dir))
    else:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(_URL, model_dir=model_dir)
        )
    model = model.to(device=device)
    model.eval()

    # Load and preprocess example images
    scene_name = "1"
    if args.vid_dir is not None:
        import cv2

        image_names = []
        video_path = args.vid_dir
        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        tmp_file = os.path.join("tmp_video", os.path.basename(video_path).split(".")[0])
        os.makedirs(tmp_file, exist_ok=True)
        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            image_path = os.path.join(tmp_file, f"{video_frame_num:06}.png")
            cv2.imwrite(image_path, frame)
            image_names.append(image_path)
            video_frame_num += 1
        images = load_and_preprocess_images(image_names).to(device)
        scene_name = os.path.basename(video_path).split(".")[0]
    else:
        image_names = os.listdir(args.img_dir)
        image_names = [os.path.join(args.img_dir, f) for f in sorted(image_names)]
        images = load_and_preprocess_images(image_names).to(device)
        scene_name = os.path.basename(args.img_dir)

    # anchor image selection
    select_indices = uniform_sample(len(image_names), min(100, len(image_names)))
    anchor_images = images[select_indices]

    os.makedirs(os.path.join(args.out_dir, scene_name), exist_ok=True)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # processing anchor images to build scene representation (kv_cache)
            print("Processing anchor images ...")
            model.tmp_forward(anchor_images)
            # remove the global transformer blocks to save memory during relocalization
            del model.aggregator.global_blocks
            # relocalization on all images
            predictions = []

            with tqdm(total=len(image_names), desc="Relocalizing") as pbar:
                for img_split in images.split(20, dim=0):
                    pbar.update(20)
                    predictions += to_cpu(model.reloc(img_split, memory_save=False))

            # save the predicted point cloud and camera poses

            from eval.utils.geometry import save_pointcloud_with_plyfile

            # save_pointcloud_with_plyfile(
            #     predictions, os.path.join(args.out_dir, scene_name, "pred.ply")
            # )

            # import numpy as np

            # from eval.utils.eval_utils import save_kitti_poses

            # poses_w2c_estimated = [
            #     one_result["extrinsic"][0].cpu().numpy() for one_result in predictions
            # ]
            # poses_c2w_estimated = [
            #     np.linalg.inv(np.vstack([pose, np.array([0, 0, 0, 1])]))
            #     for pose in poses_w2c_estimated
            # ]

            # save_kitti_poses(
            #     poses_c2w_estimated,
            #     os.path.join(args.out_dir, scene_name, "pred.txt"),
            # )
            scene_dir = os.path.join(args.out_dir, scene_name)
            pred_ply_path = os.path.join(scene_dir, "pred.ply")
            
            save_pointcloud_with_plyfile(
                predictions,
                pred_ply_path,
            )
            
            import numpy as np
            
            from eval.utils.eval_utils import save_kitti_poses
            
            poses_w2c_estimated = [
                one_result["extrinsic"][0].cpu().numpy() for one_result in predictions
            ]
            poses_c2w_estimated = [
                np.linalg.inv(np.vstack([pose, np.array([0, 0, 0, 1])]))
                for pose in poses_w2c_estimated
            ]
            
            save_kitti_poses(
                poses_c2w_estimated,
                os.path.join(scene_dir, "pred.txt"),
            )
            
            export_colmap_sparse_model(
                predictions=predictions,
                image_names=image_names,
                images_tensor=images,
                pred_ply_path=pred_ply_path,
                scene_dir=scene_dir,
                colmap_exe=args.colmap_exe,
            )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--img_dir", type=str, default="samples/kitchen", help="input image folder"
    )
    args.add_argument("--vid_dir", type=str, default=None, help="input video path")
    args.add_argument("--out_dir", type=str, default="outputs", help="output folder")
    args.add_argument(
        "--ckpt", type=str, default=None, help="pretrained model checkpoint"
    )
    args.add_argument(
        "--colmap_exe",
        type=str,
        default="colmap",
        help="COLMAP executable path",
    )
    args = args.parse_args()
    demo(args)
