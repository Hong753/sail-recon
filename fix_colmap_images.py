import os
import shutil
import subprocess
import numpy as np


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat

    K = np.array(
        [
            [Rxx - Ryy - Rzz, 0.0, 0.0, 0.0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0.0, 0.0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0.0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
        ],
        dtype=float,
    ) / 3.0

    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]

    if qvec[0] < 0:
        qvec *= -1

    return qvec


scene_dir = "outputs/20260504_084740"
sparse_txt_dir = os.path.join(scene_dir, "_sparse_txt_tmp")
sparse_dir = os.path.join(scene_dir, "sparse", "0")

pred_path = os.path.join(scene_dir, "pred.txt")
old_images_path = os.path.join(sparse_txt_dir, "images.txt")
new_images_path = os.path.join(sparse_txt_dir, "images.txt")

# pred.txt from demo.py is c2w, flattened 3x4
poses_c2w = np.loadtxt(pred_path).reshape(-1, 3, 4)

# Get image names and camera ids from the existing images.txt
entries = []
with open(old_images_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        elems = line.split()
        if len(elems) >= 10:
            image_id = int(elems[0])
            camera_id = int(elems[8])
            image_name = elems[9]
            entries.append((image_id, camera_id, image_name))

assert len(entries) == len(poses_c2w), (len(entries), len(poses_c2w))

with open(new_images_path, "w") as f:
    f.write("# Image list with two lines of data per image:\n")
    f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
    f.write(f"# Number of images: {len(entries)}, mean observations per image: 0\n")

    for (image_id, camera_id, image_name), T_c2w_3x4 in zip(entries, poses_c2w):
        R_c2w = T_c2w_3x4[:3, :3]
        C = T_c2w_3x4[:3, 3]

        # Convert c2w -> COLMAP world-to-camera
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ C

        qw, qx, qy, qz = rotmat2qvec(R_w2c)
        tx, ty, tz = t_w2c.tolist()

        f.write(
            f"{image_id} "
            f"{qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} "
            f"{tx:.12f} {ty:.12f} {tz:.12f} "
            f"{camera_id} {image_name}\n"
        )
        f.write("\n")

os.makedirs(sparse_dir, exist_ok=True)

# Keep txt files in sparse/0 too
for name in ["cameras.txt", "images.txt", "points3D.txt"]:
    shutil.copy2(
        os.path.join(sparse_txt_dir, name),
        os.path.join(sparse_dir, name),
    )

# Remove old/wrong bin files if any
for name in ["cameras.bin", "images.bin", "points3D.bin"]:
    path = os.path.join(sparse_dir, name)
    if os.path.exists(path):
        os.remove(path)

subprocess.run(
    [
        "colmap",
        "model_converter",
        "--input_path",
        sparse_txt_dir,
        "--output_path",
        sparse_dir,
        "--output_type",
        "BIN",
    ],
    check=True,
)

print("Fixed COLMAP txt/bin model at:", sparse_dir)