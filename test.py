import numpy as np

def qvec2rotmat(qvec):
    q0, q1, q2, q3 = qvec
    return np.array([
        [1 - 2*q2*q2 - 2*q3*q3,     2*q1*q2 - 2*q0*q3,     2*q3*q1 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3,         1 - 2*q1*q1 - 2*q3*q3, 2*q2*q3 - 2*q0*q1],
        [2*q3*q1 - 2*q0*q2,         2*q2*q3 + 2*q0*q1,     1 - 2*q1*q1 - 2*q2*q2],
    ])

scene_dir = "outputs/20260504_084740"

# pred.txt is c2w
pred_txt = np.loadtxt(f"{scene_dir}/pred.txt").reshape(-1, 3, 4)
centers_from_pred_txt = pred_txt[:, :3, 3]

# images.txt is COLMAP w2c
centers_from_colmap = []

with open(f"{scene_dir}/sparse/0/images.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# images.txt has two lines per image. In our export, second line is empty,
# so after stripping empty lines, only pose lines remain.
for line in lines:
    elems = line.split()
    if len(elems) < 10:
        continue

    qvec = np.array(list(map(float, elems[1:5])))
    tvec = np.array(list(map(float, elems[5:8])))

    R_w2c = qvec2rotmat(qvec)
    t_w2c = tvec

    # COLMAP camera center:
    # C = -R^T t
    C = -R_w2c.T @ t_w2c
    centers_from_colmap.append(C)

centers_from_colmap = np.stack(centers_from_colmap)

diff = np.linalg.norm(centers_from_pred_txt - centers_from_colmap, axis=1)

print("num poses:", len(diff))
print("max center diff:", diff.max())
print("mean center diff:", diff.mean())
print("first pred center:", centers_from_pred_txt[0])
print("first colmap center:", centers_from_colmap[0])
