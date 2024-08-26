"""
This is the custom dataset generator module for nerfstudio.

usage:
    python custom.py <frames_dir> <out_dir>
    e.g:
        python custom.py ../../GPS-Gaussian/data/render_data/train/img/0004_000/ ./test_data

You need to prepare ...
    1. A set of images from the same scene, which sould try to cover the whole scene for better performance.
        * should locate in the same directory
    2. Your customed framename2intr and framename2extr functions.
        * implement framename2intr and framename2extr functions
"""

import os
import sys
import numpy as np
import json
from PIL import Image
from scipy.spatial.transform import Rotation as scipy_rot

import logging
logging.basicConfig(level=logging.DEBUG)

class shape_check:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            assert res.shape == self.shape, f"Invalid shape at func {func.__name__} : {res.shape}"
            return res
        return wrapper


# Customed framename2intr and framename2extr functions (should be implemented by the user)
parm_path = "../../GPS-Gaussian/data/real_data/parm/0001/"
rot_mat = scipy_rot.from_rotvec(np.pi * np.array([0, 1, 0])).as_matrix()
@shape_check(shape=(3, 3))
def framename2intr(framename) -> np.ndarray:
    """
    Return intrinsic matrix in the shape of (3, 3).
    """
    # raise NotImplementedError
    framename = framename.split(".")[0]
    intr =  np.load(os.path.join(parm_path, f"{framename}_intrinsic.npy"))
    return intr

@shape_check(shape=(4, 4))
def framename2extr(framename) -> np.ndarray:
    """
    Return extrinsic matrix in the shape of (4, 4).
    """
    # raise NotImplementedError
    framename = framename.split(".")[0]
    extr = np.load(os.path.join(parm_path, f"{framename}_extrinsic.npy"))

    return np.stack([extr[0], extr[1], extr[2], [0, 0, 0, 1]], axis=0)
# End of customed framename2intr and framename2extr functions


def read_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    image = np.array(image)
    return image


def intr2dict(intr: np.ndarray) -> dict:
    intr_dict = {
        "fl_x": intr[0, 0],
        "fl_y": intr[1, 1],
        "cx": intr[0, 2],
        "cy": intr[1, 2],
    }
    return intr_dict


def extr2list(extr: np.ndarray) -> list:
    # flip y and z axis in the camera coordinate space, according to
    # [ns-data conventions](https://docs.nerf.studio/quickstart/data_conventions.html#camera-view-space)
    flip_mat = np.eye(4)
    flip_mat[1, 1] = -1
    flip_mat[2, 2] = -1
    extr = flip_mat @ extr

    tf_mat = np.linalg.inv(extr) # note that the transform matrix in ns should be the inverse of the extrinsic matrix
    
    return tf_mat.tolist()


def main():
    frames_dir = sys.argv[1]
    out_dir = sys.argv[2]

    if not os.path.exists(out_dir):
        logging.warning(f"{out_dir} not exists, creating ...")
        os.makedirs(out_dir)

    framenames = [name for name in os.listdir(frames_dir) if name.find("hr") == -1]
    logging.info(f"Found {len(framenames)} frames in {frames_dir}")

    h, w = read_image(os.path.join(frames_dir, framenames[0])).shape[:2]

    transforms_dict = {
        "w": w,
        "h": h,
        "k1": 0, # radical distortion param, set to 0 since it is unknown
        "k2": 0,
        "p1": 0, # tangential distortion param, set to 0 since it is unknown
        "p2": 0,
        "camera_model": "OPENCV",
    }

    frames = []
    for framename in framenames:
        frame_dict = {
            "file_path": f"images/{framename}",
            "transform_matrix": extr2list(framename2extr(framename)),
        }
        intr_dict = intr2dict(framename2intr(framename))
        frame_dict.update(intr_dict)
        frames.append(frame_dict)
    
    transforms_dict["frames"] = frames

    with open(os.path.join(out_dir, "transforms.json"), "w") as f:
        json.dump(transforms_dict, f, indent=4)

    logging.info("Copying images ...")
    os.system(f"rm -rf {os.path.join(out_dir, 'images')}") if os.path.exists(os.path.join(out_dir, "images")) else None
    os.system(f"cp -r {frames_dir} {os.path.join(out_dir, 'images')}")


if __name__ == "__main__":
    main()