# This is the custom dataset generator module for nerfstudio.

## usage:
```
python custom.py <frames_dir> <out_dir>
e.g:
    python custom.py ../../GPS-Gaussian/data/render_data/train/img/0004_000/ ./test_data
```

## You need to prepare ...

1. A set of images from the same scene, which sould try to cover the whole scene for better performance.
    * Should be located in the same directory
2. Your customed framename2intr and framename2extr functions.
    * Implement framename2intr and framename2extr functions to get the intrinsic and extrinsic parameters of each frame.