# Dataset

`open/` (1,500 images) and `closed/` (1,374 images): 34x26 grayscale crops of single eyes
(label 1 = open, 0 = closed). They come from the public `kairess/eye_blink_detector`
repository (https://github.com/kairess/eye_blink_detector, `dataset/` folder) and were
converted from its `.npy` arrays to PNG files. That repository states no licence, so cite it as
the source in your report.

## Using a bigger dataset
Put any open/closed eye images into `open/` and `closed/` (any size - they are resized to 34x26)
and run `python train.py` again. A larger option is the **MRL Eye Dataset**
(http://mrl.cs.vsb.cz/eyedataset, ~85k images):

    python prepare_mrl.py --src path/to/mrlEyes_2018_01 --max-per-class 15000 --clear
    python train.py
