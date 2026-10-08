import numpy as np
from PIL import Image

from cardiomoe.data.pk_images import letterbox
from cardiomoe.data.render_ecg import IMG_H, IMG_W, PhotoAugment, render


def test_render_size_and_ink():
    sig = np.sin(np.linspace(0, 80, 1000))[None].repeat(12, 0).astype(np.float32)
    img = render(sig)
    assert img.size == (IMG_W, IMG_H) and img.mode == "RGB"
    a = np.asarray(img)
    assert (a.sum(-1) < 200).mean() > 0.005  # dark trace pixels exist


def test_photo_augment_keeps_size():
    img = render(np.zeros((12, 1000), np.float32))
    assert PhotoAugment(seed=0)(img).size == (IMG_W, IMG_H)


def test_letterbox():
    out = letterbox(Image.new("L", (2213, 1572), 128))
    assert out.size == (IMG_W, IMG_H) and out.mode == "RGB"
