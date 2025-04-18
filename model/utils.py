# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import os
import glob

import imageio
import numpy as np
from PIL import Image
from io import BytesIO


def pad_seq(seq, batch_size):
    # pad the sequence to be the multiples of batch_size
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq


def bytes_to_file(bytes_img):
    return BytesIO(bytes_img)


def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized


def read_split_image(img):
    # Replace deprecated misc.imread with imageio.imread
    mat = imageio.imread(img).astype(np.float)
    side = int(mat.shape[1] / 2)
    assert side * 2 == mat.shape[1]
    img_A = mat[:, :side]  # target
    img_B = mat[:, side:]  # source

    return img_A, img_B


def shift_and_resize_image(img, shift_x, shift_y, nw, nh):
    w, h, _ = img.shape
    # Already using PIL for resizing, which is correct
    enlarged = np.array(Image.fromarray(np.uint8(img)).resize(size=(nw, nh)))
    return enlarged[shift_x:shift_x + w, shift_y:shift_y + h]


def scale_back(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def save_concat_images(imgs, img_path):
    concated = np.concatenate(imgs, axis=1)
    imageio.imwrite(img_path, concated)


def compile_frames_to_gif(frame_dir, gif_file):
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    print(frames)
    # Replace deprecated misc.imresize with PIL resize
    images = []
    for f in frames:
        img = imageio.imread(f)
        # Calculate new dimensions (33% of original size)
        new_size = (int(img.shape[1] * 0.33), int(img.shape[0] * 0.33))
        # Resize using PIL and convert back to numpy array
        resized_img = np.array(Image.fromarray(img).resize(new_size, Image.NEAREST))
        images.append(resized_img)
    
    imageio.mimsave(gif_file, images, duration=0.1)
    return gif_file
