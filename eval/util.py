# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for frame interpolation on a set of video frames."""
import os
import shutil
from typing import Generator, Iterable, List

import interpolator as interpolator_lib
import util
import numpy as np
import tensorflow as tf

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)
_CONFIG_FFMPEG_NAME_OR_PATH = 'ffmpeg'


def read_image(filename: str) -> np.ndarray:
  """Reads an sRgb 8-bit image.

  Args:
    filename: The input filename to read.

  Returns:
    A float32 3-channel (RGB) ndarray with colors in the [0..1] range.
  """
  image_data = tf.io.read_file(filename)
  image = tf.io.decode_image(image_data, channels=3)
  image_numpy = tf.cast(image, dtype=tf.float32).numpy()
  return image_numpy / _UINT8_MAX_F


def write_image(filename: str, image: np.ndarray) -> None:
  """Writes a float32 3-channel RGB ndarray image, with colors in range [0..1].

  Args:
    filename: The output filename to save.
    image: A float32 3-channel (RGB) ndarray with colors in the [0..1] range.
  """
  image_in_uint8_range = np.clip(image * _UINT8_MAX_F, 0.0, _UINT8_MAX_F)
  image_in_uint8 = (image_in_uint8_range + 0.5).astype(np.uint8)

  extension = os.path.splitext(filename)[1]
  if extension == '.jpg':
    image_data = tf.io.encode_jpeg(image_in_uint8)
  else:
    image_data = tf.io.encode_png(image_in_uint8)
  tf.io.write_file(filename, image_data)


def image_to_patches(image: np.ndarray, block_shape: List[int], _pad: List[int]) -> np.ndarray:
  """Folds an image into patches and stacks along the batch dimension.

  Args:
    image: The input image of shape [B, H, W, C].
    block_shape: The number of patches along the height and width to extract.
      Each patch is shaped (H/block_shape[0], W/block_shape[1])

  Returns:
    The extracted patches shaped [num_blocks, patch_height, patch_width,...],
      with num_blocks = block_shape[0] * block_shape[1].
  """
  
  paddings = 2 * [[int(i) for i in _pad]]
  
  block_height, block_width = [int(i) for i in block_shape]
  num_blocks = block_height * block_width

  height, width, channel = image.shape[-3:]
  patch_height, patch_width = height//block_height, width//block_width
 
  assert height == (
      patch_height * block_height
  ), 'block_height=%d should evenly divide height=%d.'%(block_height, height)
  assert width == (
      patch_width * block_width
  ), 'block_width=%d should evenly divide width=%d.'%(block_width, width)

  patch_size = patch_height * patch_width

  patches = tf.space_to_batch(image, [patch_height, patch_width], paddings)
  patches = tf.split(patches, patch_size, 0)
  patches = tf.stack(patches, axis=3)
  patches = tf.reshape(patches,
                       [num_blocks, patch_height, patch_width, channel])
  return patches.numpy()


def patches_to_image(patches: np.ndarray, block_shape: List[int], _pad : List[int]) -> np.ndarray:
  """Unfolds patches (stacked along batch) into an image.

  Args:
    patches: The input patches, shaped [num_patches, patch_H, patch_W, C].
    block_shape: The number of patches along the height and width to unfold.
      Each patch assumed to be shaped (H/block_shape[0], W/block_shape[1]).

  Returns:
    The unfolded image shaped [B, H, W, C].
  """
  paddings = 2 * [[int(i) for i in _pad]]
  
  block_height, block_width = [int(i) for i in block_shape]
  

  patch_height, patch_width, channel = patches.shape[-3:]
  patch_size = patch_height * patch_width

  patches = tf.reshape(patches,
                       [1, block_height, block_width, patch_size, channel])
  patches = tf.split(patches, patch_size, axis=3)
  patches = tf.stack(patches, axis=0)
  patches = tf.reshape(patches,
                       [patch_size, block_height, block_width, channel])
  image = tf.batch_to_space(patches, [patch_height, patch_width], paddings)
  return image.numpy()


def _recursive_generator(
    frame1: np.ndarray, frame2: np.ndarray, num_recursions: int,
    interpolator: interpolator_lib.Interpolator, pad: List[int]
) -> Generator[np.ndarray, None, None]:
  """Splits halfway to repeatedly generate more frames.

  Args:
    frame1: Input image 1.
    frame2: Input image 2.
    num_recursions: How many times to interpolate the consecutive image pairs.
    interpolator: The frame interpolator instance.

  Yields:
    The interpolated frames, including the first frame (frame1), but excluding
    the final frame2.
  """
  if num_recursions == 0:
    yield frame1
  else:
    # Adds the batch dimension to all inputs before calling the interpolator,
    # and remove it afterwards.
    time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
    mid_frame = interpolator.interpolate(
        np.expand_dims(frame1, axis=0), np.expand_dims(frame2, axis=0), time)[0]
    yield from _recursive_generator(frame1, mid_frame, num_recursions - 1,
                                    interpolator,pad)
    yield from _recursive_generator(mid_frame, frame2, num_recursions - 1,
                                    interpolator,pad)

def _recursive_generator_split(
    frame1: np.ndarray, frame2: np.ndarray, num_recursions: int,
    interpolator: interpolator_lib.Interpolator, blockshape: List[int], pad: List[int]
) -> Generator[np.ndarray, None, None]:
  """Splits halfway to repeatedly generate more frames.

  Args:
    frame1: Input image 1.
    frame2: Input image 2.
    num_recursions: How many times to interpolate the consecutive image pairs.
    interpolator: The frame interpolator instance.

  Yields:
    The interpolated frames, including the first frame (frame1), but excluding
    the final frame2.
  """
  if num_recursions == 0:
    yield frame1
  else:
    # Adds the batch dimension to all inputs before calling the interpolator,
    # and remove it afterwards.
    time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
    ##
    patches__1 = np.expand_dims(frame1, axis=0)
    patches_1 = util.image_to_patches(patches__1, blockshape,pad)
    patches__2 = np.expand_dims(frame2, axis=0)
    patches_2 = util.image_to_patches(patches__2, blockshape,pad)
    output_patches = []
    for image_1, image_2 in zip(patches_1, patches_2):
      image_batch_1 = np.expand_dims(image_1, axis=0)
      image_batch_2 = np.expand_dims(image_2, axis=0)
      mid_patch = interpolator.interpolate(image_batch_1, image_batch_2, time)
      output_patches.append(mid_patch)
    output_patches = np.concatenate(output_patches, axis=0)
    mid_frame = util.patches_to_image(output_patches, blockshape,pad)[0]
    ##
    yield from _recursive_generator_split(frame1, mid_frame, num_recursions - 1,
                                    interpolator,blockshape,pad)
    yield from _recursive_generator_split(mid_frame, frame2, num_recursions - 1,
                                    interpolator,blockshape,pad)


def interpolate_recursively_from_files(
    frames: List[str], times_to_interpolate: int,
    interpolator: interpolator_lib.Interpolator, block=[0,0], pad=[0,0]) -> Iterable[np.ndarray]:
  """Generates interpolated frames by repeatedly interpolating the midpoint.

  Loads the files on demand and uses the yield paradigm to return the frames
  to allow streamed processing of longer videos.

  Recursive interpolation is useful if the interpolator is trained to predict
  frames at midpoint only and is thus expected to perform poorly elsewhere.

  Args:
    frames: List of input frames. Expected shape (H, W, 3). The colors should be
      in the range[0, 1] and in gamma space.
    times_to_interpolate: Number of times to do recursive midpoint
      interpolation.
    interpolator: The frame interpolation model to use.

  Yields:
    The interpolated frames (including the inputs).
  """
  n = len(frames)
  for i in range(1, n):
    if block==[0,0] or block==[1,1]:
      yield from _recursive_generator(
          util.read_image(frames[i - 1]), util.read_image(frames[i]),
          times_to_interpolate, interpolator,block,pad)
    else:
      yield from _recursive_generator_split(
          util.read_image(frames[i - 1]), util.read_image(frames[i]),
          times_to_interpolate, interpolator,block,pad)
  # Separately yield the final frame.
  yield util.read_image(frames[-1])


def interpolate_recursively_from_memory(
    frames: List[np.ndarray], times_to_interpolate: int,
    interpolator: interpolator_lib.Interpolator) -> Iterable[np.ndarray]:
  """Generates interpolated frames by repeatedly interpolating the midpoint.

  This is functionally equivalent to interpolate_recursively_from_files(), but
  expects the inputs frames in memory, instead of loading them on demand.

  Recursive interpolation is useful if the interpolator is trained to predict
  frames at midpoint only and is thus expected to perform poorly elsewhere.

  Args:
    frames: List of input frames. Expected shape (H, W, 3). The colors should be
      in the range[0, 1] and in gamma space.
    times_to_interpolate: Number of times to do recursive midpoint
      interpolation.
    interpolator: The frame interpolation model to use.

  Yields:
    The interpolated frames (including the inputs).
  """
  n = len(frames)
  for i in range(1, n):
    yield from _recursive_generator(
        frames[i - 1], frames[i],
        times_to_interpolate, interpolator)
  # Separately yield the final frame.
  yield frames[-1]


def get_ffmpeg_path() -> str:
  path = shutil.which(_CONFIG_FFMPEG_NAME_OR_PATH)
  if not path:
    raise RuntimeError(
        f"Program '{_CONFIG_FFMPEG_NAME_OR_PATH}' is not found;"
        " perhaps install ffmpeg using 'apt-get install ffmpeg'.")
  return path
