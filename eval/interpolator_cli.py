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
r"""Runs the FILM frame interpolator on a pair of frames on beam.

This script is used evaluate the output quality of the FILM Tensorflow frame
interpolator. Optionally, it outputs a video of the interpolated frames.

A beam pipeline for invoking the frame interpolator on a set of directories
identified by a glob (--pattern). Each directory is expected to contain two
input frames that are the inputs to the frame interpolator. If a directory has
more than two frames, then each contiguous frame pair is treated as input to
generate in-between frames.

The output video is stored to interpolator.mp4 in each directory. The number of
frames is determined by --times_to_interpolate, which controls the number of
times the frame interpolator is invoked. When the number of input frames is 2,
the number of output frames is 2^times_to_interpolate+1.

This expects a directory structure such as:
  <root directory of the eval>/01/frame1.png
                                  frame2.png
  <root directory of the eval>/02/frame1.png
                                  frame2.png
  <root directory of the eval>/03/frame1.png
                                  frame2.png
  ...

And will produce:
  <root directory of the eval>/01/interpolated_frames/frame0.png
                                                      frame1.png
                                                      frame2.png
  <root directory of the eval>/02/interpolated_frames/frame0.png
                                                      frame1.png
                                                      frame2.png
  <root directory of the eval>/03/interpolated_frames/frame0.png
                                                      frame1.png
                                                      frame2.png
  ...

And optionally will produce:
  <root directory of the eval>/01/interpolated.mp4
  <root directory of the eval>/02/interpolated.mp4
  <root directory of the eval>/03/interpolated.mp4
  ...

Usage example:
  python3 -m frame_interpolation.eval.interpolator_cli \
    --model_path <path to TF2 saved model> \
    --pattern "<root directory of the eval>/*" \
    --times_to_interpolate <Number of times to interpolate>
"""

import functools
import os
from typing import List, Sequence

import interpolator as interpolator_lib
import util
from absl import app
#from absl import flags
from absl import logging
import apache_beam as beam
import mediapy as media
import natsort
import numpy as np
import tensorflow as tf
from gooey import Gooey,GooeyParser

# Add other extensions, if not either.
_INPUT_EXT = ['png', 'jpg', 'jpeg']

""" parser = argparse.ArgumentParser()
parser.add_argument("--pattern", default=None,
                    help="The pattern to determine the directories with the input frames.",type=str,dest="pattern")
parser.add_argument("--model_path", default=None,
                    help='The path of the TF2 saved model to use.',type=str,dest="model_path")
parser.add_argument("--times_to_interpolate", default=5,
                    help='The number of times to run recursive midpoint interpolation. '
    'The number of output frames will be 2^times_to_interpolate+1.',type=int,dest="times_to_interpolate")
parser.add_argument("--fps", default=30,
                    help='Frames per second to play interpolated videos in slow motion.',type=int,dest="fps")
parser.add_argument("--align", default=64,
                    help='If >1, pad the input size so it is evenly divisible by this value.',type=int,dest="align")
parser.add_argument("--output_video",
                    help='If true, creates a video of the frames in the interpolated_frames/ '
    'subdirectory',dest="output_video",action="store_true")
parser.add_argument("--blockw", default=1,
                    help='Width of patches.',type=int,dest="blockw")
parser.add_argument("--blockh", default=1,
                    help='Height of patches.',type=int,dest="blockh")
parser.add_argument("--gpu", default=0,
                    help='GPU to use',type=int,dest="gpu")
args = parser.parse_args()
 """

def _output_frames(frames: List[np.ndarray], frames_dir: str):
  """Writes PNG-images to a directory.

  If frames_dir doesn't exist, it is created. If frames_dir contains existing
  PNG-files, they are removed before saving the new ones.

  Args:
    frames: List of images to save.
    frames_dir: The output directory to save the images.

  """
  if tf.io.gfile.isdir(frames_dir):
    old_frames = tf.io.gfile.glob(os.path.join(frames_dir, 'frame_*.png'))
    if old_frames:
      logging.info('Removing existing frames from %s.', frames_dir)
      for old_frame in old_frames:
        tf.io.gfile.remove(old_frame)
  else:
    tf.io.gfile.makedirs(frames_dir)
  for idx, frame in enumerate(frames):
    util.write_image(
        os.path.join(frames_dir, f'frame_{idx:03d}.png'), frame)
  logging.info('Output frames saved in %s.', frames_dir)

class ProcessDirectory(beam.DoFn):
  """DoFn for running the interpolator on a single directory at the time."""

  def setup(self):
    self.interpolator = interpolator_lib.Interpolator(
        "./saved_model", 64)

    if args.output_video:
      ffmpeg_path = util.get_ffmpeg_path()
      media.set_ffmpeg(ffmpeg_path)

  def process(self, directory: str):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    
    input_frames_list = [
        natsort.natsorted(tf.io.gfile.glob(f'{directory}/*.{ext}'))
        for ext in _INPUT_EXT
    ]
    input_frames = functools.reduce(lambda x, y: x + y, input_frames_list)
    logging.info('Generating in-between frames for %s.', directory)
    frames = list(
        util.interpolate_recursively_from_files(
            input_frames, args.times_to_interpolate, self.interpolator,[args.blockw,args.blockh],[0,0]))
    _output_frames(frames, os.path.join(directory, 'interpolated_frames'))
    if args.output_video:
      media.write_video(f'{directory}/interpolated.mp4', frames, fps=args.fps)
      logging.info('Output video saved at %s/interpolated.mp4.', directory)


def _run_pipeline() -> None:
  directories = tf.io.gfile.glob(args.pattern)
  pipeline = beam.Pipeline('DirectRunner')
  (pipeline | 'Create directory names' >> beam.Create(directories)  # pylint: disable=expression-not-assigned
   | 'Process directories' >> beam.ParDo(ProcessDirectory()))

  result = pipeline.run()
  result.wait_until_finish()

@Gooey
def main() -> None:
  global args
  parser = GooeyParser(description="Image Interpolation")
  parser.add_argument("--pattern", default=None,
                      help="The folder to use as input.",type=str,dest="pattern",widget="DirChooser")
  parser.add_argument("--times_to_interpolate", default=5,
                      help= 'The number of output frames will be 2^times_to_interpolate+1.',type=int,dest="times_to_interpolate",widget="IntegerField")
  parser.add_argument("--fps", default=30,
                      help='Frames per second to play interpolated videos in slow motion.',type=int,dest="fps",widget="IntegerField")
  parser.add_argument("--output_video",
                      help='If true, creates a video of the frames'
      'subdirectory',dest="output_video",action="store_true",widget="BlockCheckbox")
  parser.add_argument("--blockw", default=1,
                      help='Number of patches (Width)',type=int,dest="blockw")
  parser.add_argument("--blockh", default=1,
                      help='Number of patches (Height)',type=int,dest="blockh")
  parser.add_argument("--gpu", default=0,
                      help='GPU to use',type=int,dest="gpu")
  args = parser.parse_args()
  _run_pipeline()


if __name__ == '__main__':
  main()
