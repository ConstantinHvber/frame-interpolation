import argparse
from gooey import Gooey,GooeyParser
@Gooey 
def main():
    parser = GooeyParser(description="Image Interpolation")
    parser.add_argument("--inputfolder", default=None,
                        help="The folder to use as input.",type=str,dest="pattern",widget="DirChooser")
    parser.add_argument("--model_path", default="./models/saved_model",
                        help='The path of the TF2 saved model to use.',type=str,dest="model_path")
    parser.add_argument("--times_to_interpolate", default=5,
                        help= 'The number of output frames will be 2^times_to_interpolate+1.',type=int,dest="times_to_interpolate",widget="IntegerField")
    parser.add_argument("--fps", default=30,
                        help='Frames per second to play interpolated videos in slow motion.',type=int,dest="fps",widget="IntegerField")
    parser.add_argument("--align", default=64,
                        help='If >1, pad the input size so it is evenly divisible by this value.',type=int,dest="align")
    parser.add_argument("--output_video",
                        help='If true, creates a video of the frames'
        'subdirectory',dest="output_video",action="store_true",widget="BlockCheckbox")
    parser.add_argument("--blockw", default=1,
                        help='Width of patches.',type=int,dest="blockw")
    parser.add_argument("--blockh", default=1,
                        help='Height of patches.',type=int,dest="blockh")
    parser.add_argument("--gpu", default=0,
                        help='GPU to use',type=int,dest="gpu")
    args = parser.parse_args()
    print(args)
main()