import argparse
from annotator import Annotator 
from dvImage import dvImage  

parser = argparse.ArgumentParser(description="Run annotation on a dvImage.")
parser.add_argument('-i', '--imgname', required=True, help='Name of the image file (without path)')

args = parser.parse_args()
imgname = args.imgname


image = dvImage(f"data/{imgname}")
annotator = Annotator(image, save_dir=f"annotations/{imgname}_annotations")
annotator.run_annotation()
annotator.run_verification()
