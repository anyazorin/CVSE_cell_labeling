import json
import os
import argparse
from find_fit_from_crops import findFit



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compile annotations for image(s).")
    parser.add_argument(
        '-i', '--imgname',
        help='Name of the image file (without path). If omitted, all folders under annotations/ will be processed.'
    )
    args = parser.parse_args()

    base_dir = 'annotations'

    if args.imgname:
        # single-image mode
        dirs = [f"{args.imgname}_annotations"]
    else:
        # batch mode: every *_annotations folder
        dirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.endswith('_annotations')
        ]

    if not dirs:
        print("No annotation folders found to process.")

    for d in dirs:
        img_name = d[:-len('_annotations')]
        #  get the img name by taking the start of the dir name ((imgname_annotations))
        ff = findFit(img_name)
        ff.save_ellipse_point()
