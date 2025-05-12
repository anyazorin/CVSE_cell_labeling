import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import argparse
import cv2

def plot_from_combined(annotations_dir):
    """
    annotations_dir should contain:
      ├─ combined_nucleus_annotation.json
      ├─ cell0/
      │   ├─ cell0_z0.png, cell0_z1.png, …
      ├─ cell1/
      │   ├─ cell1_z0.png, …
      └─ etc.
    """
    combined_path = os.path.join(annotations_dir, "combined_nucleus_annotation_ellipse_points.json")
    with open(combined_path, 'r') as f:
        data = json.load(f)
        entries = data.get("frames", data) if isinstance(data, dict) else data

    for entry in entries:
        cell_idx = entry.get("cell_num")
        z        = entry.get("z")
        if cell_idx is None or z is None:
            print(f"Skipping entry with missing cell_num or z: {entry}")
            continue

        # center guard
        cx = entry.get("center", {}).get("x")
        cy = entry.get("center", {}).get("y")
        if cx is None or cy is None:
            print(f"Skipping cell{cell_idx} z={z}: center missing")
            continue

        # ellipse gua
        ellipse = entry.get("ellipse")
        if not ellipse:
            print(f"Skipping cell{cell_idx} z={z}: ellipse missing")
            continue
        A, B, C, D, E, F = ellipse

        # check conic type
        denom = B*B - 4*A*C
        if denom >= 0:
            print(f"Skipping cell{cell_idx} z={z}: not ellipse (B²−4AC={denom:.3g})")
            continue

        # compute geometric ellipse params
        x0 = (2*C*D - B*E) / denom
        y0 = (2*A*E - B*D) / denom
        up   = 2*(A*E*E + C*D*D + F*B*B - B*D*E - A*C*F)
        term = np.sqrt((A-C)**2 + B*B)
        down1 = denom*( term - (A+C) )
        down2 = denom*(-term - (A+C) )

        a_len = np.sqrt(max(up/down1, 0))
        b_len = np.sqrt(max(up/down2, 0))
        theta = 0.5 * np.degrees(np.arctan2(B, A-C))
        # major axis normalization + image‐flip
        if b_len > a_len:
            a_len, b_len = b_len, a_len
            theta += 90
        theta = (360 - theta) % 360

        # load the PNG
        cell_dir = os.path.join(annotations_dir, f"cell{cell_idx}")
        png_path = os.path.join(cell_dir, f"cell{cell_idx}_z{z}.png")
        if not os.path.exists(png_path):
            print(f"Skipping cell{cell_idx} z={z}: {png_path} not found")
            continue
        img = plt.imread(png_path)

        # adjust for bbox offset
        bbox = entry.get("bbox", {})
        x1 = bbox.get("x1", 0)
        y1 = bbox.get("y1", 0)
        ex = x0 
        ey = y0
        cx_rel = cx - x1
        cy_rel = cy - y1

        # plot
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"cell{cell_idx}  z={z}")

        ax.plot(cx_rel, cy_rel,
                marker='+', markersize=12,
                markeredgewidth=2, color='red')

        ell = Ellipse((ex, ey),
                      width=.5*a_len, height=.5*b_len,
                      angle=theta,
                      edgecolor='red',
                      facecolor='none',
                      linewidth=1.5)
        ax.add_patch(ell)

    plt.tight_layout()
    plt.show()


def verify_annotations(annotations_dir):
    combined_path = os.path.join(
        annotations_dir,
        "combined_nucleus_annotation_ellipse_points.json"
    )
    output_path = os.path.join(
        annotations_dir,
        "output.json"
    )
    with open(combined_path, 'r') as f:
        data = json.load(f)
    entries = data.get("frames", data) if isinstance(data, dict) else data

    base = os.path.basename(annotations_dir)
    img_name = base[:-len("_annotations")] if base.endswith("_annotations") else base
    results = []
    

    for entry in entries:
        cell_idx = entry.get("cell_num")
        z        = entry.get("z")
        if cell_idx is None or z is None:
            continue

        # skip bad centers or ellipses
        cx = entry.get("center",{}).get("x")
        cy = entry.get("center",{}).get("y")
        ell = entry.get("ellipse")
        if cx is None or cy is None or ell is None:
            continue

        A,B,C,D,E,F = ell
        denom = B*B - 4*A*C
        if denom >= 0:
            continue

        # compute geometric ellipse params
        x0 = (2*C*D - B*E)/denom
        y0 = (2*A*E - B*D)/denom
        up   = 2*(A*E*E + C*D*D + F*B*B - B*D*E - A*C*F)
        term = np.sqrt((A-C)**2 + B*B)
        down1 = denom*(term - (A+C))
        down2 = denom*(-term - (A+C))
        a_len = np.sqrt(max(up/down1,0))
        b_len = np.sqrt(max(up/down2,0))
        theta = 0.5 * np.degrees(np.arctan2(B, A-C))
        if b_len > a_len:
            a_len, b_len = b_len, a_len
            theta += 90
        theta = (360 - theta) % 360

        # load the PNG
        cell_dir = os.path.join(annotations_dir, f"cell{cell_idx}")
        png_path = os.path.join(cell_dir, f"cell{cell_idx}_z{z}.png")
        if not os.path.isfile(png_path):
            print(f"Skipping cell{cell_idx} z={z}: {png_path} not found")
            continue
        img = cv2.imread(png_path)

        # draw center as red cross
        bbox = entry.get("bbox",{})
        x1,y1 = bbox.get("x1",0), bbox.get("y1",0)
        ex = x0 
        ey = y0
        cx_rel = cx - x1
        cy_rel = cy - y1
        cv2.drawMarker(
            img, (cx_rel, cy_rel),
            color=(0,0,255), markerType=cv2.MARKER_CROSS,
            markerSize=20, thickness=1
        )

        # draw ellipse in yellow
        # cv2.ellipse expects integer center & axes=(semi-major, semi-minor)
        axes = (int(.25*a_len), int(.25*b_len))
        print(axes)
        print(a_len, b_len)
    
        
        center_coordinates = (int(ex), int(ey)) 
        
    
        color = (0, 0, 255) 
        
        image = cv2.ellipse(img, center_coordinates, axes, 
                theta, 0, 360, (0,0,255), 1) 
        

        # show and wait for Y/N
        win = f"cell{cell_idx} z={z}"

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        # scale it up by, say, 1.5×
        h, w = img.shape[:2]
        cv2.resizeWindow(win, int(h * 5), int(w * 5))

        cv2.imshow(win, img)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('y'), ord('Y')):
                success = True
                break
            elif key in (ord('n'), ord('N')):
                success = False
                break
        cv2.destroyWindow(win)

        # record result
        results.append({
            "img_name":      img_name,
            "cell_num":      cell_idx,
            "z":             z,
            "bbox":          bbox,
            "center":        entry["center"],
            "ellipse":       ell,
            "success":       success
        })

    # write them all out
    with open(output_path, 'w') as out:
        json.dump(results, out, indent=2)
    print(f"Written {len(results)} results to {output_path}")


if __name__ == "__main__":
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
        # plot_from_combined(os.path.join(base_dir, d))
        verify_annotations(os.path.join(base_dir, d))
