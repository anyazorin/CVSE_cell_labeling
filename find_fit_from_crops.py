import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import argparse
import os
from matplotlib.patches import Ellipse
from scipy.linalg import eig
class findFit:
    def __init__(self, img_name):
        self.ref_hsv_green = [60, 100, 200]
        self.tol_green     = [30, 100,  60]
        self.ref_hsv_red   = [170, 208, 140]
        self.tol_red       = [40,100,175]
        self.img_name = img_name
        self.annotation_path = os.path.join('annotations', f'{img_name}_annotations', 'combined_nucleus_annotation.json') 
        with open(self.annotation_path, 'r') as f:
            data = json.load(f)
        rows = []
        for entry in data:
            bbox = entry['bbox']
            rows.append([
                entry['cell_num'],
                entry['z'],
                bbox['x1'], bbox['y1'],
                bbox['x2'], bbox['y2']
            ])
        self.annotations = np.array(rows, dtype=int)
        base, _ = os.path.splitext(self.annotation_path)
        self.save_filepath = base + '_ellipse_points.json'

    # THIS SHLD INSTEA TAKE CELL NUM AND Z, AND RETURN THE PNG with filepath cell{cell_num}/cell{ cell_num}_z{z}.png as rgb data. 
    def crop_bbox(self, cell_num, z):
        # build the expected file path
        filename = f"cell{cell_num}_z{z}.png"
        filepath = os.path.join('annotations', f'{self.img_name}_annotations',f"cell{cell_num}", filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Crop file not found: {filepath}")
        # load image (BGR or BGRA)
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to load image: {filepath}")
        # drop alpha if present
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        # convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # normalize to [0,1] floats
        return img.astype(np.float32) / 255.0
    def threshold_color_channel(self, crop, channel=1, threshold=0.5):
        """Simple float‐threshold on one channel (0=R,1=G,2=B)."""
        mask = crop[..., channel] >= threshold
        out = np.zeros_like(crop)
        out[mask] = crop[mask]
        return out

    def hsv_inrange_mask(self, crop, color = "green", paint_color=[0,1,0], kernel_size=1):
        """
        Full HSV in‐range + morphology.
        ref_hsv: tuple (H in [0..179], S [0..255], V [0..255])
        tol:    tuple of tolerances (dH, dS, dV)
        paint_color: the RGB color to paint mask (0–1 floats)
        """
        cu8 = (crop * 255).astype(np.uint8)
        hsv = cv2.cvtColor(cu8, cv2.COLOR_RGB2HSV)

        h, s, v = cv2.split(hsv)

        if color == "green":
            green_ch = crop[..., 1]
            y_g, x_g = np.unravel_index(np.argmax(green_ch), green_ch.shape)
            ref_hsv = hsv[y_g, x_g]
            tol = self.tol_green
            # print(f"Brightest GREEN at (y={y_g}, x={x_g}), HSV={tuple(ref_hsv)}")
            #take max of 255 and ref for hue value

        if color == "red":    
            red_ch = crop[..., 0]
            y_r, x_r = np.unravel_index(np.argmax(red_ch), red_ch.shape)
            ref_hsv = hsv[y_r, x_r]
            tol = self.tol_red
            if ref_hsv[0] <50:
                ref_hsv[0] = 170
            # print(f"Brightest   RED at (y={y_r}, x={x_r}), HSV={tuple(ref_hsv)}")


        lo = np.array([max(0, ref_hsv[0]-tol[0]),
                       max(0, ref_hsv[1]-tol[1]),
                       max(0, ref_hsv[2]-tol[2])], np.uint8)
        hi = np.array([min(255, ref_hsv[0]+tol[0]),
                       min(255, ref_hsv[1]+tol[1]),
                       min(255, ref_hsv[2]+tol[2])], np.uint8)

        mask = cv2.inRange(hsv, lo, hi)
        kernel = np.ones((kernel_size,kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask_bool = mask.astype(bool)
        out = np.zeros_like(crop)
        out[mask_bool] = paint_color 

        return out
    
    def debug_show_hsv_channels(self, crop):
        """
        Show Original, H, S, V; mark brightest-green and brightest-red;
        and plot hue histogram with their hue values.
        """

        cu8 = (crop * 255).astype(np.uint8)
        hsv = cv2.cvtColor(cu8, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        
        green_ch = crop[..., 1]
        y_g, x_g = np.unravel_index(np.argmax(green_ch), green_ch.shape)
        hsv_g = hsv[y_g, x_g]

        
        red_ch = crop[..., 0]
        y_r, x_r = np.unravel_index(np.argmax(red_ch), red_ch.shape)
        hsv_r = hsv[y_r, x_r]

        print(f"Brightest GREEN at (y={y_g}, x={x_g}), HSV={tuple(hsv_g)}")
        print(f"Brightest   RED at (y={y_r}, x={x_r}), HSV={tuple(hsv_r)}\n")
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(crop)
        axs[0].set_title("Original")
        axs[0].axis("off")

        for ax, channel, title in zip(
            axs[1:], [h, s, v], ["Hue", "Sat", "Val"]
        ):
            ax.imshow(channel, cmap="gray")
            # overlay green and red points
            ax.plot(x_g, y_g, "go", markersize=8, label="bright G")
            ax.plot(x_r, y_r, "ro", markersize=8, label="bright R")
            ax.set_title(title)
            ax.axis("off")

       
        axs[1].legend(loc="upper right")
        plt.tight_layout()
        plt.show()
        # Plot hue histogram
        hue_vals = hsv[:, :, 0].ravel()
        plt.figure(figsize=(6, 4))
        plt.hist(hue_vals, bins=180, color="lightgray", edgecolor="k")
        plt.axvline(hsv_g[0], color="g", linestyle="--", label=f"G hue={hsv_g[0]}")
        plt.axvline(hsv_r[0], color="r", linestyle="--", label=f"R hue={hsv_r[0]}")
        plt.title("Hue Distribution")
        plt.xlabel("Hue (0–179)")
        plt.ylabel("Pixel count")
        plt.legend()
        plt.show()


    def show_crops(self,
                   mask_method='hsv_inrange',
                   # for threshold_method:
                   color_channel=1,
                   threshold=0.5,
                   # for hsv_inrange:
                   ref_hsv=[60, 100, 200],  # e.g. green ≈ (30,180,200)
                   tol=[40, 200,  80],
                   color = "green",
                   max_to_show=6,
                   show_original=False,
                   kernel_size=1,
                   ):
        """
        mask_method: one of 'threshold', 'hsv_morph', 'hsv_inrange'
        """
        print(color)
        if color == "green":
            ref_hsv = self.ref_hsv_green
            tol = self.tol_green
            paint_color = (0, 1, 0)
        elif color == "red":   
            print(color)
            ref_hsv = self.ref_hsv_red
            tol = self.tol_red
            paint_color = (1, 0, 0)
            kernel_size = 2
        n = min(max_to_show, len(self.annotations))
        if mask_method == 'debug_hsv':
                for i, ann in enumerate(self.annotations[:n]):
                    cell_num, z, x1, y1, x2, y2 = ann.astype(int)
                    crop = self.crop_bbox(cell_num, z)
                    self.debug_show_hsv_channels(crop)
                return
        cols = 2 if show_original else 1
        fig, axs = plt.subplots(n, cols, figsize=(4*cols, 4*n))
        if n==1:
            axs = np.expand_dims(axs, 0)
        for i, ann in enumerate(self.annotations[:n]):
            cell_num, z, x1, y1, x2, y2 = ann.astype(int)
            crop = self.crop_bbox(cell_num, z)

            if mask_method == 'threshold':
                masked = self.threshold_color_channel(crop, channel=color_channel, threshold=threshold)
            elif mask_method == 'hsv_inrange':
                masked = self.hsv_inrange_mask(crop, color=color, paint_color=paint_color, kernel_size=kernel_size)
            else:
                raise ValueError(mask_method)

            ax_row = axs[i]
            if show_original:
                ax_row[0].imshow(crop)
                ax_row[0].set_title(f"Cell {cell_num} z={z}\n(original)")
                ax_row[0].axis('off')
                ax_row[1].imshow(masked)
                ax_row[1].set_title(f"{mask_method}")
                ax_row[1].axis('off')  

            else:
                ax_row.imshow(masked)
                ax_row.set_title(f"Cell {cell_num} z={z}\n({mask_method})")
                ax_row.axis('off')

        plt.tight_layout()
        plt.show()

    def find_ellipse(self, mask, show=False):
        if mask.ndim == 3:
            mask_bool = np.any(mask != 0, axis=-1)
        else:
            mask_bool = mask
        ys, xs = np.nonzero(mask_bool)
        if xs.size < 6:
            return None
        x = xs.astype(np.float64)
        y = ys.astype(np.float64)
        # Design matrix
        D = np.vstack([x*x, x*y, y*y, x, y, np.ones_like(x)]).T
        # Scatter matrix
        S = D.T @ D
        # Constraint matrix
        Cmat = np.zeros((6,6), dtype=np.float64)
        Cmat[0,2] = Cmat[2,0] = -2
        Cmat[1,1] = 1
        eigvals, eigvecs = eig(S, Cmat)
        # Select the eigenvector with negative real eigenvalue
        cond = np.isfinite(eigvals) & (eigvals < 0)
        if not np.any(cond):
            return None
        # pick first if multiple
        idx = np.where(cond)[0][0]
        a = eigvecs[:, idx].real
        if show:
            # convert conic to geometric params
            A, B, C, D, E, F = a
            # center
            denom = B*B - 4*A*C
            x0 = (2*C*D - B*E) / denom
            y0 = (2*A*E - B*D) / denom
            # axes
            up = 2*(A*E*E + C*D*D + F*B*B - B*D*E - A*C*F)
            term = np.sqrt((A-C)**2 + B*B)
            down1 = (B*B - 4*A*C)*(term - (A+C))
            down2 = (B*B - 4*A*C)*(-term - (A+C))
            a_len = np.sqrt(up / down1)
            b_len = np.sqrt(up / down2)
            # angle in degrees
            theta = 0.5 * np.degrees(np.arctan2(B, A-C))
            # plot
            fig, ax = plt.subplots()
            ax.imshow(mask if mask.ndim==3 else mask, cmap=None if mask.ndim==3 else 'gray')
            ell = Ellipse((x0, y0), width=.5*b_len, height=.5*a_len,
                          angle=theta, edgecolor='r', facecolor='none')
            ax.add_patch(ell)
            ax.set_title(f"Fitted ellipse: center=({x0:.1f},{y0:.1f}), axes=({a_len:.1f},{b_len:.1f}), angle={theta:.1f}\u00b0")
            ax.axis('off')
            plt.show()
        return a

    def find_center(self, mask, show=False):
        """
        mask: H×W×3 painted mask array or boolean mask array
        show: if True, display the mask with the center point overlaid
        Returns: (cx, cy) or None if mask is empty
        """
        # Convert to boolean mask if needed
        if mask.ndim == 3:
            mask_bool = np.any(mask != 0, axis=-1)
        else:
            mask_bool = mask
        # Label connected components with stats
        mask_u8 = mask_bool.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8)
        # stats: [label, (x, y, width, height, area)]
        if num_labels <= 1:
            # no components
            if show:
                plt.figure()
                plt.imshow(mask if mask.ndim==3 else mask, cmap=None if mask.ndim==3 else 'gray')
                plt.title("No masked pixels found")
                plt.axis('off')
                plt.show()
            return None
        # Compute bounding-box areas for each label (skip background=0)
        # stats[label, cv2.CC_STAT_WIDTH] * stats[label, cv2.CC_STAT_HEIGHT]
        bbox_areas = stats[1:, cv2.CC_STAT_WIDTH] * stats[1:, cv2.CC_STAT_HEIGHT]
        max_bbox = bbox_areas.max()
        candidates = np.where(bbox_areas == max_bbox)[0] + 1  # +1 to offset skip of background
        # Tie-break by distance to crop center
        if len(candidates) > 1:
            h, w = mask_bool.shape
            img_center = np.array([w/2, h/2])
            # compute centroid of each candidate cluster
            dists = []
            for lab in candidates:
                # use computed centroids
                cx_lab, cy_lab = centroids[lab]
                dists.append(np.linalg.norm(np.array([cx_lab, cy_lab]) - img_center))
            chosen = candidates[int(np.argmin(dists))]
        else:
            chosen = candidates[0]
        # Compute final centroid of chosen cluster (mean of pixels)
        ys, xs = np.nonzero(labels == chosen)
        cx, cy = int(xs.mean()), int(ys.mean())
        if show:
            plt.figure()
            plt.imshow(mask if mask.ndim==3 else mask, cmap=None if mask.ndim==3 else 'gray')
            plt.plot(cx, cy, 'r+', markersize=12)
            plt.title(f"Center at (x={cx}, y={cy})")
            plt.axis('off')
            plt.show()
        return (cx, cy)


    def save_ellipse_point(self, mask_method='hsv_inrange'):
        with open(self.annotation_path, 'r') as f:
            data = json.load(f)

        out_data = []
        for i, entry in enumerate(data):
            # pull the bbox info from self.annotations
            cell_num, z, x1, y1, x2, y2 = self.annotations[i]
            crop = self.crop_bbox(cell_num, z)
            if mask_method == 'threshold':
                green_masked = self.threshold_color_channel(
                    crop, channel=1, threshold=0.5)
                red_masked = self.threshold_color_channel(
                    crop, channel=0, threshold=0.5)
            elif mask_method == 'hsv_inrange':
                green_masked = self.hsv_inrange_mask(
                    crop,
                    color = "green",
                    paint_color=(0,1,0),
                    kernel_size=1
                )
                red_masked = self.hsv_inrange_mask(
                    crop,
                    color = "red",
                    paint_color=(1,0,0),
                    kernel_size=1
                )
            else:
                raise ValueError(mask_method)

            green_center = self.find_center(green_masked, show=False)
            a = self.find_ellipse(red_masked, show=False)
            if green_center is not None:
                # convert to global image coords
                cx_img = int(x1 + green_center[0])
                cy_img = int(y1 + green_center[1])
            else:
                cx_img = None
                cy_img = None

            # augment the entry
            entry['center'] = {'x': cx_img, 'y': cy_img}
            entry['ellipse'] = a.tolist() if a is not None else None

            out_data.append(entry)

        with open(self.save_filepath, 'w') as f:
            json.dump(out_data, f, indent=4)
        print(f"Saved {len(out_data)} entries with centers -> {self.save_filepath}")

    def interactive_classify(self,
                             mask_method='hsv_inrange',
                             kernel_size=1,
                             save_root='mask_dataset'):
        """
        Loop through every annotation, show:
           [original] | [green+red masks] | [mask+ellipse+center]
        prompt y/n, and if 'y' save original, green_mask, red_mask into
        save_root/{original,green_mask,red_mask} with filenames
        [img_name]_[cell#]_[z].png
        """
        # prepare output directories
        dirs = {
            'original':   os.path.join(save_root, 'original'),
            'green_mask': os.path.join(save_root, 'green_mask'),
            'red_mask':   os.path.join(save_root, 'red_mask'),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        for cell_num, z, x1, y1, x2, y2 in self.annotations:
            crop = self.crop_bbox(cell_num, z)

            # build masks
            green_mask = self.hsv_inrange_mask(crop, color='green',
                                               paint_color=(0,1,0),
                                               kernel_size=kernel_size)
            red_mask   = self.hsv_inrange_mask(crop, color='red',
                                               paint_color=(1,0,0),
                                               kernel_size=2)
            overlay = green_mask + red_mask

            # find center & ellipse on the fly
            center = self.find_center(green_mask, show=False)       # (cx,cy) or None
            a = self.find_ellipse(red_mask, show=False)             # [A,B,C,D,E,F] or None

            # compute geometric ellipse params if available
            ellipse_params = None
            if a is not None:
                A,B,C,D,E,F = a
                denom = B*B - 4*A*C
                if denom < 0:
                    x0 = (2*C*D - B*E) / denom
                    y0 = (2*A*E - B*D) / denom
                    up   = 2*(A*E*E + C*D*D + F*B*B - B*D*E - A*C*F)
                    term = np.sqrt((A-C)**2 + B*B)
                    down1 = denom*( term - (A+C) )
                    down2 = denom*(-term - (A+C) )
                    a_len = np.sqrt(max(up/down1, 0))
                    b_len = np.sqrt(max(up/down2, 0))
                    theta = 0.5 * np.degrees(np.arctan2(B, A-C))
                    # enforce a_len ≥ b_len
                    if b_len > a_len:
                        a_len, b_len = b_len, a_len
                        theta += 90
                    theta = (360 - theta) % 360
                    ellipse_params = (x0, y0, a_len, b_len, theta)

            # --- display side by side ---
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            # 1) original
            axs[0].imshow(crop)
            axs[0].set_title(f'Original\ncell{cell_num} z={z}')
            axs[0].axis('off')

            # 2) masks overlay
            axs[1].imshow(overlay)
            axs[1].set_title('Green + Red mask')
            axs[1].axis('off')

            # 3) overlay + ellipse & center
            axs[2].imshow(overlay)
            if center is not None:
                cx, cy = center
                axs[2].plot(cx, cy,
                            marker='+', markersize=12,
                            markeredgewidth=2, color='yellow',
                            label='green center')
            if ellipse_params is not None:
                x0,y0,a_len,b_len,theta = ellipse_params
                ell = Ellipse((x0, y0),
                              width=.5*b_len, height=.5*a_len,
                              angle=theta,
                              edgecolor='cyan',
                              facecolor='none',
                              linewidth=1.5,
                              label='red fit')
                axs[2].add_patch(ell)
            axs[2].set_title('Mask + fit')
            axs[2].axis('off')
            if center or ellipse_params:
                axs[2].legend(loc='upper right', fontsize='small')

            plt.tight_layout()
            plt.show()

            # --- prompt & save ---
            resp = input("Save as GOOD? (y/n): ").strip().lower()
            plt.close(fig)

            if resp == 'y':
                fname = f"{self.img_name}_{cell_num}_{z}.png"

                # save original
                orig_uint8 = (crop * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(dirs['original'], fname),
                            cv2.cvtColor(orig_uint8, cv2.COLOR_RGB2BGR))

                # save green mask
                green_uint8 = (green_mask * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(dirs['green_mask'], fname),
                            cv2.cvtColor(green_uint8, cv2.COLOR_RGB2BGR))

                # save red mask
                red_uint8 = (red_mask * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(dirs['red_mask'], fname),
                            cv2.cvtColor(red_uint8, cv2.COLOR_RGB2BGR))

                print(f"  → saved: {fname}")




        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load annotations for an image and display its cell crops."
    )
    parser.add_argument(
        "--img_name",
        type=str,
        help="Base name of the image (e.g. without extension) matching your annotation folder."
    )
    parser.add_argument(
        "--mask_method",
        choices=["threshold", "hsv_inrange", "hsv_morph", "debug_hsv"],
        default="hsv_inrange",
        help="Which masking method to apply to each crop."
    )
    parser.add_argument(
        "--show_original",
        action="store_true",
        help="If set, display the original crop alongside the masked version."
    )
    parser.add_argument(
        "--max_to_show",
        type=int,
        default=20,
        help="Maximum number of annotated crops to display."
    )
    args = parser.parse_args()

    # instantiate and display
    ff = findFit(args.img_name)
    # ff.show_crops(
    #     mask_method=args.mask_method,
    #     show_original=args.show_original,
    #     max_to_show=args.max_to_show,
    #     color = "green"
    # )
    ff.interactive_classify(
        mask_method=args.mask_method,
        kernel_size=1  # or expose as arg
    )
