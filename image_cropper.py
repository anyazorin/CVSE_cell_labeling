import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
from scipy.linalg import eig
from matplotlib.patches import Ellipse

class ImageMasker:
    def __init__(self, dvimage_obj, json_path):
        self.ref_hsv_green = [40, 100, 120]
        self.tol_green     = [40, 200,  80]
        # self.ref_hsv_red   = [160, 208, 140]
        # self.tol_red       = [25,  80,  80]
        self.ref_hsv_red   = [160, 208, 140]
        self.tol_red       = [10,80,30]
        self.annotation_path = json_path
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
        self.dvimage = dvimage_obj

    def crop_bbox(self, z, x1, y1, x2, y2):
        return self.dvimage.rgb_data[0, z, y1:y2, x1:x2, :]

    def threshold_color_channel(self, crop, channel=1, threshold=0.5):
        """Simple float‐threshold on one channel (0=R,1=G,2=B)."""
        mask = crop[..., channel] >= threshold
        out = np.zeros_like(crop)
        out[mask] = crop[mask]
        return out

    def hsv_morphology_mask(self, crop, ref_color='green', thresh=10):
        """
        Simple hue‐only mask + close/open.
        ref_color: 'green' or 'red'
        thresh: hue tolerance in [0..179]
        """
        cu8 = (crop * 255).astype(np.uint8)
        hsv = cv2.cvtColor(cu8, cv2.COLOR_RGB2HSV)

        if ref_color == 'green':
            ref_h = 60 // 2    # ~Hue=60° → 30 in OpenCV’s 0–179
            paint = np.array([0,1,0], np.float32)
        elif ref_color == 'red':
            ref_h = 0
            paint = np.array([1,0,0], np.float32)
        else:
            raise ValueError(ref_color)

        h = hsv[...,0].astype(int)
        # circular distance
        dh = np.minimum(np.abs(h - ref_h), 180 - np.abs(h - ref_h))
        mask = (dh <= thresh).astype(np.uint8)*255

        k = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = mask>0

        out = np.zeros_like(crop)
        out[mask] = paint
        return out

    def hsv_inrange_mask(self, crop, ref_hsv=[0,0,0], tol=[179,255,255], paint_color=[0,1,0], kernel_size=1):
        """
        Full HSV in‐range + morphology.
        ref_hsv: tuple (H in [0..179], S [0..255], V [0..255])
        tol:    tuple of tolerances (dH, dS, dV)
        paint_color: the RGB color to paint mask (0–1 floats)
        """
        cu8 = (crop * 255).astype(np.uint8)
        hsv = cv2.cvtColor(cu8, cv2.COLOR_RGB2HSV)

        lo = np.array([max(0, ref_hsv[0]-tol[0]),
                       max(0, ref_hsv[1]-tol[1]),
                       max(0, ref_hsv[2]-tol[2])], np.uint8)
        hi = np.array([min(179, ref_hsv[0]+tol[0]),
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
                   # for hsv_morphology:
                   ref_color='green',
                   # for hsv_inrange:
                   ref_hsv=(30,180,200),  # e.g. green ≈ (30,180,200)
                   tol=(0,50,50),
                   paint_color=(0,1,0),
                   max_to_show=6,
                   show_original=False,
                   kernel_size=1,
                   ):
        """
        mask_method: one of 'threshold', 'hsv_morph', 'hsv_inrange'
        """
        n = min(max_to_show, len(self.annotations))
        if mask_method == 'debug_hsv':
                for i, ann in enumerate(self.annotations[:n]):
                    cell_num, z, x1, y1, x2, y2 = ann.astype(int)
                    crop = self.crop_bbox(z, x1, y1, x2, y2)
                    self.debug_show_hsv_channels(crop)
                return
        cols = 2 if show_original else 1
        fig, axs = plt.subplots(n, cols, figsize=(4*cols, 4*n))
        if n==1:
            axs = np.expand_dims(axs, 0)
        for i, ann in enumerate(self.annotations[:n]):
            cell_num, z, x1, y1, x2, y2 = ann.astype(int)
            crop = self.crop_bbox(z, x1, y1, x2, y2)

            if mask_method == 'threshold':
                masked = self.threshold_color_channel(crop, channel=color_channel, threshold=threshold)
            elif mask_method == 'hsv_morph':
                masked = self.hsv_morphology_mask(crop, ref_color=ref_color, thresh=int(threshold))
            elif mask_method == 'hsv_inrange':
                masked = self.hsv_inrange_mask(crop, ref_hsv=ref_hsv, tol=tol, paint_color=paint_color, kernel_size=kernel_size)
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
            ell = Ellipse((x0, y0), width=2*a_len, height=2*b_len,
                          angle=theta, edgecolor='r', facecolor='none')
            ax.add_patch(ell)
            ax.set_title(f"Fitted ellipse: center=({x0:.1f},{y0:.1f}), axes=({a_len:.1f},{b_len:.1f}), angle={theta:.1f}\u00b0")
            ax.axis('off')
            plt.show()
        return a
    
    def plot_general_ellipse(a, b, c, d, e, f, ax=None, **plot_kwargs):
        """
        Plot the ellipse defined by
            a x^2 + b x y + c y^2 + d x + e y + f = 0
        """
        # 1) Compute center by solving ∇(quadratic) = 0:
        #    [2a  b ] [x0] + [ d ] = 0
        #    [ b 2c] [y0]   [ e ]
        M = np.array([[2*a, b],
                    [b,   2*c]])
        rhs = np.array([-d, -e])
        x0, y0 = np.linalg.solve(M, rhs)

        # 2) Compute the constant term at the center:
        f0 = (a*x0**2 + b*x0*y0 + c*y0**2 + d*x0 + e*y0 + f)

        # 3) Build the 2×2 “shape” matrix Q for the quadratic form:
        Q = np.array([[a,   b/2],
                    [b/2,  c  ]])

        # Check that this is an ellipse:
        #    Q must be positive-definite and f0 < 0
        eigvals, eigvecs = np.linalg.eigh(Q)
        if not (eigvals[0] > 0 and eigvals[1] > 0 and f0 < 0):
            raise ValueError("These coefficients do not define a (real) ellipse.")

        # 4) Semi-axes lengths are sqrt(−f0/λ_i):
        axes = np.sqrt(-f0 / eigvals)

        # 5) Parameterize the unit circle, stretch, rotate, translate:
        θ = np.linspace(0, 2*np.pi, 400)
        circle = np.vstack((axes[0]*np.cos(θ), axes[1]*np.sin(θ)))
        ellipse_pts = eigvecs @ circle
        xs = ellipse_pts[0, :] + x0
        ys = ellipse_pts[1, :] + y0

        # 6) Plot
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(xs, ys, **plot_kwargs)
        ax.set_aspect('equal', 'datalim')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('General ellipse')
        return ax

    def norm_find_ellipse(self, mask, show=False):
        """
        Fit an ellipse to the True pixels in mask using Direct Least Squares
        with normalization for numerical stability.

        mask: H×W painted mask array or boolean mask array
        show: if True, overlay the fitted ellipse on the mask and display it
        Returns a numpy array [A, B, C, D, E, F] of conic coefficients,
        or None if fitting fails.
        """
        # Boolean mask
        if mask.ndim == 3:
            mask_bool = np.any(mask != 0, axis=-1)
        else:
            mask_bool = mask
        ys, xs = np.nonzero(mask_bool)
        if xs.size < 6:
            return None
        x = xs.astype(np.float64)
        y = ys.astype(np.float64)
        # Normalize: translate mean to origin, scale so max range = 2
        xm, ym = x.mean(), y.mean()
        dx, dy = x.max() - x.min(), y.max() - y.min()
        if dx == 0 or dy == 0:
            return None
        scale = max(dx, dy) / 2.0
        xn = (x - xm) / scale
        yn = (y - ym) / scale
        # Build normalized design matrix
        Dn = np.vstack([xn*xn, xn*yn, yn*yn, xn, yn, np.ones_like(xn)]).T
        Sn = Dn.T @ Dn
        # Constraint
        Cmat = np.zeros((6, 6), dtype=np.float64)
        Cmat[0, 2] = Cmat[2, 0] = -2
        Cmat[1, 1] = 1
        from scipy.linalg import eig
        eigvals, eigvecs = eig(Sn, Cmat)
        cond = np.isfinite(eigvals) & (eigvals < 0)
        if not np.any(cond):
            return None
        a_norm = eigvecs[:, cond][:, 0].real
        # Un-normalize coefficients: xn = (x - xm)/scale
        A , B , C , D , E , F = a_norm
        # A = A_ * scale * scale
        # B = B_ * scale * scale
        # C = C_ * scale * scale
        # D = D_ * scale - 2*A*xm - B*ym
        # E = E_ * scale - B*xm - 2*C*ym
        # F = F_ - D_*xm*scale - E_*ym*scale + A*xm*xm + B*xm*ym + C*ym*ym
        a = np.array([A, B, C, D, E, F])
        if show:
            # Convert to geometric and display
            denom = B*B - 4*A*C
            x0 = (2*C*D - B*E) / denom
            y0 = (2*A*E - B*D) / denom
            up = 2*(A*E*E + C*D*D + F*B*B - B*D*E - A*C*F)
            term = np.sqrt((A - C)**2 + B*B)
            down1 = (B*B - 4*A*C)*(term - (A + C))
            down2 = (B*B - 4*A*C)*(-term - (A + C))
            a_len = np.sqrt(up / down1)
            b_len = np.sqrt(up / down2)
            theta = 0.5 * np.degrees(np.arctan2(B, A - C))
            fig, ax = plt.subplots()
            ax.imshow(mask if mask.ndim==3 else mask_bool,
                      cmap=None if mask.ndim==3 else 'gray')
            ell = Ellipse((x0, y0), width=2*a_len, height=2*b_len,
                          angle=theta, edgecolor='r', facecolor='none')
            ax.add_patch(ell)
            ax.set_title(
                f"Fitted ellipse: center=({x0:.1f},{y0:.1f}), "
                f"axes=({a_len:.1f},{b_len:.1f}), angle={theta:.1f}°"
            )
            ax.axis('off')
            plt.show()
        return a



    def openCV_find_ellipse(self, mask, show=False):
        """
        Fit an ellipse to the True pixels in mask using OpenCV's fitEllipse for robustness.
        mask: H×W painted mask array or boolean mask array
        show: if True, overlay the fitted ellipse on the mask and display it
        Returns: ((x0, y0), (width, height), angle) or None if fitting fails.
        """
        # get boolean mask
        if mask.ndim == 3:
            mask_bool = np.any(mask != 0, axis=-1)
        else:
            mask_bool = mask
        # prepare for contour finding
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        # find external contours
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        # pick the largest contour by area
        cnt = max(contours, key=lambda c: cv2.contourArea(c))
        # need at least 5 points for fitEllipse
        if len(cnt) < 5:
            return None
        # fit ellipse
        ellipse = cv2.fitEllipse(cnt)
        # ellipse: ((x_center, y_center), (major_axis, minor_axis), angle_degrees)
        if show:
            # overlay on mask or optional crop
            fig, ax = plt.subplots()
            ax.imshow(mask if mask.ndim==3 else mask_bool, cmap=None if mask.ndim==3 else 'gray')
            (x0, y0), (w, h), angle = ellipse
            ell_patch = Ellipse((x0, y0), width=w, height=h,
                                   angle=angle, edgecolor='r', facecolor='none')
            ax.add_patch(ell_patch)
            ax.set_title(f"Ellipse: center=({x0:.1f},{y0:.1f}), axes=({w:.1f},{h:.1f}), angle={angle:.1f}°")
            ax.axis('off')
            plt.show()
        return ellipse
    
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
            crop = self.crop_bbox(z, x1, y1, x2, y2)
            if mask_method == 'threshold':
                green_masked = self.threshold_color_channel(
                    crop, channel=1, threshold=0.5)
                red_masked = self.threshold_color_channel(
                    crop, channel=0, threshold=0.5)
            elif mask_method == 'hsv_morph':
                green_masked = self.hsv_morphology_mask(
                    crop, ref_color='green', thresh=0.5)
                red_masked = self.hsv_morphology_mask(
                    crop, ref_color='red', thresh=0.5)
            elif mask_method == 'hsv_inrange':
                green_masked = self.hsv_inrange_mask(
                    crop,
                    ref_hsv=self.ref_hsv_green,
                    tol=self.tol_green,
                    paint_color=(0,1,0),
                    kernel_size=1
                )
                red_masked = self.hsv_inrange_mask(
                    crop,
                    ref_hsv=self.ref_hsv_red,
                    tol=self.tol_red,
                    paint_color=(1,0,0),
                    kernel_size=1
                )
            else:
                raise ValueError(mask_method)

            green_center = self.find_center(green_masked, show=True)
            # self.norm_find_ellipse(red_masked, show=True)
            self.find_ellipse(red_masked, show=True)
            if green_center is not None:
                # convert to global image coords
                cx_img = int(x1 + green_center[0])
                cy_img = int(y1 + green_center[1])
            else:
                cx_img = None
                cy_img = None

            # augment the entry
            entry['center'] = {'x': cx_img, 'y': cy_img}
            entry['ellipse'] = None  # placeholder

            out_data.append(entry)

        with open(self.save_filepath, 'w') as f:
            json.dump(out_data, f, indent=4)
        print(f"Saved {len(out_data)} entries with centers -> {self.save_filepath}")


        
