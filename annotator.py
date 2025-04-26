import os
import json
import cv2
import numpy as np

class Annotator:
    def __init__(self, dv_image, save_dir="annotations"):
        self.dv_image = dv_image
        self.save_dir = save_dir
        self.rgb_stack = self.dv_image.rgb_data[0]  # T=0 always
        self.z_depth = self.rgb_stack.shape[0]
        self.current_z = 0
        self.annotations = {z: [] for z in range(self.z_depth)}
        self.drawing = False
        self.start_point = None
        self.current_rect = None
        self.mode = 'annotate'  # 'annotate' or 'verify'
        os.makedirs(self.save_dir, exist_ok=True)

    def draw_annotations(self, img, z):
        for (x1, y1, x2, y2) in self.annotations[z]:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_rect = (self.start_point[0], self.start_point[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1, x2, y2 = self.start_point[0], self.start_point[1], x, y
            self.annotations[self.current_z].append((min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)))
            self.current_rect = None

    def save_annotations(self):
        folder_counter = 0
        for z, boxes in self.annotations.items():
            for box in boxes:
                cell_folder = os.path.join(self.save_dir, f"cell{folder_counter}")
                os.makedirs(cell_folder, exist_ok=True)
                meta = {"z": z, "nucleus": None, "bbox": box}
                with open(os.path.join(cell_folder, f"cell{folder_counter}_annotation.json"), 'w') as f:
                    json.dump(meta, f)
                x1, y1, x2, y2 = box
                for zi in range(self.z_depth):
                    cropped = self.rgb_stack[zi, y1:y2, x1:x2, :]
                    cv2.imwrite(os.path.join(cell_folder, f"cell{folder_counter}_z{zi}.png"), cv2.cvtColor((cropped * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                folder_counter += 1
    def on_trackbar(self, val):
        self.current_z = val
    
    def run_annotation(self):
        cv2.namedWindow("Annotator")
        cv2.setMouseCallback("Annotator", self.mouse_callback)
        cv2.createTrackbar("Z", "Annotator", 0, self.z_depth - 1, self.on_trackbar)
        cv2.setTrackbarPos("Z", "Annotator", self.current_z)
        while True:
            img = (self.rgb_stack[self.current_z] * 255).astype(np.uint8).copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for z in range(self.z_depth):
                img = self.draw_annotations(img, z)
            if self.current_rect:
                x1, y1, x2, y2 = self.current_rect
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

            cv2.putText(img, f"Z: {self.current_z}/{self.z_depth - 1}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Annotator", img)

            key = cv2.waitKeyEx(1)
            if key == 27:  # ESC
                break
            elif key == ord('c'):
                self.annotations = {z: [] for z in range(self.z_depth)}
            elif key == 13:  # Enter
                self.save_annotations()
                break
            elif key == ord('w'):
                self.current_z = min(self.z_depth - 1, self.current_z + 1)
            elif key == ord('s'):
                self.current_z = max(0, self.current_z - 1)

        cv2.destroyAllWindows()

    def run_verification(self):
        for folder in sorted(os.listdir(self.save_dir)):
            if not folder.startswith("cell"):
                continue
            annotation_path = os.path.join(self.save_dir, folder, f"{folder}_annotation.json")
            with open(annotation_path, 'r') as f:
                    annotation_data = json.load(f)
                    bbox = annotation_data["bbox"]
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            f.close()
            for zi in range(self.z_depth):
                img_name = f"{folder}_z{zi}.png"
                img_path = os.path.join(self.save_dir, folder, img_name)

                img = cv2.imread(img_path)

                label = None

                while True:
                    disp_img = img.copy()
                    cv2.namedWindow(f"{folder}  z: {zi+1}/{self.z_depth}", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(f"{folder}  z: {zi+1}/{self.z_depth}", 800, 600)
                    cv2.imshow(f"{folder}  z: {zi+1}/{self.z_depth}", disp_img)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('g'):
                        label = True
                        cv2.destroyAllWindows()
                        break
                    elif key == ord('b'):
                        label = False
                        cv2.destroyAllWindows()
                        break

               
                with open(annotation_path, 'r') as f:
                    try:
                        existing_data = json.load(f)
                        if isinstance(existing_data, dict):
                            existing_data = [existing_data]
                    except json.JSONDecodeError:
                        # If the file is empty or invalid, initialize as an empty list
                        existing_data = []

                # Append the new annotation data
                new_data = {
                    "z": int(zi),
                    "nucleus": label,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                }
                existing_data.append(new_data)

                # Overwrite the file with the updated content
                with open(annotation_path, 'w') as f:
                    json.dump(existing_data, f, indent=4)

        cv2.destroyAllWindows()
