from aicsimageio import AICSImage
import numpy as np

class dvImage:
    def __init__(
        self,
        filepath,
        data_save_dir = "data/loaded_data",
    ):
        self.img = AICSImage(filepath)
        self.data = self.img.get_image_data("TZYXC")  # "CTZYX" = Time, Channel, Z-stack, Y, X
        self.rgb_data = self._normalize_rgb()

    def _normalize(self, img):
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    def _normalize_rgb(self):
        t, z, y, x, c = self.data.shape
        assert c == 3, "Expected 3 channels (RGB). Got {}".format(c)
        rgb_data = np.zeros((t, z, y, x, 3), dtype=np.float32)
        for ti in range(t):
            for zi in range(z):
                r = self._normalize(self.data[ti, zi, :, :, 0])
                g = self._normalize(self.data[ti, zi, :, :, 1])
                b = self._normalize(self.data[ti, zi, :, :, 2])
                rgb_data[ti, zi] = np.stack([r, g, b], axis=-1)

        return rgb_data

