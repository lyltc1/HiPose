from typing import Dict, List, Tuple, Union
import numpy as np
import random
import cv2


class DepthTransform():
    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, depth) -> np.ndarray:
        depth = self._transform_depth(depth)
        return depth

class DepthBlurTransform(DepthTransform):
    def __init__(self, factor_interval=(3, 7)):
        self.factor_interval = factor_interval

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.copy(depth)
        k = random.randint(*self.factor_interval)
        depth = cv2.blur(depth, (k, k))
        return depth

class DepthEllipseDropoutTransform(DepthTransform):
    def __init__(
        self,
        ellipse_dropout_mean: float = 10.0,
        ellipse_gamma_shape: float = 5.0,
        ellipse_gamma_scale: float = 1.0,
    ) -> None:
        self._noise_params = {
            "ellipse_dropout_mean": ellipse_dropout_mean,
            "ellipse_gamma_scale": ellipse_gamma_scale,
            "ellipse_gamma_shape": ellipse_gamma_shape,
        }

    @staticmethod
    def generate_random_ellipses(
        depth_img: np.ndarray, noise_params: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Sample number of ellipses to dropout
        num_ellipses_to_dropout = np.random.poisson(noise_params["ellipse_dropout_mean"])

        # Sample ellipse centers
        nonzero_pixel_indices = np.array(np.where(depth_img > 0)).T  # Shape: [#nonzero_pixels x 2]
        dropout_centers_indices = np.random.choice(
            nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout
        )
        # Shape: [num_ellipses_to_dropout x 2]
        dropout_centers = nonzero_pixel_indices[dropout_centers_indices, :]

        # Sample ellipse radii and angles
        x_radii = np.random.gamma(
            noise_params["ellipse_gamma_shape"],
            noise_params["ellipse_gamma_scale"],
            size=num_ellipses_to_dropout,
        )
        y_radii = np.random.gamma(
            noise_params["ellipse_gamma_shape"],
            noise_params["ellipse_gamma_scale"],
            size=num_ellipses_to_dropout,
        )
        angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

        return x_radii, y_radii, angles, dropout_centers

    @staticmethod
    def dropout_random_ellipses(
        depth_img: np.ndarray, noise_params: Dict[str, float]
    ) -> np.ndarray:
        """Randomly drop a few ellipses in the image for robustness.
        @param depth_img: a [H x W] set of depth z values
        """

        depth_img = depth_img.copy()

        (
            x_radii,
            y_radii,
            angles,
            dropout_centers,
        ) = DepthEllipseDropoutTransform.generate_random_ellipses(
            depth_img, noise_params=noise_params
        )

        num_ellipses_to_dropout = x_radii.shape[0]

        # Dropout ellipses
        for i in range(num_ellipses_to_dropout):
            center = dropout_centers[i, :]
            x_radius = np.round(x_radii[i]).astype(int)
            y_radius = np.round(y_radii[i]).astype(int)
            angle = angles[i]

            depth_img = cv2.ellipse(
                depth_img,
                tuple(center[::-1]),
                (x_radius, y_radius),
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=0,
                thickness=-1,
            )

        return depth_img

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = self.dropout_random_ellipses(depth, self._noise_params)
        return depth

class DepthGaussianNoiseTransform(DepthTransform):
    """Adds random Gaussian noise to the depth image."""

    def __init__(self, std_dev: float = 0.02):
        self.std_dev = std_dev

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.copy(depth)
        noise = np.random.normal(scale=self.std_dev, size=depth.shape)
        depth[depth > 0] += noise[depth > 0]
        depth = np.clip(depth, 0, np.finfo(np.float32).max)
        return depth


class DepthMissingTransform(DepthTransform):
    """Randomly drop-out parts of the depth image."""

    def __init__(self, max_missing_fraction: float = 0.2, debug: bool = False):
        self.max_missing_fraction = max_missing_fraction
        self.debug = debug

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.copy(depth)
        v_idx, u_idx = np.where(depth > 0)
        if not self.debug:
            missing_fraction = np.random.uniform(0, self.max_missing_fraction)
        else:
            missing_fraction = self.max_missing_fraction
        dropout_ids = np.random.choice(
            np.arange(len(u_idx)), int(missing_fraction * len(u_idx)), replace=False
        )
        depth[v_idx[dropout_ids], u_idx[dropout_ids]] = 0
        return depth


def depth_augmentation(depth, depth_augmentation_level):
    if depth_augmentation_level == 0:
        depth = np.copy(depth)
        k = random.randint(3, 7)


def add_noise_depth(depth, level=0.005, depth_valid_min=0):
    # from DeepIM-PyTorch and se3tracknet
    # in deepim: level=0.1, valid_min=0
    # in se3tracknet, level=5/1000, depth_valid_min = 100/1000 = 0.1

    if len(depth.shape) == 3:
        mask = depth[:, :, -1] > depth_valid_min
        row, col, ch = depth.shape
        noise_level = random.uniform(0, level)
        gauss = noise_level * np.random.randn(row, col)
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
    else:  # 2
        mask = depth > depth_valid_min

        change_mask = np.random.uniform(0, 1, size=depth.shape[:2])
        change_mask = change_mask < 0.1

        mask = mask & change_mask

        row, col = depth.shape
        noise_level = random.uniform(0, level)
        gauss = noise_level * np.random.randn(row, col)
        gauss = gauss * 1000.
    noisy = depth.copy()
    noisy[mask] = depth[mask] + gauss[mask]
    return noisy

