import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from scipy.ndimage import convolve
from skimage.restoration import denoise_bilateral


# ============================================================
# Exercise 1: Bayer pattern
# Detected from IMG_9939.npy
# Pattern: GRBG
#
# G R
# B G
# ============================================================


def create_bayer_masks(shape):
    h, w = shape

    red_mask = np.zeros((h, w), dtype=np.float32)
    green_mask = np.zeros((h, w), dtype=np.float32)
    blue_mask = np.zeros((h, w), dtype=np.float32)

    # GRBG pattern
    green_mask[0::2, 0::2] = 1
    green_mask[1::2, 1::2] = 1
    red_mask[0::2, 1::2] = 1
    blue_mask[1::2, 0::2] = 1

    return red_mask, green_mask, blue_mask


# ============================================================
# Exercise 2: Lecture demosaicing method
# C = ((Mc * X) convolved K) / (Mc convolved K)
# ============================================================


def demosaic(raw_data, kernel_size=3):
    raw_data = raw_data.astype(np.float32)

    red_mask, green_mask, blue_mask = create_bayer_masks(raw_data.shape)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

    def interpolate_channel(mask):
        numerator = convolve(mask * raw_data, kernel, mode="mirror")
        denominator = convolve(mask, kernel, mode="mirror")
        denominator[denominator == 0] = 1
        return numerator / denominator

    red = interpolate_channel(red_mask)
    green = interpolate_channel(green_mask)
    blue = interpolate_channel(blue_mask)

    return np.stack([red, green, blue], axis=-1)


# ============================================================
# Utility functions
# ============================================================


def percentile_bounds(data, low=0.01, high=99.99):
    data = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    a = np.percentile(data, low)
    b = np.percentile(data, high)
    return a, b


def normalize_with_bounds(data, a, b):
    if b <= a:
        return np.zeros_like(data, dtype=np.float32)

    data = (data - a) / (b - a)
    data = np.clip(data, 0, 1)
    return data.astype(np.float32)


def percentile_normalize(data, low=0.01, high=99.99):
    a, b = percentile_bounds(data, low, high)
    return normalize_with_bounds(data, a, b)


def normalize_for_export(data):
    return percentile_normalize(data, 0.01, 99.99)


# ============================================================
# Exercise 3: Gamma correction
# Sheet workflow:
# normalize -> apply gamma -> invert back to previous range
# ============================================================


def gamma_correction(data, gamma=0.3):
    data = data.astype(np.float32)
    a, b = percentile_bounds(data, 0.01, 99.99)

    normalized = normalize_with_bounds(data, a, b)
    corrected = np.power(normalized, gamma)

    # invert back to the original percentile range
    corrected = corrected * (b - a) + a

    return corrected.astype(np.float32)


# ============================================================
# Exercise 3: Alternative curve
# Logarithmic curve with same normalize -> curve -> invert workflow
# ============================================================


def logarithmic_curve(data):
    data = data.astype(np.float32)
    a, b = percentile_bounds(data, 0.01, 99.99)

    normalized = normalize_with_bounds(data, a, b)
    corrected = np.log1p(5 * normalized) / np.log1p(5)

    corrected = corrected * (b - a) + a

    return corrected.astype(np.float32)


# ============================================================
# Exercise 4: Gray-world white balance
# Channels are scaled to match the green-channel mean.
# Values are clipped to the current valid range after multiplication.
# ============================================================


def gray_world_white_balance(rgb, clip_max=None):
    rgb = rgb.astype(np.float32)

    mean_r = np.mean(rgb[:, :, 0])
    mean_g = np.mean(rgb[:, :, 1])
    mean_b = np.mean(rgb[:, :, 2])

    eps = 1e-6
    rgb[:, :, 0] *= mean_g / (mean_r + eps)
    rgb[:, :, 2] *= mean_g / (mean_b + eps)

    if clip_max is None:
        clip_max = np.percentile(rgb, 99.99)

    rgb = np.clip(rgb, 0, clip_max)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)

    return rgb.astype(np.float32)


# ============================================================
# Save image: uint8 conversion only happens here
# ============================================================


def save_image(path, rgb):
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = np.clip(rgb, 0, 1)

    rgb_uint8 = (rgb * 255).astype(np.uint8)
    imageio.imwrite(path, rgb_uint8, quality=98)


# ============================================================
# Load RAW CR3 file
# ============================================================


def load_raw(path):
    raw = rawpy.imread(path)
    return np.array(raw.raw_image_visible).astype(np.float32)


# ============================================================
# Exercise 5: Sensor linearity
# Use mean raw sensor value, not channel-wise values.
# ============================================================


def sensor_linearity_plot(data_folder):
    files = [
        "IMG_3044.CR3",
        "IMG_3045.CR3",
        "IMG_3046.CR3",
        "IMG_3047.CR3",
        "IMG_3048.CR3",
        "IMG_3049.CR3",
    ]

    exposure_times = np.array(
        [1 / 10, 1 / 20, 1 / 40, 1 / 80, 1 / 160, 1 / 320],
        dtype=np.float32,
    )

    mean_values = []

    for file in files:
        path = os.path.join(data_folder, file)
        if not os.path.exists(path):
            print("Missing file:", path)
            return

        raw_data = load_raw(path)
        mean_values.append(np.mean(raw_data))

    plt.figure(figsize=(8, 5))
    plt.plot(exposure_times, mean_values, "o-")
    plt.xlabel("Exposure Time (s)")
    plt.ylabel("Mean Sensor Value")
    plt.title("Sensor Linearity")
    plt.grid(True)
    plt.savefig("sensor_linearity.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved sensor_linearity.png")


# ============================================================
# Exercise 6: HDR combination using lecture method
# Start with brightest image h.
# Scale shorter exposures to match the first exposure.
# Replace values in h above fixed threshold t.
# t = 0.8 * max(initial h)
# ============================================================


def combine_hdr(raw_images, exposure_times, threshold_ratio=0.8):
    hdr = raw_images[0].copy().astype(np.float32)
    first_exposure = exposure_times[0]

    # Fixed threshold based on the original brightest image
    threshold = threshold_ratio * np.max(hdr)

    for i in range(1, len(raw_images)):
        current = raw_images[i].astype(np.float32)
        scale = first_exposure / exposure_times[i]
        current_scaled = current * scale

        saturated_mask = hdr > threshold
        hdr[saturated_mask] = current_scaled[saturated_mask]

    hdr = np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0)

    return hdr.astype(np.float32)


# ============================================================
# Exercise 6: Log tone mapping
# ============================================================


def log_tone_mapping(rgb):
    rgb = np.maximum(rgb.astype(np.float32), 1e-6)
    log_rgb = np.log1p(rgb)
    return percentile_normalize(log_rgb, 0.01, 99.99)


# ============================================================
# Exercise 7: iCAM06 from lecture pseudocode
# Lecture intensity formula:
# input_intensity = 1/61 * (20*R + 40*G + B)
# ============================================================


def icam06(rgb, output_range=4, sigma_color=0.1, sigma_spatial=2):
    rgb = np.maximum(rgb.astype(np.float32), 1e-6)

    input_intensity = (
        20 * rgb[:, :, 0]
        + 40 * rgb[:, :, 1]
        + rgb[:, :, 2]
    ) / 61.0

    input_intensity = np.maximum(input_intensity, 1e-6)

    r = rgb[:, :, 0] / input_intensity
    g = rgb[:, :, 1] / input_intensity
    b = rgb[:, :, 2] / input_intensity

    log_intensity = np.log(input_intensity)

    log_base = denoise_bilateral(
        log_intensity,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        channel_axis=None,
    )

    log_details = log_intensity - log_base
    base_range = np.max(log_base) - np.min(log_base)

    if base_range <= 1e-6:
        return percentile_normalize(rgb)

    compression = np.log(output_range) / base_range
    log_offset = -np.max(log_base) * compression

    output_intensity = np.exp(
        log_base * compression + log_offset + log_details
    )

    out_r = r * output_intensity
    out_g = g * output_intensity
    out_b = b * output_intensity

    output = np.stack([out_r, out_g, out_b], axis=-1)

    return percentile_normalize(output, 0.01, 99.99)


# ============================================================
# Exercise 8: Required function
# process_raw(input_path, output_path)
# ============================================================


def process_raw(input_path, output_path):
    raw_data = load_raw(input_path)

    rgb = demosaic(raw_data, kernel_size=3)
    rgb = gamma_correction(rgb, gamma=0.3)
    rgb = gray_world_white_balance(rgb)

    rgb_out = normalize_for_export(rgb)
    save_image(output_path, rgb_out)

    print("Saved", output_path)


# ============================================================
# Main execution
# Expected folder:
# data/
#   IMG_4782.CR3
#   IMG_3044.CR3 ... IMG_3049.CR3
#   00.CR3 ... 10.CR3
# ============================================================


if __name__ == "__main__":
    data_folder = "data"

    # --------------------------------------------------------
    # Exercise 2, 3, 4
    # --------------------------------------------------------

    raw_path = os.path.join(data_folder, "IMG_4782.CR3")

    if os.path.exists(raw_path):
        raw_data = load_raw(raw_path)
        rgb = demosaic(raw_data, kernel_size=3)

        gamma_rgb = gamma_correction(rgb, gamma=0.3)
        gamma_rgb = gray_world_white_balance(gamma_rgb)
        gamma_rgb_out = normalize_for_export(gamma_rgb)
        save_image("exercise_gamma.jpg", gamma_rgb_out)
        print("Saved exercise_gamma.jpg")

        log_rgb = logarithmic_curve(rgb)
        log_rgb = gray_world_white_balance(log_rgb)
        log_rgb_out = normalize_for_export(log_rgb)
        save_image("exercise_log.jpg", log_rgb_out)
        print("Saved exercise_log.jpg")

        process_raw(raw_path, "final_result.jpg")

    else:
        print("Missing file:", raw_path)

    # --------------------------------------------------------
    # Exercise 5
    # --------------------------------------------------------

    sensor_linearity_plot(data_folder)

    # --------------------------------------------------------
    # Exercise 6 and 7
    # --------------------------------------------------------

    hdr_files = [f"{i:02d}.CR3" for i in range(11)]

    exposure_times = np.array(
        [
            1,
            1 / 2,
            1 / 4,
            1 / 8,
            1 / 16,
            1 / 32,
            1 / 64,
            1 / 128,
            1 / 256,
            1 / 512,
            1 / 1024,
        ],
        dtype=np.float32,
    )

    raw_images = []
    used_exposures = []

    for file, exposure in zip(hdr_files, exposure_times):
        path = os.path.join(data_folder, file)

        if os.path.exists(path):
            raw_images.append(load_raw(path))
            used_exposures.append(exposure)
        else:
            print("Missing HDR file:", path)

    if len(raw_images) > 0:
        used_exposures = np.array(used_exposures, dtype=np.float32)

        hdr_raw = combine_hdr(raw_images, used_exposures, threshold_ratio=0.8)
        hdr_rgb = demosaic(hdr_raw, kernel_size=3)
        hdr_rgb = gray_world_white_balance(hdr_rgb)

        hdr_log = log_tone_mapping(hdr_rgb)
        save_image("hdr_log.jpg", hdr_log)
        print("Saved hdr_log.jpg")

        hdr_icam = icam06(
            hdr_rgb,
            output_range=4,
            sigma_color=0.1,
            sigma_spatial=2,
        )
        save_image("hdr_icam06.jpg", hdr_icam)
        print("Saved hdr_icam06.jpg")

    else:
        print("HDR skipped because no HDR CR3 files were found.")
