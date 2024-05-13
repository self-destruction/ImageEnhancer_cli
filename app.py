import numpy as np
from PIL import Image, ImageEnhance, ImageCms
import argparse
import torch
from utils.utils import merge_adjustments_with_blend_modes, apply_gamma_correction, sRGB_profile, Lab_profile

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str)
parser.add_argument("--save_file_name", type=str)
# ID image (main)
parser.add_argument("--image", type=str)
# [0.0, 5.0]
parser.add_argument("--hdr_intensity", type=float, default=1.25)
# [0.00, 1.00]
parser.add_argument("--shadow_intensity", type=float, default=0.5)
# [0.00, 1.00]
parser.add_argument("--highlight_intensity", type=float, default=1.00)
# [0.00, 1.00]
parser.add_argument("--gamma_intensity", type=float, default=0.25)
# [0.00, 1.00]
parser.add_argument("--contrast", type=float, default=0.1)
# [0.00, 1.00]
parser.add_argument("--enhance_color", type=float, default=0.25)
args = parser.parse_args()
print(args)


def apply_hdr2(img, hdr_intensity=0.5, shadow_intensity=0.25, highlight_intensity=0.75, gamma_intensity=0.25,
               contrast=0.1, enhance_color=0.25):
    # Step 1: Convert RGB to LAB for better color preservation
    img_lab = ImageCms.profileToProfile(img, sRGB_profile, Lab_profile, outputMode='LAB')

    # Extract L, A, and B channels
    luminance, a, b = img_lab.split()

    # Convert luminance to a NumPy array for processing
    lum_array = np.array(luminance, dtype=np.float32)

    merged_adjustments = merge_adjustments_with_blend_modes(lum_array, hdr_intensity, shadow_intensity, highlight_intensity)

    # Apply gamma correction with a base_gamma value (define based on desired effect)
    gamma_corrected = apply_gamma_correction(merged_adjustments, gamma_intensity)
    gamma_corrected = Image.fromarray(gamma_corrected).resize(a.size)

    # Merge L channel back with original A and B channels
    adjusted_lab = Image.merge('LAB', (gamma_corrected, a, b))

    # Step 3: Convert LAB back to RGB
    img_adjusted = ImageCms.profileToProfile(adjusted_lab, Lab_profile, sRGB_profile, outputMode='RGB')

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img_adjusted)
    contrast_adjusted = enhancer.enhance(1 + contrast)

    # Enhance color saturation
    enhancer = ImageEnhance.Color(contrast_adjusted)
    color_adjusted = enhancer.enhance(1 + enhance_color * 0.2)

    return color_adjusted


img = apply_hdr2(
    Image.open(args.image),
    args.hdr_intensity,
    args.shadow_intensity,
    args.highlight_intensity,
    args.gamma_intensity,
    args.contrast,
    args.enhance_color,
)

img.save(f'{args.save_dir}/{args.save_file_name}.png')
