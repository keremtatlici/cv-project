import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functions.connected_components_func import connected_components_labeling, labels_to_color

def main():
    parser = argparse.ArgumentParser(description="Connected Components Labeling in Images")
    parser.add_argument("--image", type=str, default="datasets/connected_comps.png",
                        help="Path to input image (default: 'datasets/connected_comps.png')")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=8,
                        help="Connectivity type: 4 or 8 (default: 8)")
    parser.add_argument("--bg_color", type=int, nargs=3, default=[255, 255, 255],
                        help="Background color as RGB tuple (default: 255 255 255)")

    args = parser.parse_args()

    # Create results directory if it does not exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Load image
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image '{args.image}' could not be loaded!")

    # Invert thresholding to make sure foreground is white
    _, binary_inv = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Perform connected components labeling
    labels, n_labels = connected_components_labeling(binary_inv, connectivity=args.connectivity)
    print("Number of detected components (excluding background):", n_labels)

    # Colorize components
    colored_result = labels_to_color(labels, n_labels, bg_color=tuple(args.bg_color))

    # Save images to results folder
    binary_inv_path = os.path.join(results_dir, "binary_inverted.png")
    colored_result_path = os.path.join(results_dir, "connected_components.png")
    cv2.imwrite(binary_inv_path, binary_inv)
    cv2.imwrite(colored_result_path, colored_result)

    print(f"Saved binary inverted image to {binary_inv_path}")
    print(f"Saved connected components image to {colored_result_path}")

    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Grayscale Image")
    axes[0].axis('off')

    axes[1].imshow(binary_inv, cmap='gray')
    axes[1].set_title("Binary Inverted Image (Foreground=255, Background=0)")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(colored_result, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Labeled Components with Random Colors")
    axes[2].axis('off')

    # Save figure as PNG
    figure_path = os.path.join(results_dir, "results_figure.png")
    plt.savefig(figure_path, bbox_inches='tight')
    print(f"Saved results figure to {figure_path}")

    plt.show()

if __name__ == "__main__":
    main()
