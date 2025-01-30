import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functions.hough_transformation_func import hough_circle_transform

def main():
    parser = argparse.ArgumentParser(description="Hough Circle Transform with Custom Parameters")
    parser.add_argument("--image", type=str, default="datasets/image_hough_small.png",
                        help="Path to input image (default: 'datasets/image_hough_small.png')")
    parser.add_argument("--r_min", type=int, default=30, help="Minimum circle radius (default: 30)")
    parser.add_argument("--r_max", type=int, default=40, help="Maximum circle radius (default: 40)")
    parser.add_argument("--threshold", type=int, default=150, help="Accumulator threshold (default: 150)")

    args = parser.parse_args()

    # Create results directory if it does not exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Load Image
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Error: Could not load image '{args.image}'!")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Apply Hough Circle Transform
    found_circles = hough_circle_transform(edges, r_min=args.r_min, r_max=args.r_max, threshold=args.threshold)

    # Draw detected circles
    output = image.copy()
    for (cx, cy, r) in found_circles:
        cv2.circle(output, (cx, cy), 2, (0, 255, 0), -1)  # Center point
        cv2.circle(output, (cx, cy), r, (255, 0, 0), 2)   # Circle perimeter

    # Save images to results folder
    edges_path = os.path.join(results_dir, "edges.png")
    output_path = os.path.join(results_dir, "hough_circles.png")
    cv2.imwrite(edges_path, edges)
    cv2.imwrite(output_path, output)

    print(f"Saved edges image to {edges_path}")
    print(f"Saved Hough Circles result to {output_path}")

    # Display original and output images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Hough Circle Transformation")
    axes[1].axis("off")

    # Save figure as PNG
    figure_path = os.path.join(results_dir, "hough_results.png")
    plt.savefig(figure_path, bbox_inches='tight')
    print(f"Saved results figure to {figure_path}")

    plt.show()

if __name__ == "__main__":
    main()
