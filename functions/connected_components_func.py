import numpy as np
from collections import deque


def connected_components_labeling(binary_img, connectivity=8):
    """
    Performs connected components labeling on a binary image.

    Args:
        binary_img (numpy.ndarray): Binary image with values 0 or 255 (foreground=255, background=0).
        connectivity (int): Defines the neighborhood connectivity (4 or 8).

    Returns:
        tuple: A tuple containing:
            - labels (numpy.ndarray): 2D array of int32 with labeled components.
            - n_labels (int): Number of components found.
    """
    h, w = binary_img.shape
    labels = np.zeros((h, w), dtype=np.int32)

    if connectivity == 8:
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      ( 0, -1),          ( 0, 1),
                      ( 1, -1), ( 1, 0), ( 1, 1)]
    else:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    current_label = 0

    for y in range(h):
        for x in range(w):
            if binary_img[y, x] == 255 and labels[y, x] == 0:
                current_label += 1
                labels[y, x] = current_label

                queue = deque()
                queue.append((y, x))

                while queue:
                    cy, cx = queue.popleft()
                    for dy, dx in directions:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if binary_img[ny, nx] == 255 and labels[ny, nx] == 0:
                                labels[ny, nx] = current_label
                                queue.append((ny, nx))

    return labels, current_label

def labels_to_color(labels, n_labels, bg_color=(255, 255, 255)):
    """
    Assigns random colors to labeled components for visualization.

    Args:
        labels (numpy.ndarray): 2D array with labeled components.
        n_labels (int): Number of unique labels.
        bg_color (tuple): Background color as an RGB tuple (default is white).

    Returns:
        numpy.ndarray: Colorized image with random colors assigned to components.
    """
    h, w = labels.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    color_img[:] = bg_color

    rng = np.random.default_rng()
    random_colors = rng.integers(0, 256, size=(n_labels+1, 3), dtype=np.uint8)
    random_colors[0] = bg_color

    for y in range(h):
        for x in range(w):
            lbl = labels[y, x]
            if lbl != 0:
                color_img[y, x] = random_colors[lbl]

    return color_img
