import numpy as np

def hough_circle_transform(edges, r_min, r_max, threshold=100):
    """
    Performs the Hough Circle Transform on an edge-detected binary image.
    
    Args:
        edges (numpy.ndarray): Binary image with detected edges (e.g., from Canny).
        r_min (int): Minimum circle radius.
        r_max (int): Maximum circle radius.
        threshold (int): Minimum vote count in the accumulator for a circle to be considered valid.
    
    Returns:
        list: A list of detected circles in the format (center_x, center_y, radius).
    """
    # Image dimensions
    height, width = edges.shape
    
    # Create a 3D accumulator array (b, a, r)
    accumulator = np.zeros((height, width, r_max+1), dtype=np.uint64)
    
    # Find edge pixels
    edge_points = np.argwhere(edges > 0)  # (y, x) points where edges exist

    # Convert angles to radians
    thetas = np.deg2rad(np.arange(0, 360))  # Angles from 0° to 359°
    
    # Iterate through edge pixels
    for y, x in edge_points:
        # Vote for centers for each radius in the given range
        for r in range(r_min, r_max+1):
            if r == 0:
                continue
            
            # Iterate through all angles to determine potential centers
            for theta in thetas:
                a = int(x - r * np.cos(theta))
                b = int(y - r * np.sin(theta))
                
                # Ensure the center (a, b) is within image bounds
                if 0 <= a < width and 0 <= b < height:
                    accumulator[b, a, r] += 1
    
    # Extract circles from accumulator exceeding threshold
    circles = []
    
    for b in range(height):
        for a in range(width):
            for r in range(r_min, r_max+1):
                if accumulator[b, a, r] >= threshold:
                    circles.append((a, b, r))
    
    return circles
