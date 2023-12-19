# Generates Ground Truth Heatmap

import numpy as np
import cv2

def generate_ground_truth_heatmap(image, kernel_size, sigma):
    height, width = image.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            if image[i, j] > 0:
                x, y = int(j), int(i)
                heatmap[max(0, y-kernel_size):min(height, y+kernel_size+1),
                        max(0, x-kernel_size):min(width, x+kernel_size+1)] = 1

    heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), sigma)
    return heatmap