import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("source.jpg", 0).astype(float)

# Add Gaussian noise
noisy10 = np.clip(img + np.random.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
noisy25 = np.clip(img + np.random.normal(0, 25, img.shape), 0, 255).astype(np.uint8)

# Apply averaging filters
sizes = [3, 5, 9]

plt.figure(figsize=(10, 6))
p = 1

for noisy, sigma in [(noisy10, 10), (noisy25, 25)]:
    for size in sizes:
        result = cv2.blur(noisy, (size, size))

        plt.suptitle("Task 1: Noise Suppression Using Averaging Filter")
        plt.subplot(2, 3, p)
        plt.imshow(result, cmap="gray")
        plt.title(f"σ={sigma}, {size}×{size}")
        plt.axis("off")
        p += 1

plt.tight_layout()
plt.show()