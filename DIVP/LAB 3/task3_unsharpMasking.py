import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("source.jpg", 0).astype(float)

# Add Gaussian noise: sigma = 10
noisy = np.clip(img + np.random.normal(0, 10, img.shape), 0, 255)

# Averaging filter
def average(img, size=3):
    kernel = np.ones((size, size)) / (size * size)
    return cv2.filter2D(img, -1, kernel)

# High-boost / Unsharp masking
def high_boost(img, k):
    smooth = average(img, 3)
    mask = img - smooth
    return np.clip(img + k * mask, 0, 255)

# Sharpness using gradient magnitude
def sharpness(img):
    gx, gy = np.gradient(img)
    return np.mean(np.sqrt(gx**2 + gy**2))

# k values
k_values = [1, 1.5, 2, 3]
sharp_values = []

plt.figure(figsize=(10, 6))

for i, k in enumerate(k_values):
    result = high_boost(noisy, k)
    sharp_values.append(sharpness(result))

    plt.subplot(2, 2, i + 1)
    plt.imshow(result, cmap="gray")
    plt.title(f"High Boost k = {k}")
    plt.axis("off")

plt.suptitle("Task 3: Unsharp Masking / High-Boost Filtering")
plt.tight_layout()
plt.show()

# Sharpness graph
plt.plot(k_values, sharp_values, marker="o")
plt.xlabel("Boost Factor k")
plt.ylabel("Sharpness")
plt.title("Sharpness vs High-Boost Factor")
plt.grid()
plt.show()