import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("source.jpg", 0).astype(float)

# Motion blur
kernel = np.zeros((9, 9))
kernel[4, :] = 1 / 9
blurred = correlate = None

def correlate2d(img, kernel):
    p = kernel.shape[0] // 2
    padded = np.pad(img, p, mode="edge")
    out = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j] = np.sum(
                padded[i:i+kernel.shape[0],
                       j:j+kernel.shape[1]] * kernel
            )
    return out

blurred = correlate2d(img, kernel)

# Laplacian kernels
L4 = np.array([[0,-1,0],
               [-1,4,-1],
               [0,-1,0]])

L8 = np.array([[-1,-1,-1],
               [-1, 8,-1],
               [-1,-1,-1]])

# Laplacian responses
response4 = correlate2d(blurred, L4)
response8 = correlate2d(blurred, L8)

# Sharpened images
sharp4 = np.clip(blurred + response4, 0, 255)
sharp8 = np.clip(blurred + response8, 0, 255)

# Display
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(blurred, cmap="gray")
plt.title("Motion Blurred")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(response4, cmap="gray")
plt.title("4-Neighbour Response")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(response8, cmap="gray")
plt.title("8-Neighbour Response")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(sharp4, cmap="gray")
plt.title("4-Neighbour Sharpened")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(sharp8, cmap="gray")
plt.title("8-Neighbour Sharpened")
plt.axis("off")

plt.suptitle("Task 2: Boundary Recovery Using Laplacian Sharpening")
plt.tight_layout()
plt.show()