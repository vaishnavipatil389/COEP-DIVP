import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image as grayscale
# image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("image.jpg", 0)

if image is None:
    print("Image not found!")
    exit()

# 1. Negative Transformation
negative = cv2.bitwise_not(image)

# 2. Power-Law Transformation
gamma = 0.5
normalized = image / 255.0
power_law = np.power(normalized, gamma) * 255
power_law = np.uint8(power_law)

# 3. Log Transformation
c = 255 / np.log(1 + 255)
log_image = c * np.log(1 + image)
log_image = np.uint8(log_image)

# Display results
plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(negative, cmap="gray")
plt.title("Negative")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(power_law, cmap="gray")
plt.title("Power-Law (Gamma=0.5)")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(log_image, cmap="gray")
plt.title("Log Transformation")
plt.axis("off")

plt.tight_layout()
plt.show()