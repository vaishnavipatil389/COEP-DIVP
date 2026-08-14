import cv2
import matplotlib.pyplot as plt
from skimage import exposure

# ===================================================
# Read Images
# ===================================================

source = cv2.imread("source.jpg", 0)
reference = cv2.imread("reference.jpg", 0)

if source is None:
    print("source2.jpg not found!")
    exit()

if reference is None:
    print("reference.jpg not found!")
    exit()


# ===================================================
# 1. Histogram Equalization
# ===================================================

equalized = cv2.equalizeHist(source)


# ===================================================
# 2. Histogram Matching / Specification
# ===================================================

matched = exposure.match_histograms(source, reference).astype("uint8")


# ===================================================
# Display Results
# ===================================================

plt.figure(figsize=(14, 10))

# ---------- Original Image ----------
plt.subplot(3, 3, 1)
plt.imshow(source, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# ---------- Reference Image ----------
plt.subplot(3, 3, 2)
plt.imshow(reference, cmap="gray")
plt.title("Reference Image")
plt.axis("off")

# ---------- Equalized Image ----------
plt.subplot(3, 3, 3)
plt.imshow(equalized, cmap="gray")
plt.title("Histogram Equalized")
plt.axis("off")

# ---------- Original Histogram ----------
plt.subplot(3, 3, 4)
plt.hist(source.ravel(), 256, [0, 256])
plt.title("Original Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

# ---------- Reference Histogram ----------
plt.subplot(3, 3, 5)
plt.hist(reference.ravel(), 256, [0, 256])
plt.title("Reference Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

# ---------- Equalized Histogram ----------
plt.subplot(3, 3, 6)
plt.hist(equalized.ravel(), 256, [0, 256])
plt.title("Equalized Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

# ---------- Matched Image ----------
plt.subplot(3, 3, 7)
plt.imshow(matched, cmap="gray")
plt.title("Histogram Matched")
plt.axis("off")

# ---------- Matched Histogram ----------
plt.subplot(3, 3, 8)
plt.hist(matched.ravel(), 256, [0, 256])
plt.title("Matched Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()