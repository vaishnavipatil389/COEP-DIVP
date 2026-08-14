import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# DIGITAL IMAGE AND VIDEO PROCESSING
# SPATIAL FILTERING FOR AGV CAMERA IMAGE
# ============================================================


# ------------------------------------------------------------
# 1. READ CLEAN IMAGE
# ------------------------------------------------------------

# Put your image in the same folder as this Python program
image_path = "source.jpg"

clean = cv2.imread(image_path, 0)

if clean is None:
    print("ERROR: Image not found!")
    print("Check image name/path:", image_path)
    exit()

clean = clean.astype(np.float64)

print("Image loaded successfully!")
print("Image size:", clean.shape)


# ============================================================
# 2. CORRELATION FROM SCRATCH
# ============================================================

def correlate2d(image, kernel):

    image = image.astype(np.float64)
    kernel = np.array(kernel, dtype=np.float64)

    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    # Padding image
    padded = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode='edge'
    )

    output = np.zeros_like(image, dtype=np.float64)

    rows, cols = image.shape

    # Perform correlation manually
    for i in range(rows):
        for j in range(cols):

            region = padded[i:i+kh, j:j+kw]

            output[i, j] = np.sum(region * kernel)

    return output


# ============================================================
# 3. CONVOLUTION FROM SCRATCH
# ============================================================

def convolution2d(image, kernel):

    # Flip kernel horizontally and vertically
    flipped_kernel = np.flipud(np.fliplr(kernel))

    return correlate2d(image, flipped_kernel)


# ============================================================
# 4. ADD GAUSSIAN NOISE
# ============================================================

def add_gaussian_noise(image, sigma):

    noise = np.random.normal(
        0,
        sigma,
        image.shape
    )

    noisy = image + noise

    return np.clip(noisy, 0, 255)


# Generate required noisy images
noisy10 = add_gaussian_noise(clean, 10)
noisy25 = add_gaussian_noise(clean, 25)


# ============================================================
# 5. MOTION BLUR
# ============================================================

def motion_blur(image, length=9):

    kernel = np.zeros((length, length))

    # Horizontal motion blur
    kernel[length // 2, :] = 1

    kernel = kernel / length

    blurred = correlate2d(image, kernel)

    return np.clip(blurred, 0, 255)


motion_blurred = motion_blur(clean, 9)


# ============================================================
# DISPLAY ORIGINAL DEGRADATIONS
# ============================================================

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(clean, cmap='gray')
plt.title("Clean Ground Truth")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(noisy10, cmap='gray')
plt.title("Gaussian Noise σ = 10")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(noisy25, cmap='gray')
plt.title("Gaussian Noise σ = 25")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(motion_blurred, cmap='gray')
plt.title("Motion Blur (Length = 9)")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# TASK 1
# NOISE SUPPRESSION USING AVERAGING FILTER
# ============================================================

def averaging_filter(image, size):

    kernel = np.ones((size, size), dtype=np.float64)

    kernel = kernel / (size * size)

    result = correlate2d(image, kernel)

    return np.clip(result, 0, 255)


kernel_sizes = [3, 5, 9]

task1_results = {}


plt.figure(figsize=(12, 8))

plot_number = 1

for sigma, noisy_image in [(10, noisy10), (25, noisy25)]:

    for size in kernel_sizes:

        filtered = averaging_filter(noisy_image, size)

        task1_results[(sigma, size)] = filtered

        plt.subplot(2, 3, plot_number)

        plt.imshow(filtered, cmap='gray')

        plt.title(
            "σ = " + str(sigma) +
            ", Kernel = " + str(size) + "x" + str(size)
        )

        plt.axis("off")

        plot_number += 1


plt.suptitle("Task 1: Averaging Filter Results")

plt.tight_layout()

plt.show()


# ============================================================
# TASK 2
# LAPLACIAN SHARPENING
# ============================================================

# 4-neighbour Laplacian
laplacian4 = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=np.float64)


# 8-neighbour Laplacian
laplacian8 = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float64)


# Calculate Laplacian response
response4 = correlate2d(
    motion_blurred,
    laplacian4
)

response8 = correlate2d(
    motion_blurred,
    laplacian8
)


# Sharpen image
sharp4 = motion_blurred + response4

sharp8 = motion_blurred + response8


sharp4 = np.clip(sharp4, 0, 255)
sharp8 = np.clip(sharp8, 0, 255)


# Display Task 2
plt.figure(figsize=(12, 8))


plt.subplot(2, 3, 1)

plt.imshow(motion_blurred, cmap='gray')

plt.title("Motion Blurred Image")

plt.axis("off")


plt.subplot(2, 3, 2)

plt.imshow(response4, cmap='gray')

plt.title("4-Neighbour Laplacian Response")

plt.axis("off")


plt.subplot(2, 3, 3)

plt.imshow(response8, cmap='gray')

plt.title("8-Neighbour Laplacian Response")

plt.axis("off")


plt.subplot(2, 3, 5)

plt.imshow(sharp4, cmap='gray')

plt.title("4-Neighbour Sharpened")

plt.axis("off")


plt.subplot(2, 3, 6)

plt.imshow(sharp8, cmap='gray')

plt.title("8-Neighbour Sharpened")

plt.axis("off")


plt.suptitle("Task 2: Laplacian Sharpening")

plt.tight_layout()

plt.show()


# ============================================================
# TASK 3
# UNSHARP MASKING AND HIGH-BOOST FILTERING
# ============================================================

def high_boost_filter(image, k):

    # Use 3x3 averaging filter
    smooth = averaging_filter(image, 3)

    # Mask
    mask = image - smooth

    # High boost
    result = image + k * mask

    return np.clip(result, 0, 255)


k_values = [1, 1.5, 2, 3]

task3_results = {}


plt.figure(figsize=(12, 8))


for index, k in enumerate(k_values):

    result = high_boost_filter(noisy10, k)

    task3_results[k] = result

    plt.subplot(2, 2, index + 1)

    plt.imshow(result, cmap='gray')

    plt.title("High Boost k = " + str(k))

    plt.axis("off")


plt.suptitle("Task 3: Unsharp Masking / High-Boost")

plt.tight_layout()

plt.show()


# ============================================================
# TASK 4
# PSNR CALCULATION
# ============================================================

def calculate_psnr(original, processed):

    mse = np.mean(
        (original - processed) ** 2
    )

    if mse == 0:
        return float('inf')

    psnr = 10 * np.log10(
        (255 ** 2) / mse
    )

    return psnr


# ============================================================
# SHARPNESS METRIC
# MEAN GRADIENT MAGNITUDE
# ============================================================

def sharpness_metric(image):

    # Simple gradient kernels
    gx_kernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])

    gy_kernel = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])

    gx = correlate2d(image, gx_kernel)

    gy = correlate2d(image, gy_kernel)

    magnitude = np.sqrt(
        gx ** 2 + gy ** 2
    )

    return np.mean(magnitude)


# ============================================================
# TASK 4A
# METRICS FOR AVERAGING FILTER
# ============================================================

print("\n")
print("=" * 65)
print("TASK 1: AVERAGING FILTER METRICS")
print("=" * 65)

print(
    "{:<10} {:<12} {:<15} {:<15}".format(
        "Sigma",
        "Kernel",
        "PSNR",
        "Sharpness"
    )
)

print("-" * 65)


for sigma in [10, 25]:

    for size in kernel_sizes:

        result = task1_results[(sigma, size)]

        psnr = calculate_psnr(
            clean,
            result
        )

        sharpness = sharpness_metric(
            result
        )

        print(
            "{:<10} {:<12} {:<15.3f} {:<15.3f}".format(
                sigma,
                str(size) + "x" + str(size),
                psnr,
                sharpness
            )
        )


# ============================================================
# TASK 4B
# LAPLACIAN METRICS
# ============================================================

print("\n")
print("=" * 65)
print("TASK 2: LAPLACIAN SHARPENING METRICS")
print("=" * 65)


print(
    "{:<20} {:<15} {:<15}".format(
        "Variant",
        "PSNR",
        "Sharpness"
    )
)

print("-" * 65)


psnr4 = calculate_psnr(clean, sharp4)

sharpness4 = sharpness_metric(sharp4)


psnr8 = calculate_psnr(clean, sharp8)

sharpness8 = sharpness_metric(sharp8)


print(
    "{:<20} {:<15.3f} {:<15.3f}".format(
        "4-Neighbour",
        psnr4,
        sharpness4
    )
)


print(
    "{:<20} {:<15.3f} {:<15.3f}".format(
        "8-Neighbour",
        psnr8,
        sharpness8
    )
)


# ============================================================
# TASK 4C
# HIGH BOOST METRICS
# ============================================================

print("\n")
print("=" * 65)
print("TASK 3: HIGH-BOOST FILTER METRICS")
print("=" * 65)


print(
    "{:<10} {:<15} {:<15}".format(
        "k",
        "PSNR",
        "Sharpness"
    )
)

print("-" * 65)


sharpness_values = []


for k in k_values:

    result = task3_results[k]

    psnr = calculate_psnr(
        clean,
        result
    )

    sharpness = sharpness_metric(
        result
    )

    sharpness_values.append(sharpness)

    print(
        "{:<10} {:<15.3f} {:<15.3f}".format(
            k,
            psnr,
            sharpness
        )
    )


# ============================================================
# TASK 3 GRAPH
# SHARPNESS vs k
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    k_values,
    sharpness_values,
    marker='o',
    linewidth=2
)

plt.xlabel("Boost Factor k")

plt.ylabel("Mean Gradient Magnitude")

plt.title("Sharpness vs High-Boost Factor k")

plt.grid(True)

plt.show()


# ============================================================
# TASK 5
# EXAMPLE TWO-STAGE PIPELINE
# DENOISE -> SHARPEN
# ============================================================

# First denoise using 3x3 averaging
denoised = averaging_filter(
    noisy10,
    3
)

# Then sharpen using high boost
final_output = high_boost_filter(
    denoised,
    1.5
)


final_psnr = calculate_psnr(
    clean,
    final_output
)

final_sharpness = sharpness_metric(
    final_output
)


print("\n")
print("=" * 65)
print("TASK 5: TWO-STAGE AGV PIPELINE")
print("=" * 65)

print("Pipeline:")
print("Noisy Image -> 3x3 Averaging -> High Boost k=1.5")

print("\nFinal PSNR =", round(final_psnr, 3))

print(
    "Final Sharpness =",
    round(final_sharpness, 3)
)


# Display final pipeline
plt.figure(figsize=(12, 4))


plt.subplot(1, 3, 1)

plt.imshow(noisy10, cmap='gray')

plt.title("Noisy σ=10")

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(denoised, cmap='gray')

plt.title("3x3 Denoised")

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(final_output, cmap='gray')

plt.title("Final Sharpened")

plt.axis("off")


plt.suptitle("Task 5: Denoise → Sharpen Pipeline")

plt.tight_layout()

plt.show()