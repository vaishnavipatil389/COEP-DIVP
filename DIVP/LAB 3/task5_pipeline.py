import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# TASK 5 — TWO-STAGE AGV PIPELINE
# DENOISE → SHARPEN
# ============================================================

print("\n")
print("=" * 70)
print("TASK 5: TWO-STAGE AGV PIPELINE RECOMMENDATION")
print("=" * 70)


# ============================================================
# 1. READ ORIGINAL IMAGE
# ============================================================

original = cv2.imread("source.jpg", 0)

if original is None:
    print("Error: Image not found!")
    exit()

original = original.astype(np.float32)


# ============================================================
# 2. ADD GAUSSIAN NOISE
# ============================================================

np.random.seed(0)

sigma = 10

noise = np.random.normal(
    0,
    sigma,
    original.shape
)

noisy = original + noise

noisy = np.clip(
    noisy,
    0,
    255
).astype(np.uint8)


# ============================================================
# 3. PARAMETERS TO TEST
# ============================================================

# Different averaging filter sizes
filter_sizes = [3, 5, 7]

# Different high-boost sharpening values
k_values = [1.0, 1.5, 2.0]


# ============================================================
# 4. VARIABLES FOR BEST RESULT
# ============================================================

best_score = -np.inf

best_psnr = 0
best_sharpness = 0

best_filter_size = 0
best_k = 0

best_denoised = None
best_output = None


# ============================================================
# 5. TEST ALL DENOISE → SHARPEN COMBINATIONS
# ============================================================

print("\nTesting pipeline combinations...\n")

for size in filter_sizes:

    # --------------------------------------------------------
    # Stage 1: Averaging Filter
    # --------------------------------------------------------

    kernel = np.ones(
        (size, size),
        np.float32
    ) / (size * size)

    denoised = cv2.filter2D(
        noisy,
        -1,
        kernel
    )

    # --------------------------------------------------------
    # Stage 2: High-Boost Sharpening
    # --------------------------------------------------------

    # Create blurred version
    blurred = cv2.GaussianBlur(
        denoised,
        (3, 3),
        0
    )

    for k in k_values:

        # High-boost filtering
        sharpened = (
            denoised.astype(np.float32)
            + k * (
                denoised.astype(np.float32)
                - blurred.astype(np.float32)
            )
        )

        # Keep pixels between 0 and 255
        sharpened = np.clip(
            sharpened,
            0,
            255
        ).astype(np.uint8)


        # ====================================================
        # 6. CALCULATE PSNR
        # ====================================================

        mse = np.mean(
            (
                original
                - sharpened.astype(np.float32)
            ) ** 2
        )

        if mse == 0:
            psnr = float("inf")
        else:
            psnr = 10 * np.log10(
                (255 ** 2) / mse
            )


        # ====================================================
        # 7. CALCULATE SHARPNESS
        # ====================================================

        sharpness = cv2.Laplacian(
            sharpened,
            cv2.CV_64F
        ).var()


        # ====================================================
        # 8. COMBINED SCORE
        # ====================================================

        # We want both:
        # Higher PSNR
        # Higher Sharpness

        score = psnr + 0.1 * sharpness


        # ====================================================
        # 9. PRINT CURRENT RESULT
        # ====================================================

        print(
            f"Filter = {size}x{size}, "
            f"k = {k:.1f}, "
            f"PSNR = {psnr:.3f}, "
            f"Sharpness = {sharpness:.3f}"
        )


        # ====================================================
        # 10. SAVE BEST PIPELINE
        # ====================================================

        if score > best_score:

            best_score = score

            best_psnr = psnr
            best_sharpness = sharpness

            best_filter_size = size
            best_k = k

            best_denoised = denoised.copy()
            best_output = sharpened.copy()


# ============================================================
# 11. FINAL RECOMMENDED PIPELINE
# ============================================================

print("\n")
print("=" * 70)
print("RECOMMENDED AGV PIPELINE")
print("=" * 70)

print(
    f"\nNoisy Image"
    f" -> {best_filter_size}x{best_filter_size} Averaging"
    f" -> High-Boost k={best_k}"
)

print("\nSelected Parameters:")
print(
    f"Denoising Filter : "
    f"{best_filter_size}x{best_filter_size} Averaging"
)

print(
    f"High-Boost k     : "
    f"{best_k}"
)

print("\nFinal Calculated Results:")

print(
    f"Final PSNR       : "
    f"{best_psnr:.3f}"
)

print(
    f"Final Sharpness  : "
    f"{best_sharpness:.3f}"
)


# ============================================================
# 12. JUSTIFICATION
# ============================================================

print("\n")
print("=" * 70)
print("TASK 5 JUSTIFICATION")
print("=" * 70)

print(
    f"\nThe {best_filter_size}x{best_filter_size} averaging filter "
    f"was selected as the denoising stage."
)

print(
    f"High-boost sharpening with k={best_k} "
    f"was selected as the second stage."
)

print(
    f"The final pipeline produced a PSNR of "
    f"{best_psnr:.3f} and a sharpness value of "
    f"{best_sharpness:.3f}."
)

print(
    "\nTherefore, this combination provides the best "
    "balance between noise reduction and edge enhancement "
    "among the tested pipeline combinations."
)


# ============================================================
# 13. DISPLAY PIPELINE
# ============================================================

plt.figure(figsize=(14, 4))


# Original / Noisy Image
plt.subplot(1, 3, 1)

plt.imshow(
    noisy,
    cmap="gray"
)

plt.title(
    f"Noisy Image (σ={sigma})"
)

plt.axis("off")


# Denoised Image
plt.subplot(1, 3, 2)

plt.imshow(
    best_denoised,
    cmap="gray"
)

plt.title(
    f"{best_filter_size}x{best_filter_size} Averaging"
)

plt.axis("off")


# Final Image
plt.subplot(1, 3, 3)

plt.imshow(
    best_output,
    cmap="gray"
)

plt.title(
    f"High-Boost k={best_k}"
)

plt.axis("off")


plt.suptitle(
    "Task 5: Recommended Denoise → Sharpen Pipeline"
)

plt.tight_layout()

plt.show()