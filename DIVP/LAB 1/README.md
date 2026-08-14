#Image Intensity Transformations

**Aim**
To implement Negative, Power-Law (Gamma), and Logarithmic intensity transformations using build in functions

1. Read the image
{
image = cv2.imread("image.jpg", 0)
}

2. Negative transformation :Reverses brightness levels.
                    __s = (L - 1) - r__
{
negative = cv2.bitwise_not(image)
}

3. Power-Law transformation
                    __s = c . r^gamma__
{
gamma = 0.5
normalized = image / 255.0
power_law = np.power(normalized, gamma) * 255
}

First, pixel values are converted from 0–255 to 0–1:
Here:
        γ=0.5

Finally, it is converted back to 0–255.
Because γ = 0.5 (< 1), the image becomes brighter.

4. Log transformation
    s=clog(1+r)
where:

c= 255 / log(256)

This mainly enhances darker regions and compresses brighter intensity values.

{
c = 255 / np.log(1 + 255)
log_image = c * np.log(1 + image)
}

5. Convert results to image format
{
power_law = np.uint8(power_law)
log_image = np.uint8(log_image)
}

6. Display

          Input Image
               ↓
        Grayscale Image
               ↓
     ┌─────────┼─────────┐
     ↓         ↓         ↓
 Negative   Power-Law    Log
     ↓         ↓         ↓
     └─────────┼─────────┘
               ↓
        Display Results