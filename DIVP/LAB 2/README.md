
Image Histogram Processing (Equalization & Matching)

**1. Histogram Equalization**

The built-in OpenCV function:
{
    equalized = cv2.equalizeHist(source)
}

automatically redistributes the intensity values of the source image to improve its contrast.

Output:
Original Image
Equalized Image
Original Histogram
Equalized Histogram

**2. Histogram Matching / Specification**

The built-in Scikit-image function:
{
    matched = exposure.match_histograms(source, reference)
}

adjusts the intensity distribution of the source image so that its histogram becomes similar to the reference image histogram.

Output:
Source Image
Reference Image
Histogram Matched Image
Source Histogram
Reference Histogram
Matched Histogram

Processing Flow
                 Source Image
                      ↓
              ┌───────┴────────────┐
              ↓                    ↓
      Histogram Equalization   Histogram Matching
              ↓                    ↓
       Equalized Image         Reference Image
              ↓                    ↓
       Equalized Histogram      Matched Histogram
