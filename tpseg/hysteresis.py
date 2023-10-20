import matplotlib.pyplot as plt
import numpy as np
from skimage import io, filters, morphology

# Load the image
image = io.imread('spot.tif')

# Define the radius, low threshold, and high threshold values to test
radius_values = [5]
low_values = [0.5,3,5,10]
high_values = [3]
def tophat(image, radius):
    se=morphology.square(radius)
    ero=morphology.erosion(image,se)
    dil=morphology.dilation(ero,se)
    tophat=dil-image
    return tophat

overlays = []

# Loop over the radius, low threshold, and high threshold values
for i, radius in enumerate(radius_values):
    for j, low in enumerate(low_values):
        for k, high in enumerate(high_values):
            # Apply the top-hat transform with the current parameters
            top=tophat(image,radius)
            lowt = (top > low).astype(int)
            hight = (top > high).astype(int)
            hyst = filters.apply_hysteresis_threshold(top, low, high)

            # Create an overlayed image showing the hysteresis threshold
            overlay = np.zeros_like(image)
            overlay[hyst] = 1
            overlay = filters.gaussian(overlay, sigma=2)
            overlays.append(overlay)

            # Create a figure with the original image and the top-hat transform
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
            fig.suptitle('Hysteresis threshold with radius={}, low={}, high={}'.format(radius, low, high))
            ax = axes.ravel()
            ax[0].imshow(image, cmap='gray')
            ax[0].set_title('Original image')
            ax[1].imshow(top, cmap='magma')
            ax[1].set_title('Top-hat transform')

            ax[2].imshow(lowt, cmap='magma')
            ax[2].set_title('Low threshold')
            ax[3].imshow(hight, cmap='magma')
            ax[3].set_title('High threshold')
            ax[4].imshow(hyst, cmap='magma')
            ax[4].set_title('Hysteresis threshold')
            ax[5].imshow(image, cmap='magma')
            ax[6].imshow(hyst, cmap='jet', alpha=0.5)
            ax[6].set_title('Overlay')

            # Adjust the spacing between subplots
            plt.subplots_adjust(wspace=0.4, hspace=0.4)

            # Show the figures
            plt.show()
        
 