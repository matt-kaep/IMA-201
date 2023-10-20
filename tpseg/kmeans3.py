#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Authors: Robert Layton <robertlayton@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
from skimage.io import imread

n_colors_list = [10]

ima = imread("carte.tif")

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
ima = np.array(ima, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(ima.shape)
assert d == 3
image_array = np.reshape(ima, (w * h, d))

# Create a figure with the original image and the compressed images

plt.imshow(ima)


for i, n_colors in enumerate(n_colors_list):
    print("Fitting model on a small sub-sample of the data with n_colors={}".format(n_colors))
    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    print("done in %0.3fs." % (time() - t0))

    # Get labels for all points
    print("Predicting color indices on the full image (k-means) with n_colors={}".format(n_colors))
    t0 = time()
    labels = kmeans.predict(image_array)
    print("done in %0.3fs." % (time() - t0))

    # Recreate the compressed image from the code book & labels
    recreated_image = recreate_image(kmeans.cluster_centers_, labels, w, h)

    # Show the compressed image
    plt.imshow(recreated_image)

    plt.show()
# Show the figure
