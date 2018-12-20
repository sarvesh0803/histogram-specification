import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 15]

# Importing images as greyscale
given = cv2.imread("/home/sarvesh/Projects/Image Processing/givenhist.jpg", 0)
final = cv2.imread("/home/sarvesh/Projects/Image Processing/sphist.jpg", 0)

# Initializing the output container
new_image = given - given
new_image.setflags(write=True)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(given, cmap='gray')
ax1.set_title('The given image')

ax2.imshow(final, cmap='gray')
ax2.set_title('The specified image')

plt.show()

# Reshape into column of intensities
g_col = np.reshape(given, 256 * 512, order='C').astype(float)
f_col = np.reshape(final, 256 * 512, order='C').astype(float)
L = int(max(g_col) - min(g_col))

# Histogram of the given array of intensities with 256 bins
h_g, h_g_bins = np.histogram(g_col, bins=256)
h_f, h_f_bins = np.histogram(f_col, bins=256)

# PDF of above histogram is calculated
pdf_g = h_g / sum(h_g)
pdf_f = h_f / sum(h_f)

# CDF of the above PDF is calculated
c_g = pdf_g
c_f = pdf_f
for i in range(1, 256):
    c_g[i] = c_g[i - 1] + pdf_g[i]
    c_f[i] = c_f[i - 1] + pdf_f[i]

# Rounding of CDF
c_g_r = c_g * (L)
c_f_r = c_f * (L)

temp = np.ones(256, order='C')
index = np.ones(256, order='C')

# Min difference mapping
for i in range(0, 256):
    temp = abs(c_f_r - c_g_r[i] * np.ones(256, order='C'))
    index[i] = c_f_r[np.argmin(temp)]

# New image construction
for i in range(0, 256):
    for j in range(0, 512):
        if (given[i, j] == 0):
            new_image[i, j] = 30
        else:
            new_image[i, j] = index[given[i, j]]

plt.imshow(final, cmap='gray')
plt.title('Final Output')
plt.show()