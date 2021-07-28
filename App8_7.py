# Importing modules
%matplotlib inline
import mahotas as mh
import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import interact, fixed
from google.colab import files
import cv2
import seaborn as sns
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})

# Importing image files
uploaded = files.upload()
img = cv2.imread("Image_CHnuclei.png",0)

# Fig 6d
print(type(img))
print(img.shape)
plt.imshow(img)
plt.xlabel('X-Koordinate [px]')
plt.ylabel('Y-Koordinate [px]')
plt.title('Signalstärke des nativen Bildes')
plt.tight_layout()

# Gaussian filtering
imgf = mh.gaussian_filter(img, 2.)
T_meanf = imgf.mean()
print(T_meanf)

# Fig 6e
plt.imshow(img > T_meanf)

# Counting objects
bin_image = img > T_meanf
labeled, nr_objects = mh.label(bin_image)
print(nr_objects)

# Fig 6f
plt.imshow(labeled)
plt.jet()
plt.xlabel('X-Koordinate [px]')
plt.ylabel('Y-Koordinate [px]')
plt.title('Klassifikation einzelner Zellen')
plt.tight_layout()

# local maxima
sigma = 7.0
imgf = mh.gaussian_filter(img.astype(float), sigma)
maxima = mh.regmax(mh.stretch(imgf))
maxima,_ = mh.label(maxima)

# Fig 6g
plt.imshow(maxima)
plt.xlabel('X-Koordinate [px]')
plt.ylabel('Y-Koordinate [px]')
plt.title('Lokale Maxima')
plt.tight_layout()

# distance transform
dist = mh.distance(bin_image)

# Fig 6h
plt.imshow(dist)
plt.xlabel('X-Koordinate [px]')
plt.ylabel('Y-Koordinate [px]')
plt.title('Dist-Plot')
plt.tight_layout()

# Watershed segmentation
dist = 255 - mh.stretch(dist)
watershed = mh.cwatershed(dist, maxima)

# Fig 6i
plt.imshow(watershed)
plt.xlabel('X-Koordinate [px]')
plt.ylabel('Y-Koordinate [px]')
plt.title('Watershed-Segmentation')
plt.tight_layout()

# Watershed filtering on objects
watershed *= bin_image

# Fig 6j
plt.imshow(watershed)
plt.xlabel('X-Koordinate [px]')
plt.ylabel('Y-Koordinate [px]')
plt.title('Übertragung auf ursprüngliche Zellgrenzen')
plt.tight_layout()

# Final cell count after filtering
sizes = mh.labeled.labeled_size(watershed)
min_size = 1000
filtered = mh.labeled.remove_regions_where(watershed, sizes < min_size)
labeled,nr_objects = mh.labeled.relabel(filtered)
print("Number of cells: {}".format(nr_objects))

# Fig 6k
plt.imshow(filtered)
plt.xlabel('X-Koordinate [px]')
plt.ylabel('Y-Koordinate [px]')
plt.title('Bereinigung von zu kleinen Fragmenten')
plt.tight_layout()
