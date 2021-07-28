# Importing modules
import matplotlib.pyplot as plt
import numpy as np
import cv2
from google.colab import files
from PIL import Image
import tifffile as tiff
import seaborn as sns
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})

# Importing datasets
uploaded = files.upload()

#import image as BW to minimize unnecessary dimensions
img_G = cv2.imread("Copy (2) of Overlay_HR_20x_F__CH3_G.tif",0)
img_R = cv2.imread("Copy (2) of Overlay_HR_20x_F__CH3_R.tif",0)

# Checking if matrix dimensions are equal
print(img_G.shape)
print(img_R.shape)

#Create image with empty values
IRA_CH = np.zeros([img_G.shape[0],img_G.shape[1],3])
print(IRA_CH.shape)

# Writing new channel
for i in range(0,img_G.shape[0]):
  for j in range(0,img_G.shape[1]):
    val = list((img_G[i][j], img_R[i][j]))
    IRA_CH[i][j] = [min(val),0,0]
print(IRA_CH.shape)

# Fig 12b
ax=sns.heatmap(IRA_CH[:,:,0], cmap='gist_heat',
                cbar_kws={'label': 'Signalstärke'})

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft=False)
plt.tight_layout()

# Importing tSNE modules
from sklearn.manifold import TSNE
from sklearn import preprocessing
uploaded = files.upload()
