# Importing modules
import pandas as pd
import matplotlib.pyplot as plt import seaborn as sns
import numpy as np
import statistics as stat import glob  import re
from PIL import Image
import matplotlib.image as mpimg import statsmodels
import math

# Loading data
path_gesamt = '/Users/User/Desktop/DataFolder/'
specimen_list = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N','O', 'P', 'Q', 'R', 'S'] # blinded specimen names

# Computing positive pixel counts
def creatingDF_pxTreshold(specimen_list, channel):
  paths_CH1, treshold_value, pixel_count = [[],[],[]]
  for specimen in specimen_list:
    path = path_gesamt + specimen
    files_in_path = glob.glob(path + "/*")
    for file in files_in_path:
      datas_in_path = glob.glob(file + "/*")
      for data_path in datas_in_path:
        if 'mcf' not in data_path and channel in data_path:
          paths_CH1.append(data_path)
         else:
           print(‚Error encountered:‘+str(data_path))
for img in paths_CH1:
images = mpimg.imread(img) counter = 0
for x in range(0,257):
      treshold_value.append(counter)
      pixel_count.append(np.count_nonzero(images > x))
      counter = counter +1
  df = pd.DataFrame(list(zip(treshold_value, pixel_count)),
       columns=['TresholdValue', 'PixelCount'])
return df

# Computing area under curve (AUC)
auc=[]
for specimen in specimen_list:
auc = df[‚PixelCount‘].sum()
