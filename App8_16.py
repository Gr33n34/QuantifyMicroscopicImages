# Importing modules
import random
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import string
from scipy.stats import ttest_ind
import io
from google.colab import files

# function that returns frequency of adjacent points labels
def getting_neighbor_labelfreq(point, radius):
    # Creating dict with counts of neighbour types
    freq={}
    for i in tree.query_ball_point(point, radius):
        ctype = c_df['Label'][i]
        if ctype in freq:
            freq[ctype]+=1
        else:
            freq[ctype]=0
    # Getting total number of values
    val=[]
    for j in freq:
        val.append(freq[j])
    total_num = sum(val)
    # assign frequencies
    freq_a = getting_around_zeros(freq,'Ignore',total_num)
    freq_b = getting_around_zeros(freq,'B220',total_num)
    freq_c = getting_around_zeros(freq,'Other',total_num)
    return freq_a,freq_b,freq_c

# function to avoid errors when frequency is zero
def getting_around_zeros(freq, cat, totalnumber):
    try:
        freq_cat = freq[cat]/totalnumber
    except:
        freq_cat = 0
    return freq_cat

# checking for error values
df1 = df.drop(['Class', 'Parent', 'ROI'], axis=1)
df2 = df1.apply(pd.to_numeric, errors=‚ignore')
print(df2.isnull().sum().sum())

# Checking classifications
df2[‚Name'].value_counts()

# Writing coords in appropriate array
coords = np.array([(i, j) for i, j in zip(df2['Centroid X Âμm'],df2['Centroid Y Âμm'])])

# Computing graph tree
tree = scipy.spatial.cKDTree(coords, leafsize=100)

# Now write into the dataframe
fr_a, fr_b, fr_c, fr_d = [[],[],[],[]]
for point in coords:
    a,b,c=getting_neighbor_labelfreq(point,10)
    fr_a.append(a)
    fr_b.append(b)
    fr_c.append(c)
data = list(zip(df2['Name'], fr_a,fr_b,fr_c))
df3 = pd.DataFrame(data, columns=['Name','nc','B220','Other'])
cat1 = df3[df3['Name']=='Other']
cat1.describe()
