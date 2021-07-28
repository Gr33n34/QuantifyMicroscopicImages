# Importing modules
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import skimage.filters as skmage
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})

# Uploading segmentation results
uploaded = files.upload()
data = pd.read_excel('Test01.xlsx')
print(type(data))
print(len(data))
print(len(data.columns))

# Fig 9a
a1 = sns.color_palette("crest", as_cmap=True)
green = sns.scatterplot(x=data['Centroid X ¬µm'], y=data['Centroid Y ¬µm'], hue=data['Cytoplasm: Green mean'],s=8,palette=a1)
lgd = plt.legend(loc=4,title='Intensität', edgecolor='black')
plt.ylabel('Zentroid Y-Koordinate [µm]')
plt.xlabel('Zentroid X-Koordinate [µm]')
plt.xlim((0,700))
plt.ylim((0,400))
plt.title('Zytoplasma: mittlere G-Intensität')

#Fig 9b
a1 = sns.color_palette("flare", as_cmap=True)
green = sns.scatterplot(x=data['Centroid X ¬µm'], y=data['Centroid Y ¬µm'], hue=data['Cytoplasm: Red mean'],s=8,palette=a1)
lgd = plt.legend(loc=4,title='Intensität', edgecolor='black')
plt.ylabel('Zentroid Y-Koordinate [µm]')
plt.xlabel('Zentroid X-Koordinate [µm]')
plt.xlim((0,700))
plt.ylim((0,400))
plt.title('Zytoplasma: mittlere R-Intensität')

# Fig 9c
a1 = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
green = sns.scatterplot(x=data['Centroid X ¬µm'], y=data['Centroid Y ¬µm'], hue=data['Cytoplasm: Blue mean'],s=8,palette=a1)
lgd = plt.legend(loc=4,title='Intensität', edgecolor='black')
plt.ylabel('Zentroid Y-Koordinate [µm]')
plt.xlabel('Zentroid X-Koordinate [µm]')
plt.xlim((0,700))
plt.ylim((0,400))
plt.title('Zytoplasma: mittlere B-Intensität')

# Writing a classification function
def classify_by_threshold(threshold):
  types=[]
  for int_blue,int_green in zip(data['Cytoplasm: Blue mean'], data['Cytoplasm: Green mean']):
    if int_blue > threshold and int_green < threshold:
      types.append('B')
    elif int_blue < threshold and int_green > threshold:
      types.append('G')
    elif int_blue > threshold and int_green > threshold:
      types.append('D')
    else:
      types.append('N')

  d={}
  for x in ['B','G','D','N']:
    d[x]=0
  for i in types:
    d[i]=d[i]+1
  stat_val = d['B']+d['G']-d['D']
  return stat_val

# Classifying data
index=[]
val=[]

for i in range(0,130):
  index.append(i)
  val.append(classify_by_threshold(i))
df_val = pd.DataFrame(list(zip(index,val)), columns=('Threshold','Score'))
df_val.head()

# Fig 9d
sns.lineplot(x=df_val['Threshold'],y=df_val['Score'])
plt.plot(df_val['Score'].idxmax(),df_val['Score'].max(),color='red', marker='o')
plt.axhline(y=df_val['Score'].max(),xmin=0,xmax=0.132,linewidth=1, c='grey', ls='--', alpha=.7)
plt.axvline(x=df_val['Score'].idxmax(),ymin=0,ymax=0.94,linewidth=1, c='grey', ls='--', alpha=.7)
plt.xlabel('Schwellenwert')
plt.title('Äquivalenter Schwellenwert für G- und B-Werte')

# New classification function for cycling threshold
def classify(threshold_B, threshold_G):
  types=[]
  for int_blue,int_green in zip(data['Cytoplasm: Blue mean'], data['Cytoplasm: Green mean']):
    if int_blue > threshold_B and int_green < threshold_G:
      types.append('B')
    elif int_blue < threshold_B and int_green > threshold_G:
      types.append('G')
    elif int_blue > threshold_B and int_green > threshold_G:
      types.append('D')
    else:
      types.append('N')

  d={}
  for x in ['B','G','D','N']:
    d[x]=0
  for i in types:
    d[i]=d[i]+1
  stat_val = d['B']+d['G']-d['D']-d['N']
  return stat_val

# traditional threshold calculation approaches
def math_models(parameter_B,parameter_G):
  array_b = np.array(data[parameter_B])
  array_g = np.array(data[parameter_G])

  b_yen = skmage.threshold_yen(array_b)
  g_yen = skmage.threshold_yen(array_g)
  s_yen = classify(threshold_B=b_yen, threshold_G=g_yen)
  print('Yen-Method')
  print('B-val: '+str(b_yen))
  print('G-val: '+str(g_yen))
  print('Score: '+str(s_yen))

  b_yen = tt
  g_yen = skmage.threshold_otsu(array_g)
  s_yen = classify(threshold_B=b_yen, threshold_G=g_yen)
  print('\nOtsu-Method')
  print('B-val: '+str(b_yen))
  print('G-val: '+str(g_yen))
  print('Score: '+str(s_yen))

  b_yen = skmage.threshold_isodata(array_b)
  g_yen = skmage.threshold_isodata(array_g)
  s_yen = classify(threshold_B=b_yen, threshold_G=g_yen)
  print('\nIsodata-Method')
  print('B-val: '+str(b_yen))
  print('G-val: '+str(g_yen))
  print('Score: '+str(s_yen))

  b_yen = skmage.threshold_li(array_b)
  g_yen = skmage.threshold_li(array_g)
  s_yen = classify(threshold_B=b_yen, threshold_G=g_yen)
  print('\nLi-Method')
  print('B-val: '+str(b_yen))
  print('G-val: '+str(g_yen))
  print('Score: '+str(s_yen))

  b_yen = skmage.threshold_minimum(array_b)
  g_yen = skmage.threshold_minimum(array_g)
  s_yen = classify(threshold_B=b_yen, threshold_G=g_yen)
  print('\nMinimum-Method')
  print('B-val: '+str(b_yen))
  print('G-val: '+str(g_yen))
  print('Score: '+str(s_yen))

  b_yen = skmage.threshold_triangle(array_b)
  g_yen = skmage.threshold_triangle(array_g)
  s_yen = classify(threshold_B=b_yen, threshold_G=g_yen)
  print('\nTriangle-Method')
  print('B-val: '+str(b_yen))
  print('G-val: '+str(g_yen))
  print('Score: '+str(s_yen))

# Compute thresholds
b1 = 'Cytoplasm: Blue mean'
g1 = 'Cytoplasm: Green mean'
math_models(parameter_B=b1, parameter_G=g1)

# Computing heatmap data
score, xcoord, ycoord = ([] for i in range(3))
array = np.zeros((257,257))
for x in range (0,257):
  for y in range(0,257):
    array[y][x] = classify(threshold_B=x,threshold_G=y)

# Fig 9e
ax = sns.heatmap(array,vmin=-3000, vmax=3000,center=0,xticklabels=50,yticklabels=50,cbar_kws={'label': 'Score'})
ax.invert_yaxis()
plt.ylabel('Schwellenwert G-Intensität')
plt.xlabel('Schwellenwert B-Intensität')

# writing new classification function for cell visualization
def classify(threshold_B, threshold_G):
  types=[]
  for int_blue,int_green in zip(data['Cytoplasm: Blue mean'], data['Cytoplasm: Green mean']):
    if int_blue > threshold_B and int_green < threshold_G:
      types.append('T-Zelle')
    elif int_blue < threshold_B and int_green > threshold_G:
      types.append('B-Zelle')
    elif int_blue > threshold_B and int_green > threshold_G:
      types.append('doppelt-positiv')
    else:
      types.append('nicht-klassifiziert')
  return types

# Computing classes
classlist = classify(20,14)
data['Classification'] = classlist
print(len(classlist))

# Fig 9f
green = sns.scatterplot(x=data['Centroid X ¬µm'], y=data['Centroid Y ¬µm'], hue=data['Classification'],s=8)
lgd = plt.legend(loc=4,title='Klassifikation', edgecolor='black')
plt.ylabel('Zentroid Y-Koordinate [µm]')
plt.xlabel('Zentroid X-Koordinate [µm]')
plt.xlim((0,700))
plt.ylim((0,400))
plt.title('Zellklassifikation')
