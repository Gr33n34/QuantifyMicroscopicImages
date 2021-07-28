# Importing modules
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import seaborn as sns
import io
from google.colab import files
import matplotlib as mpl
from sklearn.datasets import make_classification
import matplotlib
from sklearn.cluster import KMeans
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})

# Importing segmentation datasets
uploaded = files.upload()
file = io.BytesIO(uploaded['segm_file.txt'])
df = pd.read_csv(file, delimiter='\t',encoding = "ISO-8859-1", cache_dates=False)
df1 = df.drop(['Name', 'Class', 'Parent', 'ROI'], axis=1)
df1.head()

# Checking for errors
df2 = df1.apply(pd.to_numeric, errors='coerce')
df2.isnull().sum()

# Correcting errors
print(df2.isnull().sum().sum())
df3 = df2.fillna(0)

# Identify B- / T- and non-classified cells
classification = []
Green_Thres = 40
Blue_Thres = 50

for x,y in zip(df3['Cytoplasm: Green mean'], df3['Cytoplasm: Blue mean']):
  x = float(x)
  y = float(y)
  if x > Green_Thres and y > Blue_Thres:
    cell_type = 'B-Zelle'
  elif x > Green_Thres and y < Blue_Thres:
    cell_type = 'B-Zelle'
  elif x < Green_Thres and y > Blue_Thres:
    cell_type = 'T-Zelle'
  else:
    cell_type = 'nicht-klassifiziert'
  classification.append(cell_type)

df3['Classification'] = classification
df3.head()

# Fig 8a
plt.figure(figsize=(6.5,4.5))
points = sns.scatterplot(x=df3['Centroid X åµm'], y=df3['Centroid Y åµm'], hue=df3['Classification'], s=20, alpha=0.7)
plt.legend(loc = 'upper left',edgecolor='black')
plt.ylabel('Zentroid-Y [µm]')
plt.xlabel('Zentroid-X [µm]')
plt.title('Räumliche Verteilung der Zentroide')

# classify datasets
targets = df3['Classification'].astype('category')
# save target-values as color for plotting
# red: disease,  green: no disease

label_color = []
for i in targets:
  if i == 'B cell':
    color = 'orange'
  elif i == 'T cell':
    color = 'green'
  elif i == 'Double':
    color = 'red'
  else:
    color = 'blue'
  label_color.append(color)

print(label_color[0:3], label_color[-3:-1])

# Performing PCA
df1 = df3.drop(labels=['Classification'], axis=1)
pca = PCA(n_components=2)
pca.fit(df1)
T = pca.transform(df1)
T = pd.DataFrame(T, columns= ['PCA component 1', 'PCA component 2'])
T['Type'] = df3['Classification']
T.head()

# Fig 8b
plt.figure(figsize=(6.5,4.5))
sns.scatterplot(x='PCA component 1', y='PCA component 2',data=T, hue='Type', alpha=0.7)
plt.title('Nicht-normalisierte PCA')
plt.legend(loc = 'upper left',edgecolor='black')

# preprocessing datasets
df2 = preprocessing.StandardScaler().fit_transform(df1)
pca1 = PCA(n_components=2)
pca1.fit(df2)
comp1, comp2 = pca1.components_[0], pca1.components_[1]
labels = list(df3.columns.values)
labels.pop()
df7 = pd.DataFrame(list(zip(labels, comp1,comp2)), columns =['Label', 'Comp1','Comp2'])
df7=df7.sort_values(by='Comp1',ascending=True)
df7.head(10)

# Fig 8e
ax = sns.scatterplot(x='Comp1',y='Comp2',data=df7,color='black',s=1)

for i in range(0,len(df7)):
  if 'Red' in df7['Label'][i]:
    plt.plot(df7['Comp1'][i],df7['Comp2'][i],marker='^',color='#d62728')
  elif 'Blue' in df7['Label'][i]:
    plt.plot(df7['Comp1'][i],df7['Comp2'][i],marker='^',color='#1f77b4')
  elif 'Green' in df7['Label'][i]:
    plt.plot(df7['Comp1'][i],df7['Comp2'][i],marker='^',color='#2ca02c')
  else:
    plt.plot(df7['Comp1'][i],df7['Comp2'][i],marker='^',color='black')

rect1 = matplotlib.patches.Rectangle((.004, -.06), 0.035, 0.038,facecolor='white',edgecolor='#2ca02c')
rect2 = matplotlib.patches.Rectangle((.175, -.17), 0.09, 0.07,facecolor='white',edgecolor='#2ca02c')
rect3 = matplotlib.patches.Rectangle((-.115, -.0223), 0.073, 0.085,facecolor='white',edgecolor='#1f77b4')
rect4 = matplotlib.patches.Rectangle((-.14, .145), 0.11, 0.12,facecolor='white',edgecolor='black')
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 2')
plt.text(-.125,.019,'1',color='#1f77b4')
plt.text(-.007,-.045,'2',color='#2ca02c')
plt.text(.163,-.138,'3',color='#2ca02c')
plt.plot(-.13,.25,marker='^',color='black')
plt.plot(-.13,.22,marker='^',color='#d62728')
plt.plot(-.13,.19,marker='^',color='#2ca02c')
plt.plot(-.13,.16,marker='^',color='#1f77b4')
plt.plot(-0.145651,0.141395,marker='o',color='white')
plt.plot(-0.145651,0.137,marker='o',color='white')
plt.text(-.12,.245,'geometrisch',color='black')
plt.text(-.12,.215,'R-Marker',color='black')
plt.text(-.12,.185,'G-Marker',color='black')
plt.text(-.12,.155,'B-Marker',color='black')
plt.title('Beträge der einzelnen Komponenten')

# Fig 8c
T1 = pca1.transform(df2)
T1 = pd.DataFrame(T1)
T1.columns = ['PCA component 1', 'PCA component 2']
T1['Type'] = df3['Classification']
plt.figure(figsize=(6.5,4.5))
sns.scatterplot(x='PCA component 1', y='PCA component 2',data=T1, hue='Type', alpha=0.7)
plt.title('Normalisierte PCA')
plt.legend(loc ='upper left',edgecolor='black')

#  calculate "explained variance ratio"
print('Explained variation per principal component: {}'.format(pca1.explained_variance_ratio_))

# Predict labels with sklearn
pca2 = PCA(n_components=2)
pca2.fit(df2)
T1 = pca2.transform(df2)
T1 = pd.DataFrame(T1)
T1.columns = ['PCA component 1', 'PCA component 2']
T1.head()

#Initialize the class object
kmeans = KMeans(n_clusters= 2)

#predict the labels of clusters.
label = kmeans.fit_predict(T1)
print(type(label))
kmean_label = []
for i in label:
  kmean_label.append(i)
T1['Label']=kmean_label
T1['Label'] = T1['Label'].replace({0: 'Cluster 1', 1: 'Cluster 2'})
T1.head()

# Fig 8d
plt.figure(figsize=(6.5,4.5))
sns.scatterplot(x='PCA component 1', y='PCA component 2', data=T1, hue='Label', marker='o',alpha=0.7)
plt.title('K-means Clustering')
plt.legend(loc ='upper left',edgecolor='black',title='Einteilung')
