# Importing modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from math import sqrt
from google.colab import files
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
path = '/Users/User/Desktop/'

# Generate some random points
np.random.seed(4321)
pts = 0.1 + 0.8*np.random.rand(15, 2)
ch = ConvexHull(pts)

# Fig 14a
plt.plot(pts[:, 0], pts[:, 1], 'ko', markersize=5,label='Punkt')
plt.xlim(0, 1)
plt.ylim(0, 1)
legend = plt.legend(loc='center left', fontsize='large')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, ncol=4,edgecolor='black')
plt.tight_layout()

# Getting hull points
hull_indices = np.unique(ch.simplices.flat)
hull_pts = pts[hull_indices, :]

# Fig 14b
plt.plot(pts[:, 0], pts[:, 1], 'ko', markersize=5, label='Punkt')
plt.plot(hull_pts[:, 0], hull_pts[:, 1], 'ro', alpha=.25, markersize=10,label='Hull-Punkt')
plt.xlim(0, 1)
plt.ylim(0, 1)
legend = plt.legend(loc='center left', fontsize='large')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=4,edgecolor='black')
plt.tight_layout()

# Get the indices of the hull points.
hull_indices = ch.vertices

# These are the actual points.
hull_pts = pts[hull_indices, :]

# Fig 14c
plt.plot(pts[:, 0], pts[:, 1], 'ko', markersize=5,label='Punkt')
plt.fill(hull_pts[:,0], hull_pts[:,1], fill=False, edgecolor='b',label='Hull')
plt.xlim(0, 1)
plt.ylim(0, 1)
legend = plt.legend(loc='center left', fontsize='large')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, ncol=4,edgecolor='black')
plt.tight_layout()

# Writing function to find centroid
def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

# Computing centroid
centroid = list(centeroidnp(np.array(hull_pts)))
print('Centroid: ' + str(centroid))

# Fig 14d
hull_indices = np.unique(ch.simplices.flat)
hull_pts = pts[hull_indices, :]
plt.plot(pts[:, 0], pts[:, 1], 'ko', markersize=5, label='Punkt')
plt.plot(centroid[0], centroid[1], 'ro', alpha=.25, markersize=5,label='Zentroid')
plt.xlim(0, 1)
plt.ylim(0, 1)
legend = plt.legend(loc='center left', fontsize='large')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, ncol=4,edgecolor='black')
plt.tight_layout()

# Writing dataframe with Hull points
df_hullpts = pd.DataFrame(hull_pts)
df_hullpts.head()

# Writing function for points by way of direction vector times length
def compute_newpoints(M,H1, desired_length):
    vector_x = H1[0] - M[0]
    vector_y = H1[1] - M[1]
    length = sqrt(vector_x**2 + vector_y**2)
    if length > 1 or length < 1:
        centeroidnp(arr):
        length = arr.shape[0]
    vector_x = vector_x * factor
    vector_y = vector_y * factor
    vector = []
    vector.append(vector_y)
    new_point_x = H1[0] + desired_length * vector[0]
    new_point_y = H1[1] + desired_length * vector[1]
    new_points = []
    new_points.append(new_point_x)
    new_points.append(new_point_y)
    return new_points

# Computing points
points_expanded = []
for element in hull_pts:
    points_expanded.append(compute_newpoints(centroid,element, 0.5))

df_expanded = pd.DataFrame(points_expanded)
df_expanded.head()

# Fig 14e
plt.plot(pts[:, 0], pts[:, 1], 'ko', markersize=5, label='Punkt')
plt.plot(hull_pts[:, 0], hull_pts[:, 1], 'ro', alpha=.25, markersize=10, label ='Hull-Punkt')
plt.plot(df_expanded[0], df_expanded[1], 'go', alpha=.25, markersize=5,label='erweiterter Hull-Punkt')
plt.plot(centroid[0], centroid[1], 'r+', alpha=.25, markersize=5,label='Zentroid')
legend = plt.legend(loc='center left', fontsize='large')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, ncol=3,edgecolor='black')
plt.tight_layout()

# Fig 14f
ch = ConvexHull(pts)
hull_indices = ch.vertices
hull_pts = pts[hull_indices, :]
exp_pts = np.array(df_expanded)
ch_exp = ConvexHull(exp_pts)
hullexp_indices = ch_exp.vertices
hullexp_pts = exp_pts[hullexp_indices, :]
plt.plot(pts[:, 0], pts[:, 1], 'ko', markersize=5, label='Punkt')
plt.plot(hull_pts[:, 0], hull_pts[:, 1], 'ro', alpha=.25, markersize=10, label ='Hull-Punkt')
plt.fill(hull_pts[:,0], hull_pts[:,1], fill=False, edgecolor='r',label='Hull')
plt.plot(df_expanded[0], df_expanded[1], 'go', alpha=.25, markersize=5,label='erweiterter Hull-Punkt')
plt.fill(hullexp_pts[:,0], hullexp_pts[:,1], fill=False,edgecolor='g',label='erweiterte Hull')
legend = plt.legend(loc='center left', fontsize='large')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, ncol=3,edgecolor='black')
plt.tight_layout()
