import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

def plot_bygroup(X,metadata_file,num_clusters):
	'''creates a 2D plot of the inputted data matrix labeled by categories
	indicated in the inputted metadata file'''

	from sklearn.mixture import GaussianMixture
	from matplotlib.colors import LogNorm

	f = open(metadata_file,'r')
	reader = csv.reader(f,delimiter='\t')
	reader.next()
	group_labels = [line[1] for line in reader]
	groups = list(set(group_labels))

	df = pd.DataFrame(dict(X1 = X[:,0],X2 = X[:,1],color=group_labels))
	groups = df.groupby('color')

	fig, ax = plt.subplots()
	for name, group in groups:
		ax.scatter(group.X1,group.X2,s=10,label=name)
	ax.legend()

	gmm = GaussianMixture(n_components=num_clusters, covariance_type='full').fit(X)

	# contour plots
	a = np.linspace(min(X[:,0]),max(X[:,0]),num=100)
	b = np.linspace(min(X[:,1]),max(X[:,1]),num=100)
	A,B = np.meshgrid(a,b)

	AA = np.array([A.ravel(),B.ravel()]).T
	Z = -gmm.score_samples(AA)
	Z = Z.reshape(A.shape)

	CS = plt.contour(A, B, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),levels=np.linspace(3,10,10))
	ax.plot(gmm.means_[:,0],gmm.means_[:,1],'+',markersize=13,color='black')
	ax.clabel(CS)

	plt.show()

	# 

X = np.loadtxt('all8_rep_pca.txt',delimiter='\t')

metadata_file = 'all8_metadata.tsv'

plot_bygroup(X,metadata_file,8)