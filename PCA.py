import numpy as np
import ntpath as nt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

file_path = r"----Place the file path here----"		#please do not remove "r" at the start of the string by mistake
pca_alg = pd.read_csv(file_path,sep='\t',header=None)
file_name=nt.basename(file_path) 

pca_alg = np.asmatrix(pca_alg)
features = np.delete(pca_alg,-1,axis=1)  
features = features.astype(np.float)  

diseases_col = pca_alg[:, -1] #saving last column as disease

#Finding the mean of each feature
mean_values = np.mean(features, axis=0) 

#Calculating mean centered data
mean_centered_data = features - mean_values

#Finding the covariance matrix
covariance_matrix = np.cov(mean_centered_data.T) 

#Finding the corresponding eigen values and eigen vectors
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
eigen_values_indexes = eigen_values.argsort()[::-1] 
sorted_eigen_values = eigen_values[eigen_values_indexes]
sorted_eigen_vectors = eigen_vectors[:,eigen_values_indexes]
eigen_vector_subset = sorted_eigen_vectors[:,0:2]

#Calculating the principal components
principal_components = np.dot(eigen_vector_subset.transpose(),mean_centered_data.transpose()).transpose()
principal_components = np.asarray(principal_components)

#creating groups of each disease
diseases_col = diseases_col.flatten().tolist()[0]
df = pd.DataFrame(dict(pc1=principal_components[:,0], pc2=principal_components[:,1], label=diseases_col))
disease_groups= df.groupby('label')

#PCA plotting
fig, ax = plt.subplots()
fig.suptitle('PCA of ' + file_name)
for disease, disease_class in disease_groups:
    ax.plot(disease_class.pc1, disease_class.pc2, marker='^', linestyle='', ms=6, label=disease)

plt.xlabel('PC1')
plt.ylabel('PC2')
ax.legend()
plt.savefig('PCA_' + file_name + ".png", dpi = 400)


#SVD plotting
svd = TruncatedSVD(n_components=2)
svd_features = svd.fit_transform(features)
df = pd.DataFrame(dict(svd1=svd_features[:,0], svd2=svd_features[:,1], label=diseases_col))
disease_groups = df.groupby('label')

fig, ax = plt.subplots()
fig.suptitle('SVD of ' + file_name)
for disease, disease_class in disease_groups:
    ax.plot(disease_class.svd1, disease_class.svd2, marker='^', linestyle='', ms=6, label=disease)

plt.xlabel('SVD1')
plt.ylabel('SVD2')
ax.legend()
plt.savefig('SVD_' + file_name + ".png", dpi = 400)


#tSNE plotting
tsne = TSNE(n_components=2)
tsne_features = tsne.fit_transform(features)
df = pd.DataFrame(dict(tsne1=tsne_features[:,0], tsne2=tsne_features[:,1], label=diseases_col))
disease_groups = df.groupby('label')
    
fig, ax = plt.subplots()
fig.suptitle('t-SNE of ' + file_name)
for disease, disease_class in disease_groups:
    ax.plot(disease_class.tsne1, disease_class.tsne2, marker='^', linestyle='', ms=6, label=disease)

plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
ax.legend()
plt.savefig('t-SNE_' + file_name + ".png", dpi = 400)
