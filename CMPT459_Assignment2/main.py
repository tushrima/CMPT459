#References
#CMPT 459 Martin Ester Lecture notes
#CMPT 419 Angelica Lim Assignment 2
#https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670

import anndata
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA as pca
import argparse
from kmeans import KMeans
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='number of clusters to find')
    parser.add_argument('--n-clusters', type=int,
                        help='number of features to use in a tree',
                        default=2)
    parser.add_argument('--data', type=str, default='scRNAseq_human_pancreas.csv',
                        help='data path') #human pancreas tissue dataset

    a = parser.parse_args()
    return(a.n_clusters, a.data)

def read_data(data_path):
    return anndata.read_csv(data_path)

def preprocess_data(adata: anndata.AnnData, scale :bool=True):
    """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0

    return adata


def PCA(X, num_components: int):
    return pca(num_components).fit_transform(X)

def main():
    n_classifiers, data_path = parse_args()
    heart = read_data(data_path)
    heart = preprocess_data(heart)
    X = PCA(heart.X, 100)
    # Your code

    # Clustering using KMeans
    random_= []
    for k in range(2, 10):
        print("k =",k)
        kmean_random = KMeans(n_clusters=k, init='random')
        labels = kmean_random.fit(X)
        score = kmean_random.silhouette(labels, X)
        random_.append(score)
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(2, 10), random_) #clusterings for k from 2 to 10
    ax.set_xlabel('k')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score for various k value - random')
    plt.savefig('kmeans.png')
    print(random_)
    # part 3 Produce clusterings for k from 2 to 10 using KMeans++ initialization.
    kmpp_scores = []
    for k in range(2, 10):
        kmpp = KMeans(n_clusters=k, init='kmeans++')
        labels = kmpp.fit(X)
        score = kmpp.silhouette(labels, X)
        kmpp_scores.append(score)

    fig, ax = plt.subplots()
    ax.plot(np.arange(2, 10), kmpp_scores)
    ax.set_xlabel('k')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score for various k value - kmeans++')
    plt.savefig('kmeans++.png')
    print(kmpp_scores)
   
    #part 4 Use a scatter plot to visualize the clusters with the best k from task 2 and 3.
    n_classifiers = np.argmax(random_) + 2
    X_2 = PCA(heart.X, 2)
    model = KMeans(n_clusters=n_classifiers, init='random')
    labels = model.fit(X_2)
    visualize_cluster(X_2, labels, labels)     # Visualize clustering
    

def visualize_cluster(x, y, clustering):
    #Your code
    plt.figure(figsize=(8, 8))
    plt.scatter(x[:, 0], x[:, 1], c=clustering, cmap='viridis', s=50, alpha=0.5)
    plt.xlabel('Principal Component 1', fontsize=15)
    plt.ylabel('Principal Component 2', fontsize=15)
    plt.title('2 component PCA', fontsize=20)
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.savefig('ScatterPlot.png')

if __name__ == '__main__':
    main()
