
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def euclidean_distance_by_mean(x1, x2):
    mean = (x1 + x2) / 2 
    distance = np.sqrt(np.sum((x1 - mean) ** 2))  
    return distance

def covar_matrix(df):
    mean = np.mean(df, axis=0)  
    centered = df - mean  
    cov_matrix = np.cov(centered, rowvar=False)  
    return cov_matrix

def mahalanobis_dist(x1, x2, df):
    mean_diff = x1 - x2  
    cov_matrix = covar_matrix(df)
    cov_matrix_inv = np.linalg.inv(cov_matrix)  
    distance = np.sqrt(np.dot(np.dot(mean_diff.T, cov_matrix_inv), mean_diff))
    return distance

class ref_temp:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.X = X 

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [mahalanobis_dist(x, x_train, self.X) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        label_counts = {}
        for label in k_nearest_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        most_common = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        return most_common[0][0]

    def plot_decision_boundary(self, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                             np.arange(y_min, y_max, 1))
        
        # Predict for each point in the mesh grid
        mesh_predictions = np.array([self._predict(np.array([i, j])) for i, j in np.c_[xx.ravel(), yy.ravel()]])
        mesh_predictions = mesh_predictions.reshape(xx.shape)
        
        # Plot the decision boundary
        plt.figure(figsize=(10, 6))
        cmap_background = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_points = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        
        plt.contourf(xx, yy, mesh_predictions, alpha=0.8, cmap=cmap_background)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor='k', cmap=cmap_points)
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title("KNN Decision Boundary")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()
