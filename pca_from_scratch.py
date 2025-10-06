import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
import time

class PCA:
    """
    Principal Component Analysis implementation from scratch
    """
    
    def __init__(self, n_components: Optional[Union[int, float, str]] = None):
        """
        Initialize PCA
        
        Parameters:
        n_components: int, float, or str
            - int: Number of components to keep
            - float: Percentage of variance to preserve (0 < n_components < 1)
            - str: 'mle' for automatic selection using MLE (not implemented)
            - None: All components are kept
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_components_ = None
        
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit PCA model to data
        
        Parameters:
        X: numpy array of shape (n_samples, n_features)
        """
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        n_samples = X.shape[0]
        covariance_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Determine number of components
        if self.n_components is None:
            self.n_components_ = X.shape[1]
        elif isinstance(self.n_components, float) and 0 < self.n_components < 1:
            # Select components based on explained variance ratio
            explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
            cumulative_variance = np.cumsum(explained_variance_ratio)
            self.n_components_ = np.argmax(cumulative_variance >= self.n_components) + 1
        elif isinstance(self.n_components, int):
            self.n_components_ = min(self.n_components, X.shape[1])
        else:
            raise ValueError("Invalid n_components parameter")
            
        # Store results
        self.components_ = eigenvectors[:, :self.n_components_]
        self.explained_variance_ = eigenvalues[:self.n_components_]
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted PCA model
        """
        if self.components_ is None:
            raise ValueError("PCA must be fitted before transforming data")
            
        X_centered = X - self.mean_
        return X_centered @ self.components_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA model and transform data
        """
        return self.fit(X).transform(X)
    
    def get_cumulative_variance(self) -> np.ndarray:
        """
        Get cumulative explained variance ratio
        """
        return np.cumsum(self.explained_variance_ratio_)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space
        """
        if self.components_ is None:
            raise ValueError("PCA must be fitted before inverse transforming")
            
        return (X_transformed @ self.components_.T) + self.mean_


def benchmark_pca(X: np.ndarray, n_components: int = 2) -> Tuple[float, float]:
    """
    Benchmark PCA performance
    
    Returns:
    tuple: (fit_time, transform_time)
    """
    pca = PCA(n_components=n_components)
    
    # Time fitting
    start_time = time.time()
    pca.fit(X)
    fit_time = time.time() - start_time
    
    # Time transformation
    start_time = time.time()
    _ = pca.transform(X)
    transform_time = time.time() - start_time
    
    return fit_time, transform_time
