# Impelement Singular Value Decomposition from scratch

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# img = np.array(Image.open("todo.jpg")) / 255.
img = cv2.imread("todo.jpg", 0) / 255.
noised_img = img + 0.3 * np.random.normal(loc=0., scale=1., size=img.shape)
# plt.imshow(noised_img, cmap="gray")
# plt.show()




class SVD:
    # 1. standardize data
    # 2. compute covariance matrix
    # 3. compute eigenvalues and eigenvectors
    # 4. sort eigenvalues
    # 5. project data onto the principal components

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - X.mean(axis=0)) / X.std(axis=0)

    def covariance_matrix(self, X: np.ndarray) -> np.ndarray:
        return X@X.T / len(X)
        

        # return X.mean(axis=-1), X.std(axis=-1)

svd = SVD()
print(img.shape)
print(img)
scaled_img = svd.standardize(img)
cov_matrix = svd.covariance_matrix(scaled_img)
plt.imshow(cov_matrix, cmap="gray")
plt.show()
