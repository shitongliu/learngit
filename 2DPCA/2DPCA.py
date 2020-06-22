# Jian Yang, David Zhang, Alejandro F. Frangi, and Jing-yu Yang
# "Two-Dimensional PCA: A New Approach to Appearance-Based Face
# Representation and Recognition".
# IEEE Transactions on Pattern Analysis and Machine Intelligence,
# 2004, 26(1):131-137.
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_images_from_disk(directory, max_value=255.):
    """
    read images from disk, all images must have the same size

    :param directory:   a special directory
    :param max_value:   for regularization
    :return:            3D array, shape (n_samples, height, width)
    """
    # load the images(all images are the same size)
    all_imgs = []
    all_paths = list(((Path.cwd()) / Path(directory)).glob('*'))
    all_paths = [str(path) for path in all_paths]
    for path in all_paths:
        img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE) / max_value
        all_imgs.append(np.expand_dims(img, 0))
    all_imgs = np.concatenate(all_imgs, 0)
    return all_imgs


def pca_2d(imgs, n_component):
    """
    Two-Dimensional PCA

    :param imgs:            training samples
    :param n_component:     number of principle component
    :return:                all_V: feature matrix of all samples
                                    shape (n_samples, hei or wid, n_component)
                            U: projection matrix
    """
    # get the shape
    M = imgs.shape[0]       # total number of training samples
    m, n = imgs.shape[1:]   # image size

    # 2D-PCA
    # 1) compute the scatter matrix Gt
    mean = np.mean(imgs, 0)     # mean face
    Gt = np.zeros(shape=(n, n))
    for i in range(0, M):
        face = imgs[i]
        Gt += (face - mean).T @ (face - mean)
    Gt /= M
    # 2) compute the eigenvalue and eigenvector X of Gt
    eig_val, X = np.linalg.eig(Gt)
    ind = np.argsort(eig_val)   # sort
    X = X[:, ind[::-1]]         # projection vector X[i]
    U = X[:, :n_component]      # projection matrix U
    all_V = []                  # feature matrix of all samples
    for i in range(0, M):
        V = imgs[i] @ U     # feature matrix
        all_V.append(np.expand_dims(V, 0))
    all_V = np.concatenate(all_V, 0)
    return all_V, U


print('start.')
d1, d2, d3 = 10, 30, 60     # number of components
all_imgs = read_images_from_disk('ORL')

all_V1, U1 = pca_2d(all_imgs, d1)
all_V2, U2 = pca_2d(all_imgs, d2)
all_V3, U3 = pca_2d(all_imgs, d3)

# 2DPCA-based image reconstruction
no = np.int(np.round(
    np.random.rand() * (all_imgs.shape[0] - 1)
))
A = all_imgs[no]
A_cap1 = all_V1[no] @ U1.T
A_cap2 = all_V2[no] @ U2.T
A_cap3 = all_V3[no] @ U3.T

# display the result
plt.figure()
plt.subplot(1, 4, 1)
plt.imshow(A, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.xlabel('original image')

plt.subplot(1, 4, 2)
plt.imshow(A_cap1, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.xlabel('d={}'.format(d1))

plt.subplot(1, 4, 3)
plt.imshow(A_cap2, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.xlabel('d={}'.format(d2))

plt.subplot(1, 4, 4)
plt.imshow(A_cap3, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.xlabel('d={}'.format(d3))

plt.show()

# 2DPCA-based image recognition
# get the training and test image
all_training_imgs = []
all_test_imgs = []
for i in range(0, all_imgs.shape[0]):
    if i % 10 >= 8:
        all_test_imgs.append(
            np.expand_dims(all_imgs[i, :, :], 0)
        )
    else:
        all_training_imgs.append(
            np.expand_dims(all_imgs[i, :, :], 0)
        )
all_training_imgs = np.concatenate(all_training_imgs, 0)
all_test_imgs = np.concatenate(all_test_imgs, 0)

# feature extraction by 2D-PCA
all_training_V, U = pca_2d(all_training_imgs, 20)
all_test_V =[]
for i in range(0, all_test_imgs.shape[0]):
    all_test_V.append(
        np.expand_dims((all_test_imgs[i] @ U), 0)
    )
all_test_V = np.concatenate(all_test_V, 0)

# test
cls_mean = np.zeros(shape=(40, all_training_V.shape[1], all_training_V.shape[2]))
dist = np.zeros(40)
acc = 0
for i in range(0, 40):
    cls_mean[i] = np.mean(all_training_V[8*i:8*i+8, :, :], 0)

for i in range(0, all_test_V.shape[0]):
    for j in range(0, 40):
        dist[j] = np.sum(np.linalg.norm(cls_mean[j] - all_test_V[i], axis=0, ord=2))
    if np.argmin(dist) == (i // 2):
        acc += 1
    else:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(all_test_imgs[i], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('true class:{}'.format(i//2))
        plt.subplot(1, 2, 2)
        plt.imshow(all_training_imgs[np.argmin(dist) * 8], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('predict class:{}'.format(np.argmin(dist)))
        plt.show()

print('accuracy:', acc / all_test_V.shape[0])




print('done.')
