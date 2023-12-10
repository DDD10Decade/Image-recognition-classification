import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import string
from mlxtend.plotting import plot_decision_regions
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
from sklearn.svm import SVC
# HOG

import json
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from subprocess import check_output

dim = 100


def GetFruit(fruitList, datatype, print_n=False, k_fold=False):
    images = []
    labels = []
    val = ['Training', 'test']
    if not k_fold:
        PATH = "./Fruit-Images-Dataset-master/" + datatype + "/"
        for i, fruit in enumerate(fruitList):
            p = PATH + fruit
            j = 0
            for image_path in glob.glob(os.path.join(p, "*.jpg")):
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (dim, dim))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                images.append(image)
                labels.append(i)
                j += 1
            if (print_n):
                print("There are", j, "", datatype.upper(), " images of ", fruitList[i].upper())
        images = np.array(images)
        labels = np.array(labels)
        return images, labels
    else:
        for v in val:
            PATH = "./Fruit-Images-Dataset-master/" + v + "/"
            for i, fruit in enumerate(fruitList):
                p = PATH + fruit
                j = 0
                for image_path in glob.glob(os.path.join(p, "*.jpg")):
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    image = cv2.resize(image, (dim, dim))
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    images.append(image)
                    labels.append(i)
                    j += 1
            images = np.array(images)
            labels = np.array(labels)
            return images, labels


def GetAll():
    fruitList = []
    for fruit_path in glob.glob("./Fruit-Images-Dataset-master/Training/*"):
        fruit = fruit_path.split("/")[-1]
        fruitList.append(fruit)
    return fruitList


def getClassNumber(y):
    v = []
    i = 0
    count = 0
    for index in y:
        if index == i:
            count += 1
        else:
            v.append(count)
            count = 1
            i += 1
    v.append(count)
    return v


def plotPrincipalComponents(X, dim):
    v = getClassNumber(y_train)
    colors = 'orange', 'purple', 'r', 'c', 'm', 'y', 'k', 'grey', 'b', 'g'
    markers = ['o', 'x', 'v', 'd']
    tot = len(X)
    start = 0
    if dim == 2:
        for i, index in enumerate(v):
            end = start + index
            plt.scatter(X[start:end, 0], X[start:end, 1], color=colors[i % len(colors)],
                        marker=markers[i % len(markers)], label=fruitList[i])
            start = end
        plt.xlabel('PC1')
        plt.ylabel('PC2')

    if (dim == 3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, index in enumerate(v):
            end = start + index
            ax.scatter(X[start:end, 0], X[start:end, 1], X[start:end, 2], color=colors[i % len(colors)],
                       marker=markers[i % len(markers)], label=fruitList[i])
            start = end
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

    plt.legend(loc='lower left')
    plt.xticks()
    plt.yticks()
    plt.show()


# 绘制混淆矩阵

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix

    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=fruitList, yticklabels=fruitList,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return cm, ax


sample = cv2.imread("./Fruit-Images-Dataset-master/Training/Kiwi/0_100.jpg", cv2.IMREAD_COLOR)
plt.hist(sample.ravel(), bins=256, range=[0, 256]);
plt.show()
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([sample], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
plt.xlim([0, 256])
plt.show()

print("kiwi0_100\n")

# Binary classification
fruitList = ['Pineapple', 'kiwi']
# Get image and Labels
X_train_raw, y_train = GetFruit(fruitList, 'Training', print_n=True, k_fold=False)
X_test_raw, y_test = GetFruit(fruitList, 'test', print_n=True, k_fold=False)
# Get data for k-fold
X_raw, y = GetFruit(fruitList, '', print_n=True, k_fold=True)

from skimage import feature

ppc = 16
hog_features_train = []
hog_features_test = []
hog_images_train = []
hog_images_test = []
for image in X_train_raw:
    # print("1")
    # print(image)
    fd, hog_image = feature.hog(image, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(4, 4),
                                block_norm='L2', visualize=True, channel_axis=2)

    hog_images_train.append(hog_image)
    hog_features_train.append(fd)

print("2")
for image in X_test_raw:
    fd, hog_image = feature.hog(image, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(4, 4),
                                block_norm='L2', visualize=True, channel_axis=2)
    hog_images_test.append(hog_image)
    hog_features_test.append(fd)

# Scale data Images
scaler = StandardScaler()
X_train = scaler.fit_transform([i.flatten() for i in X_train_raw])
X_test = scaler.fit_transform([i.flatten() for i in X_test_raw])
X_train_hog = scaler.fit_transform([i.flatten() for i in hog_images_train])
X_test_hog = scaler.fit_transform([i.flatten() for i in hog_images_test])
X = scaler.fit_transform([i.flatten() for i in X_raw])

print("Shape of the data:")
print((X_train.shape, y_train.shape))
print((X_test.shape, y_test.shape))
print(X_train_hog.shape)
print(X_test_hog.shape)

print("\nData sample:")
print((X_train[0], y_train[0]))
print((X_train_hog[0], y_train[0]))


# x中是数据，每张图片由100*100像素，3条RGB通道构成，一共是30000维的向量构成
# y中是预测结果，二分类中被编码为0和1


# 查看数据样例
def plot_image_grid(images, nb_rows, nb_cols, figsize=(15, 15)):
    assert len(images) == nb_rows * nb_cols, "Number of images should be the same as (nb_rows*nb_cols)"
    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=figsize)

    n = 0
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            axs[i, j].axis('off')
            axs[i, j].imshow(images[n])
            n += 1


print(fruitList)
plot_image_grid(X_train_raw[0:100], 10, 10)
plot_image_grid(X_train_raw[490:590], 10, 10)

plt.imshow(X_train_raw[1])

# linear SVM using hog features
svm = SVC(gamma='auto', kernel='linear', probability=True)
svm.fit(X_train_hog, y_train)
y_pred = svm.predict(X_test_hog)

# Evaluation
precision = metrics.accuracy_score(y_pred, y_test) * 100
print("Accuracy with SVM: {0:.2f}%".format(precision))
cm, _ = plot_confusion_matrix(y_test, y_pred, classes=y_train, normalize=True, title="Normalized confusion matrix")
plt.show()

# calculate FPR and TPR
probs = svm.predict_proba(X_test_hog)
probs = probs[:, 1]
svm_fpr, svm_tpr, thresholds = metrics.roc_curve(y_test, probs)
svm_auc = metrics.roc_auc_score(y_test, probs)
