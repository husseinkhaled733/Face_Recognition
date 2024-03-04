import numpy as np  # linear algebra
import scipy as sp
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import time

number_of_persons = 40


def isSymmetric(matrix, tol=1e-4):
    return np.allclose(matrix, matrix.T, atol=tol)


def read_single_image(image_path):
    ans = []
    with open(image_path, 'rb') as f:
        assert f.readline() == b'P5\n'
        assert f.readline() == b'92 112\n'
        assert f.readline() == b'255\n'
        # print(f.readline())
        # print(f.readline())

        for i in range(10304):
            ans.append(ord(f.read(1)))
    if image_path == "/home/hussein/PatternDatasets/Faces/s40/10.pgm":
        print(image_path)
        print(ans)
        print(len(ans))

    return ans  # np.array(ans, dtype='float')


def construct_data_frame():
    images = []
    persons = []

    path = "/home/hussein/PatternDatasets/Faces/"
    print('Reading Started')
    for x in range(1, number_of_persons + 1):
        current_person_path = path + 's' + str(x) + '/'
        for y in range(1, 11):
            persons.append(str(x))
            images.append(read_single_image(current_person_path + str(y) + '.pgm'))

    images = np.array(images)

    return images, persons


def split_data(D, labels):
    train_Data = []
    train_Label = []
    test_Data = []
    test_Label = []
    for i in range(len(D)):
        if i % 2 == 0:
            train_Data.append(D[i])
            train_Label.append(labels[i])
        else:
            test_Data.append(D[i])
            test_Label.append(labels[i])
    return np.array(train_Data), np.array(train_Label), np.array(test_Data), np.array(test_Label)


def LDA(Data, label, eigen_values_count=39):
    print(eigen_values_count)
    Data = np.array(Data)
    label = np.array(label)
    unique_values, count = np.unique(label, return_counts=True)
    number_of_classes = len(np.unique(label))
    print("Number of Classes = ", number_of_classes)
    number_of_features = len(Data[0])

    # Calculate Means for every class
    means = np.zeros((number_of_classes, number_of_features))

    for i in range(1, number_of_classes + 1):
        data = Data[np.where(label == str(i))]
        print("data shape = ", data.shape)
        meanI = np.mean(data, axis=0)
        means[i - 1] = meanI

    # Calculate mean for each feature
    mean = np.mean(Data, axis=0)

    # Calculate Sb
    Sb = np.zeros((number_of_features, number_of_features))
    for i in range(number_of_classes):
        x = np.array(means[i] - mean).reshape(number_of_features, 1)
        y = x.dot(x.T) * count[i]
        Sb = np.add(Sb, y)

    # Calculate S
    S = np.zeros((number_of_features, number_of_features))
    for i in range(number_of_classes):
        dataI = Data[label == str(i + 1)]
        Zi = np.array(dataI - means[i])
        S += Zi.T.dot(Zi)

    # Calculate Eigen Values and Eigen Vectors

    try:
        inverse = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        print("S is Singular")
        inverse = np.linalg.pinv(S)

    print("Is S Symmetric = ", isSymmetric(S))
    print("Is Sb Symmetric = ", isSymmetric(Sb))
    print("Is inverse Symmetric = ", isSymmetric(inverse))
    print(inverse)

    X = inverse.dot(Sb)
    print("Is X Symmetric = ", isSymmetric(X))
    eigen_Values, eigen_Vectors = np.linalg.eigh(X)

    idx = eigen_Values.argsort()[::-1]
    eigen_Values = eigen_Values[idx]
    eigen_Vectors = eigen_Vectors[:, idx]
    print("Eigen Values = ", eigen_Values)
    print("Eigen Vectors = ", eigen_Vectors)
    eigen_Vectors = np.real(eigen_Vectors)

    U = eigen_Vectors[:, 0:eigen_values_count]
    return U

# (D, labels) = construct_data_frame()
#
# print(D)
# print(labels)
# (train_Data, train_Label, test_Data, test_Label) = split_data(D, labels)
#
# start_time = time.time()
#
# U = LDA(train_Data, train_Label, eigen_values_count=39)
#
# end_time = time.time()
#
# print("time taken = ",end_time-start_time)
#
# Projected_train_Data = train_Data.dot(U)  # ==  ((U.T).dot(train_Data.T)).T
# Projected_test_Data = test_Data.dot(U)
#
# K = [1, 3, 5, 7]
# for i in K:
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(Projected_train_Data, train_Label)
#     predicted_labels = knn.predict(Projected_test_Data)
#     acc = accuracy_score(test_Label, predicted_labels)
#
#     print("Accuracy = ", acc, "  at k = ", i)
