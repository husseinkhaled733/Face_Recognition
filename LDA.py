import numpy as np  # linear algebra
import scipy
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

    return ans


def construct_data_frame():
    images = []
    persons = []

    path = "C:\\Users\\DELL\\Desktop\\Pattern_Lab1\\archive\\"
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
        if i % 2 == 1:
            train_Data.append(D[i])
            train_Label.append(labels[i])
        else:
            test_Data.append(D[i])
            test_Label.append(labels[i])
    return np.array(train_Data), np.array(train_Label), np.array(test_Data), np.array(test_Label)

def split_data_70_30(D, labels):
    train_Data = []
    train_Label = []
    test_Data = []
    test_Label = []
    for i in range(len(D)):
        if i % 10 <= 6:
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
        # print("data shape = ", data.shape)
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

    inverse = scipy.linalg.pinv(S)

    X = inverse.dot(Sb)

    eigen_Values, eigen_Vectors = np.linalg.eig(X)

    idx = eigen_Values.argsort()[::-1]
    eigen_Values = eigen_Values[idx]
    eigen_Vectors = eigen_Vectors[:, idx]
    print("Eigen Values = ", eigen_Values)
    print("Eigen Vectors = ", eigen_Vectors)
    eigen_Vectors = np.real(eigen_Vectors)

    U = eigen_Vectors[:, 0:eigen_values_count]
    return U


def calc_Accuracy(Projected_train_Data, Projected_test_Data, train_Label, test_Label):
    K = [1, 3, 5, 7]
    accuracies = []

    for i in K:
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(Projected_train_Data, train_Label)
        predicted_labels = knn.predict(Projected_test_Data)
        acc = accuracy_score(test_Label, predicted_labels)
        accuracies.append(acc)

        print("Accuracy = ", acc, "  at k = ", i)

    # Plotting accuracy against K
    plt.plot(K, accuracies, marker='o')
    plt.title('Accuracy vs. K')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.xticks(K)
    plt.grid(True)
    plt.show()



(D, labels) = construct_data_frame()


#(train_Data, train_Label, test_Data, test_Label) = split_data(D, labels)
(train_Data, train_Label, test_Data, test_Label) = split_data_70_30(D, labels)

start_time = time.time()

U = LDA(train_Data, train_Label, eigen_values_count=39)

end_time = time.time()

print("time taken = ", end_time - start_time)

Projected_train_Data = train_Data.dot(U)  # ==  ((U.T).dot(train_Data.T)).T
Projected_test_Data = test_Data.dot(U)

calc_Accuracy(Projected_train_Data, Projected_test_Data,train_Label ,test_Label)
