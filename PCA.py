"""
* We have 40 labels. Every label has 10 pictures. Total of 400 pictures. Every picture has 92x112 = 10304 features.
* We will split our data matrix into a train data matrix and a test data matrix.
* D is the train data matrix with size n x d, n = 200 and d = 10304
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import cupy as cp

labels_num = 40  # The number of persons
features_num = 10304  # The number of features in every picture
dataset_dir = "archive\\"  # The dataset directory path


def read_image(image_path):
    """
    :param: image_path: path to image
    :return: A list of length 10304 contains the features of this image
    """
    image_vec = []
    with open(image_path, 'rb') as f:
        assert f.readline() == b'P5\n'
        assert f.readline() == b'92 112\n'
        assert f.readline() == b'255\n'

        for i in range(features_num):
            image_vec.append(ord(f.read(1)))
    return image_vec


def construct_dataset():
    """
    :return: The total data matrix, the total labels vector
    """
    ds, lb = [], []  # ds: dataset, lb: labels
    for i in range(1, labels_num + 1):
        label_path = dataset_dir + f's{i}/'
        for j in range(1, 11):
            lb.append(str(i))
            ds.append(read_image(f'{label_path}{j}.pgm'))

    return np.array(ds), lb


def split_data(ds, lb):
    """
    :param ds: dataset
    :param lb: labels
    :return: training data matrix, training labels vector, testing data matrix, testing labels vector
    """
    trd, trl, tsd, tsl = [], [], [], []  # Train data, Train labels, Test data, Test labels
    for i in range(len(ds)):
        # Odd for train
        if i & 1:
            trd.append(ds[i])
            trl.append(lb[i])
            continue
        # Even for test
        tsd.append(ds[i])
        tsl.append(lb[i])
    return np.array(trd), np.array(trl), np.array(tsd), np.array(tsl)


def get_eigen_variables(ds):
    """
    :param ds: Dataset (n x d) (200 x 10304)
    :return eigenvalues, eigenvectors: Those are for the covariance matrix of the dataset
    """

    # 1) Calculate the mean vector
    means = np.mean(ds, axis=0, dtype=np.float64)

    # 2) Center the data
    z = ds - means
    z = np.array(z).transpose()

    # 3) Calculate the covariance matrix
    cov = cp.cov(z, bias=True)

    # 4) Get the eigenvalues and eigenvectors of the covariance matrix in decreasing order of eigen values
    eigenvalues, eigenvectors = cp.linalg.eigh(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = cp.ndarray.get(eigenvalues[idx])
    eigenvectors = cp.ndarray.get(eigenvectors[:, idx]).transpose()
    return eigenvalues, eigenvectors


def get_projection_matrix(eigenvalues, eigenvectors, alpha):
    """
    :param eigenvalues:
    :param eigenvectors:
    :param alpha: Threshold for significance
    :return projection_matrix: The matrix used for reducing data dimensions
    """
    p = []  # p: projection matrix
    sm, total = 0, sum(eigenvalues)  # sm: sum of eigenvalues, total: total sum of all eigenvalues
    for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors):
        if sm / total >= alpha:
            break
        sm += eigenvalue
        if len(p) == 0:
            p.append(eigenvector)
        else:
            p = np.append(p, [eigenvector], axis=0)

    return np.array(p).transpose()


def PCA(ds, alpha):
    """
    :param ds: Dataset (n x d) (200 x 10304)
    :param alpha:
    :return projection_matrix: The matrix used for reducing data dimensions
    """
    eigenvalues, eigenvectors = get_eigen_variables(ds)
    return get_projection_matrix(eigenvalues, eigenvectors, alpha)


dataset, labels = construct_dataset()

train_data, train_labels, test_data, test_labels = split_data(dataset, labels)
alphas = [0.8, 0.85, 0.9, 0.95]
e_val, e_vec = get_eigen_variables(train_data)
# Comment out next line when testing multiple alphas, uncomment the next line when testing one alpha
# projection_matrix = PCA(train_data, 0.8)
for a in alphas:
    # Comment out next line and remove the loop when testing one alpha, un-comment and add the loop for multiple alphas
    projection_matrix = get_projection_matrix(e_val, e_vec, a)
    projected_train_data = np.real(np.dot(train_data, projection_matrix))
    projected_test_data = np.real(np.dot(test_data, projection_matrix))

    K = [1, 3, 5, 7]
    for k in K:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(projected_train_data, train_labels)
        predicted_labels = knn.predict(projected_test_data)
        acc = accuracy_score(test_labels, predicted_labels)
        print("Accuracy = ", acc, "  at k = ", k, " and alpha = ", a)
