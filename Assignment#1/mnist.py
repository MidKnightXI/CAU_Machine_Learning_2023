import sys

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def train_svm(train_images, train_labels, test_images, test_labels) -> None:
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.001, 0.01, 0.1]
    }
    svm_model_rbf = svm.SVC(kernel='rbf')
    svm_model_linear = svm.SVC(kernel='linear')
    svm_grid_search_rbf = GridSearchCV(svm_model_rbf, param_grid, cv=5)
    svm_grid_search_linear = GridSearchCV(svm_model_linear, param_grid, cv=5)

    svm_grid_search_rbf.fit(train_images, train_labels)
    svm_grid_search_linear.fit(train_images, train_labels)
    print("SVM Best hyperparameters for RBF kernel: ", svm_grid_search_rbf.best_params_)
    print("SVM Best hyperparameters for linear kernel: ", svm_grid_search_linear.best_params_)

    svm_model_rbf_best = svm.SVC(kernel='rbf', C=svm_grid_search_rbf.best_params_['C'], gamma=svm_grid_search_rbf.best_params_['gamma'])
    svm_model_linear_best = svm.SVC(kernel='linear', C=svm_grid_search_linear.best_params_['C'])
    svm_model_rbf_best.fit(train_images, train_labels)
    svm_model_linear_best.fit(train_images, train_labels)

    svm_predictions_rbf = svm_model_rbf_best.predict(test_images)
    svm_predictions_linear = svm_model_linear_best.predict(test_images)
    svm_accuracy_rbf = accuracy_score(test_labels, svm_predictions_rbf)
    svm_accuracy_linear = accuracy_score(test_labels, svm_predictions_linear)
    print("SVM Test accuracy for RBF kernel: ", svm_accuracy_rbf)
    print("SVM Test accuracy for linear kernel: ", svm_accuracy_linear)


def train_decision_tree(train_images, train_labels, test_images, test_labels) -> None:
    depths = set([3, 6, 9, 12])
    param_grid = {
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_leaf_nodes': [5, 10, None]
    }

    for depth in depths:
        print("Max depth:", depth)
        tree_model = DecisionTreeClassifier(max_depth=depth)
        tree_grid_search = GridSearchCV(tree_model, param_grid, cv=5)
        tree_grid_search.fit(train_images, train_labels)
        print("Decision Tree Best hyperparameters: ", tree_grid_search.best_params_)

        tree_model_best = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=tree_grid_search.best_params_['min_samples_split'],
            min_samples_leaf=tree_grid_search.best_params_['min_samples_leaf'],
            max_leaf_nodes=tree_grid_search.best_params_['max_leaf_nodes']
        )
        tree_model_best.fit(train_images, train_labels)
        tree_train_predictions = tree_model_best.predict(train_images)
        tree_train_accuracy = accuracy_score(train_labels, tree_train_predictions)
        print("Decision Tree Train accuracy: ", tree_train_accuracy)

        tree_test_predictions = tree_model_best.predict(test_images)
        tree_test_accuracy = accuracy_score(test_labels, tree_test_predictions)
        print("Decision Tree Test accuracy: ", tree_test_accuracy)




def get_dataloaders():
    mnist_train_transform = transforms.Compose([transforms.ToTensor()])
    mnist_test_transform = transforms.Compose([transforms.ToTensor()])

    trainset_mnist = datasets.MNIST(
        root='./datasets',
        train=True,
        download=True,
        transform=mnist_train_transform
    )
    testset_mnist = datasets.MNIST(
        root='./datasets',
        train=False,
        download=True,
        transform=mnist_test_transform
    )

    MNIST_train = DataLoader(trainset_mnist, batch_size=32, shuffle=True, num_workers=8)
    MNIST_test = DataLoader(testset_mnist, batch_size=32, shuffle=False, num_workers=8)

    return MNIST_train, MNIST_test



def main() -> None:
    MNIST_train_loader, MNIST_test_loader = get_dataloaders()
    MNIST_train_images, MNIST_train_labels = None, None

    print("Loading MNIST training data:")
    for batch in tqdm(MNIST_train_loader):
        images, labels = batch
        images_flat = images.view(images.shape[0], -1)
        if MNIST_train_images is None:
            MNIST_train_images = images_flat.numpy()
            MNIST_train_labels = labels.numpy()
        else:
            MNIST_train_images = np.vstack([MNIST_train_images, images_flat.numpy()])
            MNIST_train_labels = np.concatenate([MNIST_train_labels, labels.numpy()])

    MNIST_test_images, MNIST_test_labels = None, None
    print("Loading MNIST test data:")
    for batch in tqdm(MNIST_test_loader):
        images, labels = batch
        images_flat = images.view(images.shape[0], -1)
        if MNIST_test_images is None:
            MNIST_test_images = images_flat.numpy()
            MNIST_test_labels = labels.numpy()
        else:
            MNIST_test_images = np.vstack([MNIST_test_images, images_flat.numpy()])
            MNIST_test_labels = np.concatenate([MNIST_test_labels, labels.numpy()])

    print("Training SVM:")
    train_svm(MNIST_train_images, MNIST_train_labels, MNIST_test_images, MNIST_test_labels)
    print("Training Decision Tree:")
    train_decision_tree(MNIST_train_images, MNIST_train_labels, MNIST_test_images, MNIST_test_labels)



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
