import sys
import time

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


def load_data():
      CIFAR_transform_train = transforms.Compose([transforms.ToTensor()])
      CIFAR_transform_test = transforms.Compose([transforms.ToTensor()])

      trainset_CIFAR = datasets.CIFAR10(
            root='./datasets/cifar10/train',
            train=True,
            download=True,
            transform=CIFAR_transform_train
      )

      testset_CIFAR = datasets.CIFAR10(
            root='./datasets/cifar10/test',
            train=False,
            download=True,
            transform=CIFAR_transform_test
      )

      CIFAR_train = DataLoader(trainset_CIFAR, batch_size=32, shuffle=True, num_workers=2)
      CIFAR_test = DataLoader(testset_CIFAR, batch_size=32, shuffle=False, num_workers=2)

      return CIFAR_train, CIFAR_test


def main() -> None:
      CIFAR_train_loader, CIFAR_test_loader = load_data()
      CIFAR_train_images, CIFAR_train_labels = None, None

      print("Loading CIFAR training data:")
      for batch in tqdm(CIFAR_train_loader):
            images, labels = batch
            images_flat = images.view(images.shape[0], -1)
            if CIFAR_train_images is None:
                  CIFAR_train_images = images_flat.numpy()
                  CIFAR_train_labels = labels.numpy()
            else:
                  CIFAR_train_images = np.vstack([CIFAR_train_images, images_flat.numpy()])
                  CIFAR_train_labels = np.concatenate([CIFAR_train_labels, labels.numpy()])

      CIFAR_test_images, CIFAR_test_labels = None, None
      print("Loading CIFAR test data:")
      for batch in tqdm(CIFAR_test_loader):
            images, labels = batch
            images_flat = images.view(images.shape[0], -1)
            if CIFAR_test_images is None:
                  CIFAR_test_images = images_flat.numpy()
                  CIFAR_test_labels = labels.numpy()
            else:
                  CIFAR_test_images = np.vstack([CIFAR_test_images, images_flat.numpy()])
                  CIFAR_test_labels = np.concatenate([CIFAR_test_labels, labels.numpy()])

      print("Training SVM:")
      start_time = time.time()
      train_svm(CIFAR_train_images, CIFAR_train_labels, CIFAR_test_images, CIFAR_test_labels)
      end_time = time.time()
      print("Time for SVM training: ", end_time - start_time, " seconds")

      print("Training Decision Tree:")
      start_time = time.time()
      train_decision_tree(CIFAR_train_images, CIFAR_train_labels, CIFAR_test_images, CIFAR_test_labels)
      end_time = time.time()
      print("Time for Decision Tree training: ", end_time - start_time, " seconds")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)