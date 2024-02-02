import ctypes
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

linear_model_dll = ctypes.CDLL("D:\\MachineLearning\\ProjetML\\MachineLearningDll\\x64\\Debug\\MachineLearningDll.dll")


linear_model_dll.create_linear_model.restype = ctypes.POINTER(ctypes.c_void_p)
linear_model_dll.create_linear_model.argtypes = [ctypes.c_int]

linear_model_dll.train_regression.restype = None
linear_model_dll.train_regression.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
]

linear_model_dll.train_classification.restype = None
linear_model_dll.train_classification.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
]

linear_model_dll.predict_regression.restype = ctypes.c_double
linear_model_dll.predict_regression.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
]

linear_model_dll.predict_classification.restype = ctypes.c_int
linear_model_dll.predict_classification.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
]

linear_model_dll.destroy_linear_model.restype = None
linear_model_dll.destroy_linear_model.argtypes = [ctypes.c_void_p]

def load_images_from_folder(folder, label):
    X = []
    y = []

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)

        try:
            img = cv2.imread(img_path)

            if img is not None:

                img = cv2.resize(img, (64, 64))
                # Flatten the image into a 1D array
                features = img.flatten()
                X.append(features)
                y.append(label)
            else:
                print(f"Error loading image: {img_path}")

        except Exception as e:
            print(f"Error processing image: {img_path}. Error: {str(e)}")

    return X, y


def load_image_data():

    cars_folder = "D:\\TST\\Application\\Voitures"
    motorcycles_folder = "D:\\TST\\Application\\Motos"


    X_cars, y_cars = load_images_from_folder(cars_folder, label=0)
    X_motorcycles, y_motorcycles = load_images_from_folder(motorcycles_folder, label=1)


    X = np.concatenate([X_cars, X_motorcycles])
    y = np.concatenate([y_cars, y_motorcycles])

    X = np.array(X)
    y = np.array(y)

    return X, y

def principal_function(X_train, y_train, X_test):
    num_features = X_train.shape[1]
    model = linear_model_dll.create_linear_model(num_features)

    y_train = y_train.astype(np.int32)
    linear_model_dll.train_classification(
        model,
        X_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_train.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        X_train.shape[0],
        X_train.shape[1],
        0.0001,
        10000
    )

    predictions = np.zeros(X_test.shape[0], dtype=np.int32)
    for i in range(X_test.shape[0]):
        predictions[i] = linear_model_dll.predict_classification(
            model,
            X_test[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            X_test.shape[1],
        )

    linear_model_dll.destroy_linear_model(model)

    return predictions

def simple_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split_index = int((1 - test_size) * num_samples)

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

def display_images_with_predictions(X, y_actual, predictions):
    num_samples = len(X)


    predictions = np.asarray(predictions)
    if predictions.ndim == 0:
        predictions = np.array([predictions])

    num_rows = 2
    num_cols = (num_samples + 1) // 2

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5))

    for i in range(num_samples):
        img_shape = (64, 64, 3)

        img = X[i].reshape(img_shape)
        label_actual = "Car" if y_actual[i] == 0 else "Motorcycle"

        label_pred = "Car" if predictions[i] == 0 else "Motorcycle"

        axes[i // num_cols, i % num_cols].imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        axes[i // num_cols, i % num_cols].axis('off')
        axes[i // num_cols, i % num_cols].set_title(f"Actual: {label_actual}\nPrediction: {label_pred}")

    plt.show()

def main():
    # Load image data
    X, y = load_image_data()


    X_train, X_test, y_train, y_test = simple_train_test_split(X, y, test_size=0.2, random_state=42)

    # Classification
    X_classification = np.concatenate([np.random.random((50, 64, 64, 3)) * 0.9 + np.ones((50, 1, 1, 3)), np.random.random((50, 64, 64, 3)) * 0.9 + np.ones((50, 1, 1, 3)) * 2])
    y_classification = np.concatenate([np.ones((50, 1)), np.zeros((50, 1))])
    X_test_classification = X_test.reshape((X_test.shape[0], 64, 64, 3))

    prediction_classification = principal_function(X_classification, y_classification, X_test_classification)
    print("Classification Prediction:", prediction_classification)

    # Display images with predictions for classification
    display_images_with_predictions(X_test_classification, y_test, prediction_classification)


if __name__ == "__main__":
    main()
