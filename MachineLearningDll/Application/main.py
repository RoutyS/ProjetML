import ctypes
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

ND_POINTER_INT = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C")
ND_POINTER_FLOAT = np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C")

#linear_model_dll = ctypes.CDLL("D:\\MachineLearning\\ProjetML\\MachineLearningDll\\x64\\Debug\\MachineLearningDll.dll")
linear_model_dll = ctypes.CDLL("C:\\Users\\ruth9\\Desktop\\ProjetML\\MachineLearningDll\\x64\\Debug\\MachineLearningDll.dll")

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


#PMC
linear_model_dll.CreatePMC.restype = ctypes.c_void_p
linear_model_dll.CreatePMC.argtypes = [ND_POINTER_INT, ctypes.c_int]

linear_model_dll.PredictPMC.restype = ND_POINTER_FLOAT
linear_model_dll.PredictPMC.argtypes = [ND_POINTER_FLOAT, ctypes.c_int, ctypes.c_bool]

linear_model_dll.TrainPMC.restype = None
linear_model_dll.TrainPMC.argtypes = [ctypes.c_int, ND_POINTER_FLOAT, ctypes.c_int, ctypes.c_int]

linear_model_dll.PredictionPMCSize.restype = ctypes.c_int
linear_model_dll.PredictionPMCSize.argtypes = [ctypes.c_void_p]

linear_model_dll.DestroyPMC.restype = None
linear_model_dll.DestroyPMC.argtypes = [ctypes.c_void_p]


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

    #cars_folder = "D:\\TST\\Application\\Voitures"
    cars_folder = "C:\\ProjetML\\MachineLearningDll\\Application\\Voitures"
    #motorcycles_folder = "D:\\TST\\Application\\Motos"
    motorcycles_folder = "C:\\ProjetML\\MachineLearningDll\\Application\\Motos"


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

def create_pmc(layers):
    return linear_model_dll.CreatePMC(layers, len(layers))

def train_pmc(pmc, inputs, input_size, num_inputs, raw_all_input, raw_all_output, size_output_subarray, num_outputs, is_classification, alpha, max_iter):
    linear_model_dll.TrainPMC(pmc, inputs, input_size, num_inputs, raw_all_input, raw_all_output, size_output_subarray, num_outputs, is_classification, alpha, max_iter)

def predict_pmc(pmc, input, input_size, output_size, is_classification):
    output = np.zeros(output_size, dtype=np.float64)
    linear_model_dll.PredictPMC(pmc, input, input_size, output, output_size, is_classification)
    return output

def destroy_pmc(pmc):
    linear_model_dll.DestroyPMC(pmc)

def PMClinearSimpleClassification(pmc_model):
    # Données pour le test de classification linéaire simple
    X = np.array([[1, 1], [2, 3], [3, 3]])
    Y = np.array([1, -1, -1])

    # Entraîner le modèle PMC
    train_pmc(pmc_model, X, Y)

    # Effectuer des prédictions
    predictions = np.zeros(Y.shape)
    for i, x in enumerate(X):
        predict_pmc(pmc_model, x, predictions[i], len(x), 1, True)

    # Visualiser les résultats
    for i, (x, pred) in enumerate(zip(X, predictions)):
        color = 'blue' if pred > 0 else 'red'
        plt.scatter(x[0], x[1], color=color)

    plt.show()
    plt.clf()

def PMClinearMultiClassification(pmc_model):
    # Données pour le test "linéaire multi"
    X = np.concatenate([np.random.random((50, 2)) * 0.9 + np.array([1, 1]),
                        np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    # Entraîner le PMC sur ces données
    train_pmc(pmc_model, X, Y)

    # Effectuer des prédictions
    predictions = np.zeros(Y.shape)
    for i, x in enumerate(X):
        predictions[i] = predict_pmc(pmc_model, x, len(x), 1, True)

    # Visualiser les résultats
    plt.scatter(X[0:50, 0], X[0:50, 1], color='blue')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='red')
    plt.show()
    plt.clf()

def train_pmc(X_train, y_train, layer_sizes, alpha, max_iter, is_classification):

    layer_sizes_np = np.array(layer_sizes, dtype=np.int32)

    # Création du PMC
    pmc_model = linear_model_dll.CreatePMC(layer_sizes_np, len(layer_sizes_np))

    # Formatage des données pour l'entraînement
    X_train_prepared = X_train.ctypes.data_as(ND_POINTER_FLOAT)
    y_train_prepared = y_train.astype(np.int32).ctypes.data_as(ND_POINTER_INT)

    # Entraînement du PMC
    if is_classification:
        linear_model_dll.TrainPMC(
            pmc_model,
            X_train_prepared,
            X_train.shape[1],
            X_train.shape[0],
            y_train_prepared,
            y_train.shape[0],
            alpha,
            max_iter
        )
    else:
        linear_model_dll.TrainPMCRegression(
            pmc_model,
            X_train_prepared,
            X_train.shape[1],
            X_train.shape[0],
            y_train_prepared,
            alpha,
            max_iter
        )

    # Prédiction avec le PMC
    predictions = []
    for i in range(X_test.shape[0]):
        X_test_prepared = X_test[i].ctypes.data_as(ND_POINTER_FLOAT)
        if is_classification:
            prediction = linear_model_dll.PredictPMC(pmc_model, X_test_prepared, X_test.shape[1], True)
        else:
            prediction = linear_model_dll.PredictPMCRegression(pmc_model, X_test_prepared, X_test.shape[1])
        predictions.append(prediction)

    # Destruction du PMC
    linear_model_dll.DestroyPMC(pmc_model)

    return predictions

def main():
    # Load image data
    #X, y = load_image_data()

    #X_train, X_test, y_train, y_test = simple_train_test_split(X, y, test_size=0.2, random_state=42)

    # Classification
    #X_classification = np.concatenate([np.random.random((50, 64, 64, 3)) * 0.9 + np.ones((50, 1, 1, 3)), np.random.random((50, 64, 64, 3)) * 0.9 + np.ones((50, 1, 1, 3)) * 2])
    #y_classification = np.concatenate([np.ones((50, 1)), np.zeros((50, 1))])
    #X_test_classification = X_test.reshape((X_test.shape[0], 64, 64, 3))

    #prediction_classification = principal_function(X_classification, y_classification, X_test_classification)
    #print("Classification Prediction:", prediction_classification)

    # Display images with predictions for classification
    #display_images_with_predictions(X_test_classification, y_test, prediction_classification)

    # Chargement des données d'images
    X, y = load_image_data()

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = simple_train_test_split(X, y, test_size=0.2, random_state=42)

    # Configuration pour la régression PMC
    layer_sizes_regression = [X_train.shape[1], 64, 1]
    alpha_regression = 0.01
    max_iter_regression = 1000

    # Entraînement du PMC pour la régression
    pmc_regression = train_pmc(X_train, y_train, layer_sizes_regression, alpha_regression, max_iter_regression, is_classification=False)

    # Configuration pour la classification PMC
    layer_sizes_classification = [X_train.shape[1], 64, 2]  # Supposant 2 classes pour la classification
    alpha_classification = 0.01
    max_iter_classification = 1000

    # Entraînement du PMC pour la classification
    #pmc_classification = train_pmc(X_train, y_train, layer_sizes_classification, alpha_classification,max_iter_classification, is_classification=True)

    # Prédictions avec le PMC de régression
    #predictions_regression = pmc_regression.predict(X_test)

    # Prédictions avec le PMC de classification
    #predictions_classification = pmc_classification.predict(X_test)

    # Affichage des images avec prédictions pour la classification
    #display_images_with_predictions(X_test, y_test, predictions_classification)

    # Affichage des images avec prédictions pour la régression
    #display_images_with_predictions(X_test, y_test, predictions_regression)


if __name__ == "__main__":
    main()