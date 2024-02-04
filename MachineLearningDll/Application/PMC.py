import ctypes
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

PMC_dll = ctypes.CDLL("C:\\ProjetML\\MachineLearningDll\\x64\Debug\\MachineLearningDll.dll")

ND_POINTER_INT = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C")
ND_POINTER_FLOAT = np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C")

#PMC
PMC_dll.CreatePMC.restype = ctypes.c_void_p
PMC_dll.CreatePMC.argtypes = [ND_POINTER_INT, ctypes.c_int]

PMC_dll.PredictPMC.restype = ND_POINTER_FLOAT
PMC_dll.PredictPMC.argtypes = [ND_POINTER_FLOAT, ctypes.c_int, ctypes.c_bool]

PMC_dll.TrainPMC.restype = None
PMC_dll.TrainPMC.argtypes = [ctypes.c_int, ND_POINTER_FLOAT, ctypes.c_int, ctypes.c_int]

PMC_dll.PredictionPMCSize.restype = ctypes.c_int
PMC_dll.PredictionPMCSize.argtypes = ctypes.c_int

PMC_dll.Destroy.restype = None
PMC_dll.Destroy.argtypes = ctypes.c_int

def create_pmc(layers):
    return PMC_dll.CreatePMC(layers, len(layers))

def train_pmc(pmc, inputs, input_size, num_inputs, raw_all_input, raw_all_output, size_output_subarray, num_outputs, is_classification, alpha, max_iter):
    PMC_dll.TrainPMC(pmc, inputs, input_size, num_inputs, raw_all_input, raw_all_output, size_output_subarray, num_outputs, is_classification, alpha, max_iter)

def predict_pmc(pmc, input, input_size, output_size, is_classification):
    output = np.zeros(output_size, dtype=np.float64)
    PMC_dll.PredictPMC(pmc, input, input_size, output, output_size, is_classification)
    return output

def destroy_pmc(pmc):
    PMC_dll.DestroyPMC(pmc)

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


def main ():
    # Définir la structure du PMC test linear simple
    layers = np.array([2, 1], dtype=np.int32)

    # Définir la structure du PMC test linear simple
    layers = np.array([2, 1], dtype=np.int32)

    # Créer le modèle PMC
    pmc_model = create_pmc(layers)

    # Effectuer le test de classification linéaire
    PMClinearSimpleClassification(pmc_model)

    # Effectuer le test linéaire multi
    test_multi_linear_classification(pmc_model)

    # Libérer les ressources du modèle PMC
    destroy_pmc(pmc_model)

if __name__ == "__main__":
    main()
