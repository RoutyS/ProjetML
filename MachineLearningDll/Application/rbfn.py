import ctypes
import numpy as np
import matplotlib.pyplot as plt

rbfn_dll = ctypes.CDLL("D:\\MachineLearning\\ProjetML\\MachineLearningDll\\x64\\Debug\\MachineLearningDll.dll")
class RBFN(ctypes.Structure):
    _fields_ = [
        ('num_centers', ctypes.c_int),
        ('centers', ctypes.c_float * 5),
        ('sigmas', ctypes.c_float * 5),
        ('weights', ctypes.c_float * 5)
    ]




rbfn_dll.rbf_approximation_instance.argtypes = [ctypes.c_float, ctypes.POINTER(RBFN)]
rbfn_dll.train_rbfn.argtypes = [ctypes.POINTER(RBFN), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float, ctypes.c_int]
rbfn_dll.train_classification_rbfn.argtypes = [ctypes.POINTER(RBFN), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_float, ctypes.c_int]
rbfn_dll.predict_classification_rbfn.argtypes = [ctypes.POINTER(RBFN), ctypes.c_float]
rbfn_dll.predict_classification_rbfn.restype = ctypes.c_int


if __name__ == "__main__":
    rbf_instance = RBFN()

    # Données d'entraînement
    X_train = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ], dtype=np.float32)
    Y_train = np.array([
        1,
        -1,
        -1
    ], dtype=np.int32)


    rbfn_dll.train_classification_rbfn(ctypes.byref(rbf_instance),
                                       X_train.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                       Y_train.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                       len(X_train), 0.01, 10000)

    # Points de test
    X_test = np.array([
        [1.0, 1.5],
        [2.5, 3.5]
    ], dtype=np.float32)


    predictions = [rbfn_dll.predict_classification_rbfn(ctypes.byref(rbf_instance), float(x[0])) for x in X_test]


    plt.scatter(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], color='blue', label='Classe 1')
    plt.scatter(X_train[Y_train == -1, 0], X_train[Y_train == -1, 1], color='red', label='Classe -1')


    for i, pred in enumerate(predictions):
        color = 'blue' if pred == 1 else 'red'
        plt.scatter(X_test[i, 0], X_test[i, 1], color=color, marker='x', label=f'Point de Test {i+1}')

    plt.legend()
    plt.show()
