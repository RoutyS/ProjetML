import ctypes
import numpy as np


rbfn_dll = ctypes.CDLL("D:\\MachineLearning\\ProjetML\\MachineLearningDll\\x64\\Debug\\MachineLearningDll.dll")  # Replace "YourRBFNDll.dll" with the actual DLL name
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

    inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    targets = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

    rbfn_dll.train_rbfn(ctypes.byref(rbf_instance), inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        targets.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(inputs), 0.1, 100)

    input_value = 2.5
    result = rbfn_dll.rbf_approximation_instance(input_value, ctypes.byref(rbf_instance))
    print(f'RBF Approximation result  {input_value}: {result}')

    classification_targets = np.array([0, 1, 0, 1, 1], dtype=np.int32)
    rbfn_dll.train_classification_rbfn(ctypes.byref(rbf_instance), inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                       classification_targets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                       len(inputs), 0.1, 100)

    classification_input = 3.5
    prediction = rbfn_dll.predict_classification_rbfn(ctypes.byref(rbf_instance), classification_input)
    print(f'Classification prediction for input {classification_input}: {prediction}')
