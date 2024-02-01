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


