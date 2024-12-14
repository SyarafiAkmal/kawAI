import subprocess
import pandas as pd
from model.model_library import *

go_file = "main.go"

def KNN(k, filename):
    params = [k, filename, 'KNN'] 
    result = subprocess.run(['go', 'run', go_file] + params, capture_output=True, text=True)
    print(result.stdout)

    if result.stderr:
        print("Go program error:")
        print(result.stderr)

def NB(k, filename):
    params = [k, filename, 'NB'] 
    result = subprocess.run(['go', 'run', go_file] + params, capture_output=True, text=True)
    print(result.stdout)

    if result.stderr:
        print("Go program error:")
        print(result.stderr)

def ID3(k, filename):
    params = [k, filename, 'ID3'] 
    result = subprocess.run(['go', 'run', go_file] + params, capture_output=True, text=True)
    print(result.stdout)

    if result.stderr:
        print("Go program error:")
        print(result.stderr)

def KNN_library(k, filename) :
    data = pd.read_csv("data/" + filename + ".csv")
    KNNs(k, data)


KNN_library(5, 'dummy')
# KNN('3', 'dummy')