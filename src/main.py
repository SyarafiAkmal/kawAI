import subprocess
import pandas as pd
from model.model_library import *

go_file = "main.go"

def KNN(k, filename):
    params = [filename, 'KNN', str(k)] 
    result = subprocess.run(['go', 'run', go_file] + params, capture_output=True, text=True)
    print(result.stdout)

    if result.stderr:
        print("Go program error:")
        print(result.stderr)

def NB(filename):
    params = [filename, 'NB'] 
    result = subprocess.run(['go', 'run', go_file] + params, capture_output=True, text=True)
    print(result.stdout)

    if result.stderr:
        print("Go program error:")
        print(result.stderr)

def ID3(filename):
    params = [filename, 'ID3'] 
    result = subprocess.run(['go', 'run', go_file] + params, capture_output=True, text=True)
    print(result.stdout)

    if result.stderr:
        print("Go program error:")
        print(result.stderr)

def KNN_library(filename) :
    data = pd.read_csv("data/" + filename + ".csv")
    KNNs(data)

def NB_library(filename) :
    data = pd.read_csv("data/" + filename + ".csv")
    NBs(data)
    # print(data.head())
    
def ID3_library(filename) :
    data = pd.read_csv("data/" + filename + ".csv")
    ID3s(data)

# KNN_library(5, 'data')
# NB_library('final_data_set')
# ID3_library('data')   

# NB('data2')


