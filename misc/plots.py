import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#---plot

def plotAtividade(values, atividades, titulo):
    atividades_unicas = np.unique(atividades.astype(int))
    dados = [values[atividades == a] for a in atividades_unicas]

    plt.figure(figsize=(12, 6))
    plt.boxplot(dados, labels=atividades_unicas)
    plt.xlabel("Atividade")
    plt.ylabel("Valor")
    plt.title(titulo)

def plotOutliers(valores, mask_outlier, titulo):
    """
    Mostra scatter com pontos normais a azul e outliers a vermelho.
    - valores: array 1D (ex.: |acc|)
    - mask_outlier: array booleano (True = outlier)
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(valores)), valores, c=np.where(mask_outlier, "red", "blue"), s=1)
    plt.title(titulo)
    plt.xlabel("Índice da amostra")
    plt.ylabel("Valor")


def plot3D(data, outlierArray, titulo):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    
    
    ax.scatter3D(x[~outlierArray], y[~outlierArray], z[~outlierArray], c= "blue", alpha = 0.6, s = 15)
    ax.scatter3D(x[outlierArray], y[outlierArray], z[outlierArray], c= "red", marker = 'o', alpha = 0.3, s = 5)
    
    ax.set_title(titulo)             
    plt.show()
    