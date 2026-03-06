import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from misc.ficheiros.initializar import *
from misc.ficheiros.constantes import COL
from misc.ficheiros.caracteristicas import calcularAccelerometer, calcularMagnometer, calcularGyroscope
from misc.ficheiros.plots import plotAtividade



def plot_by_device_activity(grid, devices, activities, function, sensor_name):
    for d in devices:
        grupos = []
        labels = []
        for a in activities:
            bloco = grid[d][a]
            values = function(bloco)

            grupos.append(values)
            labels.append(str(a))

        plt.figure(figsize=(12, 6))
        plt.boxplot(grupos, labels=labels)
        plt.title(f"Dispositivo {d} — {sensor_name} por atividade")
        plt.xlabel("Atividade")
        plt.ylabel("modulo")
    return
    
'''
tabela = importParaTabela(15, 5)

devices, activities, grid = group_by_device_activity(tabela, COL)    

plot_by_device_activity(grid, devices, activities, calcularAccelerometer, "accel")
plot_by_device_activity(grid, devices, activities, calcularGyroscope, "gyro")
plot_by_device_activity(grid, devices, activities, calcularMagnometer, "magno")

plt.show()
'''

