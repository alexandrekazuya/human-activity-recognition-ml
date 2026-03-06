import numpy as np
import matplotlib.pyplot as plt
from ficheiros.initializar import *
from ficheiros.constantes import COL
from ficheiros.caracteristicas import calcularAccelerometer, calcularMagnometer, calcularGyroscope
from ficheiros.plots import plotAtividade

tabela = importParaTabela(15, 5)

devices, activities, grid = groupByDeviceActivity(tabela, COL)

def plotByDeviceActivity(grid, devices, activities, function, sensor_name):
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

plotByDeviceActivity(grid, devices, activities, calcularAccelerometer, "accel")
plotByDeviceActivity(grid, devices, activities, calcularGyroscope, "gyro")
plotByDeviceActivity(grid, devices, activities, calcularMagnometer, "magno")

plt.show()
