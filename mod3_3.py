import matplotlib.pyplot as plt
import numpy as np
from ficheiros.initializar import groupByDeviceActivity, importParaTabela
from ficheiros.constantes import *
from ficheiros.caracteristicas import *
from ficheiros.outliers import getZscore
from ficheiros.plots import plotOutliers


def plot_outliers_por_atividade(k, function, devices, grid, activities):
    if function.__name__ == "calcularAccelerometer":
        nome = "Accelerometer"
    elif function.__name__ == "calcularGyroscope":
        nome = "Gyroscope"
    elif function.__name__ == "calcularMagnometer":
        nome = "Magnometer"

    activity_to_idx = {a: i for i, a in enumerate(activities)}

    for d in devices:
        allvalues = []
        outliers = []
        x_positions = []

        for a in activities:
            bloco = grid[d][a]
            values = function(bloco)
            zscore = getZscore(values, k)

            allvalues.extend(values)
            outliers.extend(zscore)

            x_positions.extend([activity_to_idx[a]] * len(values))

        colors = np.where(outliers, "red", "blue")

        # plot
        plt.figure(figsize=(10, 4))
        plt.scatter(x_positions, allvalues, c=colors, s=2)
        plt.xticks(range(len(activities)), activities, rotation=45)
        plt.title(f"{nome}, Dispositivo {d} — Outliers por atividade (k={k})")
        plt.xlabel("Atividade")
        plt.ylabel(f"|{nome}|")
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    tabela = importParaTabela(15, 5)
    devices, activities, grid = groupByDeviceActivity(tabela, COL)
    plot_outliers_por_atividade(3.0, calcularAccelerometer, devices, grid, activities)
    plot_outliers_por_atividade(3.0, calcularGyroscope, devices, grid, activities)
    plot_outliers_por_atividade(3.0, calcularMagnometer, devices, grid, activities)

    plt.show()
