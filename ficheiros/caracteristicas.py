import numpy as np
from ficheiros.constantes import COL

#----formula para o accelerometer
def calcularAccelerometer(tabela):
    ax = tabela[:, COL["ax"]]
    ay = tabela[:, COL["ay"]]
    az = tabela[:, COL["az"]]
    return np.sqrt(ax*ax + ay*ay + az*az)


#----formula para o gyroscope----
def calcularGyroscope(tabela):
    gx = tabela[:, COL["gx"]]
    gy = tabela[:, COL["gy"]]
    gz = tabela[:, COL["gz"]]
    return np.sqrt(gx*gx + gy*gy + gz*gz)


#----formula para o magnometer----
def calcularMagnometer(tabela):
    mx = tabela[:, COL["mx"]]
    my = tabela[:, COL["my"]]
    mz = tabela[:, COL["mz"]]
    return np.sqrt(mx*mx + my*my + mz*mz)
