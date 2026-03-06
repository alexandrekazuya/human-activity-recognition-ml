from ficheiros.initializar import importParaTabela
from ficheiros.constantes import COL
from ficheiros.caracteristicas import calcularAccelerometer, calcularMagnometer, calcularGyroscope
from ficheiros.outliers import densidadeOutliersPorAtividade

tabela = importParaTabela(15, 5)

#---so no pulso direito (id 2)---
tabelaPulsoDireito = tabela[tabela[:, COL["dispositivo"]] == 2]

#---formulas no pulso direito
acc = calcularAccelerometer(tabelaPulsoDireito)
gyro = calcularGyroscope(tabelaPulsoDireito)
mag = calcularMagnometer(tabelaPulsoDireito)
atividades = tabelaPulsoDireito[:, COL["atividade"]]

#--densidade
dens_accel = densidadeOutliersPorAtividade(acc, atividades)
dens_gyro = densidadeOutliersPorAtividade(gyro, atividades)
dens_magno = densidadeOutliersPorAtividade(mag, atividades)

print("Densidade na acellerometer:", dens_accel)
print("Densidade no gyroscope", dens_gyro)
print("Densidade no magnometer:", dens_magno)