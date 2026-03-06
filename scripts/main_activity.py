import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from misc.ficheiros.constantes import *
from misc.ficheiros.initializar import importParaTabela, groupByDeviceActivity, importParaTabelaComParticipante
from misc.ficheiros.plots import plotAtividade, plotOutliers, plot3D
from misc.ficheiros.caracteristicas import calcularAccelerometer, calcularGyroscope, calcularMagnometer
from misc.ficheiros.outliers import getOutliers, densidadeOutliersPorAtividade, getZscore, outlierKmeansDistancia, outlierKmeansTamanho
from scripts.module3_outliers_boxplot import plot_by_device_activity
from scripts.module3_zscore_plots import plot_outliers_por_atividade
from scripts.module3_kmeans_outliers import kMeans
from scripts.module4_feature_statistics import *
from scripts.module4_pca_relief import normalizar_e_PCA_e_variancia, plot4_4, fisher_relieff
from multiprocessing import Pool, cpu_count
from functools import partial
from misc.ficheiros.constantes import *
from src.har.smote import smote
from src.har.deployment import predict

#---colunas-
COL = {
    "dispositivo": 0,
    "ax": 1, "ay": 2, "az": 3,
    "gx": 4, "gy": 5, "gz": 6,
    "mx": 7, "my": 8, "mz": 9,
    "tempo": 10,
    "atividade": 11,
}

#--------------3.1----------------
tabela = importParaTabela(15, 5)

devices, activities, grid = groupByDeviceActivity(tabela, COL)

plot_by_device_activity(grid, devices, activities, calcularAccelerometer, "accel")
plot_by_device_activity(grid, devices, activities, calcularGyroscope, "gyro")
plot_by_device_activity(grid, devices, activities, calcularMagnometer, "magno")

plt.show()

#---- ---3.2---------
#---so no pulso direito (id 2)---
tabelaPulsoDireito = tabela[tabela[:, COL["dispositivo"]] == 2]

#---formulas no pulso direito
accPulsoDireito = calcularAccelerometer(tabelaPulsoDireito)
gyroPulsoDireito = calcularGyroscope(tabelaPulsoDireito)
magPulsoDireito = calcularMagnometer(tabelaPulsoDireito)
atividadesPulsoDireito = tabelaPulsoDireito[:, COL["atividade"]]

#--densidade
dens_accel = densidadeOutliersPorAtividade(accPulsoDireito, atividadesPulsoDireito)
dens_gyro = densidadeOutliersPorAtividade(gyroPulsoDireito, atividadesPulsoDireito)
dens_magno = densidadeOutliersPorAtividade(magPulsoDireito, atividadesPulsoDireito)

print("Densidade na acellerometer:", dens_accel)
print("Densidade no gyroscope", dens_gyro)
print("Densidade no magnometer:", dens_magno)




#-----------------3.4--------------
tabela = importParaTabela(15, 5)
pulsoDireito = tabela[tabela[:, COL["dispositivo"]] == 2]
acc  = calcularAccelerometer(tabela)
gyro = calcularGyroscope(tabela)
mag  = calcularMagnometer(tabela)
atividades = tabela[:, COL["atividade"]]

plot_outliers_por_atividade(3.0, calcularAccelerometer, devices, grid, activities)
plot_outliers_por_atividade(3.0, calcularGyroscope, devices, grid, activities)
plot_outliers_por_atividade(3.0, calcularMagnometer, devices, grid, activities)

plt.show()



#----------------3.7------------------

def cotovelo(data, minClusters,maxClusters, titulo="Plot"):
    
    #inicializar arrays 
    x = [i for i in range(minClusters, maxClusters+1)]
    y = []
    
    #calcular distancia média para cada numero de clusters
    for n in range(minClusters, maxClusters+1):
        print(n)
        arrayCentros, centros = kMeans(data, n)
        distancias = np.sqrt(np.sum((data - centros[arrayCentros]) ** 2, axis = 1))
        y.append(distancias.mean())
    
    #plot
    plt.plot(x,y, marker= 'o')
    plt.xlabel("Número de clusters")
    plt.ylabel("Distancia média de cada ponto ao seu centro")
    plt.title(titulo)
    plt.savefig(f"cotovelo_{titulo.replace(' ', '_')}.png")
    plt.close()
    return


minClusters = 1
maxClusters = 25

def executar_cotovelo(args):
    """Wrapper para executar cotovelo com argumentos desempacotados"""
    data, minClusters, maxClusters, titulo = args
    cotovelo(data, minClusters, maxClusters, titulo)
    return

# Retirar o comentário para fazer o teste do cotovelo
'''
tarefas = []
for idDispositivo in range(1, 6):
    mask = tabela[:, COL["dispositivo"]] == idDispositivo
    
    data = tabela[mask]
    dataAcc = data[:, COL["ax"]: COL["az"] + 1]
    dataGyr = data[:, COL["gx"]: COL["gz"] + 1]
    dataMag = data[:, COL["mx"]: COL["mz"] + 1]
    
    tarefas.append((dataAcc, minClusters, maxClusters, 
                   f"Dispositivo {idDispositivo}, acelerómetro"))
    tarefas.append((dataGyr, minClusters, maxClusters, 
                   f"Dispositivo {idDispositivo}, giroscópio"))
    tarefas.append((dataMag, minClusters, maxClusters, 
                   f"Dispositivo {idDispositivo}, magnetómetro"))

with Pool(processes=int(cpu_count()-1)) as pool:  # Ajustar número de processos conforme CPU
    pool.map(executar_cotovelo, tarefas)


 ''' 
#Criado com base na observação dos gráficos de cotovelo
arrayNClusters = [[15, 16, 16], [13, 18, 16], [15, 15, 12], [11, 15, 13], [25, 20, 18]]


dictSensor = {1: "Acelerómetro", 2: "Giroscópio", 3: "Magnetómetro"}

for idDispositivo in range(1,6):
    for sensor in range(1,4):
        nClusters = arrayNClusters[idDispositivo-1][sensor-1]
        
        mask = tabela[:, COL["dispositivo"]] == idDispositivo
        data = tabela[mask]
        
        match(sensor):
            case 1:
                data = data[:, COL["ax"] : COL["az"] + 1]
            case 2:
                data = data[:, COL["gx"] : COL["gz"] + 1]
            case 3:
                data = data[:, COL["mx"] : COL["mz"] + 1]
            
        arrayCentros, centros = kMeans(data, nClusters)
        
        outlierArray = outlierKmeansDistancia(data, arrayCentros, centros) | outlierKmeansTamanho(arrayCentros)

        print(f"|||||||Dispositivo {idDispositivo}, {dictSensor[sensor]}|||||||")
        print(f"Percentagem de outliers: {100 * np.sum(outlierArray)/len(data)}%\n")
        plot3D(data, outlierArray, f"Dispositivo {idDispositivo}, {dictSensor[sensor]}")


#---------------4.1--------------------
def testeNormalidade(data):
    teste = stats.kstest(data, stats.norm.cdf)
    return teste

def testeKruskalWallis(data):
    teste = stats.kruskal(*data)
    return teste


accel = calcularAccelerometer(tabela)
gyro = calcularGyroscope(tabela)
magno = calcularMagnometer(tabela)
modPorAtividade = np.column_stack([accel, magno, gyro, atividades])

#A testar se os dados de cada sensor por atividade seguem uma distribuição normal

#cada teste de normalidade vai corresponder a um teste estatistico
for atividade in range(1,17):
    for sensor in range(0,3):
        mask = modPorAtividade[:,3] == atividade
        teste = testeNormalidade(modPorAtividade[mask,sensor])
        if teste.pvalue >= 0.05:
            print(f"Atividade {atividade}, {dictSensor[sensor+1]}: a distribuição é normal")
        else:
            print(f"Atividade {atividade}, {dictSensor[sensor+1]}: a distribuição não é normal")

#Distribuição não normal e data unpaired,usar teste kruskal-wallis 
for sensor in range(0,3):
    dadosTeste = []
    for atividade in range(1,17):
        mask = modPorAtividade[:,3] == atividade
        dadosTeste.append(modPorAtividade[mask,sensor])
    teste = testeKruskalWallis(dadosTeste)
    if teste.pvalue >= 0.05:
        print(f"{dictSensor[sensor+1]}: Os valores médios não são relevantes para distinguir a atividade")
    else:
        print(f"{dictSensor[sensor+1]}: Os valores médios são relevantes para distinguir a atividade. Estatística: {teste.statistic}")
        


#---------------------4.2------------------------


janelas, atividades = criarJanelas(tabela)

featuresJanelas = []
for janela in janelas:
    featuresJanelas.append(extrairFeaturesJanela(janela))

featuresJanelas = np.array(featuresJanelas)

nomes_features = []
    
nomes_sensores = [
     'acc_x', 'acc_y', 'acc_z',      
     'gyro_x', 'gyro_y', 'gyro_z',   
     'mag_x', 'mag_y', 'mag_z'       
 ]

# Features individuais dos sensores
for nome_sensor in nomes_sensores:
    for nome_funcao in nomes_funcoes:
        nomes_features.append(f"{nome_sensor}_{nome_funcao}")

# Correlações pairwise
for i in range(len(nomes_sensores)):
    for j in range(i + 1, len(nomes_sensores)):
        nomes_features.append(f"corr_{nomes_sensores[i]}_{nomes_sensores[j]}")


#-----------------4.3------------------#

varianciaCumulativa = normalizar_e_PCA_e_variancia(featuresJanelas)
plot4_4(varianciaCumulativa)

manterAte75 = np.argmax(varianciaCumulativa >= 0.75) + 1
print(f"Variância de 75% atingida em {manterAte75} componentes.")


#----4.5-----4.6-----

top10fisher, top10relief = fisher_relieff(featuresJanelas, atividades)

top10fisher = np.array(top10fisher).flatten().astype(int)
top10relief = np.array(top10relief).flatten().astype(int)

#print("Top 10 Fisher:",[f"{int(i)}: {nomes_features[i]}" for i in top10fisher]) 
#print("Top 10 ReliefF:", [f"{int(i)}: {nomes_features[i]}" for i in top10relief])
print("Top 10 Fisher:")
for idx in top10fisher:
    print(f"{idx}: {nomes_features[idx]}")

print("\nTop 10 ReliefF:")
for idx in top10relief:
    print(f"{idx}: {nomes_features[idx]}")
    
    
    

#---Part B----
tabela_com_participantes = importParaTabelaComParticipante(15,5)
tabela_com_participantes = tabela_com_participantes[tabela_com_participantes[:,COL_WITH_PARTICIPANT["atividade"]] <= 7]


#-----1.1------

samples_per_activity = {}

print("Samples per activity:")

for i in range(1,8):
    count = np.sum(tabela_com_participantes[:,COL_WITH_PARTICIPANT["atividade"]]==i)
    samples_per_activity[i] = count
    print(f"{i}: {count}")
    

#----1.3------
tabela_participante3 = tabela_com_participantes[tabela_com_participantes[:,COL_WITH_PARTICIPANT["participante"]] == 3]

dados_participante3 = tabela_participante3[:, COL_WITH_PARTICIPANT["ax"] : COL_WITH_PARTICIPANT["mz"] +1]
atividades_participante3 = tabela_participante3[:,COL_WITH_PARTICIPANT["atividade"]]

valores_sinteticos = smote(dados_participante3, atividades_participante3, class_to_generate=4, k=5, num_samples=3)

# Calcular duas primeiras features (media e mediana) e plot

# Medias
plt.figure()
for i in range(1,8):
    dados_mask = atividades_participante3 == i
    dados = dados_participante3[dados_mask]
    media=[]
    colunas = [0,1,2,3,4,5,6,7,8]
    for col in colunas: 
        media.append(calcularMean(dados[:,col]))
    plt.scatter(colunas,media, label=f'Atividade {i}')

media = []
for col in colunas:
    media.append(calcularMean(valores_sinteticos[:,col]))
    
plt.scatter(colunas,media, marker='*')
plt.legend()
plt.title("Medias")
plt.show()


#Medianas
plt.figure()
for i in range(1,8):
    dados_mask = atividades_participante3 == i
    dados = dados_participante3[dados_mask]
    media=[]
    colunas = [0,1,2,3,4,5,6,7,8]
    for col in colunas: 
        media.append(calcularMedian(dados[:,col]))
    plt.scatter(colunas,media, label=f'Atividade {i}')

media = []
for col in colunas:
    media.append(calcularMedian(valores_sinteticos[:,col]))
    
plt.scatter(colunas,media, marker='*')
plt.legend()
plt.title("Medianas")
plt.show()

# 2 - build_embeddings_data.py e build_features_data.py
# 3 - splitEmbeddings.py e splitFeatures.py
# 4 - classifier.py
# 5 - evaluation.py e hypothesis_testing.py

# 6 - Usage case
pred = predict(janelas[5][:,COL["ax"]:COL["mz"]+1])
print(f"Prediction: {pred}")
