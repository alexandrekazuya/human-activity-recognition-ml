import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np

def getOutliers(valores):
    q1 = np.percentile(valores, 25)
    q3 = np.percentile(valores, 75)
    intervalo = q3 - q1
    limite_inferior = q1 - 1.5 * intervalo #acho q o stor tinha falado disto q era um k = 1.5
    limite_superior = q3 + 1.5 * intervalo
    outliers = (valores < limite_inferior) | (valores > limite_superior)
    return outliers

def densidadeOutliersPorAtividade(values, atividades):
    resultados = {}
    for a in np.unique(atividades.astype(int)):
        dados = values[atividades == a]
        if dados.size == 0:
            continue
        outliers = getOutliers(dados)
        densidade = (np.sum(outliers) / dados.size) * 100
        resultados[int(a)] = round(densidade, 3)  #3 casas decimais, 
                                                  #sem isso ficava tipo 1.1111111111111111
    return resultados

def getZscore(dados, k):
    media = np.mean(dados)
    std = np.std(dados)
    if std == 0:
        return np.zeros_like(dados, dtype=bool) #no caso do std ser zero, não dividimos
    zScore = (dados - media) / std
    return np.abs(zScore) > k #fica um array de true/false, indica so se é outlier ou nao

def normalizeZscore(dados):
    media = np.mean(dados, axis=0)
    std = np.std(dados, axis=0)
    std[std == 0] = 1.0
    return (dados - media) / std

def normalizeZscore_with_stats(dados):
    media = np.mean(dados, axis=0)
    std = np.std(dados, axis=0)
    safe = std.copy()
    safe[safe == 0] = 1.0 #da preplace aos 0s por 1
    normalized = (dados - media) / safe
    return normalized, media, safe

def outlierKmeansDistancia(data, arrayCentros, centros):
    '''Deteta outliers com base nos pontos mais distantes dos seus centros'''
    
    fatorLimite = 1.5 #Valor para verificar se um ponto está distante o suficiente para ser considerado outlier
    
    nClusters = len(centros)
    
    #Calcular a distancia entre cada ponto e o seu centro
    distancias = np.sqrt(np.sum((data - centros[arrayCentros]) ** 2, axis = 1))
    
    outliers = np.zeros(len(data), dtype = bool)
    
    
    for i in range(nClusters):
        mask = (arrayCentros == i) #Bool array, indíces True indicam que esse ponto está no cluster atual
        distanciasCluster = distancias[mask]
        
        #calcular média e std das distancias no cluster atual
        mean = np.mean(distanciasCluster)
        std = np.std(distanciasCluster)
        
        limiteOutlier = mean + (fatorLimite * std) 
        
        #Marcar os outlier no array
        outliers[mask] = distanciasCluster > limiteOutlier
    
    return outliers

def outlierKmeansTamanho(arrayCentros):
    '''Deteta outliers através da procura por clusters com poucos elementos'''
    
    tamanhoLimite = 5 #percentagem do tamanho medio dos clusters abaixo da qual os elementos do cluster sao considerados outliers
    
    #Numero de elementos em cada cluster
    tamanhoClusters = np.bincount(arrayCentros) 
    
    mean = np.mean(tamanhoClusters)
    
    limiteOutlier = (tamanhoLimite / 100) * mean
    
    clustersOutliers = tamanhoClusters < limiteOutlier #bool array -> true = cluster de outliers
    
    return clustersOutliers[arrayCentros] #bool array -> true = ponto outlier
