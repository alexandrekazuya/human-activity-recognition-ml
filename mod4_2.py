import numpy as np
from scipy.stats import skew, kurtosis, iqr, entropy
from scipy.fft import fft
from ficheiros.constantes import COL, COL_WITH_PARTICIPANT
from ficheiros.caracteristicas import *

def calcularMean(data):
    return np.mean(data)

def calcularMedian(data):
    return np.median(data)

def calcularSTD(data):
    return np.std(data)

def calcularVariance(data):
    return np.var(data)

def calcularRMS(data):
    return np.sqrt(np.mean(data**2))

def calcularAveragedDerivatives(data):
    t = 1/51.2 #na pagina github diz que a frequencia é 51.2Hz
    
    return np.mean(np.abs(np.gradient(data,t))) 
    
def calcularSkewness(data):
    return skew(data)

def calcularKurtosis(data):
    return kurtosis(data)

def calcularInterquartileRange(data):
    return iqr(data)

def calcularZeroCrossingRate(data):
    
    sinal = np.sign(data)
    
    mudancaSinal = np.abs(np.diff(sinal))
    
    zcr = np.count_nonzero(mudancaSinal) / len(data)
    
    return zcr

def calcularMeanCrossingRate(data):
    
    dataNormal = data - np.mean(data)
    
    sinal = np.sign(dataNormal)
    
    mudancaSinal = np.abs(np.diff(sinal))
    
    mcr = np.count_nonzero(mudancaSinal) / len(data)
    
    return mcr

def calcularPairwiseCorrelation(data):
    '''Recebe todas as colunas dos sensores'''
    
    return np.corrcoef(data, rowvar = False)

def calcularSpectralEntropy(data):
    
    fourier = fft(data)
    
    espectro = np.abs(fourier) ** 2
    
    normalizado = espectro / np.sum(espectro)
    
    
    return entropy(normalizado, base = 2)







def criarJanelas(data):
    '''Cria as janelas e um array com a atividades de cada janela'''
    
    tempoJanela= 5 #segundos
    overlap = 0.5
    fs = 51.2
    
    
    tamanhoJanela = int(fs * tempoJanela)
    step =int(tamanhoJanela *   (1-overlap))
    
    janelas = []
    atividades = []
    
    for inicio in range(0,len(data) - tamanhoJanela, step):
        fim = inicio + tamanhoJanela
        janela = data[inicio:fim]
        if (janela[0,COL["atividade"]] == janela[-1,COL["atividade"]]):
            janelas.append(janela)
            atividades.append(int(janela[0,COL["atividade"]]))
    
    return np.array(janelas), np.array(atividades)

def criarJanelasComParticipantes(data):
    '''Cria as janelas e um array com a atividades de cada janela, mas recebe uma tabela em que a coluna 0 é o id de participante e tambem qretorna um array com esses ids por janela'''
    
    tempoJanela= 5 #segundos
    overlap = 0.5
    fs = 51.2
    
    
    tamanhoJanela = int(fs * tempoJanela)
    step =int(tamanhoJanela *   (1-overlap))
    
    janelas = []
    atividades = []
    participantes = []
    
    for inicio in range(0,len(data) - tamanhoJanela, step):
        fim = inicio + tamanhoJanela
        janela = data[inicio:fim]
        if (janela[0,COL_WITH_PARTICIPANT["atividade"]] == janela[-1,COL_WITH_PARTICIPANT["atividade"]]):
            participantes.append(janela[0, 0])
            janelas.append(janela[:, 1:])
            atividades.append(int(janela[0,COL_WITH_PARTICIPANT["atividade"]]))
    
    return np.array(janelas), np.array(atividades), np.array(participantes)


def extrairFeaturesJanela(window):
    
    features = []
    colunasSensores = [1,2,3,4,5,6,7,8,9]
    
    for col in colunasSensores:
        #calcular para cada coluna
        data = window[:, col]
        
        features.append(calcularMean(data))
        features.append(calcularMedian(data))
        features.append(calcularSTD(data))
        features.append(calcularVariance(data))
        features.append(calcularRMS(data))
        features.append(calcularAveragedDerivatives(data))
        features.append(calcularSkewness(data))
        features.append(calcularKurtosis(data))
        features.append(calcularInterquartileRange(data))
        features.append(calcularZeroCrossingRate(data))
        features.append(calcularMeanCrossingRate(data))
        features.append(calcularSpectralEntropy(data))
    
    
    #todas as colunas para correlacao
    
    data = window[:,colunasSensores]
    
    corr = calcularPairwiseCorrelation(data)
    
    #pegar apenas nos valores acima da diagonal, diaonal é 1 e abaixo é igual a em cima
    for i in range(len(colunasSensores)):
        for j in range(i+1, len(colunasSensores)):
            features.append(corr[i][j])
            
    return np.array(features)


    
