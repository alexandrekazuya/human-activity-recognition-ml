import numpy as np

def inicializaCentros(data, nClusters):
    '''Obtem os centros iniciais de forma aleatória'''
    
    #Obter numero de linhas np array = numero de elementos
    nElementos = data.shape[0]
    
    #Escolkher nCLusters indices random para serem os centros iniciais, sem repetidos
    indices = np.random.choice(nElementos, nClusters, replace = False)
    
    return data[indices].copy()
    

def calcularDistancias(data, centros):
    '''Calcula a distancia de cada ponto com todos os centros'''
    
    data3D = data[:,np.newaxis, :]
    centros3D = centros[np.newaxis, :, :]
    
    distancias = np.sqrt(np.sum((data3D - centros3D) ** 2, axis = 2))
    
    return distancias
    
def atribuirCentros(distancias):
    '''Retorna um array com o indice do centro a que cada ponto pertence'''
    
    return np.argmin(distancias, axis=1)



def calculaCentros(data, arrayCentros, nClusters):
    '''Calcula a média dos pontos de cada cluster para e calcula o novo centro'''
    
    dimensoes = data.shape[1]
    
    novosCentros = np.zeros((nClusters, dimensoes))
    
    for i in range(nClusters):
        #Selecionar os pontos com o mesmo centro
        pontosCentro = data[arrayCentros == i]
        
        # Cluster vazio -> novo centro aleatório
        if len(pontosCentro) == 0:
            novosCentros[i] = data[np.random.choice(data.shape[0])]
        
        else:
            novosCentros[i] = pontosCentro.mean(axis = 0)

    return novosCentros


def kMeans(data, nClusters):
    
    maxIteracoes = 100
    i = 0
    
    centros = inicializaCentros(data, nClusters)
    
    while(i < maxIteracoes):
        i+=1
        
        centrosAntigos = centros.copy()
        
        distancias = calcularDistancias(data, centros)
        
        arrayCentros = atribuirCentros(distancias)
        
        centros = calculaCentros(data, arrayCentros, nClusters)
        
        diferencaCentros = np.sqrt(np.sum((centros - centrosAntigos) ** 2))
        
        
        #Tolerancia para evitar erros de floating point e mudanças negligenciáveis
        if diferencaCentros < 1e-4:
            break
    return arrayCentros, centros
    
    