import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from scripts.module4_feature_statistics import criarJanelas, extrairFeaturesJanela
from misc.ficheiros.initializar import importParaTabela
from misc.ficheiros.outliers import normalizeZscore

#from skfeature.function.similarity_based import fisher_score
from skrebate import ReliefF
from sklearn.feature_selection import f_classif




def normalizar_e_PCA_e_variancia(featuresJanelas):

    # Manual z-score normalization
    FJnormalizado = normalizeZscore(featuresJanelas)
    # Print mean and std to verify z-score normalization
    # print("Mean of normalized features (should be ~0):", np.round(FJnormalizado.mean(axis=0), 4))
    # print("Std of normalized features (should be ~1):", np.round(FJnormalizado.std(axis=0), 4))

    #aplicar pca
    pca = PCA()
    pca.fit(FJnormalizado)

    varianciaExplicada = pca.explained_variance_ratio_
    varianciaCumulativa = np.cumsum(varianciaExplicada)

    return varianciaCumulativa

def plot4_4(varianciaCumulativa):   
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(varianciaCumulativa)+1), varianciaCumulativa, marker='o')
    plt.title("Variância acumulada das componentes principais")
    plt.xlabel("Número de componentes")
    plt.ylabel("Variância explicada acumulada")
    plt.grid(True)
    plt.show()

'''
if __name__ == '__main__':    
    tabela = importParaTabela(15,5)
    janelas, atividades = criarJanelas(tabela)
    
    featuresJanelas = []
    for janela in janelas:
        featuresJanelas.append(extrairFeaturesJanela(janela))
    
    featuresJanelas = np.array(featuresJanelas)
    varianciaCumulativa = normalizar_e_PCA_e_variancia(featuresJanelas)

    plot4_4(varianciaCumulativa)

    #para responder à questao "Quantas dimensões deverá usar para explicar 75% do feature set?""
    manterAte75 = np.argmax(varianciaCumulativa >= 0.75) + 1
    print(f"Variância de 75% atingida em {manterAte75} componentes.")
'''
#Deve-se usar manterAte75 dimensões para explicar 75% da variancia



#-----4.4.1----- acho q é para o relatorio?

#FJnormalizado = scaler.fit_transform(featuresJanelas)
#pca.fit(FJnormalizado)

#FJreduzido = pca.transform(FJnormalizado)[:, :manterAte75]

#Exemplo “para um instante à sua escolha”

instante = 0  #por exemplo, o primeiro
#print(f"Features originais ({featuresJanelas.shape[1]} dimensões):")
#print(featuresJanelas[instante, :5], "...")  # só mostrar as 5 primeiras para não ficar gigante

#print(f"\nFeatures comprimidas ({manterAte75} dimensões):")
#print(FJreduzido[instante])

'''A variância explicada indica a importância individual de cada componente principal,
enquanto a variância acumulada permite determinar quantas componentes são necessárias
 para preservar uma fração desejada da informação total (por exemplo, 75 %).
'''
#------4.4.2-------- para por no relatorio

'''O PCA permite comprimir os dados mantendo a maior parte da variancia,
 mas perde-se a relaçao direta com as variaveis originais'''

# --- 4.5 e 4.6 ---

def fisher_relieff(featuresJanelas, atividades):
    featuresNorm = normalizeZscore(featuresJanelas)
    ativ = np.array(atividades).astype(int)

    fscores, _ = f_classif(featuresNorm, ativ)
    top10fisher = np.argsort(fscores)[::-1][:10]

    relief = ReliefF(n_neighbors=10, n_features_to_select=10)
    relief.fit(featuresNorm, ativ)
    top10relief = relief.top_features_[:10]

    return top10fisher, top10relief

if __name__ == '__main__':
    
    tabela = importParaTabela(15,5)
    janelas, atividades = criarJanelas(tabela)
    
    featuresJanelas = []
    for janela in janelas:
        featuresJanelas.append(extrairFeaturesJanela(janela))
    
    featuresJanelas = np.array(featuresJanelas)
    varianciaCumulativa = normalizar_e_PCA_e_variancia(featuresJanelas)

    plot4_4(varianciaCumulativa)

    #para responder à questao "Quantas dimensões deverá usar para explicar 75% do feature set?""
    manterAte75 = np.argmax(varianciaCumulativa >= 0.75) + 1
    print(f"Variância de 75% atingida em {manterAte75} componentes.")
    pca = PCA()
    scaler = StandardScaler()
    '''
    tabela = importParaTabela(15,5)
    janelas, atividades = criarJanelas(tabela)
    
    featuresJanelas = []
    for janela in janelas:
        featuresJanelas.append(extrairFeaturesJanela(janela))
    
    featuresJanelas = np.array(featuresJanelas)
    '''
    
    featuresNorm = normalizeZscore(featuresJanelas)
    ativ = np.array(atividades).astype(int)
    
    #top10 fisher
    fscores, _ = f_classif(featuresNorm, ativ)
    top10fisher = np.argsort(fscores)[::-1][:10]
    
    #top10 Relieff
    relief = ReliefF(n_neighbors=10, n_features_to_select=10)
    relief.fit_transform(featuresJanelas, ativ)
    top10relief = relief.top_features_[:10] #top_features_ indices das features ordenadas por importância
    
    print("top 10 Fisher:", top10fisher)
    print("top 10 ReliefF: ", top10relief)
    
    # ---comparar a sobreposicao--- para comparar os resultados entre os 2
    intersecao = sorted(set(top10relief).intersection(set(top10fisher)))
    
    print("\nFeatures em comum: ", intersecao)

#---4.6.1
'''
fscores = fisher_score.fisher_score(featuresNorm, ativ)
top10fisher = np.argsort(fscores)[::-1][:10]
print("top 10 Fisher:", top10fisher)

Isto devolve o vetor de 10 features selecionadas para a primeira janela
'''

#------4.6.2

'''Vantagens:
- As features continuam a ter significado físico (ex.: aceleração média, desvio-padrão do giroscópio).
isto é - No PCA, as novas dimensões são combinações matemáticas (tipo “0.7*ax + 0.3*g”),
ou seja — já não sabemos que sensor representam.
Aqui, com o fisher e relief, escolhemos diretamente features reais:
por exemplo, “média do acelerómetro X do pé” ou “desvio padrão do giroscópio Z do braço”.

- Reduz ruído. Como escolhe o top10, vai eliminar as que acrescentam pouca informação

- Utiliza informação supervisionada (por atividades) ao contrário do PCA.

Limitacoes:
- Analisa cada feature isoladamente e pode ignorar combinações importantes.
como os metodos avaliam cada feature individualmente, pode-se perder uma combinação entre 
duas features fracas que juntas conseguem distinguir um movimento.

por exemplo:
ax sozinho não distingue andar de correr muito bem e o gx sozinho também não.
Mas (ax, gx) em conjunto distingue perfeitamente porque o movimento é um ciclo.
'''
