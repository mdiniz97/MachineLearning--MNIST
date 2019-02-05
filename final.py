import os
import time
import numpy as np
import mnist
from mnist import MNIST
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report

def tempo(tempo):
    horas = tempo//3600
    tempo -= horas *3600

    minutos = tempo//60
    tempo -= minutos*60

    print(horas, "h:", minutos, "m:", tempo, "s", sep='')

def imprime_matriz(matriz):
    linhas = len(matriz)
    colunas = len(matriz[0])

    for i in range(linhas):
        for j in range(colunas):
            if(j == colunas - 1):
                print("%d" %matriz[i][j], end = "\t")
            else:
                print("%d" %matriz[i][j], end = "\t")
        print()

def printResul(saida):
    matrix = np.zeros((10, 10), dtype=np.int)
    for i in range(len(saida)):
        matrix[saida[i]][testLabels[i]] += 1
    imprime_matriz(matrix)

tempoInicial0 = time.time()

# pegando o diretório que contem as imagens no formato MNIST
path = os.path.join("images-mnist")

# passando o diretório para a função MNIST para trabalhar com as imagens
data = MNIST(path)

print("Loading dataset")
trainImages, trainLabels = data.load_training() # carregando imagens de treinamento
testImages, testLabels = data.load_testing() #carregando imagens de teste
print("Dataset is load")


tempoInicial = time.time()

lda = LinearDiscriminantAnalysis()
# definindo a função LDA

print("Training LDA")
lda.fit(trainImages, trainLabels)
ldaResult = lda.predict(testImages)
#print_report(predictions, testLabels)
printResul(ldaResult)
tempoAux = time.time()
tempo(int(tempoAux-tempoInicial))
print(int(tempoAux-tempoInicial))

k = 1
resultKnn = list()

knn = KNeighborsClassifier(n_neighbors = k, n_jobs = -1)
#definindo a função Knn e seus parametros
# n_neighbors = numeros de vizinhos
# n_jobs = O número de trabalhos paralelos a serem executados para pesquisa de vizinhos. 
# Se n_jobs = -1, então, o número de trabalhos é definido para o número de núcleos da CPU
# fonte: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

tempoInicial = time.time()
print("Training for k = 1")
knn.fit(trainImages, trainLabels.tolist())
# realizando o treinamento usando o k = 1
resultKnn.append(knn.predict(testImages))
# pegando o resultado das predições e salvando a lista.
printResul(resultKnn[len(resultKnn)-1])
tempoAux = time.time()
tempo(int(tempoAux-tempoInicial))


tempoInicial = time.time()
#mudando o valor de k para um novo treinamento
knn.n_neighbors = 10
print("Training for k = 10")
knn.fit(trainImages, trainLabels.tolist())
# realizando o treinamento usando o k = 10
resultKnn.append(knn.predict(testImages))
# pegando o resultado das predições e salvando a lista.
printResul(resultKnn[len(resultKnn)-1])
tempoAux = time.time()
tempo(int(tempoAux-tempoInicial))


tempoInicial = time.time()
#mudando o valor de k para um novo treinamento
knn.n_neighbors = 100
print("Training for k = 100")
knn.fit(trainImages, trainLabels.tolist())
# realizando o treinamento usando o k = 100
resultKnn.append(knn.predict(testImages))
# pegando o resultado das predições e salvando a lista.
printResul(resultKnn[len(resultKnn)-1])
tempoAux = time.time()
tempo(int(tempoAux-tempoInicial))


tempoInicial = time.time()
#mudando o valor de k para um novo treinamento
knn.n_neighbors = 245
print("Training for k = 245")
knn.fit(trainImages, trainLabels.tolist())
# realizando o treinamento usando o k = 245
resultKnn.append(knn.predict(testImages))
# pegando o resultado das predições e salvando a lista.
printResul(resultKnn[len(resultKnn)-1])
tempoAux = time.time()
tempo(int(tempoAux-tempoInicial))

tempoInicial = time.time()
knn.n_neighbors = 490
print("Training for k = 490")
knn.fit(trainImages, trainLabels.tolist())
# realizando o treinamento usando o k = 490
resultKnn.append(knn.predict(testImages))
# pegando o resultado das predições e salvando a lista.
printResul(resultKnn[len(resultKnn)-1])
tempoAux = time.time()
tempo(int(tempoAux-tempoInicial))


tempoInicial = time.time()
knn.n_neighbors = 1000
print("Training for k = 1000")
knn.fit(trainImages, trainLabels.tolist())
# realizando o treinamento usando o k = 1000
resultKnn.append(knn.predict(testImages))
# pegando o resultado das predições e salvando a lista.
printResul(resultKnn[len(resultKnn)-1])
tempoAux = time.time()
tempo(int(tempoAux-tempoInicial))

fim = time.time()
print("Tempo total: ", tempo(int(fim - tempoInicial0)))









