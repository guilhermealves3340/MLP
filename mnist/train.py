import csv
from datetime import datetime
import numpy as np


start = str(datetime.now())[0:19]

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# Target
t0 = [1,0,0,0,0,0,0,0,0,0]
t1 = [0,1,0,0,0,0,0,0,0,0]
t2 = [0,0,1,0,0,0,0,0,0,0]
t3 = [0,0,0,1,0,0,0,0,0,0]
t4 = [0,0,0,0,1,0,0,0,0,0]
t5 = [0,0,0,0,0,1,0,0,0,0]
t6 = [0,0,0,0,0,0,1,0,0,0]
t7 = [0,0,0,0,0,0,0,1,0,0]
t8 = [0,0,0,0,0,0,0,0,1,0]
t9 = [0,0,0,0,0,0,0,0,0,1]
t = [t0,t1,t2,t3,t4,t5,t6,t7,t8,t9]

# Neuronios da camada de entrada
neuronInput = 1

# Neuronios da camada escondida
neuronHidden = 15

# Neuronios da camada de saída
neuronOutput = 10

# Paramentros
alfa = 0.05                    # Taxa de aprendizagem         
ErroQuadratico = []            # Erro quadratico total
EqTotal = 10
ciclo = 0
for i in range(neuronOutput):
    ErroQuadratico.append(0)
Acc = 0                        # Acurácia

# Lendo o arquivo o dataset.csv de train
f = open('dataset/mnist_train.csv','rt')
try:
    reader = csv.reader(f)
    data = list(reader)
    reader = None
finally:
    f.close()

# Excluindo linha de labels da tabela
data.pop(0)

f = open('dataset/mnist_test.csv','rt')
try:
    reader = csv.reader(f)
    test = list(reader)
    reader = None
finally:
    f.close()

# Excluindo linha de labels da tabela
test.pop(0)

n = 784        # 28x28 PIXELS

# Inicialização dos pesos
# Pesos das conecções
V = (np.random.rand(neuronInput,neuronHidden)) -0.5
W = (np.random.rand(neuronHidden,neuronOutput)) -0.5

# Pesos das bias
Bv = (np.random.rand(neuronHidden)) -0.5
Bw = (np.random.rand(neuronOutput)) -0.5

# Valores dos neuronios das camadas intermediarias
Zin = np.zeros((neuronHidden),dtype=np.float64)
Z = np.zeros((neuronHidden),dtype=np.float64)
Yin = np.zeros((neuronOutput),dtype=np.float64)
Y = np.zeros((neuronOutput),dtype=np.float64)

deltaV = np.zeros((neuronInput, neuronHidden), dtype=np.float64)
deltaW = np.zeros((neuronHidden, neuronOutput), dtype=np.float64)

deltinhaW = np.zeros((neuronOutput), dtype=np.float64)
deltaBw = np.zeros((neuronOutput), dtype=np.float64)

deltinhaV = np.zeros((neuronHidden), dtype=np.float64)
deltaBv = np.zeros((neuronHidden), dtype=np.float64)

grafico = [[],[]]
c=0

for row in range(len(data)):
    number = int(data[row][0])
    ciclo = ciclo +1
    EqTotal = 0

    for pixel in range(n):

        Xpad = int(data[row][pixel+1])/255

        for i in range(neuronHidden):
            ac = 0
            for j in range(neuronInput):
                ac = ac + V[j][i] * Xpad

            Zin[i] = ac + Bv[i]
            Z[i] = sigmoid(Zin[i])

        for i in range(neuronOutput):
            ac = 0
            for j in range(neuronHidden):
                ac = ac + Z[j] * W[j][i]

            Yin[i] = ac + Bw[i]
            Y[i] = sigmoid(Yin[i])

        # Fase da retropropagação do erro
        # da saida para a camada escondida

        deltinhaW = (t[number] - Y) * (Y * (1 - Y))

        for i in range(neuronHidden):
            for j in range(neuronOutput):
                deltaW[i][j] = alfa * deltinhaW[j]*Z[i]

        deltaBw = alfa * deltinhaW

        # Da camada escondida para a camada de entrada
        for i in range(neuronHidden):
            for j in range(neuronOutput):
                deltinhaV[i] = deltinhaW[j]*W[i][j]*(Z[i]*(1-Z[i]))

        for i in range(neuronInput):
            for k in range(neuronHidden):
                deltaV[i][k] = alfa*deltinhaV[k]*Xpad

        deltaBv = alfa*deltinhaV

        # Atualização dos pesos
        # Da camada de saida

        for i in range(neuronHidden):
            for k in range(neuronOutput):
                W[i][k] = W[i][k]+deltaW[i][k]

        Bw = Bw + deltaBw

        # Camada escondida
        for i in range(neuronInput):
            for j in range(neuronHidden):
                V[i][j] = V[i][j] + deltaV[i][j]

        Bv = Bv + deltaBv

        EqTotal = EqTotal + 0.5*((t[number]-Y)**2)

        print("Img: {}\nPixel: {}\n".format(row,pixel+1))

print("\n")

# Salvando o modelo
f = open('model/model.csv','wt')
try:
    w = csv.writer(f)
    for i in range(neuronInput):
        w.writerow(V[i])

    for i in range(neuronHidden):
        w.writerow(W[i])

    w.writerow(Bv)
    w.writerow(Bw)

finally:
    f.close()

end = str(datetime.now())[0:19]

print("\nINICIO:     {}\nFIM:        {}".format(start,end))