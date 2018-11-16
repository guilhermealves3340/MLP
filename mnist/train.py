import csv
from datetime import datetime
import numpy as np


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
V = np.zeros((neuronInput, neuronHidden), dtype=np.float64)
for i in range(neuronInput):
    for j in range(neuronHidden):
        V[i][j] = np.random.uniform(-0.5,0.5)

W = np.zeros((neuronHidden, neuronOutput), dtype=np.float64)
for i in range(neuronHidden):
    for j in range(neuronOutput):
        W[i][j] = np.random.uniform(-0.5,0.5)

# Pesos das bias
Bv = []

for i in range(neuronHidden):
    Bv.append(np.random.uniform(-0.5,0.5))

Bw = []
for i in range(neuronOutput):
    Bw.append(np.random.uniform(-0.5,0.5))

# Valores dos neuronios das camadas intermediarias
Zin = []
Z = []
for i in range(neuronHidden):
    Zin.append(0)
    Z.append(0)

# Valores dos neuronios das camadas de entrada
Yin = []
Y = []
for i in range(neuronOutput):
    Yin.append(0)
    Y.append(0)

deltaV = np.zeros((neuronInput, neuronHidden), dtype=np.float64)
deltaW = np.zeros((neuronHidden, neuronOutput), dtype=np.float64)

deltinhaW = []
deltaBw = []
for i in range(neuronOutput):
    deltinhaW.append(0)

    deltaBw.append(0)

deltinhaV = []
deltaBv = []
for i in range(neuronHidden):
    deltaBv.append(0)
    deltinhaV.append(0)

grafico = [[],[]]
c=0

for row in range(len(data)):
    number = int(data[row][0])
    ciclo = ciclo +1
    EqTotal = 0


    if data[row][0] == '0':
        c+=1

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
            for i in range(neuronOutput):
                deltinhaW[i] = (t[number][i]-Y[i])*(Y[i]*(1-Y[i]))

            for i in range(neuronHidden):
                for j in range(neuronOutput):
                    deltaW[i][j] = alfa * deltinhaW[j]*Z[i]

            for i in range(neuronOutput):
                deltaBw[i] = alfa * deltinhaW[i]

            # Da camada escondida para a camada de entrada
            for i in range(neuronHidden):
                for j in range(neuronOutput):
                    deltinhaV[i] = deltinhaW[j]*W[i][j]*(Z[i]*(1-Z[i]))

            for i in range(neuronInput):
                for k in range(neuronHidden):
                    deltaV[i][k] = alfa*deltinhaV[k]*Xpad

            for i in range(neuronHidden):
                deltaBv[i] = alfa*deltinhaV[i]

            # Atualização dos pesos
            # Da camada de saida

            for i in range(neuronHidden):
                for k in range(neuronOutput):
                    W[i][k] = W[i][k]+deltaW[i][k]

            for i in range(neuronOutput):
                Bw[i] = Bw[i] + deltaBw[i]

            # Camada escondida
            for i in range(neuronInput):
                for j in range(neuronHidden):
                    V[i][j] = V[i][j] + deltaV[i][j]

            for i in range(neuronHidden):
                Bv[i] = Bv[i] + deltaBv[i]

            for i in range(neuronOutput):
                EqTotal = EqTotal + 0.5*((t[number][i]-Y[i])**2)

            print("Img: {}\nPixel: {}\n".format(c,pixel+1))

    print("ERRO QUADRATICO TOTAL: {}\nCICLOS: {}\n".format(EqTotal,ciclo))
    


print("\n")

#row = 0
#media = []
#for i in range(10):
#    media.append(0)
#while test[row][0] != '0':
#    row += 1
#for pixel in range(n):
#
#    Xpad = test[row][pixel+1]
#
#    for i in range(neuronHidden):
#        ac = 0
#        for j in range(neuronInput):
#            ac = ac + V[j][i] * Xpad
#
#        Zin[i] = ac + Bv[i]
#        Z[i] = sigmoid(Zin[i])
#
#    for i in range(neuronOutput):
#        ac = 0
#        for j in range(neuronHidden):
#            ac = ac + Z[j] * W[j][i]
#
#        Yin[i] = ac + Bw[i]
#        Y[i] = sigmoid(Yin[i])
#
#    for i in range(10):
#        if pixel != 0:
#            media[i] = (media[i] + Y[i])*0.5
#        else:
#            media = Y
#
#for i in range(10):
#    print("{}   {}".format(t[0][i],media[i]))

# Salvando o modelo
f = open('model/model.csv','wt')
try:
    w = csv.writer(f)
    for i in range(neuronInput):
        w.writerow(V[i])

    for i in range(neuronHidden):
        w.writerow(W[i])

finally:
    f.close()