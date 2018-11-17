import csv
import numpy as np
from datetime import datetime


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
V = np.zeros((neuronInput,neuronHidden),dtype=np.float64)
W = np.zeros((neuronHidden,neuronOutput),dtype=np.float64)

# Pesos das bias
Bv = np.zeros((neuronHidden),dtype=np.float64)
Bw = np.zeros((neuronOutput),dtype=np.float64)

# Valores dos neuronios das camadas intermediarias
Zin = np.zeros((neuronHidden),dtype=np.float64)
Z = np.zeros((neuronHidden),dtype=np.float64)
Yin = np.zeros((neuronOutput),dtype=np.float64)
Y = np.zeros((neuronOutput),dtype=np.float64)

print("\n")

f = open('model/model.csv','rt')
try:
    r = csv.reader(f)
    data = list(r)
    r = None
finally:
    f.close()

for i in range(neuronInput):
    for j in range(neuronHidden):
        V[i][j] = float(data[i][j])

for i in range(neuronHidden):
    for j in range(neuronOutput):
        W[i][j] = float(data[i+len(V)][j])

k = neuronInput + neuronHidden
for i in range(neuronHidden):
    Bv[i] = float(data[i+k])

k = k + neuronHidden
for i in range(neuronOutput):
    Bw[i] = float(data[i+k])

row = 0
media = np.zeros((neuronOutput),dtype=np.float64)
while test[row][0] != '0':
    row += 1
for pixel in range(n):


    Xpad = int(test[row][pixel+1])/255

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

    for i in range(10):
        if pixel != 0:
            media[i] = (media[i] + Y[i])*0.5
        else:
            media = Y

for i in range(10):
    print("{}   {}".format(t[0][i],media[i]))
