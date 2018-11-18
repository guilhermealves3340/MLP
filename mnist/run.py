import csv
import numpy as np
from datetime import datetime


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-5*x))

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
neuronInput = 784

# Neuronios da camada escondida
neuronHidden = 12

# Neuronios da camada de saída
neuronOutput = 10

f = open('dataset/mnist_test.csv','rt')
try:
    reader = csv.reader(f)
    test = list(reader)
    reader = None
finally:
    f.close()
    f = None

# Excluindo linha de labels da tabela
test.pop(0)

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

f = open('model/model.csv','rt')
try:
    r = csv.reader(f)
    data = list(r)
    r = None
finally:
    f.close()
    f = None
K = 0
for i in range(neuronInput):
    for j in range(neuronHidden):
        V[i][j] = float(data[i][j])

K = neuronInput + K
for i in range(neuronHidden):
    for j in range(neuronOutput):
        W[i][j] = float(data[i+K][j])

for i in range(neuronHidden):
    Bv[i] = float(data[796][i])

for i in range(neuronOutput):
    Bw[i] = float(data[797][i])

c = 0
for i in range(1000):

    x = input("DIGITE QUALQUER COISA PRA TESTAR 10 DIGITOS ")
    
    for i in range(10):
        comp = int(test[c][0])

        Xpad = test[c][1:785]

        for i in range(neuronHidden):
            ac = 0
            for j in range(neuronInput):
                ac = ac + V[j][i] * int(Xpad[j])/255
            Zin[i] = ac + Bv[i]
            Z[i] = sigmoid(Zin[i])

        for i in range(neuronOutput):
            ac = 0
            for j in range(neuronHidden):
                ac = ac + Z[j] * W[j][i]
            Yin[i] = ac + Bw[i]
            Y[i] = sigmoid(Yin[i])

        dif = 0
        for i in range(10):
            print("{}   {}".format(t[comp][i],Y[i]))
            dif = dif + ((t[comp][i]  - Y[i])**2)**0.5
        dif = dif/10
        dif = (1-dif)*100
        print("Acc: {}%".format(dif))
        c += 1