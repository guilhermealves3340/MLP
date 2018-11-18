# Guilherme Alves da Silva - Eng. de Computação UFU
# Nov 2018

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-5*x))

# Inputs
x0 = [1,0,1,0]
x1 = [1,1,0,0]
x = [x0,x1]

# Target
t = [0,1,1,0]

# Neuronios da camada de entrada
neuronInput = 2

# Neuronios da camada escondida
neuronHidden = 4

# Neuronios da camada de saída
neuronOutput = 1

# Paramentros
alfa = 0.10
cicloMax = 5000
ciclo = 0
EqTotal = 10
EqAlvo = 0.005

# Inicialização dos pesos da camada escondida
V = np.zeros((neuronInput, neuronHidden), dtype=np.float64)
for i in range(neuronInput):
    for j in range(neuronHidden):
        V[i][j] = np.random.uniform(-0.5,0.5)

Bv = []
for i in range(neuronHidden):
    Bv.append(np.random.uniform(-0.5,0.5))

# Inicialização dos pesos da camada de saida
W = np.zeros((neuronHidden, neuronOutput), dtype=np.float64)
for i in range(neuronHidden):
    for j in range(neuronOutput):
        W[i][j] = np.random.uniform(-0.5,0.5)

Bw = []
for i in range(neuronOutput):
    Bw.append(np.random.uniform(-0.5,0.5))

Zin = []
Z = []
for i in range(neuronHidden):
    Zin.append(0)
    Z.append(0)

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

matriz = [[],[]]

while (EqTotal > EqAlvo):
    EqTotal = 0
    ciclo = ciclo +1

    # Fase forward

    # Calculo das saidas dos neuronios escondidos
    for i in range(len(x[0])):      # Percorrer cada elemento de cada entrada
        for j in range(neuronHidden):
            buffer = 0
            for k in range(neuronInput):
                buffer = buffer + x[k][i] * V[k][j]

            # Cálculo da saida dos neuronios da camada escondida
            Zin[j] = buffer + Bv[j]
            Z[j] = sigmoid(Zin[j])

        # Calculo da saida Y da rede
        buffer = 0
        for j in range(neuronHidden):
            buffer = buffer + Z[j] * W[j]
        Yin = buffer + Bw[0]
        Y = sigmoid(Yin)

        # Fase da retropropagação do erro
        # da saida para a camada escondida
        for j in range(neuronOutput):
            deltinhaW[j] = (t[i] - Y[j]) * (Y[j] * (1-Y[j]))

        for j in range(neuronHidden):
            for k in range(neuronOutput):
                deltaW[j][k] = alfa*deltinhaW[k]*Z[j]

        for j in range(neuronOutput):
            deltaBw[j] = alfa*deltinhaW[j]

        # Da camada escondida para a camada de entrada
        for j in range(neuronHidden):
            for k in range(neuronOutput):
                deltinhaV[j] = deltinhaW[k] * W[j][k] * (Z[j] * (1 - Z[j]))

        for j in range(neuronInput):
            for k in range(neuronHidden):
                deltaV[j][k] = alfa * deltinhaV[k] * x[j][i]
        
        for j in range(neuronHidden):
            deltaBv[j] = alfa * deltinhaV[j]

        # Atualização dos pesos
        # Da camada de saida

        for j in range(neuronHidden):
            for k in range(neuronOutput):
                W[j][k] = W[j][k] + deltaW[j][k]

        #W = W + deltaW
        for j in range(neuronHidden):
            for k in range(neuronOutput):
                W[j][k] = W[j][k] + deltaW[j][k]

        for j in range(neuronOutput):
            Bw[j] = Bw[j] +deltaBw[j]

        # Camada escondida
        for k in range(neuronInput):
            for j in range(neuronHidden):
                V[k][j] = V[k][j] + deltaV[k][j]
        
        for j in range(neuronHidden):
            Bv[j] = Bv[j] + deltaBv[j]
        
        # Calculo do erro total
        for j in range(neuronOutput):
            EqTotal = EqTotal + 0.5*((t[i] - Y[j]) ** 2)

    print("ERRO QUADRATICO TOTAL: {}\nCICLOS: {}\n".format(EqTotal,ciclo))
    matriz[0].append(ciclo)
    matriz[1].append(EqTotal)

plt.plot(matriz[0], matriz[1], 'ro')
plt.axis([0, 1.3* max(matriz[0]), min(matriz[1]), max(matriz[1])])
plt.show()

print("\n")

for entrada in range(len(x0)):
    for i in range(neuronHidden):
        buffer = 0
        for j in range(neuronInput):
            buffer = buffer + x[j][entrada] * V[j][i]
        Zin[i] = buffer + Bv[i]
        Z[i] = sigmoid(Zin[i])

    for i in range(neuronOutput):
        buffer = 0
        for j in range(neuronHidden):
            buffer = buffer + Z[j] * W[j][i]
        Yin[i] = buffer + Bw[i]
        Y[i] = sigmoid(Yin[i])
    print("{}   {}".format(t[entrada],Y[0]))



