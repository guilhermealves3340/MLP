"""
        Verificar dif sigmoid bipolar
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def sigmoid(x):
    y = float(x)
    return 2.0/(1.0 + np.exp(-y)) -1.0

pi = np.pi

def funcao(x):
    return np.sin(x) * np.sin(2*x)
    
start = str(datetime.today())[0:19]

n = 50
x1 = []
x2 = []
y1 = []
y2 = []
for i in range(n):
    y2.append(0)

for i in range(n):
    x1.append(round(np.random.uniform(0, 2*pi),1))
    y1.append(funcao(x1[i]))
    x2.append(round(np.random.uniform(0, 2*pi),1))

# Target
t = y1

# Neuronios da camada de entrada
neuronInput = 1

# Neuronios da camada escondida
neuronHidden = 10

# Neuronios da camada de saída
neuronOutput = 1

# Paramentros
alfa = 0.05             # Taxa de aprendizagem
cicloMax = 50000         
ciclo = 0
EqTotal = 10            # Erro quadratico total
EqAlvo = 0.5

# Inicialização dos pesos e deltas

# Pesos das conexões
V = np.zeros((neuronInput, neuronHidden), dtype=np.float64)
for i in range(neuronInput):
    for j in range(neuronHidden):
        V[i][j] = round(np.random.uniform(-0.5,0.5),1)

W = np.zeros((neuronHidden, neuronOutput), dtype=np.float64)
for i in range(neuronHidden):
    for j in range(neuronOutput):
        W[i][j] = round(np.random.uniform(-0.5,0.5),1)

# Pesos das bias
Bv = []
for i in range(neuronHidden):
    Bv.append(round(np.random.uniform(-0.5,0.5),1))

Bw = []
for i in range(neuronOutput):
    Bw.append(round(np.random.uniform(-0.5,0.5),1))

# Valores dos neuronios das camadas intermediarias
Zin = []
for i in range(neuronHidden):
    Zin.append(0)

Z = []
for i in range(neuronHidden):
    Z.append(0)

Yin = []
for i in range(neuronOutput):
    Yin.append(0)

Y = []
for i in range(neuronOutput):
    Y.append(0)

Vanterior = V
BVanterior = Bv
Wanterior = W
BWanterior = Bw

delta_y = []
altera_y = []
for i in range(neuronOutput):
    delta_y.append(0)
    altera_y.append(0)

delta_z = []
altera_z = []
for i in range(neuronHidden):
    delta_z.append(0)
    altera_z.append(0)


matriz = [[],[]]

while EqAlvo < EqTotal:
    ciclo = ciclo +1
    EqTotal = 0

    # Para cada padrao de treinamento
    for i in range(n):
        Xpad = x1[i]    # Selecionando o padrao de entrada
        Ypad = t[i]     # Selecionando o padrao de saida

        # Calculo das saidas do neuronios da camada intermediario
        for j in range(neuronHidden):
            ac = 0          # ac: acumulador
            for k in range(neuronInput):
                ac = ac + V[k][j] * Xpad

            Zin[j] = ac + Bv[j]
            Z[j] = sigmoid(Zin[j])

        # Calculo das saidas do neuronios da camada de saida
        for j in range(neuronOutput):
            ac = 0          # ac: acumulador
            for k in range(neuronHidden):
                ac = ac + W[k][j] * Z[k]

            Yin[j] = ac + Bw[j]
            Y[j] = sigmoid(Yin[j])

        ## Analisar melhor o codigo apartir daqui
        # Atualizando os pesos da camada final
        for j in range(neuronOutput):
            delta_y[j] = (Ypad - Y[j]) * (0.5*(1 + Y[j])*(1-Y[j]))          # Dif_func(y_in)
            altera_y[j] = alfa * delta_y[j]

        # Atualizando os pesos da camada intermediaria
        for j in range(neuronHidden):
            for k in range(neuronOutput):
                delta_z[j] = (delta_y[k]*W[j][k]) * (0.5*(1+Z[j]) * (1-Z[j]))   # Dif_func(z_in)
                altera_z[j] = alfa * delta_z[j]

        for j in range(neuronHidden):
            for k in range(neuronOutput):
                W[j][k] = Wanterior[j][k] + altera_y[k] * Z[j]

        for j in range(neuronOutput):
            Bw[j] = BWanterior[j] + altera_y[j]

        for j in range(neuronInput):
            for k in range(neuronHidden):
                V[j][k] = Vanterior[j][k] + altera_z[k] * Xpad

        for j in range(neuronHidden):
            Bv = BVanterior + altera_z[j]

        Vanterior = V
        BVanterior = Bv
        Wanterior = W
        BWanterior = Bw

        # Calculo do erro quadratico total
        EqTotal = EqTotal + 0.5 * ((Ypad - Y[0])**2)
    
    print('ERRO QUADRATICO TOTAL: {} \nCICLO: {}\n'.format(EqTotal,ciclo))
    matriz[0].append(ciclo)
    matriz[1].append(EqTotal)

    if ciclo%50000 == 0:

        Zin2 = Zin
        Z2 = Z
        Yin2 = Yin
        Y2 = Y

        end = str(datetime.today())[0:19]

        plt.plot(matriz[0], matriz[1],'bo')
        plt.axis([0,1.3 * max(matriz[0]), min(matriz[1]),1.3* max(matriz[1])])
        plt.show()

        # Teste da rede
        print("##########################")

        print('\nTEMPO DE TRAINAMENTO:\n\nINICIO: {}\nFIM:    {}'.format(start,end))

        for entrada in range(n):        # As n(50) entradas
            for i in range(neuronHidden):
                Zin2[i] = x2[entrada] * V[0][i] + Bv[i]
                Z2[i] = sigmoid(Zin2[i])

            ac = 0
            for i in range(neuronHidden):
                ac = ac + Z2[i] * W[i][0]
            Yin2 = ac + Bw[0]   
            Y2 = sigmoid(Yin2)
            
            y2[entrada] = Y2

        plt.plot(x1,y1,'go')
        plt.plot(x2,y2,'ro')
        plt.axis([0,1.3 * max(x1), 1.3 * min(y1),1.3* max(y1)])
        plt.show()
