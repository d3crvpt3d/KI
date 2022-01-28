from xml.dom import xmlbuilder
import numpy as np


#var
inputlenght = 10

cost = 0.0

Input_1 = np.array([
    [130, 100, 500, 100,  10], #1
    [123, 745, 412, 324, 564], #0
    [ 91,  94, 513, 412, 511], #0
    [123, 123, 600, 128, 453], #1
    [138, 105, 345,  95,  64], #1
    [123, 513,  46,  17,  46], #0
    [843, 513, 453,  83, 628], #0
    [  3,  20, 400,  60, 246], #1
    [138, 500, 700, 550,  68], #1
    [185,   1, 486,  20, 101], #1
    ])

Real = np.array([
    [1.0, 0.0], #1
    [0.0, 1.0], #0
    [0.0, 1.0], #0
    [1.0, 0.0], #1
    [1.0, 0.0], #1
    [0.0, 1.0], #0
    [0.0, 1.0], #0
    [1.0, 0.0], #1
    [1.0, 0.0], #1
    [1.0, 0.0]  #1
    ])

layer1 = np.zeros(5)
layer2 = np.zeros(3)
layerO = np.zeros(2)

Weights = np.zeros((np.random.random((Input_1[0].size, layer1.size)),
                    np.random.random((layer1.size, layer2.size)),
                    np.random.random((layer2.size,   layerO.size))))

#matix multiplikation
def dot(input, weight):
    return np.dot(input, weight)


#ReLU
def relu(x):
    for i in range(x.size):
        if(x[i] <= 0):
            x[i] = 0
    return x


#sigmoid
def sigmoid(x):
    tmp = [] * x
    for i in range(tmp.size):
        tmp[i] = 1.0 / (1.0 + np.exp(-x[i]))
    return tmp


#change
def changeW(layer_in, layer_out, Weights_tmp):
    for j in range(layer_out.size):
        diff = Real[j] - layer_out[j]#änderungsrate
        for i in range(layer_in.size):
            Weights_tmp[i][j] = Weights_tmp[i][j] * diff
    return Weights_tmp


#start
def Wbackprop(_Input_, W1, W2, WO):
    WO = changeW(dot(dot(_Input_, W1), W2), dot(dot(dot(_Input_, W1), W2), WO), WO) #layer2,    layerO, Weights2
    W2 = changeW(dot(_Input_, W1),          dot(dot(_Input_, W1), W2),          W2) #layer1,    layer2, Weights1
    W1 = changeW(_Input_,                   dot(_Input_, W1),                   W1) #Input_1,   layer1, Weights0


#gothrough
def gothrough(W1, W2, WO, InputArray):
    dot(dot(dot(InputArray, W1), W2), WO)


#debug
print("Gothrough 1: " + str(gothrough(Weights[0], Weights[1], Weights[2], Input_1[0])))
Weights = Wbackprop(Input_1[0], Weights[0], Weights[1], Weights[2])
print("Gothrough 1: " + str(gothrough(Weights[0], Weights[1], Weights[2], Input_1[0])))