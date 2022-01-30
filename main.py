from time import process_time_ns
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

layer0 = np.zeros(5)
layer1 = np.zeros(3)
layer2 = np.zeros(2)

Weights0 = np.random.random((Input_1[0].size,  layer0.size))
Weights1 = np.random.random((layer0.size,      layer1.size))
Weights2 = np.random.random((layer1.size,      layer2.size))


#groupW
Weights = []
Weights.append(Weights0)
Weights.append(Weights1)
Weights.append(Weights2)

#matix multiplikation
def dot(input, weight):
    dotproduct = np.dot(input, weight)

    return sigmoid(dotproduct)


#ReLU
def relu(x):
    for i in range(x.size):
        if(x[i] <= 0):
            x[i] = 0
    
    return x


#sigmoid
def sigmoid(array):
    tmp = np.zeros((array.size))
    for i in range(array.size):
        tmp[i] = 1.0 / (1.0 + np.exp(-array[i]))
    
    return tmp


#change
def changeWeights(layer_in, layer_out, Weights_tmp, layer_out_gewollt):
    for y in range(layer_out.size):
        for x in range(layer_in.size):
            diff = layer_out_gewollt[y] - layer_out[y] #änderungsrate
            Weights_tmp[x][y] = Weights_tmp[x][y] + diff * layer_in[x] #weight wird um die different mit berücksichtigung der größe des neurons verändert
    
    return Weights_tmp


#start
def Wbackprop(inputnumber, _Input_, W0, W1, W2):
    W2 = changeWeights(dot(dot(_Input_, W0), W1), dot(dot(dot(_Input_, W0), W1), W2), W2, Real[inputnumber]) #layer1,      layer2, Weights2, gewollt

    layer1_gewollt = dot(W2, Real[inputnumber]) #zurückrechnen = weights[x] "dot" layer_hinter
    W1 = changeWeights(dot(_Input_, W0),          dot(dot(_Input_, W0), W1),          W1, layer1_gewollt) #layer0,    layer1, Weights1, gewollt

    layer0_gewollt = dot(W1, layer1_gewollt) #zurückrechnen = weights[x] "dot" layer_hinter
    W0 = changeWeights(_Input_,                   dot(_Input_, W0),                   W0, layer0_gewollt) #Input_1,   layer0, Weights0, gewollt

    Weights_tmp = []
    Weights_tmp.append(W0)
    Weights_tmp.append(W1)
    Weights_tmp.append(W2)

    return Weights_tmp


#gothrough
def gothrough(W0, W1, W2, InputArray):
    output = dot(dot(dot(InputArray, W0), W1), W2)
    return softmax(output)


#softmax
def softmax(vector):
	e = np.exp(vector)
	return e / e.sum()


#debug
print("Gothrough 1: " + str(gothrough(Weights[0], Weights[1], Weights[2], Input_1[0])))
Weights = Wbackprop(0, Input_1[0], Weights[0], Weights[1], Weights[2])
print("Gothrough 2: " + str(gothrough(Weights[0], Weights[1], Weights[2], Input_1[0])))
Weights = Wbackprop(0, Input_1[0], Weights[0], Weights[1], Weights[2])
print("Gothrough 3: " + str(gothrough(Weights[0], Weights[1], Weights[2], Input_1[0])))