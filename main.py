from xml.dom import xmlbuilder
import numpy as np


#10*5 array
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


inputlenght = 10


#10*1 array
Real = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]


#Layer mit size 5, 3, 1
layer1 = np.zeros(5)
layer2 = np.zeros(3)
layerO = np.zeros(1)


#Weights mit shape vom Layer
Weights1 = np.random.random((Input_1[0].size, layer1.size))
Weights2 = np.random.random((layer1.size ,  layer2.size))
WeightsO = np.random.random((layer2.size,   layerO.size))


#get weights debug
print("W1:")
print(Weights1)
print("W2:")
print(Weights2)
print("WO:")
print(WeightsO)


#return tmp for iterate
ret = np.array


#Iterate durch
def Iterate(x):

    for xx in range(x):
        #dot product
        global layer1
        layer1 = np.dot(Input_1[xx],    Weights1)

        global layer2
        layer2 = np.dot(layer1,         Weights2)

        global layerO
        layerO = np.dot(layer2,         WeightsO)

        ret[xx] = layer1, layer2, layerO

    return ret
    



#ReLU
def relu(x):
    return max(0.0, x)


# sigmoid activation function
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))


#start
Iterate(inputlenght)[0]


#def cost_tmp
tmp = 0.0

for k in range(layer2.size):
    for l in range(layerO.size):
        for f in range(inputlenght):
            
            tmp = tmp + ret[f][2][l] #iteration;layer;neuron

        WeightsO[k][l] = tmp[l]

print(WeightsO.shape)