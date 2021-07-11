import numpy as np

def sigmoid(x):
    return 1 /(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)


training_input=np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_output=np.array([[0,1,1,0]]).T

np.random.seed(1) 


synaptic_weights=2*np.random.random((3,1))-1
print("synaptice_weights")
print(synaptic_weights)



for iteration in range(6000):
    input_layer=training_input

    output=sigmoid(np.dot(input_layer,synaptic_weights))

    error=training_output-output

    adjustments=error*sigmoid_derivative(output)

    synaptic_weights +=np.dot(input_layer.T,adjustments)

print("synaptic weights after training ")
print(synaptic_weights)

print("output after training")
print(output)