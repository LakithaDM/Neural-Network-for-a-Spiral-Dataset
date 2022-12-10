import numpy as np

#layers building and forward propogation class
class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    
    def backward_propagation(self,inputs ,output_error, learning_rate):
        relu_ouputs = np.maximum(0, output_error)
        drelu = relu_ouputs.copy()
        drelu[output_error <= 0] = 0
        input_error = np.dot(drelu, self.weights.T)
        weights_error = np.dot(inputs.T,drelu)
        dBias = np.sum(drelu, axis=0, keepdims=True)

        # update parameters
        self.weights += -learning_rate * weights_error
        self.biases += -learning_rate * dBias
    
#Rectified Linear Unit(ReLU) Activation function class
class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)
        return self.output
        
    
#SoftMax Activation function class
class Activation_SoftMax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = probabilities
        return self.output
    

        
class Loss_CategoricalCrossEntropy:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true,axis = 1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
class Loss:
    def calculate(self, output, y):
        sample_losses = Loss_CategoricalCrossEntropy.forward(self,output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
    
#Dataset building function
def spiral_data(N,K):
    D = 2
    
    X = np.zeros((N*K,D))
    y = np.zeros(N*K, dtype='uint8')

    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N)
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.4
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    
    return X, y

#Calling and building the dataset
X, y = spiral_data(100, 3)

#Creating the first neural layer
#dense1 = Layer_Dense(2,3)

#Creating an object for ReLU class
#Activation1 = Activation_ReLU()

#Creating the second neural layer
#dense2 = Layer_Dense(3,3)

#Creating an object for SoftMax class
#Activation2 = Activation_SoftMax()

#Getting the neural layer 1 outputs
#layer1_output = dense1.forward(X)

#Passing layer 1 outputs to layer 2 and getting the outputs
#layer2_output = dense2.forward(layer1_output)

#Printing the neural layer 1 outputs
#print("1st Neural Layer outputs:")
#print(layer1_output[0:5,:])

#print('\n')

#Printing the neural layer 2 outputs
#print("2nd Neural Layer outputs:")
#print(layer2_output[0:5,:])

#print('\n')

#Creating an instance for ReLU activation
#active_ReLU = Activation1.forward(layer2_output)

#active_SoftMax = Activation2.forward(layer2_output)

loss_function = Loss()
#printing ReLU activated outputs
#print("ReLU activated outputs:")
#print(active_ReLU[0:5,:])

#print('\n')

#printing SoftMax activated outputs
#print("SoftMax activated outputs:")
#print(active_SoftMax[0:5,:])

#loss = loss_function.calculate(active_ReLU, y)
#print('\n')
#print('loss:', loss)

dense1 = Layer_Dense(2,3)
Activation1 = Activation_ReLU()
#dense2 = Layer_Dense(3,3)
    
#feet forward 
for iteration in range(10):
    
    #Getting the neural layer 1 outputs
    layer1_output = dense1.forward(X)

    #Passing layer 1 outputs to layer 2 and getting the outputs
    #layer2_output = dense2.forward(layer1_output)
    
    active_ReLU = Activation1.forward(layer1_output)
    
    loss = loss_function.calculate(active_ReLU, y)
    predictions = np.argmax(active_ReLU, axis=1)
    accuracy = np.mean(predictions == y)
    print('loss:', loss)
    print('Acc', accuracy)
    
    dense1.backward_propagation(X, active_ReLU, 0.01)


#courier tranformation

