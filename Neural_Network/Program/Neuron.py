import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def deriv_mse_loss(y_true, y_pred):
    """
    :param y_true: Training data used to train the neural network
    :param y_pred: Predict result from the neural network
    :return: Derivative of Loss function with respect to prediction variable(y_pred)
    """
    return -2 * (y_true - y_pred)


def deriv_sigmoid(x):

    return sigmoid(x) * (1 - sigmoid(x))


class Neuron:
    """
    :param weight: A matrix describe the weight of this neuron
    """
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        self.total = 0

    def feedforward(self, X):
        total = np.dot(self.weight, X) + self.bias
        self.total = total

        return sigmoid(total)

    def form_d_H_d_w(self):
        """
        This function is a tool function of back propagation, which is used to calculate the partial derivative
        of active function with respect to the certain variable.
        To achieve this goal, following rule are made to simplify and proceed the program on back propagation:

            The path, which before the target variable of back propagation must strictly follow
            a straight line in neural network. In other word, divergence of back propagation's path
            must happen at its final step of reaching the target variable
            eg:
            ******************************************************************************

            Input_1 ------ Hidden layer_1 ------ Hidden layer_3

            Input_2 ------ Hidden layer_2 ------ Hidden layer_4

            ********************************************************************************
            Above path describe how back propagation push from last hidden layer to first hidden layer.
            In each layer, function calculate a derivative matrix d_H_d_w which contains the partial
            derivative wrt ith weight of ith neuron in jth layer.

        :return: partial derivative matrix wrt particular weight
        """

        d_H_d_w = np.zeros((len(self.weight), 1))
        for i in range(len(self.weight)):
            weight = self.weight[i]

            d_H_d_w[i] = weight * deriv_sigmoid(self.total)

        return d_H_d_w


class Layer:
    def __init__(self, pre_size, size):
        self.size = size
        self.pre_size = pre_size
        self.receptacle = []
        for i in range(self.size):
            temp_weight = np.random.random((pre_size, 1))
            temp_bias = np.random.normal()
            temp_neuron = Neuron(temp_weight, temp_bias)

            self.receptacle.append(temp_neuron)

    def form_result(self, X):
        """
        :param X: A matrix, which contains input data used to calculate
        :return: A matrix, formed by the result of each neuron in this layer
        """

        result = np.zeros((self.size, 1))

        for i in range(0, self.size):
            ele = self.receptacle[i].feedforward(
                X)  # extract the neuron from the receptacle and calculate result through active function
            result[i] = ele  # store the result

        return result



class Network:
    def __init__(self, layer_num, layer_size, input):
        """
        :param layer_num: number of layers of neuron network, the last layer is the output layer
        :param layer_size: This parameter is a tuple that contains the information of each hidden layers size,
        it must follow rules below:
                     1. The succession of last hidden layer should equal to the size of output layer
                     2. The precession of first hidden layer should equal to the size of input layer
        """
        self.layer_num = layer_num
        self.layer_size = layer_size
        self.sequential = []

        for i in range(layer_num):
            if i == 0:
                temp_layer = Layer(pre_size=input, size=layer_size[i])

            else:
                temp_layer = Layer(pre_size=layer_size[i - 1], size=layer_size[i])

            self.sequential.append(temp_layer)

    def feed_forward(self, X):
        """
        This function is used to proceed the upward propagation of BP neural network

        :param X: A matrix, contains the input data which to be calculated
        :return: The result after the feed forward process
        """

        temp_res = 0
        res = 0

        for i in range(self.layer_num):
            temp_layer = self.sequential[i]
            if i == 0:
                res = temp_layer.form_result(X)
            else:
                res = temp_layer.form_result(temp_res)
            temp_res = res

        return res

    def back_propagation(self, y_pred, y_true, lr):
        """
        Processing the back propagation of the neural network
        The goal of this function is to complete below formula:

            w_i = w_i - alpha * partial L / partial w_i

        :param y_pred: Prediction result from neural network
        :param y_true: Training data
        :return: Weight that adjusted by the back propagation process (matrix)
        """
        p_L_p_Y_pred = deriv_mse_loss(y_true, y_pred)
        former_derivative = []
        for i in range(self.layer_num - 1, -1, -1): # traverse layers
            layer = self.sequential[i]
            for j in range(layer.size): # traverse each neuron
                neuron = layer.receptacle[j]
                for k in range(len(neuron.weight)): # traverse weights in each neuron
                    if i == self.layer_num - 1:
                        d_H_d_w = neuron.form_d_H_d_w()
                        former_derivative.append(d_H_d_w)
                        change = p_L_p_Y_pred * d_H_d_w
                        neuron.weight -= lr * change # update weight

                    else:
                        backtrack = self.layer_num - i - 1
                        d_H_d_w = neuron.form_d_H_d_w()
                        recall = 1
                        for i in range(backtrack):
                            recall *= former_derivative[i][i]
                        change = p_L_p_Y_pred * recall * d_H_d_w
                        neuron.weight -= lr * change # update weight

    def train(self, epoch, X, y_true, lr):
        for i in range(epoch):
            y_pred = self.feed_forward(X)
            self.back_propagation(y_pred, y_true, lr)


if __name__ == '__main__':
    X = []

