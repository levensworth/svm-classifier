from linear_separability.data_processing.processes._classifier import Classifier
import numpy as np
import random


class Perceptron(Classifier):

    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)
        self.epochs = kwargs['epochs']
        self.max_error = kwargs['error']
        self.learning_rate = kwargs['learning_rate']

    def train(self, data, labels):
        # add treashold
        data = np.append(data, np.ones((len(data), 1)), axis=1)
        # basic init
        weights = np.random.random((data.shape[1], labels.shape[1])) * 2 - 1
        weights = weights.reshape((data.shape[1], 1))
        error = 1

        min_error = error
        min_weights = weights

        for i in range(self.epochs):
            if abs(error) < self.max_error:
                self.logger.info(
                    'Achieved a reasonable error {}'.format(error))
                self.weights = min_weights
                return True

            # select a random sample
            x_i = int(random.randint(0, len(data)-1))
            # compute activation
            exitacion = np.matmul(data[x_i, :], weights)
            activacion = np.sign(exitacion)
            # get update value
            d_w = self.learning_rate * \
                (labels[x_i, :] - activacion) * data[x_i, :].T
            weights = weights + d_w.reshape((data.shape[1], 1))

            error = self.calculate_error(data, labels, weights, self.epochs)
            # verify if is best than previous attempts
            if abs(error) < abs(min_error):
                min_error = error
                min_weights = weights
                self.logger.info('best error so far {}'.format(min_error))

        self.logger.info('Achieved error = {}'.format(min_error))
        print('Achieved error = {}'.format(min_error))
        self.weights = min_weights

    def calculate_error(self, data, labels, weights, epochs):
        y_hat = np.sign(np.matmul(data, weights))

        # return np.sum(np.sqrt((labels - y_hat)**2) * (1/ epochs))
        err = abs(y_hat - labels)
        return np.sum(err)

    def predict(self, data):
        # add treashold to input
        data = np.append(data, np.ones((len(data), 1)), axis=1)
        return np.sign(np.matmul(data, self.weights))
    def show_decision_boundary(self, start, end):
        '''
        Compute decision boundary and generates 100 points form start to end
        representing the decision plane.
        Params:
        - start: float
        - end : float

        Returns:
        - list of 2-d tuples
        '''
        # m = -(b / w2) / (b / w1)
        print(self.weights[2])
        print(self.weights[0])
        slope = - (self.weights[2] / self.weights[1]) / \
            (self.weights[2] / self.weights[0])
        origin_y = - self.weights[2] / self.weights[1]
        # y = (-(b / w2) / (b / w1)) x + (-b / w2)

        points = []
        for i in range(start, end):
            points.append((i, slope * i + origin_y))
        return points

    def get_decision_equation(self):
        '''
        Returns the slope and y origin for the given perceptron
        decision hyperplane
        '''
        slope = - (self.weights[2] / self.weights[1]) / \
            (self.weights[2] / self.weights[0])
        origin_y = - self.weights[2] / self.weights[1]
        return slope, origin_y
