from linear_separability.data_processing.processes._classifier import Classifier
import numpy as np
import random
from heapq import heapify, heappush, heappop
from itertools import combinations
import math


class Perceptron(Classifier):

    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)
        self.epochs = kwargs['epochs']
        self.max_error = kwargs['error']
        self.learning_rate = kwargs['learning_rate']
        self.enahnce_decision = kwargs.get('enahnce', True)

    def train(self, data, lables):
        # add treashold
        data = np.append(data, np.ones((len(data), 1)), axis=1)
        # basic init
        weights = np.random.random((data.shape[1], lables.shape[1])) * 2 - 1
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
                (lables[x_i, :] - activacion) * data[x_i, :].T
            weights = weights + d_w.reshape((data.shape[1], 1))

            error = self.calculate_error(data, lables, weights, self.epochs)
            # verify if is best than previous attempts
            if abs(error) < abs(min_error):
                min_error = error
                min_weights = weights
                self.logger.info('best error so far {}'.format(min_error))

        self.logger.info('Achieved error = {}'.format(min_error))
        print('Achieved error = {}'.format(min_error))
        self.weights = min_weights
        slope, y_origin = self.get_decision_equation()

        self.boundary  = (slope, y_origin)

        if self.enahnce_decision:
            boundary = self._enhance(self.weights, data, lables)
            # now we get the weight vector as a general form of the line's equation
            # remember a line can be informed in infinite number of vectores
            # Ax + By + C = 0 => -m X + 1y - b = 0
            self.weights = -1 * np.array([-boundary[0], 1, -boundary[1]])
            # why the -1?
            # remember that this perceptron contains the coordinate for an
            # orthogonal vector to the boundary plane. There are 2 possible directions for the
            # vector, as we know the distribution for the our data, i know the real vector direction
            self.boundary = boundary

    def _enhance(self, weights, data, lables):
        '''
        Find the best linear decision boundary given a existing decision boundary
        Params:
        - weights: a np array with +1 column for bias
        - data: training data
        - lable: results for trainin data

        Returns:
        - numpy array with same dimensions of weights
        '''
        slope, b = self.get_decision_equation()
        # the k should not be hardcoded but .... whatever
        near_neighbours = self.find_near_neighbours(slope, b, data, lables, 5)

        # now start rendering possible 3 point planes
        possible_boundaries = []
        heapify(possible_boundaries)
        lable_list = [i for i in zip(*np.unique(lables, axis=0).tolist())]
        possible_lables = list(lable_list[0])
        a_lable = possible_lables.pop()
        another_lable = possible_lables.pop()

        two_point_combinations = combinations(
            near_neighbours[another_lable], 2)
        for a_point in near_neighbours[a_lable]:
            for pair in two_point_combinations:
                points = [a_point, pair[0], pair[1]]
                possible_slope, possible_b, error = self._generate_boundary(points)
                heappush(possible_boundaries, (error, (possible_slope, possible_b)))

        two_point_combinations = combinations(near_neighbours[a_lable], 2)
        for a_point in near_neighbours[another_lable]:
            for pair in two_point_combinations:
                points = [a_point, pair[0], pair[1]]
                possible_slope, possible_b, error = self._generate_boundary(points)
                heappush(possible_boundaries, (error, (possible_slope, possible_b)))

        return heappop(possible_boundaries)[1]

    def _generate_boundary(self, points):
        '''
        generate the line of best fit for the given points
        it always assume the first point to be of one class and the others from the other class.
        Paras:
        - points: list of points (x, y)

        Returns:
        - slope
        - b
        - error: the abs sum of the distance from each point to the line
        '''
        another_class_point = points.pop(0)
        # claculate a margin
        slope, b = self._find_linear_interpolation(points.copy())

        #  now we calculate the orthogonal line
        slope_p = -1/ slope
        b_p = another_class_point[1] - slope_p * another_class_point[0]

        # intersect with the other point
        x = (b_p - b) / (slope - slope_p)
        y = slope * x + b

        another_point_projection = np.array([x, y])

        # now we calculate the middle distance to both margins
        middle_distance = np.linalg.norm(another_class_point[:-1] - another_point_projection) / 2

        b += middle_distance if another_class_point[1] > (slope * another_class_point[0] + b) else - middle_distance

        points.append(another_class_point)
        # error = self._calculate_boundary_error(points, slope, b)
        error = 1/middle_distance if middle_distance != 0 else math.inf
        return slope, b, error

    def _find_linear_interpolation(self, points):
        '''
        Given 2 points returns the slope and the y(0)
        '''
        point_a = points.pop(0)
        point_b = points.pop(0)
        #  slope = Ay / Ax
        slope = (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])
        # b = y - mx
        b = point_a[1] - slope * point_a[0]
        return slope, b

    def _calculate_boundary_error(self, points, slope, b):
        '''
        Calcualte the abs sum of each point disntance to the line
        '''
        total_error = 0
        for point in points:
            line_x = (point[1] - b) / slope
            horizontal_vertex = abs(point[0] - line_x)

            line_y = slope * point[0] + b
            vertical_vertex = abs(point[1] - line_y)

            # now apply pythagora's
            distance = math.sqrt(vertical_vertex ** 2 + horizontal_vertex ** 2)
            total_error += distance
        return total_error

    def find_near_neighbours(self, slope, b, data, lables, k):
        '''
        find the 2k nearest neighbours to the line formed as y = slope x + b
        k neighbours from each lable

        Params:
        - slope: the slope of the decision boundary
        - b : the value for y(0) of the decision boundary
        - data: np array containing x values in the first column and y values in the second
        - lables: the corresponding lables for each point in data
        k: the amount of points to return from each lable

        Returns:
        - a dicstionary containing a list for each possible lable (should be 2)
        '''
        lables_heap_dict = {}
        lable_list = [i for i in zip(*np.unique(lables, axis=0).tolist())]
        for lable in lable_list[0]:
            lables_heap_dict[lable] = []
            heapify(lables_heap_dict[lable])

        index = 0
        for point in data:
            # calculate distance between line and point
            # first we calculate the projection of the point in the line's plane
            line_x = (point[1] - b) / slope
            horizontal_vertex = abs(point[0] - line_x)

            line_y = slope * point[0] + b
            vertical_vertex = abs(point[1] - line_y)

            # now apply pythagora's
            distance = math.sqrt(vertical_vertex ** 2 + horizontal_vertex ** 2)
            # add p(distance, point) pair to the corresponding heap
            heappush(lables_heap_dict[lables[index][0]], (distance, point))
            index += 1

        result = {}
        for lable in lable_list[0]:
            points = [heappop(lables_heap_dict[lable])[1] for i in range(k)]
            result[lable] = points

        return result

    def calculate_error(self, data, lables, weights, epochs):
        y_hat = np.sign(np.matmul(data, weights))

        # return np.sum(np.sqrt((lables - y_hat)**2) * (1/ epochs))
        err = abs(y_hat - lables)
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
        # # m = -(b / w2) / (b / w1)
        # print(self.weights[2])
        # print(self.weights[0])
        # slope = - (self.weights[2] / self.weights[1]) / \
        #     (self.weights[2] / self.weights[0])
        # origin_y = - self.weights[2] / self.weights[1]
        # # y = (-(b / w2) / (b / w1)) x + (-b / w2)

        slope = self.boundary[0]
        origin_y = self.boundary[1]
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
