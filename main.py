from matplotlib import pyplot as plt
from linear_separability.data_handler.data_handler import Handler
import linear_separability.data_handler.settings as settings_handler
import logging
import numpy as np
from linear_separability.data_processing.processor import Processor
import linear_separability.data_processing.settings as settings_processor

from mlxtend.plotting import plot_decision_regions




def show_SVM(data, processor_conf):

    x = np.array(data[0])
    y = np.array(data[1])
    y1 = y
    y = y.reshape((len(y), 1))

    p = Processor(logging)
    p.load_sources(processor_conf)

    models = p.list_injectors()
    model = models.pop()
    p.train_injector(model, x, y)
    y_hat = p.predict(model, x)
    print(y - y_hat)

    plot_decision_regions(x, y1, clf=model.model, legend=2)
    plt.show()

def find_perceptron_plane(data, processor_conf):

    x = np.array(data[0])
    y = np.array(data[1])
    y = y.reshape((len(y), 1))

    p = Processor(logging)
    p.load_sources(processor_conf)

    models = p.list_injectors()
    model = models.pop()
    p.train_injector(model, x, y)
    y_hat = p.predict(model, x)
    y_hat = y_hat.reshape(y_hat.size, 1)
    print(y - y_hat)

    x1 = [i[0] for i in x]
    x2 = [i[1] for i in x]
    plt.figure(figsize=(5,5))
    plt.scatter(x1, x2, c=y)
    points = model.show_decision_boundary(0, 5)
    p_x = [i[0] for i in points]
    p_y = [i[1] for i in points]
    plt.plot(p_x, p_y)
    plt.show()
    return model.get_decision_equation()


# punto A
h = Handler(logging)
h.load_sources([ {'path': 'big_separation', 'config': {'lower_limit': 0.0, 'upper_limit': 5.0, 'size': 100}}])
sources = h.list_injectors()
x = sources.pop()
data_big = h.get_set(x)

h = Handler(logging)
h.load_sources([ {'path': 'small_separation', 'config': {'lower_limit': 0.0, 'upper_limit': 5.0, 'size': 100}}])
sources = h.list_injectors()
x = sources.pop()
data_small = h.get_set(x)

find_perceptron_plane(data_big, [{'path': 'perceptron', 'config': {'epochs': 1000, 'error': 0.0, 'learning_rate': 0.1 ,'enhance': False}}])

# punto B
# h = Handler(logging)
# h.load_sources([ {'path': 'big_separation', 'config': {'lower_limit': 0.0, 'upper_limit': 5.0, 'size': 100}}])
# {'path': 'small_separation', 'config': {'lower_limit': 0.0, 'upper_limit': 5.0, 'size': 100}}
find_perceptron_plane(data_big, [{'path': 'perceptron', 'config': {'epochs': 1000, 'error': 0.0, 'learning_rate': 0.1 ,'enhance': True}}])


# Punto C

# {'path': 'small_separation', 'config': {'lower_limit': 0.0, 'upper_limit': 5.0, 'size': 100}}
find_perceptron_plane(data_small, [{'path': 'perceptron', 'config': {'epochs': 1000, 'error': 0.0, 'learning_rate': 0.1 ,'enhance': True}}])

# Punto D a
# h = Handler(logging)
# h.load_sources([ {'path': 'big_separation', 'config': {'lower_limit': 0.0, 'upper_limit': 5.0, 'size': 100}}])
# {'path': 'small_separation', 'config': {'lower_limit': 0.0, 'upper_limit': 5.0, 'size': 100}}
# possible kernels: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
show_SVM(data_big, [{'path': 'svm_classifier', 'config': {'c': 1, 'kernel': 'linear'}}])

# D b
# h = Handler(logging)
# h.load_sources([ {'path': 'small_separation', 'config': {'lower_limit': 0.0, 'upper_limit': 5.0, 'size': 100}}])
# {'path': 'small_separation', 'config': {'lower_limit': 0.0, 'upper_limit': 5.0, 'size': 100}}
# possible kernels: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
show_SVM(data_small, [{'path': 'svm_classifier', 'config': {'c': 1, 'kernel': 'linear'}}])


'''

def test_perceptron():

    p = Perceptron(logging, **{'epochs': 100, 'error': 0.1, 'learning_rate': 0.1})
    x = np.array([1, 1, 1, 0, 0, 1, 0, 0])
    x = x.reshape((4, 2))
    y = np.array([1,1,1, 0]).T
    y = y.reshape((4, 1))
    p.train(x, y)

    y_hat =p.predict(x)

    print(y_hat)

'''

