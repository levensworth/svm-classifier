from matplotlib import pyplot as plt
from linear_separability.data_handler.data_handler import Handler
import linear_separability.data_handler.settings as settings_handler
import logging
import numpy as np
from linear_separability.data_processing.processor import Processor
import linear_separability.data_processing.settings as settings_processor

from mlxtend.plotting import plot_decision_regions


h = Handler(logging)
h.load_sources(settings_handler.SOURCES)


sources = h.list_injectors()
x = sources.pop()
data = h.get_set(x)
source = settings_processor.SOURCES[0]['config']
# c = SVMClassifier(logging, **source)

x = np.array(data[0])
y = np.array(data[1])
y1 = y
y = y.reshape((len(y), 1))

p = Processor(logging)
p.load_sources(settings_processor.SOURCES)

models = p.list_injectors()
model = models.pop()
p.train_injector(model, x, y)
y_hat = p.predict(model, x)
print(y - y_hat)

plot_decision_regions(x, y1, clf=model.model, legend=2)
plt.show()

def find_perceptron_plane(h):
    sources = h.list_injectors()
    x = sources.pop()
    data = h.get_set(x)

    x = np.array(data[0])
    y = np.array(data[1])
    y = y.reshape((len(y), 1))

    p = Processor(logging)
    p.load_sources(settings_processor.SOURCES)

    models = p.list_injectors()
    model = models.pop()
    p.train_injector(model, x, y)
    y_hat = p.predict(model, x)
    print(y - y_hat)

    x1 = [i[0] for i in x]
    x2 = [i[1] for i in x]
    plt.scatter(x1, x2, c=y)
    points = model.show_decision_boundary(0, 5)
    p_x = [i[0] for i in points]
    p_y = [i[1] for i in points]
    plt.plot(p_x, p_y)
    plt.show()
    return model.get_decision_equation()



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

