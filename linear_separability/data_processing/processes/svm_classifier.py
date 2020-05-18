from linear_separability.data_processing.processes._classifier import Classifier
from sklearn.svm import SVC

class SVMClassifier(Classifier):
    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)
        C = kwargs.get('c', 0.1)
        self.model = SVC(C=C, kernel='linear')

    def train(self, data, labels):
        self.model = self.model.fit(data, labels)

    def predict(self, data):
        return self.model.predict(data).reshape(len(data), 1)

    def show_decision_boundary(self, start , end):
        raise  NotImplementedError

    def get_decision_equation(self):
        raise  NotImplementedError
