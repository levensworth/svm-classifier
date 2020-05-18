
class Classifier():
    def __init__(self, logger, **kwargs):
        self.logger = logger

    def train(self, data, labels):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def show_decision_boundary(self, start , end):
        raise  NotImplementedError

    def get_decision_equation(self):
        raise  NotImplementedError
