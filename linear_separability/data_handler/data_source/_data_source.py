
class DataSource():
    def __init__(self, logger, **kwargs):
        self.logger = logger
        self.lower_limit = kwargs['lower_limit']
        self.upper_limit = kwargs['upper_limit']
        self.size = kwargs['size']

    def get_data(self):
        raise NotImplementedError
