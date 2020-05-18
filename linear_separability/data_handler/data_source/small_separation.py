
from linear_separability.data_handler.data_source._data_source import DataSource
import random
import numpy as np

class SmallSeparation(DataSource):
    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)
        pass

    def get_data(self):
        x_arr = []
        label_arr = []

        for x in np.arange(self.lower_limit, self.upper_limit, (self.upper_limit - self.lower_limit)/self.size):
            middle = (self.upper_limit - self.lower_limit)/2
            label_arr.append(1 if x > middle else -1)
            # x = random.uniform(self.lower_limit, self.upper_limit)
            y = self._implicit_function(x)
            x_arr.append((x, y))


        return x_arr, label_arr

    def _implicit_function(self, x):
        '''
        Implicit function used for distribution.
        At the moment we use the step function with a little bit of noise aggregation.
                    -> lower_limit + noise      if x < middle
        f(x) = {
                    -> upper_limit + noise      else
        '''

        middle = abs(self.upper_limit - self.lower_limit) / 2
        low = self.lower_limit  if x > middle else (middle + 0.1)
        high = self.upper_limit if x <= middle else (middle  - 0.1)
        possible = [i for i in np.arange(low, high, 0.1)]

        return random.choice(possible)
