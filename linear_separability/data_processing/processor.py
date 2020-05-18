import importlib
import inspect
from linear_separability.data_processing.processes._classifier import Classifier
import linear_separability.data_processing.settings as settings


class Processor(object):
    def __init__(self, logger):
        self.logger = logger
        self.sources = []


    def load_sources(self, source_list):
        '''
        Params:
        - source_list: a list of dict source. Each item of the list contains two keys:
            - path: the relative path to this script, where the data source model exists
            - cnfig: a dict containing any additional information needed for the model
        '''
        for source in source_list:
            try:
                path = source['path']
                # https://stackoverflow.com/questions/49434118/python-how-to-create-a-class-object-using-importlib
                mod = importlib.import_module('{}.{}'.format(settings.APPS_BASE_PATH, path),)
                module_classes = inspect.getmembers(mod, inspect.isclass)

                for class_implementation in module_classes.copy():
                    if not self._is_injector(class_implementation[1]):
                        module_classes.remove(class_implementation)

                instance = module_classes[0][1]
                config = source['config']
                instance = instance(self.logger, **config)
                self.sources.append(instance)

            except  ImportError as e:
                self.logger.warning('''Error getting
                {} module. Please, remember to name the file as the Class'''.format(path))
            except Exception as e:
                self.logger.error('{}'.format(e))


    def _is_injector(self, object):
        check_subclass = issubclass(object, Classifier)
        check_class = not (object == Classifier)
        return check_subclass and check_class

    def list_injectors(self):
        return self.sources.copy()

    def train_injector(self, injector, data, labels):
        injector.train(data, labels)

    def predict(self, injector, data):
        return injector.predict(data)
