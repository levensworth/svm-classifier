import importlib
import inspect
from linear_separability.data_handler.data_source._data_source import DataSource
import linear_separability.data_handler.settings as settings


class Handler(object):
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
                for class_implementation in module_classes:
                    if not self._is_injector(class_implementation[1]):
                        module_classes.remove(class_implementation)
                instance = module_classes[0][1]
                instance = instance(self.logger, **source['config'])
                self.sources.append(instance)

            except  ImportError:
                self.logger.warning('''Error getting
                {} module. Please, remember to name the file as the Class'''.format(path))

    def _is_injector(self, object):
        check_subclass = issubclass(object, DataSource)
        check_class = not (object == DataSource)
        return check_subclass and check_class

    def list_injectors(self):
        return self.sources.copy()

    def get_set(self, injector):
        return injector.get_data()
