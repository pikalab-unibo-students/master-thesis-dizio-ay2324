import logging
import pandas as pd


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('fairlib')


DataFrame = pd.DataFrame


# https://realpython.com/python-magic-methods/#managing-attributes-through-descriptors
class DataFrameExtensionProperty:
    def __init__(self, default=None, can_read=False, can_write=False, can_delete=False):
        self.__can_read = can_read
        self.__can_write = can_write
        self.__can_delete = can_delete
        self.__default = default

    def __set_name__(self, owner, name):
        assert owner is DataFrame
        self.__name = name
        logger.debug(f"Extend {DataFrame.__name__} with property {name}")

    @property
    def key(self):
        return 'fairlib-' + self.__name

    def __get__(self, instance, owner):
        if self.__can_read:
            if self.key not in instance.attrs:
                instance.attrs[self.key] = self.default(instance, self.__default)
            value = instance.attrs[self.key]
            logger.debug(f"Read property {DataFrame.__name__}#{id(instance)}.{self.__name}: {value}")
            return value
        raise AttributeError(f"Can't read property {DataFrame.__name__}.{self.__name}")
    
    def default(self, instance, default):
        return default

    def __set__(self, instance, value):
        if self.__can_write:
            instance.attrs[self.key] = self.validate(instance, value)
            logger.debug(f"Write property {DataFrame.__name__}#{id(instance)}.{self.__name}: {value}")
        else:
            raise AttributeError(f"Can't write property {DataFrame.__name__}.{self.__name}")
        
    def validate(self, instance, value):
        return value

    def __delete__(self, instance):
        if self.__can_delete:
            if self.key in instance.attrs:
                del instance.attrs[self.key]
                logger.debug(f"Delete property {DataFrame.__name__}#{id(instance)}.{self.__name}")
            else:
                raise AttributeError(f"Can't delete property {DataFrame.__name__}.{self.__name}")
        else:
            raise AttributeError(f"Can't delete property {DataFrame.__name__}.{self.__name}")

    def apply(self, name):
        setattr(DataFrame, name, self)
        self.__set_name__(DataFrame, name)


class ColumnsContainerProperty(DataFrameExtensionProperty):
    def __init__(self):
        super().__init__(can_read=True, can_write=True, default=set())

    def validate(self, instance, value):
        if isinstance(value, str):
            value = {value}
        value = set(value)
        for target in value:
            if target not in instance.columns:
                raise ValueError(f"Column {target} not found")
        return value


ColumnsContainerProperty().apply('targets')
ColumnsContainerProperty().apply('sensitive')


# let this be the last line of this file
logger.info("fairlib loaded")
