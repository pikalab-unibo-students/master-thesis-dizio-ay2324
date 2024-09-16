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
        logger.debug("Extend %s with property %s", DataFrame.__name__, name)

    @property
    def key(self):
        return 'fairlib-' + self.__name
    
    @property
    def name(self):
        return self.__name

    def __get__(self, instance, owner):
        if self.__can_read:
            if self.key not in instance.attrs:
                instance.attrs[self.key] = self.default(instance, self.__default)
            value = self.validate_get(instance, instance.attrs[self.key])
            value = self.defensive_copy(instance, value)
            logger.debug("Read property %s#%s.%s: %s", DataFrame.__name__, id(instance), self.name, value)
            return value
        raise AttributeError(name=self.name, instance=instance)
    
    def validate_get(self, instance, value):
        return value
    
    def defensive_copy(self, instance, value):
        return value
    
    def default(self, instance, default):
        return default

    def __set__(self, instance, value):
        if self.__can_write:
            value = self.validate_set(instance, value)
            value = self.defensive_copy(instance, value)
            instance.attrs[self.key] = value
            logger.debug("Write property %s#%s.%s: %r", DataFrame.__name__, id(instance), self.name, value)
        else:
            raise AttributeError(name=self.name, instance=instance)
        
    def validate_set(self, instance, value):
        return value

    def __delete__(self, instance):
        if self.__can_delete:
            if self.key in instance.attrs:
                del instance.attrs[self.key]
                logger.debug("Delete property %s#%s.%s", DataFrame.__name__, id(instance), self.name)
            else:
                raise AttributeError(f"No such a key: {DataFrame.__name__}.{self.name}")
        else:
            raise AttributeError(name=self.name, instance=instance)

    def apply(self, name):
        if hasattr(DataFrame, name):
            raise AttributeError(f"Attribute {name} already exists")
        setattr(DataFrame, name, self)
        self.__set_name__(DataFrame, name)


class DataFrameExtensionFunction(DataFrameExtensionProperty):
    def __init__(self, callable = None):
        super().__init__(can_read=True)
        self.__callable = callable
    
    def call(self, df, *args, **kwargs):
        logger.debug("Call function %s#%s.%s", DataFrame.__name__, id(df), self.name)
        if self.__callable is None:
            raise NotImplementedError
        return self.__callable(df, *args, **kwargs)
    
    def __get__(self, instance, _):
        return lambda *args, **kwargs: self.call(instance, *args, **kwargs)


class ColumnsContainerProperty(DataFrameExtensionProperty):
    def __init__(self):
        super().__init__(can_read=True, can_write=True, default=set())

    def validate_get(self, instance, value):
        return set(column for column in value if column in instance.columns)

    def validate_set(self, instance, value):
        if isinstance(value, str):
            value = {value}
        value = set(value)
        for column in value:
            if column not in instance.columns:
                raise ValueError(f"Column `{column}` not found")
        return value


ColumnsContainerProperty().apply('targets')
ColumnsContainerProperty().apply('sensitive')


class ColumnsDomainInspector:
    def __init__(self, df: DataFrame):
        assert isinstance(df, DataFrame)
        self.__df = df

    def __getitem__(self, name):
        if name in self.__df.columns:
            domain = self.__df[name].unique()
            logger.debug("Inspect domain of %s#%s[%s]: %r", DataFrame.__name__, id(self.__df), name, domain)
            return domain
        raise KeyError(f"Column {name} not found")

    def __len__(self):
        return len(self.__df.columns)
    
    def __iter__(self):
        columns = list(self.__df.columns)
        return iter(columns)
    
    def __contains__(self, name):
        return name in self.__df.columns
    
    def items(self):
        for column in self:
            yield column, self[column]
    
    def __repr__(self):
        return f"<{type(self).__name__}#{id(self)}>"
    
    def __str__(self):
        return "{" + "; ".join(f'{k}: [{", ".join(map(str, v))}]' for k, v in self.items()) + "}"
    

class ColumnDomainInspectorProperty(DataFrameExtensionProperty):
    def __init__(self):
        super().__init__(can_read=True)

    def __get__(self, instance, owner):
        return ColumnsDomainInspector(instance)
    

ColumnDomainInspectorProperty().apply('domains')


# let this be the last line of this file
logger.info("fairlib loaded")
