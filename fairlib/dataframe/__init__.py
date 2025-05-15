from pandas import DataFrame
from ..logging import logger
import numpy as np
from typing import NamedTuple


# https://realpython.com/python-magic-methods/#managing-attributes-through-descriptors
class DataFrameExtensionProperty:
    """
    A descriptor class to extend the pandas DataFrame with additional properties.

    Attributes:
        default: The default value for the property.
        can_read: Boolean indicating if the property can be read.
        can_write: Boolean indicating if the property can be written.
        can_delete: Boolean indicating if the property can be deleted.
    """

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
        return "fairlib-" + self.__name

    @property
    def name(self):
        return self.__name

    def __get__(self, instance, owner):
        if self.__can_read:
            if self.key not in instance.attrs:
                instance.attrs[self.key] = self.default(
                    sorted(instance), self.__default
                )
            value = self.validate_get(instance, instance.attrs[self.key])
            value = self.defensive_copy(instance, value)
            logger.debug(
                "Read property %s#%s.%s: %s",
                DataFrame.__name__,
                id(instance),
                self.name,
                value,
            )
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
            logger.debug(
                "Write property %s#%s.%s: %r",
                DataFrame.__name__,
                id(instance),
                self.name,
                value,
            )
        else:
            raise AttributeError(name=self.name, instance=instance)

    def validate_set(self, instance, value):
        return value

    def __delete__(self, instance):
        if self.__can_delete:
            if self.key in instance.attrs:
                del instance.attrs[self.key]
                logger.debug(
                    "Delete property %s#%s.%s",
                    DataFrame.__name__,
                    id(instance),
                    self.name,
                )
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
    """
    A descriptor class for extending the DataFrame with callable functions.

    Args:
        callable: The function to be called with the DataFrame.
    """

    def __init__(self, callable=None):
        super().__init__(can_read=True)
        self.__callable = callable

    def call(self, df, *args, **kwargs):
        logger.debug("Call function %s#%s.%s", DataFrame.__name__, id(df), self.name)
        if self.__callable is None:
            raise NotImplementedError
        return self.__callable(df, *args, **kwargs)

    def __get__(self, instance, _):
        return lambda *args, **kwargs: self.call(instance, *args, **kwargs)


def dataframe_extension(callable):
    DataFrameExtensionFunction(callable).apply(callable.__name__)
    return callable


class ColumnsContainerProperty(DataFrameExtensionProperty):
    """
    A property class to manage a collection of DataFrame columns.

    Inherits from DataFrameExtensionProperty.
    """

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


class UnpackedDataframe(NamedTuple):
    inputs: np.ndarray
    targets: np.ndarray
    inputs_names: list[str]
    targets_names: list[str]
    sensitive_names: list[str]
    sensitive_indexes: list[int]


def unpack_dataframe(
    df: DataFrame, return_as_dataframe: bool = False
) -> UnpackedDataframe:
    target_names = [name for name in df.columns if name in df.targets]
    sensitive_names = [name for name in df.columns if name in df.sensitive]
    input_names = [name for name in df.columns if name not in target_names]
    inputs = df[input_names] if return_as_dataframe else df[input_names].values
    targets = df[target_names].values
    sensitive_indexes = [input_names.index(name) for name in sensitive_names]
    return UnpackedDataframe(
        inputs, targets, input_names, target_names, sensitive_names, sensitive_indexes
    )


DataFrameExtensionFunction(callable=unpack_dataframe).apply("unpack")
