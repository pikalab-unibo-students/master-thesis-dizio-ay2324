from ..dataframe import *
from pandas import Series
from typing import Union

ColumnsContainerProperty().apply("targets")
ColumnsContainerProperty().apply("sensitive")


class ColumnsDomainInspector:
    """
    A class to inspect the domain of columns in a DataFrame.

    Args:
        df (DataFrame): The DataFrame to inspect.
    """

    def __init__(self, df: DataFrame):
        assert isinstance(df, DataFrame)
        self.__df = df

    def __getitem__(self, name):
        if name in self.__df.columns:
            domain = self.__df[name].unique()
            logger.debug(
                "Inspect domain of %s#%s[%s]: %r",
                DataFrame.__name__,
                id(self.__df),
                name,
                domain,
            )
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
        return (
            "{"
            + "; ".join(f'{k}: [{", ".join(map(str, v))}]' for k, v in self.items())
            + "}"
        )


class ColumnDomainInspectorProperty(DataFrameExtensionProperty):
    def __init__(self):
        super().__init__(can_read=True)

    def __get__(self, instance, owner):
        return ColumnsDomainInspector(instance)


ColumnDomainInspectorProperty().apply("domains")


@dataframe_extension
def separate_columns(df: DataFrame, *columns, as_array: bool = False):
    input_columns = list(columns) if len(columns) > 0 else list(df.targets)
    X = df.drop(input_columns, axis=1)
    y = df[input_columns]
    if as_array:
        return X.values, y.values
    return X, y


@dataframe_extension
def is_discrete(df: Union[DataFrame, Series]) -> bool:
    if isinstance(df, Series):
        return df.dtype.kind in "Oib"
    elif isinstance(df, DataFrame):
        return all(is_discrete(df[column]) for column in df.columns)
    raise TypeError(f"Unsupported type: {type(df)}")


@dataframe_extension
def is_binary(df: Union[DataFrame, Series]) -> bool:
    if isinstance(df, Series):
        return df.dtype.kind == "b" or len(df.unique()) == 2
    elif isinstance(df, DataFrame):
        return all(is_binary(df[column]) for column in df.columns)
    raise TypeError(f"Unsupported type: {type(df)}")


def _optionally_force_int(s: Series, force_int: bool = True) -> Series:
    return s.astype(int) if force_int else s


def ohe(series: Series, force_int: bool = True) -> DataFrame:
    values = list(series.unique())
    values.sort()
    df = DataFrame()
    for value in values:
        df[f"{series.name}=={value}"] = _optionally_force_int(
            series == value, force_int
        )
    return df


class _preserving_extension_properties:
    def __init__(self, df: DataFrame):
        self.df = df
        self.__renames: list[tuple[str, str]] = []

    def __enter__(self):
        self.targets = set(self.df.targets)
        self.sensitive = set(self.df.sensitive)
        return self

    def renaming(self, old_column: str, new_column: str):
        self.__renames.append((old_column, new_column))

    def __exit__(self, exc_type, exc_value, traceback):
        for old_column, new_column in self.__renames:
            if old_column in self.targets:
                self.df.targets |= {new_column}
            if old_column in self.sensitive:
                self.df.sensitive |= {new_column}


@dataframe_extension
def one_hot_encode(
    df: DataFrame, *columns, in_place: bool = False, force_int: bool = True
) -> DataFrame:
    def _flatten(lst):
        for item in lst:
            if isinstance(item, list):
                yield from _flatten(item)
            else:
                yield item

    if not in_place:
        df = df.copy()
    columns_to_ohe = (
        set(
            col for col in df.columns if is_discrete(df[col]) and not is_binary(df[col])
        )
        if not columns
        else set(columns)
    )
    all_columns = list(df.columns)
    with _preserving_extension_properties(df) as props:
        for column_name in all_columns:
            column = df[column_name]
            df.drop(column_name, axis=1, inplace=True)
            if column_name in columns_to_ohe:
                ohe_columns = ohe(column, force_int=force_int)
                for ohe_col in ohe_columns.columns:
                    df[ohe_col] = ohe_columns[ohe_col]
                    props.renaming(column_name, ohe_col)
            else:
                df[column_name] = column
    return df


@dataframe_extension
def discretize(
    df: DataFrame,
    *discrete_columns,
    in_place: bool = False,
    force_int: bool = False,
    **discretization_functions,
) -> DataFrame:
    if not in_place:
        df = df.copy()
    if not discrete_columns and not discretization_functions:
        discretization_functions = {col: "ohe" for col in df.columns}
    for column in discrete_columns:
        if isinstance(column, tuple):
            name, column = column
        else:
            name = None
        if isinstance(column, Series):
            name = column.name if name is None else name
            if not is_discrete(column):
                raise ValueError(f"Column {name} is not discrete")
            df[column.name] = _optionally_force_int(column, force_int)
            with _preserving_extension_properties(df) as props:
                df.rename(columns={column.name: name}, inplace=True)
                props.renaming(column.name, name)
        else:
            raise TypeError(f"Unsupported type: {type(column)}")
    for column_name, discretization in discretization_functions.items():
        if isinstance(discretization, str) and discretization.lower() in {
            "ohe",
            "onehotencode",
        }:
            one_hot_encode(df, column_name, in_place=True, force_int=force_int)
            continue
        if isinstance(discretization, tuple):
            new_name, discretization = discretization
        else:
            new_name = None
        if callable(discretization):
            column = df[column_name]
            new_name = column_name if new_name is None else new_name
            with _preserving_extension_properties(df) as props:
                df.drop(column_name, axis=1, inplace=True)
                df[new_name] = _optionally_force_int(
                    column.apply(discretization), force_int
                )
                props.renaming(column_name, new_name)
        else:
            raise ValueError(
                f"Unsupported discretization function for column {column_name}"
            )
    return df
