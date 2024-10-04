from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from typing import Mapping


@runtime_checkable
class Domain(Protocol):
    feature: str


@dataclass(frozen=True)
class Assignment(Domain):
    feature: str
    value: object

    def __str__(self) -> str:
        return f"{self.feature}={self.value}"
    
    def __repr__(self) -> str:
        return str(self)

@dataclass(frozen=True)
class Range(Domain):
    feature: str
    lower_bound: float
    upper_bound: float

    def __str__(self) -> str:
        return f"{self.feature} in [{self.lower_bound}, {self.upper_bound}]"
    
    def __repr__(self) -> str:
        return str(self)


def dict_intersection(dict1, dict2):
    return {k: v for k, v in dict1.items() if k in dict2 and v == dict2[k]}


def dict_to_domain(assignments: dict) -> tuple[Assignment]:
    return tuple(Assignment(k, v) for k, v in assignments.items())


class DomainDict(Mapping):
    def __init__(self, data):
        self.__dict = data

    def __getitem__(self, key):
        if isinstance(key, dict):
            return self[dict_to_domain(key)]
        elif isinstance(key, tuple):
            if key in self.__dict:
                return self.__dict[key]
            intersection = None
            for k in key:
                if intersection is None:
                    intersection = self[k]
                else:
                    intersection = dict_intersection(intersection, self[k])
            return intersection
        elif isinstance(key, str):
            return {k: v for k, v in self.__dict.items() for domain in k if domain.feature == key}
        elif isinstance(key, Domain):
            return {k: v for k, v in self.__dict.items() for domain in k if domain == key}
        else:
            raise KeyError(key)

    def __iter__(self):
        return iter(self.__dict)

    def __len__(self):
        return len(self.__dict)

    def __repr__(self):
        return str(self.__dict)

    def __str__(self):
        return str(self.__dict)