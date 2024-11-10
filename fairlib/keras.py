from enum import Enum
import importlib
from typing import Optional, Callable
import os
from .logging import logger


class KerasBackend(Enum):
    TENSORFLOW = ('tensorflow', 'tf', 't')
    PYTORCH = ('torch', 'pytorch', 'pt', 'p')
    NUMPY = ('numpy', 'np', 'n')
    JAX = ('jax', 'j')

    @classmethod
    def parse(cls, string: str):
        string = string.strip().lower()
        for keras_backend in cls:
            if string in keras_backend.value:
                return keras_backend
        raise ValueError(f"Unknown backend: {string}")

    @property
    def module_name(self):
        return self.value[0]

    def __str__(self):
        return self.name

    @classmethod
    def from_env(cls):
        return cls.parse(os.getenv("KERAS_BACKEND", "tensorflow"))


def keras_backend_specific(
        keras_backend: Optional[KerasBackend] = None,
        tensorflow: Optional[Callable] = None,
        pytorch: Optional[Callable] = None,
        numpy: Optional[Callable] = None,
        jax: Optional[Callable] = None):

    keras_backend = keras_backend or KerasBackend.from_env()

    def do_nothing(_):
        return None

    if keras_backend == KerasBackend.TENSORFLOW:
        return (tensorflow or do_nothing)(keras_backend)
    elif keras_backend == KerasBackend.PYTORCH:
        return (pytorch or do_nothing)(keras_backend)
    elif keras_backend == KerasBackend.NUMPY:
        return (numpy or do_nothing)(keras_backend)
    elif keras_backend == KerasBackend.JAX:
        return (jax or do_nothing)(keras_backend)
    else:
        raise ValueError(f"Unknown backend: {keras_backend}")


def load_backend(keras_backend):
    try:
        return importlib.import_module(keras_backend.module_name)
    except ImportError:
        raise RuntimeError(f"Backend {keras_backend} not found: Keras will not work.")


keras_backend_specific(
    tensorflow=load_backend,
    pytorch=load_backend,
    numpy=load_backend,
    jax=load_backend
)


class TensorFlowMixin:
    def __init__(self):
        self.__tensorflow = None

    @property
    def tensorflow(self):
        if self.__tensorflow is None:
            self.__tensorflow = load_backend(KerasBackend.TENSORFLOW)
        return self.__tensorflow

    @property
    def tf(self):
        return self.tensorflow


class PyTorchMixin:
    def __init__(self):
        self.__pytorch = None

    @property
    def pytorch(self):
        if self.__pytorch is None:
            self.__pytorch = load_backend(KerasBackend.PYTORCH)
        return self.__pytorch

    @property
    def torch(self):
        return self.pytorch


class NumpyMixin:
    def __init__(self):
        self.__numpy = None

    @property
    def numpy(self):
        if self.__numpy is None:
            self.__numpy = load_backend(KerasBackend.NUMPY)
        return self.__numpy

    @property
    def np(self):
        return self.numpy


class JaxMixin:
    def __init__(self):
        self.__jax = None

    @property
    def jax(self):
        if self.__jax is None:
            self.__jax = load_backend(KerasBackend.JAX)
        return self.__jax


logger.info(f"Using Keras backend: {KerasBackend.from_env()}")

from keras import ops
from keras import backend
from keras import losses
from keras import metrics
from keras import initializers
from keras import activations
from keras import constraints
from keras import regularizers
from keras import layers
from keras import Model
from keras import Sequential
from keras import utils
