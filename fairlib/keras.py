import enum
import os
import importlib
from .logging import logger


class KerasBackend(enum.Enum):
    TENSORFLOW = ('tensorflow', 'tf', 't')
    PYTORCH = ('torch', 'pytorch', 'pt', 'p')
    NUMPY = ('numpy', 'np', 'n')
    JAX = ('jax', 'j')

    @classmethod
    def parse(cls, string: str):
        string = string.strip().lower()
        for backend in cls:
            if string in backend.value:
                return backend
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
        backend: KerasBackend = None,
        tensorflow: callable = None,
        pytorch: callable = None,
        numpy: callable = None,
        jax: callable = None):
    backend = backend or KerasBackend.from_env()
    do_nothing = lambda _: None
    match backend:
        case KerasBackend.TENSORFLOW:
            return (tensorflow or do_nothing)(backend)
        case KerasBackend.PYTORCH:
            return (pytorch or do_nothing)(backend)
        case KerasBackend.NUMPY:
            return (numpy or do_nothing)(backend)
        case KerasBackend.JAX:
            return (jax or do_nothing)(backend)
        case _:
            raise ValueError(f"Unknown backend: {backend}")


def load_backend(backend):
    try:
        return importlib.import_module(backend.module_name)
    except ImportError:
        raise RuntimeError(f"Backend {backend} not found: Keras will not work.")


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


from keras import *
KerasBackend.current = KerasBackend.parse(backend.backend())
