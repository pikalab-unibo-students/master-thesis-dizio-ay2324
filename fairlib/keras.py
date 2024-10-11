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
        backend: KerasBackend=None,
        tensorflow: callable=None,
        pytorch: callable=None,
        numpy: callable=None,
        jax: callable=None):
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

logger.info(f"Using Keras backend: {KerasBackend.from_env()}")


from keras import *
