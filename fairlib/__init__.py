from .logging import logger
from .dataframe import DataFrame
from .metrics import *
from .inprocessing import *
from .preprocessing import *

# from .postprocessing import *
# TODO import all from preprocessing, inprocessing, and postprocessing


# let this be the last line of this file
logger.info("fairlib loaded")
