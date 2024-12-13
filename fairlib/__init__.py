from .logging import logger
from .dataframe import DataFrame
import fairlib.dataframe.extensions
from .metrics import *

# TODO import all from preprocessing, inprocessing, and postprocessing


# let this be the last line of this file
logger.info("fairlib loaded")
