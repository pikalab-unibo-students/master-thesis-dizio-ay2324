from .logging import logger
from .dataframe import DataFrame
import fairlib.dataframe.extensions
from .metrics import *
from .inprocessing import *
from .preprocessing import *

# from .postprocessing import *


# let this be the last line of this file
logger.info("fairlib loaded")
