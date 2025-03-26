"""
Module to train a bioFAM model
"""

import scipy as s
import pandas as pd
import numpy as np
from loguru import logger

from mofapy2.core.BayesNet import BayesNet


def train_model(model):
    # Sanity check on the Bayesian Network
    assert isinstance(model, BayesNet), "'model' has to be a BayesNet class"

    ####################
    ## Start training ##
    ####################

    logger.info("Training the model with seed %d." % (model.options["seed"]))

    model.iterate()

    logger.info("Training finished.")
