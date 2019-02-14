"""Remove mean vector."""
import numpy as np

import logging

__all__ = ('remove_mean')

logger = logging.getLogger(__name__)


def remove_mean(model, output_filepath):
    """Remove mean vector from model."""
    print(model.shape)
    nomean_model = model - np.mean(model, axis=0)
    print(nomean_model.shape)
    logger.info('Saving output to {}'.format(output_filepath))
    np.save(output_filepath, nomean_model)
