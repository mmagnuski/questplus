from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd

from psychometric import weibull


def reformat_params(params):
    '''Unroll multiple lists into array of their products.'''
    if isinstance(params, list):
        n_params = len(params)
        params = np.array(list(product(*params)))
    elif isinstance(params, np.ndarray):
        assert params.ndim == 1
        params = params[:, np.newaxis]
    return params


# TODO:
# - [ ] highlight lowest point in entropy in plot
class QuestPlus(object):
    def __init__(self, stim, params, function=weibull):
        self.function = function
        self.stim_domain = reformat_params(stim)
        self.param_domain = reformat_params(params)

        self._orig_params = deepcopy(params)
        self._orig_param_shape = (list(map(len, params)) if
                                  isinstance(params, list) else len(params))
        self._orig_stim_shape = (list(map(len, stim)) if
                                 isinstance(params, list) else len(stim))

        n_stim, n_param = self.stim_domain.shape[0], self.param_domain.shape[0]

        # setup likelihoods for all combinations
        # of stimulus and model parameter domains
        self.likelihoods = np.zeros((n_stim, n_param, 2))
        for p in range(n_param):
            self.likelihoods[:, p, 0] = self.function(self.stim_domain,
                                                      self.param_domain[p, :])

        # FIXME assumes (correct, incorrect) responses
        self.likelihoods[:, :, 1] = 1. - self.likelihoods[:, :, 0]

        # FIXME we also assume a flat prior (so we init posterior to flat too)
        self.posterior = np.ones(n_param)
        self.posterior /= self.posterior.sum()

        self.stim_history = list()
        self.resp_history = list()
        self.entropy = np.ones(n_stim)

    def update(self, contrast, ifcorrect):
        '''Update posterior probability with outcome of current trial.

        contrast - contrast value for the given trial
        ifcorrect   - whether response was correct or not
                      1 - correct, 0 - incorrect
        '''

        # turn ifcorrect to response index
        resp_idx = 1 - ifcorrect
        contrast_idx = self._find_contrast_index(contrast)

        # take likelihood of such resp for whole model parameter domain
        likelihood = self.likelihoods[contrast_idx, :, resp_idx]
        self.posterior *= likelihood
        self.posterior /= self.posterior.sum()

        # log history of contrasts and responses
        self.stim_history.append(contrast)
        self.resp_history.append(ifcorrect)

    def _find_contrast_index(self, contrast, approximate=False):
        return np.where((self.stim_domain == contrast).sum(axis=1))[0][0]

    def next_contrast(self):
        '''Get contrast value minimizing entropy of the posterior
        distribution.

        Expected entropy is updated in self.entropy.

        Returns
        -------
        contrast : contrast value for the next trial.'''
        full_posterior = self.likelihoods * self.posterior[
            np.newaxis, :, np.newaxis]

        norm = full_posterior.sum(axis=1, keepdims=True)
        full_posterior /= norm

        H = -np.nansum(full_posterior * np.log(full_posterior), axis=1)
        self.entropy = (norm[:, 0, :] * H).sum(axis=1)

        # choose contrast with minimal entropy
        return self.stim_domain[self.entropy.argmin(), :]

    def get_posterior(self):
    	return self.posterior.reshape(self._orig_param_shape)

    def get_fit_params(self, select='mode'):
        if select in ['max', 'mode']:
            # parameters corresponding to maximum peak in posterior probability
            return self.param_domain[self.posterior.argmax(), :]
        elif select == 'mean':
            # parameters weighted by their probability
            return (self.posterior[:, np.newaxis] *
                    self.param_domain).sum(axis=0)

    def fit(self, contrasts, responses):
        for contrast, response in zip(contrasts, responses):
            self.update(contrast, response)
