import numpy as np
import matplotlib.pyplot as plt

import psychometric
import quest_plus

def plot_csf(stim_space, true_params):
	stims = quest_plus.reformat_params(stim_space)

	probs = psychometric.csf_watson_and_ahumada(stims, true_params)
	probs = probs.reshape(len(frequencies), len(contrasts))
	extents = [
		min(frequencies), max(frequencies),
		min(contrasts), max(contrasts),
	]

	plt.imshow(probs.T, origin='lower', extent=extents, aspect='auto')

frequencies = np.arange(0, 40, 1)
contrasts = np.arange(-50, 0, 1)
stim_space = [frequencies, contrasts]

true_params = [-40, -50, 1.2]

plot_csf(stim_space, true_params)
plt.show()
