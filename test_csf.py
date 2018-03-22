import numpy as np
import matplotlib.pyplot as plt

import psychometric
from quest_plus import QuestPlus

def plot_qp(qp, true_params):
	posterior = qp.get_posterior()
#	extents = [
#		slopes[0] - 0.25, slopes[-1] + 0.25,
#		thresholds[0] - 0.025, thresholds[-1] + 0.025
#	]

	fig, ax = plt.subplots(ncols=5, figsize=(14, 2.5))

	# Plot posterior
	ax[0].imshow(posterior, origin='lower', aspect='auto')
	ax[0].set_ylabel('')
	ax[0].set_xlabel('')

	# Plot probability distribution for t
	ax[1].plot(t_space, posterior.sum(axis=0))
	ax[1].set_title('t probability')

	# Plot probability distribution for c0
	ax[2].plot(c0_space, posterior.sum(axis=1))
	ax[2].set_title('c0 probability')
	
	# Plot probability distribution for cf
#	ax[3].plot(cf_space, posterior.sum(axis=2))
#	ax[3].set_title('cf probability')

	# Plot estimation vs Ground Truth
	x = qp.stim_domain
	params = qp.get_fit_params()
	ax[4].plot(x, psychometric.csf_watson_and_ahumada(x, params), label='Best Quest+ fit')
	ax[4].plot(x, psychometric.csf_watson_and_ahumada(x, true_params), label='Ground truth')

# create a function that draws from the psychometric function:
def draw_from(stimulus_values, params, function=psychometric.csf_watson_and_ahumada):
	prob = function(stimulus_values, params)[0]
	return np.random.choice([0, 1], p=[1 - prob, prob])

contrasts = np.arange(0.05, 1.05, 0.05)
frequencies = np.arange(0, 40, 2)
stim_space = [contrasts, frequencies]

t_space = np.arange(-30, -50, -2)
c0_space = np.arange(-40, -60, -2)
cf_space = np.arange(.8, 1.6, .2)
param_space = [t_space, c0_space, cf_space]

qp = QuestPlus(stim_space, param_space, psychometric.csf_watson_and_ahumada)

true_params = [-20, -50, 1.2]
stim_params = qp.next_stim()

for group in range(3):
	for trial in range(100):
		response = draw_from(stim_params, true_params)
		qp.update(stim_params, response)
		stim_params = qp.next_stim()

	plot_qp(qp, true_params)
	print('found params:', qp.get_fit_params())

print('true params :', true_params)

plt.show()
