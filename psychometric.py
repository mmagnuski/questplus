import numpy as np

def weibull(x, params, corr_at_thresh=0.75, chance_level=0.5):
        # unpack params
        if len(params) == 3:
            t, b, lapse = params
        else:
            t, b = params
            lapse = 0.

        k = ( -np.log((1.0 - corr_at_thresh) / (1.0 - chance_level)) ) \
            ** (1.0 / b)
        expo = ((k * x) / t) ** b

        return (1 - lapse) - (1 - lapse - chance_level) * np.exp(-expo)


def weibull_db(contrast, params, guess=0.5):
    # unpack params
    if len(params) == 3:
        threshold, slope, lapse = params
    else:
        threshold, slope = params
        lapse = 0.

    return (1 - lapse) - (1 - lapse - guess) * np.exp(
        -10. ** (slope * (contrast - threshold) / 20.))

# From Watson, A. B., & Ahumada, A. J. (2016). The pyramid of visibility. Electronic Imaging, 2016 (16), 1–6, doi:10.2352/ISSN.2470-1173.2016.16.HVEI-102.
# As described in Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian adaptive psychometric method. Journal of vision, 17(3), 10-10.
def csf_watson_and_ahumada(stim_values, params, slope=3, guess=0.5, lapse=0.1):
    if len(params) == 4:
        t, c0, cf, cw = params
    else:
        t, c0, cf = params
        cw = 0

    contrast, frequency = stim_values.T
    contrast = np.atleast_1d(contrast)
    frequency = np.atleast_1d(frequency)

    min_thresh = np.array(len(contrast) * [t])
    threshold = c0 + cf * frequency + cw * contrast

    threshold = np.amax([min_thresh, threshold], 0)

    return (1 - lapse) - (1 - lapse - guess) * np.exp(
        -10. ** (slope * (contrast - threshold) / 20.))
