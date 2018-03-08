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
