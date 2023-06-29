import random
import numpy as np
from scipy.stats import norm


def generate_random_r0_around(t_max, around=2.9):
    r0_at_time = {}
    for t in range(t_max):
        r0_at_time[t] = random.uniform(around-0.6, around+0.6)

    return r0_at_time


def choose_from_distrib(values, stdev, mean):
    distrib = norm.pdf(values, mean, stdev)
    return random.choices(values, distrib)[0]


def inverse_probability_vector(vector):
    return np.array([1]*3) - vector
