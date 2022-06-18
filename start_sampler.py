import matplotlib.pyplot as plt
import numpy as np
import time

from joint_sampler import joint_sampler
from gibbs_sampler import gibbs_sampler

from tqdm import tqdm   

samples = 300
l_min = 2
l_max = 700
q_sigma = 0.03
fwhm_deg = 0.1
noise_muK = 0.1
'''
a = gibbs_sampler(fwhm_multiplier = fwhm, noise_multiplier = noise, l_min = l_min, l_max = l_max, q_sigma = q_sigma)
a.plot_spectras()
start = time.time()
a.start_gibbs_sampler(samples=samples, burnin = 10, start_q = 1)
end = time.time()
tot = end-start
print('{} seconds doing {} samples, which gives {} samples/second'.format(tot, samples, samples/tot ))
'''
b = joint_sampler(fwhm_deg = fwhm_deg, noise_muK = noise_muK, l_min = l_min, l_max = l_max, q_sigma = q_sigma)
b.plot_spectras()

start = time.time()
b.start_joint_sampler(samples=samples, burnin = 20, start_q = 1)
end = time.time()
tot = end-start
print('{} seconds doing {} samples, which gives {} samples/second'.format(tot, samples, samples/tot ))


