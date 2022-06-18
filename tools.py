import numba as nb
import numpy as np

@nb.njit
def get_idx(l_max, l, m):
    # From the Healpy library. But we copy it here so that Numba can use it
    return m * (2 * l_max + 1 - m) // 2 + l

@nb.njit
def get_scaled_flm(l_min, l_max, proposed_c_l, old_c_l, alm_size, old_f_lm):
  scaled_f_lm = np.zeros(alm_size, dtype=nb.complex64)
  for l in range(l_min, l_max+1):
      pre_factor = np.sqrt(proposed_c_l[l] / old_c_l[l])
      index = get_idx(l_max, l, np.arange(l+1))
      scaled_f_lm[index] = pre_factor * old_f_lm[index]
  return scaled_f_lm

@nb.njit(parallel=True)
def acceptance(old_s_lm, proposed_s_lm, old_f_lm, proposed_f_lm, old_c_l, proposed_c_l, l_min, l_max, B_l, N_l, d_lm):
    tot=0
    for l in nb.prange(l_min, l_max+1):
        index = get_idx(l_max, l, np.arange(l+1))
        a = 0
        a += sum(np.abs(d_lm[index] - B_l[l] * proposed_s_lm[index])**2 / N_l)
        a -= sum(np.abs(d_lm[index] - B_l[l] * old_s_lm[index])**2 / N_l)

        a += sum(np.abs(proposed_s_lm[index])**2 / proposed_c_l[l])
        a -= sum(np.abs(old_s_lm[index])**2 / old_c_l[l])

        a += sum(np.abs(proposed_f_lm[index]) ** 2 * B_l[l] ** 2 / N_l)
        a -= sum(np.abs(old_f_lm[index]) ** 2 * B_l[l]** 2 / N_l)
        tot += a
        
    A = np.real(np.exp(-1/2 * (tot)))
    eta = np.random.uniform(0, 1)
    #print('A:', A, 'Eta:', eta)
    
    return eta < A