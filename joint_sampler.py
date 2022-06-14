import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sqrt
import camb
#import healpy as hp
import time

from tqdm import tqdm

class joint_sampler:
    def __init__(self, fwhm_multiplier, noise_multiplier, l_max=350, l_min=20, q_sigma = 0.05):
        self.q_sigma = q_sigma
        self.Nside = 128
        self.l_max = l_max
        self.l_min = l_min
        self.Npix = 12 * self.Nside * 3
        self.fwhm = fwhm_multiplier * 3 * 60/self.Nside * pi / 180
        print('FWHM:', self.fwhm, 'radians')
        self.alm_size = int((self.l_max+1)*(self.l_max + 2)/2)
        self.b_l = hp.sphtfunc.gauss_beam(self.fwhm, lmax=self.l_max)

        cp = camb.set_params(tau=0.0544, ns=0.9649, H0=67.36, ombh2=0.02237,omch2=0.12, As=2.1e-09, lmax=1000)
        camb_results = camb.get_results(cp)
        self.c_l_lcdm = np.array(camb_results.get_cmb_power_spectra(lmax=self.l_max, raw_cl=True)['total'][:, 0])

        self.N_l = noise_multiplier * np.array(camb_results.get_cmb_power_spectra(lmax=250, raw_cl=True)['total'][250, 0])# * self.b_l[self.l_max]**2
        self.init_CMB_noise()

    def init_CMB_noise(self):
        a_lm = np.zeros(self.alm_size, dtype=complex)
        d_lm = np.zeros(self.alm_size, dtype=complex)
        N_l = self.N_l
        c_l_lcdm = self.c_l_lcdm
        l_max = self.l_max

        for l in range(2, l_max+1):
            for m in range(l+1):
                index = hp.sphtfunc.Alm.getidx(l_max, l, m)
                a_lm[index] = np.random.normal(0, 1) * sqrt(c_l_lcdm[l] * self.b_l[l]**2)
                d_lm[index] = a_lm[index] + np.random.normal(0, 1) * sqrt(N_l)

        self.d_lm = d_lm
        self.a_lm = a_lm
    
    def plot_spectras(self):
        l = np.arange(self.l_min, self.l_max+1)

        c_l = get_D_l(self.c_l_lcdm)
        d_l = get_D_l(hp.sphtfunc.alm2cl(self.d_lm))
        a_l = get_D_l(hp.sphtfunc.alm2cl(self.a_lm))
        D_N_l = get_D_l(np.repeat(self.N_l, self.l_max+1))

        plt.figure()
        plt.plot(l, c_l[self.l_min:], label='LCDM', linewidth=4)
        plt.plot(l, a_l[self.l_min:], label='Beam Smoothed Cosmological Signal')
        plt.plot(l, d_l[self.l_min:], label='Measured Data')
        plt.plot(l, D_N_l[self.l_min:], label='White Noise')
        
        plt.ylabel(r"$C_{\ell}\ell (\ell + 1)$")
        plt.xlabel(r"$\ell$")
        plt.legend()
        plt.savefig('spectra.pdf')
        
    def get_joint_slm_sample(self, c_l):
        N_l = self.N_l
        d_lm = self.d_lm
        l_min = self.l_min
        l_max = self.l_max
        B_l = self.b_l

        s_lm = np.zeros(self.alm_size, dtype=complex)
        f_lm = np.zeros(self.alm_size, dtype=complex)
        for l in range(l_min, l_max+1):
            first_prefactor = N_l * sqrt(c_l[l]) / (N_l + B_l[l] ** 2 * c_l[l])

            s_prefac = sqrt(c_l[l]) * B_l[l]/N_l
            omega_1_prefac = B_l[l] * sqrt(c_l[l]/N_l)

            index = hp.sphtfunc.Alm.getidx(l_max, l, np.arange(l+1))
            s_lm[index] = first_prefactor * s_prefac * d_lm[index]
            f_lm[index] = first_prefactor * (np.random.normal(0, 1) + np.random.normal(0, 1) * omega_1_prefac)

        return s_lm, f_lm

    def get_c_l_lcdm_from_param(self, ns):
        # ns = 0.9649
        cp = camb.set_params(tau=0.0544, ns=ns, H0=67.36, ombh2=0.02237,omch2=0.12, As=2.1e-09, lmax=self.l_max)
        camb_results = camb.get_results(cp)
        c_l_lcdm = np.array(camb_results.get_cmb_power_spectra(lmax=self.l_max, raw_cl=True)['total'][:, 0])

        return c_l_lcdm

    def proposal_w(self, old_q):
        new_q = np.random.normal(old_q, self.q_sigma)
        c_l = new_q * self.c_l_lcdm

        return new_q, c_l

    def acceptance(self, s_lm, f_lm, proposed_f_lm, c_l, proposed_c_l):
        l_max = self.l_max
        B_l = self.b_l
        N_l = self.N_l
        d_lm = self.d_lm

        ln_pi_ip1 = 0
        ln_pi_i = 0
        for l in range(self.l_min, l_max+1):
            index = hp.sphtfunc.Alm.getidx(l_max, l, np.arange(l+1))

            ln_pi_ip1 += sum(s_lm[index]**2 / proposed_c_l[l] +\
                proposed_f_lm[index] ** 2 * B_l[l] ** 2 / N_l)
            ln_pi_i += sum(s_lm[index]**2 / c_l[l] + f_lm[index] ** 2 * B_l[l]** 2 / N_l)


        A = np.exp(-1/2 * (ln_pi_ip1 - ln_pi_i))
        eta = np.random.uniform(0, 1)
        #print('A:', A, 'Eta:', eta)
        
        return eta < A

    def start_joint_sampler(self, samples=100, burnin=10, start_q=1):
        # Sample only ns for now
        l_min = self.l_min
        l_max = self.l_max
        cur_q = start_q
        cur_c_l = self.c_l_lcdm * cur_q

        list_of_q = np.zeros(samples+1)
        list_of_q[0] = cur_q
        
        accept_rate = 0
        tot = 0

        j = 1
        pbar = tqdm(total=samples)
        while j <= samples:
            s_lm, cur_f_lm = self.get_joint_slm_sample(cur_c_l)
            proposed_q, proposed_c_l = self.proposal_w(cur_q)

            # Scale f_lm: f_lm^(i+1) = sqrt(c^(i+1)_l / c^i_l) f_lm^i
            proposed_f_lm = np.zeros(self.alm_size, dtype=complex)
            for l in range(l_min, l_max+1):
                pre_factor = sqrt(proposed_c_l[l] / cur_c_l[l])
                index = hp.sphtfunc.Alm.getidx(l_max, l, np.arange(l+1))
                proposed_f_lm[index] = pre_factor * cur_f_lm[index]
            accepted = self.acceptance(s_lm, cur_f_lm, proposed_f_lm, cur_c_l, proposed_c_l)
            tot += 1
            if accepted:
                #if j%(samples/10) == 0: print('Progress: {}%'.format(j/samples*100))
                cur_c_l = proposed_c_l
                cur_q = proposed_q
                accept_rate += 1
                
                list_of_q[j] = cur_q
                j += 1
                pbar.update(1)
        print('Accept rate:', accept_rate/tot)
        print('Avg q:', np.mean(list_of_q[burnin:]), 'Std q:', np.std(list_of_q[burnin:]))
        plt.figure()
        plt.hist(list_of_q)
        l = np.arange(l_min, l_max+1)
        plt.figure()
        plt.plot(l, get_D_l(self.c_l_lcdm[l_min:]), label='LCDM')
        plt.plot(l, get_D_l(hp.sphtfunc.alm2cl(s_lm + cur_f_lm)[l_min:]), label='Sigma_l from s_lm+f_lm')
        plt.plot(l, get_D_l(np.mean(list_of_q)*self.c_l_lcdm[l_min:]), label='C_l from best-fit q')
        plt.legend()
        plt.figure()
        plt.plot(np.arange(len(list_of_q)), list_of_q)
        plt.xlabel('Iteration')
        plt.ylabel('q')