import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sqrt
import camb
import healpy as hp

from tqdm import tqdm

def get_D_l(c_l):
        return [c_l[l] * l * (l+1) for l in range(len(c_l))]

class gibbs_sampler():
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
        self.c_l_lcdm = np.array(camb_results.get_cmb_power_spectra(lmax=l_max, raw_cl=True)['total'][:, 0])

        self.N_l = noise_multiplier * camb_results.get_cmb_power_spectra(lmax=250, raw_cl=True)['total'][250, 0]# * self.b_l[self.l_max]**2
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

        smooth_alm = a_lm#hp.sphtfunc.smoothalm(a_lm, self.fwhm, inplace = False)
        smooth_dlm = d_lm#hp.sphtfunc.smoothalm(d_lm, self.fwhm, inplace = False)

        self.smooth_a_lm = smooth_alm
        self.smooth_d_lm = smooth_dlm

    def get_slm_sample(self, c_l):
        N_l = self.N_l
        d_lm = self.smooth_d_lm
        l_max = self.l_max

        B_l = self.b_l
        F_inv = np.array(c_l**(-1) + B_l * N_l**(-1) * B_l)**(-1)

        s_lm = np.zeros(self.alm_size, dtype=complex)
        for l in range(self.l_min, l_max+1):
            #N_B_2_C_inv = 1/(N_l + B_l[l]**2 * c_l[l])

            index = hp.sphtfunc.Alm.getidx(l_max, l, np.arange(l+1))
            s_lm[index] = F_inv[l] * (B_l[l] * N_l**(-1) * d_lm[index] + c_l[l]**(-1/2) * np.random.normal(0, 1) + B_l[l] * N_l**(-1/2) * np.random.normal(0, 1))
            #   for m in range(l+1):
            #       index = hp.sphtfunc.Alm.getidx(l_max, l, m)
            #       #s_lm[index] = N_B_2_C_inv * (c_l[l] * B_l[l] * d_lm[index] + c_l[l] * B_l[l] * sqrt(N_l) * np.random.normal(0, 1) + sqrt(c_l[l]) * N_l * np.random.normal(0, 1))
            #       s_lm[index] = F_inv[l] * (B_l[l] * N_l**(-1) * d_lm[index] + c_l[l]**(-1/2) * np.random.normal(0, 1) + B_l[l] * N_l**(-1/2) * np.random.normal(0, 1))

        return s_lm

    def get_cl_sample(self, s_lm):
        l_max = self.l_max
        new_c_l = np.zeros(l_max+1)
        for l in range(2, l_max+1):
            index = hp.sphtfunc.Alm.getidx(l_max, l, 0)
            sigma_l = np.abs(s_lm[index])**2
            for m in range(1, l+1):
                    index = hp.sphtfunc.Alm.getidx(l_max, l, m)
                    sigma_l += 2*np.abs(s_lm[index])**2
            eta = np.random.normal(0, 1, 2*l-1)
            rho_l = np.sum(eta**2)
            
            new_c_l[l] = sigma_l/rho_l
        return new_c_l

    def plot_spectras(self):
        l = np.arange(self.l_min, self.l_max+1)

        c_l = get_D_l(self.c_l_lcdm)
        smooth_d_l = get_D_l(hp.sphtfunc.alm2cl(self.smooth_d_lm))
        smooth_a_l = get_D_l(hp.sphtfunc.alm2cl(self.smooth_a_lm))
        D_N_l = get_D_l(np.repeat(self.N_l, self.l_max+1))

        plt.figure()
        plt.plot(l, c_l[self.l_min:], label='LCDM', linewidth=4)
        plt.plot(l, smooth_a_l[self.l_min:], label='Beam Smoothed Cosmological Signal')
        plt.plot(l, smooth_d_l[self.l_min:], label='Measured Data')
        plt.plot(l, D_N_l[self.l_min:], label='White Noise')
        
        plt.ylabel(r"$C_{\ell}\ell (\ell + 1)$")
        plt.xlabel(r"$\ell$")
        plt.legend()
        plt.savefig('spectra.pdf')

    def get_cosmo_param_sample(self, q_old, s_lm):
        c_l_old_sample = q_old * np.array(self.c_l_lcdm)
        sigma_l = hp.sphtfunc.alm2cl(s_lm)
        ln_P_old = self.log_likelihood(c_l_old_sample, sigma_l)

        tot = 0
        
        while True:
            q_new = np.random.normal(q_old, self.q_sigma)           
            c_l_new_sample = q_new * np.array(self.c_l_lcdm)

            a = np.exp(self.log_likelihood(c_l_new_sample, sigma_l) - ln_P_old)
            
            eta = np.random.uniform(0, 1)
            tot += 1 

            if eta < a:
                return q_new, tot

    def log_likelihood(self, c_l, sigma_l):
        ln_P = 0
        for l in range(self.l_min, self.l_max+1):
            ln_P += (2*l+1)/2 * (-sigma_l[l]/c_l[l] + np.log(sigma_l[l]/c_l[l])) - np.log(sigma_l[l])
        return ln_P

    def start_gibbs_sampler(self, samples=100, start_q=1, burnin=10, sample_cosmo = True):
        l_max = self.l_max
        l_min = self.l_min
        l = np.arange(l_min, l_max+1)

        tot = 0
        acceptance = 0

        if sample_cosmo:
            new_q = start_q
            list_q = [new_q]
            new_c_l = new_q * self.c_l_lcdm
        else:
            new_c_l = np.ones(l_max+1)*2e-13
            list_of_c_l = np.zeros((samples+1, l_max+1))
            list_of_c_l[0, :] = new_c_l
        
        plt.plot(l, get_D_l(new_c_l[l_min:]), label='Initial Spectra')      

        for j in tqdm(range(samples)):
            #if j%(samples/100) == 0: print('Progress: {}%'.format(j/samples*100))

            new_s_lm = self.get_slm_sample(new_c_l)
            acceptance += 1
            if sample_cosmo:
                new_q, new_tot = self.get_cosmo_param_sample(new_q, new_s_lm)
                new_c_l = new_q * self.c_l_lcdm
                tot += new_tot

                list_q = np.append(list_q, new_q)
            else:
                new_c_l = self.get_cl_sample(new_s_lm)
                list_of_c_l[j+1, :] = new_c_l
            

        plt.plot(l, get_D_l(self.c_l_lcdm[l_min:]), label='LCDM')
        #plt.plot(l, get_D_l(hp.sphtfunc.alm2cl(new_s_lm)[l_min:]), label='Sigma_l from s_lm')

        if sample_cosmo:
            plt.plot(l, get_D_l(np.mean(list_q)*self.c_l_lcdm[l_min:]), label='C_l from best-fit q')
        else:
            c_l_average = np.mean(list_of_c_l[burnin:, :], axis=0)
            plt.plot(l, get_D_l(c_l_average[l_min:]), label='Best fit C_l')
            plt.ylim((0, 1.5*np.max(get_D_l(self.c_l_lcdm[l_min:]))))
        
        plt.ylabel(r'$C_{\ell} \ell (\ell + 1)$')
        plt.xlabel(r"$\ell$")
        plt.legend()
        if sample_cosmo:
            print('Acceptance Rate:', acceptance/tot)
            print('Avg q:', np.mean(list_q[burnin:]), 'Std q:', np.std(list_q[burnin:]))
            plt.figure()
            plt.hist(list_q)
            plt.xlabel(r"$q$")
            plt.savefig('q_hist.pdf')
            plt.figure()
            plt.plot(np.arange(len(list_q)), list_q)
            plt.xlabel('Iteration')
            plt.ylabel('q')
            plt.savefig('q_it.pdf')
        else:
            plt.figure()
            plt.plot(np.arange(samples+1), list_of_c_l[:, 10] / c_l_average[10])
            plt.xlabel('Iteration')
            plt.ylabel(r"$C_{10}/C^{avg}_{10}$")
            plt.figure()
            plt.plot(np.arange(samples+1), list_of_c_l[:, 150] / c_l_average[150])
            plt.xlabel('Iteration')
            plt.ylabel(r"$C_{150}/C^{avg}_{150}$")
            plt.figure()
            plt.plot(np.arange(samples+1), list_of_c_l[:, 250] / c_l_average[250])
            plt.xlabel('Iteration')
            plt.ylabel(r"$C_{250}/C^{avg}_{250}$")