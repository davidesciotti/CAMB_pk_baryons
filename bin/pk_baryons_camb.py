from copy import deepcopy

import numpy as np
import camb
from camb import model, initialpower
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

matplotlib.use('Qt5Agg')

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})


# TODO check h units

def sigma8_to_As(pars, extra_args):
    params = deepcopy(pars)
    if 'As' in params:
        ini_As = params['As']
    else:
        ini_As = 2.1e-9
        params['As'] = ini_As

    final_sig8 = params['sigma8']
    params['WantTransfer'] = True
    params.update(extra_args)

    pars = camb.set_params(**{key: val for key, val in params.items() if key != 'sigma8'})
    pars.set_matter_power(redshifts=[0.], kmax=2.0)

    results = camb.get_results(pars)
    ini_sig8 = np.array(results.get_sigma8())[-1]
    final_As = ini_As * (final_sig8 / ini_sig8) ** 2.

    return final_As


# ! options
kmin, kmax, kpoints = 10 ** -5, 10 ** 2, 804  # 1/Mpc
zmin, zmax, zpoints = 0., 3, 303
zmin, zmax, zpoints = 0., 3, 150  # CAMB computes 150 redshifts at most
z_grid = np.linspace(zmin, zmax, zpoints)
pk_header = 'redshift \t log10(k) {Mpc^-1} \t P_mm nonlin(k) {Mpc^3}'
pk_output_path = '/Users/davide/Documents/Lavoro/Programmi/CAMB_pk_baryons/output'
# ! end options


#######################################################################################################################

percentages = np.asarray((-10, -5, -3.75, -3, -2.5, -1.875, -1.25, -1.0, -0.625, 0.0,
                          0.625, 1.0, 1.25, 1.875, 2.5, 3.0, 3.75, 5.0, 10.0)) / 100

# neutrino mass and density - these don't change with the variations
neutrino_mass_fac = 94.07
mnu = 0.06
nnu = 3.046
g_factor = nnu / 3
omnuh2 = mnu / neutrino_mass_fac * g_factor ** 0.75
extra_args = {'dark_energy_model': 'ppf'}

# fiducials
fid_pars_dict = {
    'Omega_M': 0.32,
    'Omega_B': 0.05,
    'Omega_DE': 0.68,
    'w0': -1.0,
    'wa': 0.0,
    'h': 0.67,
    'n_s': 0.966,
    'sigma8': 0.816,
    'logT_AGN': 7.75,
}

for name_param_to_vary in fid_pars_dict.keys():
    print(f'working on {name_param_to_vary}')

    # ! re-initialize
    # params_dict = {'Omega_m': Omega_m, 'Omega_b': Omega_b, 'w0_fld': w0_fld,
    #                'wa_fld': wa_fld, 'h': h, 'n_s': n_s, 'sigma8': sigma8,
    #                'm_ncdm': m_ncdm, 'Omega_Lambda': Omega_Lambda}

    i = 0  # to keep track of the iteration number, not important

    param_values = fid_pars_dict[name_param_to_vary] + fid_pars_dict[name_param_to_vary] * percentages

    if name_param_to_vary == "wa":  # wa is 0! take directly the percentages
        param_values = percentages

    # ricorda che, quando shifti OmegaM va messo OmegaCDM in modo che OmegaB + OmegaCDM dia il valore corretto di OmegaM,
    # mentre quando shifti OmegaB deve essere aggiustato sempre OmegaCDM in modo che OmegaB + OmegaCDM = 0.32; per OmegaX
    # lo shift ti darà un OmegaM + OmegaDE diverso da 1 il che corrisponde appunto ad avere modelli non piatti

    # initialize after having shifted a parameter
    vinc_pars_dict = deepcopy(fid_pars_dict)

    for vinc_pars_dict[name_param_to_vary] in param_values:  # producing 19 PS

        Omega_nu = omnuh2 / (vinc_pars_dict['h'] ** 2)

        # 1. adjust the values of the other parameters accordingly
        # if name_param_to_vary == 'Omega_M' or name_param_to_vary == 'Omega_B':
        Omega_CDM = vinc_pars_dict['Omega_M'] - vinc_pars_dict['Omega_B'] - Omega_nu

        # other CAMB quantities - call them with CAMB-friendly names already
        omch2 = Omega_CDM * vinc_pars_dict['h'] ** 2
        ombh2 = vinc_pars_dict['Omega_B'] * vinc_pars_dict['h'] ** 2
        H0 = vinc_pars_dict['h'] * 100

        omk = 1 - vinc_pars_dict['Omega_M'] - vinc_pars_dict['Omega_DE']  # I'll have non-flat models!
        if np.abs(omk) < 1e-8:
            omk = 0

        # 2. translate dict to camb-like dict
        camb_pars_dict_for_As = {
            'omch2': omch2,
            'ombh2': ombh2,
            'omk': omk,
            'omnuh2': omnuh2,
            'w': vinc_pars_dict['w0'],
            'wa': vinc_pars_dict['wa'],
            'H0': H0,
            'mnu': mnu,
            'nnu': nnu,
            'num_massive_neutrinos': 1,
            'sigma8': vinc_pars_dict['sigma8'],
        }

        # compute As and remove sigma8 from the dict
        # As = sigma8_to_As(camb_pars_dict_for_As, extra_args)
        As = 1

        print(f'{np.array([vinc_pars_dict[key] for key in vinc_pars_dict.keys()])}'
              f'\t{Omega_CDM:.4f}\t{omk:.6f}\t{omch2:.4f}\t{ombh2:.4f}\t{Omega_nu:.4f}\t{As:.7e}')
        from camb.dark_energy import DarkEnergyPPF, DarkEnergyFluid

        # should I do this?
        pars = camb.CAMBparams()
        pars.set_cosmology(
            omch2=omch2,
            ombh2=ombh2,
            omk=omk,
            # omnuh2=omnuh2,
            # w=vinc_pars_dict['w0'],
            # wa=vinc_pars_dict['wa'],
            H0=H0,
            mnu=mnu,
            nnu=nnu,
            num_massive_neutrinos=1,
        )
        pars.InitPower.set_params(ns=vinc_pars_dict['n_s'], As=As)
        pars.set_dark_energy(w=vinc_pars_dict['w0'], wa=vinc_pars_dict['wa'], dark_energy_model='ppf')
        pars.set_matter_power(redshifts=z_grid, kmax=kmax)
        pars.NonLinear = model.NonLinear_both
        pars.NonLinearModel.set_params(halofit_version='mead2020_feedback', HMCode_logT_AGN=vinc_pars_dict['logT_AGN'])

        results = camb.get_results(pars)
        results.calc_power_spectra(pars)
        kh_grid, z_grid, pkh = results.get_matter_power_spectrum(minkh=kmin / vinc_pars_dict['h'],
                                                                 maxkh=kmax / vinc_pars_dict['h'],
                                                                 npoints=kpoints)
        k_grid = kh_grid * vinc_pars_dict['h']
        pk = pkh / vinc_pars_dict['h'] ** 3

        z_grid_reshaped = np.repeat(z_grid, kpoints).reshape(-1, 1)
        k_grid_reshaped = np.log10(np.tile(k_grid, zpoints).reshape(-1, 1))
        pk_tosave = np.column_stack((z_grid_reshaped, k_grid_reshaped, pk.flatten()))
        np.savetxt(f'{pk_output_path}/{name_param_to_vary}/'
                   f'PddVsZedLogK-{name_param_to_vary}_{vinc_pars_dict[name_param_to_vary]:.3e}.dat',
                   pk_tosave, header=pk_header)

print('done')
