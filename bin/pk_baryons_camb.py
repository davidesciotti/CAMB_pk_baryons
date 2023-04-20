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
# zmin, zmax, zpoints = 0., 3, 303
zmin, zmax, zpoints = 0., 3, 150  # CAMB computes 150 redshifts at most
z_grid = np.linspace(zmin, zmax, zpoints)
pk_header = 'redshift \t log10(k) {Mpc^-1} \t P_mm nonlin(k) {Mpc^3}'
halofit_version = 'bird'  # or 'mead2020_feedback'
pk_output_path = f'/Users/davide/Documents/Lavoro/Programmi/CAMB_pk_baryons/output/{halofit_version}'
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
}
if halofit_version == 'mead2020_feedback':
    fid_pars_dict['logT_AGN'] = 7.75

for name_param_to_vary in fid_pars_dict.keys():
    print(f'working on {name_param_to_vary}')

    param_values = fid_pars_dict[name_param_to_vary] + fid_pars_dict[name_param_to_vary] * percentages

    if name_param_to_vary == "wa":  # wa is 0! take directly the percentages
        param_values = percentages

    # ricorda che, quando shifti OmegaM va messo OmegaCDM in modo che OmegaB + OmegaCDM dia il valore corretto di OmegaM,
    # mentre quando shifti OmegaB deve essere aggiustato sempre OmegaCDM in modo che OmegaB + OmegaCDM = 0.32; per OmegaX
    # lo shift ti darà un OmegaM + OmegaDE diverso da 1 il che corrisponde appunto ad avere modelli non piatti

    # initialize after having shifted a parameter
    vinc_pars_dict_tovary = deepcopy(fid_pars_dict)

    for vinc_pars_dict_tovary[name_param_to_vary] in param_values:  # producing 19 PS

        Omega_nu = omnuh2 / (vinc_pars_dict_tovary['h'] ** 2)

        # 1. adjust the values of the other parameters accordingly
        # if name_param_to_vary == 'Omega_M' or name_param_to_vary == 'Omega_B':
        Omega_CDM = vinc_pars_dict_tovary['Omega_M'] - vinc_pars_dict_tovary['Omega_B'] - Omega_nu

        # other CAMB quantities - call them with CAMB-friendly names already
        omch2 = Omega_CDM * vinc_pars_dict_tovary['h'] ** 2
        ombh2 = vinc_pars_dict_tovary['Omega_B'] * vinc_pars_dict_tovary['h'] ** 2
        H0 = vinc_pars_dict_tovary['h'] * 100

        omk = 1 - vinc_pars_dict_tovary['Omega_M'] - vinc_pars_dict_tovary['Omega_DE']  # I'll have non-flat models!
        if np.abs(omk) < 1e-8:
            omk = 0

        # 2. translate dict to camb-like dict
        camb_pars_dict_for_As = {
            'omch2': omch2,
            'ombh2': ombh2,
            'omk': omk,
            'omnuh2': omnuh2,
            'w': vinc_pars_dict_tovary['w0'],
            'wa': vinc_pars_dict_tovary['wa'],
            'H0': H0,
            'mnu': mnu,
            'nnu': nnu,
            'num_massive_neutrinos': 1,
            'sigma8': vinc_pars_dict_tovary['sigma8'],
        }

        # compute As and remove sigma8 from the dict
        As = sigma8_to_As(camb_pars_dict_for_As, extra_args)

        print(f'{np.array([vinc_pars_dict_tovary[key] for key in vinc_pars_dict_tovary.keys()])}'
              f'\t{Omega_CDM:.4f}\t{omk:.6f}\t{omch2:.4f}\t{ombh2:.4f}\t{Omega_nu:.4f}\t{As:.7e}')
        from camb.dark_energy import DarkEnergyPPF, DarkEnergyFluid

        # should I do this?
        pars = camb.CAMBparams()
        pars.set_cosmology(
            omch2=omch2,
            ombh2=ombh2,
            omk=omk,
            # omnuh2=omnuh2,
            # w=vinc_pars_dict_tovary['w0'],
            # wa=vinc_pars_dict_tovary['wa'],
            H0=H0,
            mnu=mnu,
            nnu=nnu,
            num_massive_neutrinos=1,
        )
        pars.InitPower.set_params(ns=vinc_pars_dict_tovary['n_s'], As=As)
        pars.set_dark_energy(w=vinc_pars_dict_tovary['w0'], wa=vinc_pars_dict_tovary['wa'], dark_energy_model='ppf')
        pars.set_matter_power(redshifts=z_grid, kmax=kmax)
        pars.NonLinear = model.NonLinear_both

        if halofit_version == 'mead2020_feedback':
            pars.NonLinearModel.set_params(halofit_version=halofit_version,
                                           HMCode_logT_AGN=vinc_pars_dict_tovary['logT_AGN'])
        elif halofit_version == 'bird':  # this is "takabird"
            pars.NonLinearModel.set_params(halofit_version='bird')
        else:
            raise ValueError('halofit_version must be either "mead2020_feedback" or "bird')

        results = camb.get_results(pars)
        results.calc_power_spectra(pars)
        kh_grid, z_grid, pkh = results.get_matter_power_spectrum(minkh=kmin / vinc_pars_dict_tovary['h'],
                                                                 maxkh=kmax / vinc_pars_dict_tovary['h'],
                                                                 npoints=kpoints)
        k_grid = kh_grid * vinc_pars_dict_tovary['h']
        pk = pkh / vinc_pars_dict_tovary['h'] ** 3

        z_grid_reshaped = np.repeat(z_grid, kpoints).reshape(-1, 1)
        k_grid_reshaped = np.log10(np.tile(k_grid, zpoints).reshape(-1, 1))
        pk_tosave = np.column_stack((z_grid_reshaped, k_grid_reshaped, pk.flatten()))
        np.savetxt(f'{pk_output_path}/{name_param_to_vary}/'
                   f'PddVsZedLogK-{name_param_to_vary}_{vinc_pars_dict_tovary[name_param_to_vary]:.3e}.dat',
                   pk_tosave, header=pk_header)

print('done')
