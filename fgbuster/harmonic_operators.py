# FGBuster
# Copyright (C) 2019 Davide Poletti, Josquin Errard and the FGBuster developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import healpy as hp

__all__ = [
    'get_correlated_noise_covariance',
    'apply_beams_noise_in_harmonic_domain',
]


def get_correlated_noise_covariance(lmax, ell_knee=140, alpha_knee=0.89):
    """
    Get the noise covariance matrix
    """
    nl_ = [(ell_knee/ell)**alpha_knee if ell != 0 else 1 for ell in range(lmax+1)]
    nl *= (np.ones((lmax+1)) + np.array(nl_))
    
    return nl


def apply_beams_noise_in_harmonic_domain(maps, beams, exp_beam=1, nl=None, 
                                        nside=64, lmax=128, iter=3):
    """
    Apply the beam operator and / or 
    the noise covariance to a set of maps
    """
    if isinstance(beams, (int, float)):
        beams = np.ones(len(maps)) * beams
    if nl is None:
        nl = np.ones((len(maps), lmax+1))
    maps_out = np.zeros((maps.shape[0], 2, 12*nside**2))
    for i, map_in in enumerate(maps):
        if isinstance(nl, (int, float)):
            nl = np.ones(lmax+1) * nl
        bl_true = hp.gauss_beam(np.deg2rad(beams[i]/60.), lmax=lmax, pol=True)
        map_in = np.vstack([np.zeros(12*nside**2), map_in])
        alm_in = hp.map2alm(map_in, lmax=lmax, pol=True, iter=iter)
        alm_out = np.zeros_like(alm_in)
        for pol in range(3):
            alm_out[pol] = hp.sphtfunc.almxfl(alm_in[pol], pow((bl_true[:, pol]), exp_beam)/nl[i])
        maps_out[i] = hp.alm2map(alm_out, nside=nside, pol=True)[1:]
    
    return maps_out
