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

""" High-level component separation routines 
    with SHTs to apply harmonic domain operators
"""
import numpy as np
import healpy as hp
import scipy as sp
import inspect
from .harmonic_operators import apply_beams_noise_in_harmonic_domain


__all__ = [
    'Wd_mixed_no_commutation',
]


def get_pixel_based_precond(A, N):
    """
    Builds the pixel-based preconditioner,
    as the system matrix but without beams 
    and without correlations in the noise
    """
    AtNA = np.einsum('ijp,ik, klp -> jlp', A, np.linalg.inv(N), A)
    inv_AtNA = np.linalg.inv(AtNA.swapaxes(0,-1)).swapaxes(0,-1)
    
    return inv_AtNA


### Component separation functions
def Wd_mixed_no_commutation(freq_maps, beams_true, beam_final, A, nl, N_pixel, 
                            nside, lmax, pcg_maxiter=1000, pcg_tol=1e-6):
    """
    Performs the second step of the component separation
    with the beam operator and the noise covariance matrix
    in the harmonic domain, exact approach 
    (no commutation of mixing matrix and harmonic operator)

    freq_maps: ndarray of shape (n_freq, n_stokes, n_pix)
    """
    ncomp = A.shape[1]
    nstokes = freq_maps.shape[1]
    npix = freq_maps.shape[2]
    
    # RHS
    BNd = apply_beams_noise_in_harmonic_domain(freq_maps, beams_true, nl=nl, 
                                               nside=nside, lmax=lmax)
    AtBNd = np.einsum('ijp,isp->jsp', A, BNd)
    BAtBNd = apply_beams_noise_in_harmonic_domain(AtBNd, beam_final, exp_beam=-1, 
                                                  nside=nside, lmax=lmax)
    BAtBNd_reshaped = BAtBNd.reshape((ncomp*nstokes*npix))

    # LHS
    def get_LHS(vec):
        vec = vec.reshape((ncomp, nstokes, npix))
        # B x
        B_x = apply_beams_noise_in_harmonic_domain(vec, beam_final, exp_beam=-1, 
                                                    nside=nside, lmax=lmax)
        # A B x
        AB_x = np.einsum('ijp,jsp->isp', A, B_x)
        # B N B A B x
        BNBAB_x = apply_beams_noise_in_harmonic_domain(AB_x, beams_true, nl=nl,
                                                        nside=nside, lmax=lmax)
        # At B N B A B x
        AtBNBAB_x = np.einsum('ijp,isp->jsp', A, BNBAB_x)
        BAtBNBAB_x = apply_beams_noise_in_harmonic_domain(AtBNBAB_x, beam_final, 
                                                          exp_beam=-1, nside=nside, 
                                                          lmax=lmax)
        return BAtBNBAB_x.reshape((ncomp*nstokes*npix))
    
    AtNA_precond = get_pixel_based_precond(A, N_pixel)

    def apply_precond(v):
        v = v.reshape(ncomp, nstokes, npix)
        Mv = np.einsum('ij...,js...->is...', AtNA_precond, v)
        
        return Mv.reshape((ncomp*nstokes*npix))

    iters = 0
    residuals = []
    def report_callback(xk):
        frame = inspect.currentframe().f_back
        residuals.append(frame.f_locals['r'])
        nonlocal iters
        iters += 1
        
    vect0 = np.zeros((ncomp*nstokes*npix))
    ### PCG
    matrix = sp.sparse.linalg.LinearOperator((ncomp*nstokes*npix,ncomp*nstokes*npix), matvec=get_LHS)
    precond = sp.sparse.linalg.LinearOperator((ncomp*nstokes*npix,ncomp*nstokes*npix), matvec=apply_precond)
    x_solution, cg_info = sp.sparse.linalg.cg(matrix, BAtBNd_reshaped, x0=vect0, M=precond, 
                                              maxiter=pcg_maxiter, tol=pcg_tol, callback=report_callback)
    print('>>> PCG info:', cg_info)

    return x_solution, residuals, iters
