# utility functions for jet injection calculations.
import numpy as np

def compute_injected_power_estimate(gamma, Bi, rho, rc, vel0, r0, rho0):
  """
  Compute an estimate of the injected energy rate (ergs/second) 
  
  Parameters:
  -----------
  gamma : float
    field injection rate parameter
  Bi : float
    characteristic field strength in injection region
  rho : float
    characteristic density in injection region
  rc : float
    radius of injection region
  vel0 : float
    velocity scale to convert from code to cgs units
  r0 : float
    length scale to convert from code to cgs units
  rho0 : float
    density scale to convert from code to cgs units
  
  Returns:
  --------
  l_inj : float
    Injected energy rate in ergs/second
  """
  l_inj = ((4 * np.pi)**2 / 6) * \
          (gamma * Bi)**(3/2) * \
          rho**(1/4) * rc**(7/2) * \
          (rho0 * r0**2 * vel0**3)
  
  return l_inj

def compute_divB(xvs, xfs, hydro, Binds=(4,5,6), coord_range=None, return_coords=False):
    """
    Compute div B over an Athena++ meshblock dataset.
    Assumes Cartesian coordinates and cell-centered magnetic fields.

    Parameters
    ----------
    xvs : list of np.ndarray
        Cell-centered coordinates [x1v, x2v, x3v], shape (Nmb, mb).
    xfs : list of np.ndarray
        Face-centered coordinates [x1f, x2f, x3f], shape (Nmb, mb+1).
    hydro : np.ndarray
        Athena++ primitive/conserved variable array, shape (Nvars, Nmb, mb, mb, mb).
    Binds : tuple of ints, optional
        Indices of (Bx, By, Bz) in hydro. Default: (4,5,6).
    coord_range : list or None, optional
        Spatial limits [x1min, x1max, x2min, x2max, x3min, x3max].
        Returns all cells in any overlapping meshblock.
    return_coords : bool, optional
        If True, also return corresponding cell-centered coordinates.

    Returns
    -------
    divB : np.ndarray
        Divergence field, shape (Nmb_selected, mb, mb, mb).
    coords : tuple of np.ndarray, optional
        If return_coords=True, returns (x1, x2, x3) with shape (Nmb_selected, mb).

    Notes
    -----
    Uses second-order centered differences in interior and first-order one-sided 
    differences at meshblock boundaries.
    """
    # Unpack fields
    Bx, By, Bz = hydro[Binds[0]], hydro[Binds[1]], hydro[Binds[2]]
    x1v, x2v, x3v, x1f, x2f, x3f = *xvs, *xfs
    Nmb, mb = hydro.shape[1], hydro.shape[-1]
    divB_list = []
    coord_lists = ([], [], []) if return_coords else None
    
    for m in range(Nmb):
        # Spatial selection: check meshblock overlap with coord_range
        if coord_range is not None:
            x1min, x1max, x2min, x2max, x3min, x3max = coord_range
            if not ((x1f[m,-1] >= x1min and x1f[m,0] <= x1max) and
                   (x2f[m,-1] >= x2min and x2f[m,0] <= x2max) and
                   (x3f[m,-1] >= x3min and x3f[m,0] <= x3max)):
                continue
        
        # Local arrays and coordinate spacing
        bx, by, bz = Bx[m], By[m], Bz[m]
        dx1, dx2, dx3 = np.diff(x1f[m]), np.diff(x2f[m]), np.diff(x3f[m])
        dBx_dx, dBy_dy, dBz_dz = np.zeros_like(bx), np.zeros_like(by), np.zeros_like(bz)
        
        # dBx/dx (axis=2): interior uses centered diff, boundaries use one-sided
        dBx_dx[:,:,1:-1] = (bx[:,:,2:] - bx[:,:,:-2]) / (x1v[m,2:] - x1v[m,:-2])[None,None,:]
        dBx_dx[:,:,0], dBx_dx[:,:,-1] = (bx[:,:,1] - bx[:,:,0]) / dx1[0], (bx[:,:,-1] - bx[:,:,-2]) / dx1[-1]
        
        # dBy/dy (axis=1)
        dBy_dy[:,1:-1,:] = (by[:,2:,:] - by[:,:-2,:]) / (x2v[m,2:] - x2v[m,:-2])[None,:,None]
        dBy_dy[:,0,:], dBy_dy[:,-1,:] = (by[:,1,:] - by[:,0,:]) / dx2[0], (by[:,-1,:] - by[:,-2,:]) / dx2[-1]
        
        # dBz/dz (axis=0)
        dBz_dz[1:-1,:,:] = (bz[2:,:,:] - bz[:-2,:,:]) / (x3v[m,2:] - x3v[m,:-2])[:,None,None]
        dBz_dz[0,:,:], dBz_dz[-1,:,:] = (bz[1,:,:] - bz[0,:,:]) / dx3[0], (bz[-1,:,:] - bz[-2,:,:]) / dx3[-1]
        
        divB_list.append(dBx_dx + dBy_dy + dBz_dz)
        if return_coords:
            for i, xv in enumerate([x1v[m], x2v[m], x3v[m]]):
                coord_lists[i].append(xv)
    
    divB_arr = np.array(divB_list)
    return (divB_arr, tuple(np.array(cl) for cl in coord_lists)) if return_coords else divB_arr

def compute_total_divB(xfs, divB, coord_system="cartesian"):
    """
    Compute the signed volume-integrated divergence of B: Σ (∇·B) dV
    Matches the Athena++ history diagnostic.

    Parameters
    ----------
    xfs : list of np.ndarray
        Face-centered coordinates [x1f, x2f, x3f], shape (Nmb, mb+1).
    divB : np.ndarray
        Divergence field from compute_divB(), shape (Nmb, mb, mb, mb).
    coord_system : str
        Currently only "cartesian" supported.

    Returns
    -------
    total_divB : float
        Signed integrated divergence over the domain.
    """
    if coord_system != "cartesian":
        raise NotImplementedError(f"coord_system='{coord_system}' not implemented")
    
    x1f, x2f, x3f = xfs
    total = 0.0
    for m in range(divB.shape[0]):
        dx1, dx2, dx3 = np.diff(x1f[m]), np.diff(x2f[m]), np.diff(x3f[m])
        dV = dx3[:,None,None] * dx2[None,:,None] * dx1[None,None,:]  # Cell volumes
        total += np.sum(divB[m] * dV)
    return total