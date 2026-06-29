"""
Method 2 (binning) axisymmetric profiles for Cartesian Athena++ (SMR) data.

Builds on athena_util_functions.load_domain_and_variables, which returns
  domain_data : (Nmb, mb, mb, mb, Nvar)   spatial axes ordered (x1, x2, x3) = (x, y, z)
  domain_grid : (Nmb, mb+1, 3)            face coordinates, last axis (x1f, x2f, x3f)

Each leaf cell carries its own dV = dx*dy*dz, so mixed refinement levels are
handled automatically (no interpolation, conservative) provided the dump holds
only leaf blocks (standard Athena++ output).

Reductions:
  'mean'     -> intensive fields (v_phi, v_r, T, beta): sum(w q)/sum(w)
  'integral' -> extensive flux (Mdot = -closed_int rho v_r dA): sum(w q)/d(coord)

These are intended to live alongside / be pasted into athena_util_functions.py.
"""

import numpy as np


# ----------------------------------------------------------------------------
# internals
# ----------------------------------------------------------------------------
def _cell_centers_and_dV(domain_grid):
  """From face coords (Nmb, mb+1, 3) return broadcast cell centers and dV.

  Returns X, Y, Z, dV each of shape (Nmb, mb, mb, mb) with axes (Nmb, x, y, z),
  matching the spatial layout of domain_data[..., k].
  """
  xf, yf, zf = domain_grid[:, :, 0], domain_grid[:, :, 1], domain_grid[:, :, 2]
  xc = 0.5 * (xf[:, :-1] + xf[:, 1:])            # (Nmb, mb)
  yc = 0.5 * (yf[:, :-1] + yf[:, 1:])
  zc = 0.5 * (zf[:, :-1] + zf[:, 1:])
  dx = np.diff(xf, axis=1)                        # (Nmb, mb)
  dy = np.diff(yf, axis=1)
  dz = np.diff(zf, axis=1)

  X = xc[:, :, None, None]
  Y = yc[:, None, :, None]
  Z = zc[:, None, None, :]
  dV = dx[:, :, None, None] * dy[:, None, :, None] * dz[:, None, None, :]

  X, Y, Z, dV = np.broadcast_arrays(X, Y, Z, dV)  # each (Nmb, mb, mb, mb)
  return X, Y, Z, dV


def _bin_coordinate(X, Y, Z, coord, eps=1e-30):
  """Return the 1D binning coordinate per cell for the requested geometry."""
  if coord == 'spherical':
    return np.sqrt(X * X + Y * Y + Z * Z)
  elif coord == 'cylindrical':
    return np.sqrt(X * X + Y * Y)
  else:
    raise ValueError("coord must be 'spherical' or 'cylindrical'")


def _weights(dV, rho, weight):
  if weight == 'volume':
    return dV
  elif weight == 'mass':
    if rho is None:
      raise ValueError("weight='mass' requires rho (same shape as field)")
    return rho * dV
  elif weight == 'uniform':
    return np.ones_like(dV)
  else:
    raise ValueError("weight must be 'volume', 'mass', or 'uniform'")


# ----------------------------------------------------------------------------
# 1D profile
# ----------------------------------------------------------------------------
def accumulate_profile(domain_grid, field, bin_edges, coord='spherical',
                       weight='volume', reduction='mean', rho=None,
                       zmax=None, return_extras=False):
  """Axisymmetrized 1D radial profile by single-pass scatter accumulation.

  Parameters
  ----------
  domain_grid : (Nmb, mb+1, 3) face coordinates from load_domain_and_variables.
  field       : (Nmb, mb, mb, mb) cell-centered scalar to profile.
  bin_edges   : (Nbin+1,) monotonic edges in the binning coordinate.
  coord       : 'spherical' (r) or 'cylindrical' (R).
  weight      : 'volume' (dV), 'mass' (rho*dV), or 'uniform'.
  reduction   : 'mean'     -> sum(w*field)/sum(w)        (intensive)
                'integral' -> sum(w*field)/d(coord)       (extensive flux,
                              with weight='volume' this is closed_int field dV / dr)
  rho         : (Nmb, mb, mb, mb), required only for weight='mass'.
  zmax        : if set, keep only cells with |z| < zmax (e.g. midplane cut).
  return_extras : also return a dict with count, sum_w, std, n_eff, sem.

  Returns
  -------
  profile : (Nbin,) with NaN in empty bins.
  centers : (Nbin,) arithmetic bin centers (use sqrt(e[:-1]*e[1:]) for log x-axis).
  extras  : dict (only if return_extras=True).
  """
  X, Y, Z, dV = _cell_centers_and_dV(domain_grid)
  c = _bin_coordinate(X, Y, Z, coord)
  w = _weights(dV, rho, weight)

  c = np.ravel(c)
  q = np.ravel(field).astype(np.float64)
  w = np.ravel(w).astype(np.float64)

  good = np.isfinite(c) & np.isfinite(q) & np.isfinite(w)
  if zmax is not None:
    good &= (np.abs(np.ravel(Z)) < zmax)

  edges = np.asarray(bin_edges, dtype=np.float64)
  nbin = edges.size - 1
  idx = np.digitize(c, edges) - 1                 # 0..nbin-1 in range, else outside
  good &= (idx >= 0) & (idx < nbin)

  idx = idx[good]
  q = q[good]
  w = w[good]
  wq = w * q

  sum_w = np.bincount(idx, weights=w, minlength=nbin)
  sum_wq = np.bincount(idx, weights=wq, minlength=nbin)

  centers = 0.5 * (edges[1:] + edges[:-1])
  empty = sum_w <= 0.0

  if reduction == 'mean':
    profile = np.where(empty, np.nan, sum_wq / np.where(empty, 1.0, sum_w))
  elif reduction == 'integral':
    dcoord = np.diff(edges)
    profile = sum_wq / dcoord                      # = sum(field*dV)/d(coord)
    profile = np.where(empty, np.nan, profile)
  else:
    raise ValueError("reduction must be 'mean' or 'integral'")

  if not return_extras:
    return profile, centers

  sum_wq2 = np.bincount(idx, weights=wq * q, minlength=nbin)
  sum_w2 = np.bincount(idx, weights=w * w, minlength=nbin)
  count = np.bincount(idx, minlength=nbin)
  mean = np.where(empty, np.nan, sum_wq / np.where(empty, 1.0, sum_w))
  var = np.where(empty, np.nan, sum_wq2 / np.where(empty, 1.0, sum_w) - mean ** 2)
  std = np.sqrt(np.clip(var, 0.0, None))
  n_eff = np.where(empty, np.nan, sum_w ** 2 / np.where(empty, 1.0, sum_w2))
  sem = std / np.sqrt(n_eff)                       # statistical only; see notes on ell_corr
  extras = dict(count=count, sum_w=sum_w, std=std, n_eff=n_eff, sem=sem)
  return profile, centers, extras


# ----------------------------------------------------------------------------
# 2D poloidal map  (r, theta) or (R, z)
# ----------------------------------------------------------------------------
def accumulate_poloidal_map(domain_grid, field, edges1, edges2,
                            grid_type='spherical_polar', weight='volume',
                            reduction='mean', rho=None, eps=1e-30):
  """2D (poloidal) map by single-pass 2D scatter accumulation.

  grid_type : 'spherical_polar' -> (edges1=r, edges2=theta in [0, pi])
              'cylindrical'      -> (edges1=R, edges2=z)
  Other arguments as in accumulate_profile. Returns map (N1, N2), c1, c2.
  """
  X, Y, Z, dV = _cell_centers_and_dV(domain_grid)
  w = _weights(dV, rho, weight)

  if grid_type == 'spherical_polar':
    r = np.sqrt(X * X + Y * Y + Z * Z)
    coord1 = r
    coord2 = np.arccos(np.clip(Z / (r + eps), -1.0, 1.0))   # theta
  elif grid_type == 'cylindrical':
    coord1 = np.sqrt(X * X + Y * Y)                          # R
    coord2 = Z
  else:
    raise ValueError("grid_type must be 'spherical_polar' or 'cylindrical'")

  c1 = np.ravel(coord1)
  c2 = np.ravel(coord2)
  q = np.ravel(field).astype(np.float64)
  w = np.ravel(w).astype(np.float64)

  e1 = np.asarray(edges1, dtype=np.float64)
  e2 = np.asarray(edges2, dtype=np.float64)
  n1, n2 = e1.size - 1, e2.size - 1

  i1 = np.digitize(c1, e1) - 1
  i2 = np.digitize(c2, e2) - 1
  good = (np.isfinite(c1) & np.isfinite(c2) & np.isfinite(q) & np.isfinite(w) &
          (i1 >= 0) & (i1 < n1) & (i2 >= 0) & (i2 < n2))

  i1, i2, q, w = i1[good], i2[good], q[good], w[good]
  flat = i1 * n2 + i2
  wq = w * q

  sum_w = np.bincount(flat, weights=w, minlength=n1 * n2).reshape(n1, n2)
  sum_wq = np.bincount(flat, weights=wq, minlength=n1 * n2).reshape(n1, n2)
  empty = sum_w <= 0.0

  if reduction == 'mean':
    out = np.where(empty, np.nan, sum_wq / np.where(empty, 1.0, sum_w))
  elif reduction == 'integral':
    out = np.where(empty, np.nan, sum_wq / np.diff(e1)[:, None])
  else:
    raise ValueError("reduction must be 'mean' or 'integral'")

  c1c = 0.5 * (e1[1:] + e1[:-1])
  c2c = 0.5 * (e2[1:] + e2[:-1])
  return out, c1c, c2c


# ----------------------------------------------------------------------------
# thin wrappers
# ----------------------------------------------------------------------------
def compute_mdot_profile(ath_file, bin_edges, coord_range=None, return_extras=False):
  """Spherical accretion-rate profile  Mdot(r) = -closed_int rho v_r dA.

  Positive return value => net inflow. Uses field 'mdotr' = rho * v_r,sph,
  volume weighting, integral reduction (sum(rho v_r dV)/dr).
  """
  from athena_util_functions import load_domain_and_variables
  dk = {} if coord_range is None else {'coord_range': coord_range}
  dom, grid, _ = load_domain_and_variables(ath_file, ['mdotr'], domain_kwargs=dk)
  out = accumulate_profile(grid, dom[..., 0], bin_edges, coord='spherical',
                           weight='volume', reduction='integral',
                           return_extras=return_extras)
  if return_extras:
    prof, cen, extras = out
    return -prof, cen, extras
  prof, cen = out
  return -prof, cen


def compute_rotation_curve(ath_file, bin_edges, zmax, coord_range=None,
                           weight='mass', return_extras=False):
  """Mass-weighted cylindrical rotation curve v_phi(R) with |z| < zmax cut."""
  from athena_util_functions import load_domain_and_variables
  dk = {} if coord_range is None else {'coord_range': coord_range}
  dom, grid, _ = load_domain_and_variables(ath_file, ['velphi', 'rho'], domain_kwargs=dk)
  return accumulate_profile(grid, dom[..., 0], bin_edges, coord='cylindrical',
                            weight=weight, reduction='mean', rho=dom[..., 1],
                            zmax=zmax, return_extras=return_extras)


def compute_poloidal_mass_flux(ath_file, r_edges, theta_edges,
                               weight='volume', coord_range=None):
  """Poloidal (r, theta) map of radial mass-flux density rho v_r,sph.

  Volume-weighted mean per bin (flux *density*, sign-resolved): inflow < 0,
  outflow > 0. Use a diverging colormap centered at 0.
  """
  from athena_util_functions import load_domain_and_variables
  dk = {} if coord_range is None else {'coord_range': coord_range}
  dom, grid, _ = load_domain_and_variables(ath_file, ['mdotr'], domain_kwargs=dk)
  return accumulate_poloidal_map(grid, dom[..., 0], r_edges, theta_edges,
                                 grid_type='spherical_polar', weight=weight,
                                 reduction='mean')
