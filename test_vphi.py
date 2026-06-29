#!/usr/bin/env python3
"""
Validate the Method-2 binning of v_phi(r) against an independent yt extraction.

Both paths use the *identical* definition so any disagreement isolates a bug in
the meshblock bookkeeping of Method 2 (axis order, cell volumes, AMR handling):

  v_phi = (x*vy - y*vx) / sqrt(x^2 + y^2)        (= -vx sin(phi) + vy cos(phi))
  r     = sqrt(x^2 + y^2 + z^2)                  (spherical, from origin)
  profile = sum(w * v_phi) / sum(w),  w = rho*dV (mass) or dV (volume)

Method 2 reconstructs cells/volumes from the raw HDF5 meshblocks; yt reconstructs
them from the AMR hierarchy. They should agree to ~round-off if Method 2 is
correct. Same log-r bin edges and (optional) |z|<zmax cut are applied to both.

Usage:
  python test_vphi.py jet.wom.00010.athdf
  python test_vphi.py jet.wom.00010.athdf --weight mass --rmin 1.5e-4 --rmax 4e-2 \
                      --nbins 40 --zmax 0.0 --out vphi_compare.png

All quantities are kept in code units.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from athena_util_functions import load_domain_and_variables
from binning_profiles import accumulate_profile


def vphi_profile_yt(fname, edges, weight='mass', zmax=None):
  """Independent v_phi(r) profile via yt cell extraction + numpy binning."""
  import yt
  try:
    yt.set_log_level(50)
  except Exception:
    pass
  ds = yt.load(fname)
  ad = ds.all_data()

  def grab(candidates, unit):
    last = None
    for f in candidates:
      try:
        return np.asarray(ad[f].to(unit).v, dtype=np.float64)
      except Exception as e:
        last = e
    raise RuntimeError("none of %s available (%s)" % (candidates, last))

  x = grab([('index', 'x')], 'code_length')
  y = grab([('index', 'y')], 'code_length')
  z = grab([('index', 'z')], 'code_length')
  dV = grab([('index', 'cell_volume'), ('gas', 'cell_volume')], 'code_length**3')
  vx = grab([('athena_pp', 'vel1'), ('gas', 'velocity_x')], 'code_velocity')
  vy = grab([('athena_pp', 'vel2'), ('gas', 'velocity_y')], 'code_velocity')
  rho = grab([('athena_pp', 'rho'), ('gas', 'density')], 'code_density')

  R = np.sqrt(x * x + y * y)
  r = np.sqrt(x * x + y * y + z * z)
  vphi = (x * vy - y * vx) / np.where(R > 0, R, np.nan)

  w = rho * dV if weight == 'mass' else (dV if weight == 'volume' else np.ones_like(dV))

  good = np.isfinite(vphi) & np.isfinite(r) & np.isfinite(w)
  if zmax:
    good &= (np.abs(z) < zmax)
  r, vphi, w = r[good], vphi[good], w[good]

  sum_w, _ = np.histogram(r, bins=edges, weights=w)
  sum_wq, _ = np.histogram(r, bins=edges, weights=w * vphi)
  empty = sum_w <= 0.0
  return np.where(empty, np.nan, sum_wq / np.where(empty, 1.0, sum_w))


def vphi_profile_method2(fname, edges, weight='mass', zmax=None):
  """Method-2 binning profile (the implementation under test)."""
  dom, grid, _ = load_domain_and_variables(fname, ['velphi', 'rho'])
  rho = dom[..., 1] if weight == 'mass' else None
  prof, _ = accumulate_profile(grid, dom[..., 0], edges, coord='spherical',
                               weight=weight, reduction='mean', rho=rho, zmax=zmax)
  return prof


def main():
  p = argparse.ArgumentParser()
  p.add_argument('dumpfile')
  p.add_argument('--rmin', type=float, default=1.5e-4)
  p.add_argument('--rmax', type=float, default=4.0e-2)
  p.add_argument('--nbins', type=int, default=40)
  p.add_argument('--weight', choices=['mass', 'volume', 'uniform'], default='mass')
  p.add_argument('--zmax', type=float, default=0.0, help='|z| cut; 0 disables')
  p.add_argument('--out', default='vphi_compare.png')
  args = p.parse_args()

  zmax = args.zmax if args.zmax > 0 else None
  edges = np.logspace(np.log10(args.rmin), np.log10(args.rmax), args.nbins + 1)
  rc = np.sqrt(edges[:-1] * edges[1:])              # geometric centers for log x-axis

  prof_m2 = vphi_profile_method2(args.dumpfile, edges, args.weight, zmax)
  prof_yt = vphi_profile_yt(args.dumpfile, edges, args.weight, zmax)

  both = np.isfinite(prof_m2) & np.isfinite(prof_yt)
  reldiff = np.full_like(prof_m2, np.nan)
  denom = np.where(np.abs(prof_yt) > 0, prof_yt, np.nan)
  reldiff[both] = (prof_m2[both] - prof_yt[both]) / np.abs(denom[both])

  finite = both & np.isfinite(reldiff)
  if np.any(finite):
    print("max |rel diff| = %.3e   median |rel diff| = %.3e"
          % (np.nanmax(np.abs(reldiff[finite])), np.nanmedian(np.abs(reldiff[finite]))))

  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1]})
  ax1.plot(rc, prof_m2, 'o-', ms=3, lw=1.2, label='Method 2 (binning)')
  ax1.plot(rc, prof_yt, 'x--', ms=4, lw=1.0, label='yt')
  ax1.set_xscale('log')
  ax1.set_ylabel(r'$v_\phi$  (%s-weighted)' % args.weight)
  ax1.legend(frameon=False)
  ax1.set_title(r'$v_\phi(r)$  validation' +
                ('' if zmax is None else r'  ($|z|<%.3g$)' % zmax))

  ax2.axhline(0.0, color='k', lw=0.6)
  ax2.plot(rc, reldiff, 's-', ms=3, lw=1.0, color='C3')
  ax2.set_xscale('log')
  ax2.set_xlabel(r'spherical radius $r$')
  ax2.set_ylabel(r'(M2 $-$ yt)/$|$yt$|$')
  fig.tight_layout()
  fig.savefig(args.out, dpi=150)
  print('wrote', args.out)


if __name__ == '__main__':
  main()
