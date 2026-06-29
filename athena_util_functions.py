import numpy as np
import sys, h5py

def extract_slice(xvs,xfs,hydro,slice_dim,loc,remain_coord_range=None,current=False):
  """
  Extracts a 2D slice of cell-centered hydrodynamic variables at a given location
  along one spatial dimension from a 3D grid of meshblocks.

  Parameters
  ----------
  xvs : list of np.ndarray or hdf5 datasets
    Cell-centered coordinates along the x1, x2, and x3 directions.
    Each has shape (Nmb, mb), where Nmb is number of meshblocks and mb is cells per block.
  xfs : list of np.ndarray or hdf5 datasets
    Face-centered coordinates along the x1, x2, and x3 directions.
    Each has shape (Nmb, mb+1), where Nmb is number of meshblocks and mb is cells per block.
  
  hydro : np.ndarray or hdf5 dataset
    5D array of hydrodynamic primitive variables with shape (Nprims, Nmb, mb, mb, mb).

  slice_dim : int
    Dimension to slice along: 0 (x1), 1 (x2), or 2 (x3).
  
  loc : float
    Physical location along `slice_dim` where the slice is taken.

  remain_coord_range : list
    List of slice domain for the remaining dimensions. None for entire domain. 
    Note that the returned slice will contain the entire meshblock that overlaps the requested domain, so output may be larger than expected.
  
  current: bool
    Whether to compute the current density J and combine it with the hydro data

  Returns
  -------
  slice_data : np.ndarray
    2D slices of hydro variables from meshblocks intersecting the slice location.
    Shape: (Nmb_selected, mb, mb, Nprims).

  slice_grid : np.ndarray
    Corresponding face-centered grid coordinates in the remaining two dimensions.
    Shape: (Nmb_selected, mb+1, 2).
  """
  # first generate the current density over the grid if needed
  if(current):
    currentJ = compute_J(xvs[0],xvs[1],xvs[2],hydro)
    hydro = np.concatenate([hydro,currentJ],axis=0)
  faces = np.array(xfs).transpose((1,2,0))
  Nprims = hydro.shape[0]
  Nmb = hydro.shape[1]
  mb = hydro.shape[-1]
  remain_dim = sorted(list(set(range(3)) - set([slice_dim])))
  slice_data = []
  slice_grid = []
  for i in range(Nmb):
    # if the meshblock straddles the location of the slice
    if faces[i,0,slice_dim]<=loc and faces[i,-1,slice_dim]>=loc:
      if remain_coord_range==None or \
      (remain_coord_range[0]<=faces[i,-1,remain_dim[0]] and remain_coord_range[1]>=faces[i,0,remain_dim[0]] and \
      remain_coord_range[2]<=faces[i,-1,remain_dim[1]] and remain_coord_range[3]>=faces[i,0,remain_dim[1]]):
        # find meshblock face closest to location
        ind = np.argmin(abs(faces[i,:,slice_dim]-loc))
        if ind == mb:
          ind-=1
        slice_grid.append(faces[i,:,remain_dim])
        if slice_dim==0:
          slice_data.append(hydro[:,i,:,:,ind])
        elif slice_dim==1:
          slice_data.append(hydro[:,i,:,ind,:])
        elif slice_dim==2:
          slice_data.append(hydro[:,i,ind,:,:])
  slice_data = np.array(slice_data).transpose(0,2,3,1)
  slice_grid = np.array(slice_grid).transpose(0,2,1)
  return slice_data,slice_grid

def extract_domain(xvs, xfs, hydro, coord_range=None, current=False):
  """
  Extracts 3D meshblocks of cell-centered hydrodynamic variables within a specified
  coordinate range from a 3D grid of meshblocks. 
  Note it returns the entire meshblock if coord_range lies within it.
  
  Parameters
  ----------
  xvs : list of np.ndarray or hdf5 datasets
    Cell-centered coordinates along the x1, x2, and x3 directions.
    Each has shape (Nmb, mb), where Nmb is number of meshblocks and mb is cells per block.
  xfs : list of np.ndarray or hdf5 datasets
    Face-centered coordinates along the x1, x2, and x3 directions.
    Each has shape (Nmb, mb+1), where Nmb is number of meshblocks and mb is cells per block.
  
  hydro : np.ndarray or hdf5 dataset
    5D array of hydrodynamic primitive variables with shape (Nprims, Nmb, mb, mb, mb).
  
  coord_range : list or None
    List of coordinate ranges [x1min, x1max, x2min, x2max, x3min, x3max].
    None for entire domain. Note that the returned domain will contain the entire 
    meshblock that overlaps the requested domain, so output may be larger than expected.
  
  current : bool
    Whether to compute the current density J and combine it with the hydro data.
  
  Returns
  -------
  domain_data : np.ndarray
    3D meshblocks of hydro variables intersecting the coordinate range.
    Shape: (Nmb_selected, mb, mb, mb, Nprims).
  
  domain_grid : np.ndarray
    Corresponding face-centered grid coordinates in all three dimensions.
    Shape: (Nmb_selected, mb+1, 3).
  """
  # first generate the current density over the grid if needed
  if current:
    currentJ = compute_J(xvs[0], xvs[1], xvs[2], hydro)
    hydro = np.concatenate([hydro, currentJ], axis=0)
  
  # transpose faces to shape (Nmb, mb+1, 3) for easier indexing
  faces = np.array(xfs).transpose((1, 2, 0))
  Nprims = hydro.shape[0]
  Nmb = hydro.shape[1]
  mb = hydro.shape[-1]
  
  domain_data = []
  domain_grid = []

  # return the full domain without iteration
  if coord_range==None:
    domain_data = np.asarray(hydro).transpose(1,-1,-2,-3,0)
    domain_grid = faces
    return domain_data,domain_grid
  
  for i in range(Nmb):
    # check if the meshblock overlaps with the coordinate range
    if (coord_range[0] <= faces[i, -1, 0] and coord_range[1] >= faces[i, 0, 0] and \
        coord_range[2] <= faces[i, -1, 1] and coord_range[3] >= faces[i, 0, 1] and \
        coord_range[4] <= faces[i, -1, 2] and coord_range[5] >= faces[i, 0, 2]):
      
      # include the entire meshblock's grid coordinates (all 3 dimensions)
      domain_grid.append(faces[i])  # Shape: (mb+1, 3 (x1-x2-x3 ordering))
      
      # include the entire meshblock's hydro data (all 3 spatial dimensions)
      domain_data.append(hydro[:, i, :, :, :])  # Shape: (Nprims, mb, mb, mb)
  
  # convert to arrays and transpose to match expected output format
  domain_data = np.array(domain_data).transpose(0, 4, 3, 2, 1)  # (Nmb_selected, mb(x1), mb(x2), mb(x3), Nprims)
  domain_grid = np.array(domain_grid)  # (Nmb_selected, mb+1, 3)
  
  return domain_data, domain_grid

def var_list_lookup(header,current=False):
  var_list = header['VariableNames']  
  if(current):
    var_list_full = var_list+['J1','J2','J3']
    return var_list_full
  else:
    return var_list

def load_domain_and_variables(ath_file, variables, domain_kwargs={}):
  """Load entire domain from Athena++ file and compute requested variables."""
  with h5py.File(ath_file) as hfp:
    hydro = hfp['hydro']
    xvs = np.array([hfp['x1v'][:], hfp['x2v'][:], hfp['x3v'][:]])
    xfs = np.array([hfp['x1f'][:], hfp['x2f'][:], hfp['x3f'][:]])

    header = {i: hfp.attrs[i] for i in hfp.attrs.keys()}
    header['VariableNames'] = [i.decode('utf-8') for i in header['VariableNames']]
    
    domain_hydro,domain_grid = extract_domain(xvs,xfs,hydro,**domain_kwargs)
    
    var_list_full = var_list_lookup(header, domain_kwargs.get("current",False))
    domain_data = np.zeros(shape=(*domain_hydro.shape[:-1], len(variables)))
    
    cyl_vars = ['velr', 'velphi', 'velz', 'Br', 'Bphi', 'Bz', 'Jr', 'Jphi', 'Jz']
    needs_cyl = any(var.replace('log', '') in cyl_vars for var in variables)
    
    if needs_cyl:
      R_grid, phi_grid = compute_cylindrical_coords(domain_grid)
    
    for ind, variableName in enumerate(variables):
      if ('log' in variableName):
        variable = variableName.replace('log', '')
        log_flag = True
      else:
        log_flag = False
        variable = variableName
      
      if variable == 'velr':
        vx_ind, vy_ind = var_list_full.index('vel1'), var_list_full.index('vel2')
        vx, vy = domain_hydro[..., vx_ind], domain_hydro[..., vy_ind]
        domain_data[..., ind] = vx * np.cos(phi_grid) + vy * np.sin(phi_grid)
      elif variable == 'velphi':
        vx_ind, vy_ind = var_list_full.index('vel1'), var_list_full.index('vel2')
        vx, vy = domain_hydro[..., vx_ind], domain_hydro[..., vy_ind]
        domain_data[..., ind] = -vx * np.sin(phi_grid) + vy * np.cos(phi_grid)
      elif variable == 'velz':
        vz_ind = var_list_full.index('vel3')
        domain_data[..., ind] = domain_hydro[..., vz_ind]
      elif variable == 'Br':
        bx_ind, by_ind = var_list_full.index('Bcc1'), var_list_full.index('Bcc2')
        bx, by = domain_hydro[..., bx_ind], domain_hydro[..., by_ind]
        domain_data[..., ind] = bx * np.cos(phi_grid) + by * np.sin(phi_grid)
      elif variable == 'Bphi':
        bx_ind, by_ind = var_list_full.index('Bcc1'), var_list_full.index('Bcc2')
        bx, by = domain_hydro[..., bx_ind], domain_hydro[..., by_ind]
        domain_data[..., ind] = -bx * np.sin(phi_grid) + by * np.cos(phi_grid)
      elif variable == 'Bz':
        bz_ind = var_list_full.index('Bcc3')
        domain_data[..., ind] = domain_hydro[..., bz_ind]
      elif variable == 'Jr':
        J1_ind, J2_ind = var_list_full.index('J1'), var_list_full.index('J2')
        J1, J2 = domain_hydro[..., J1_ind], domain_hydro[..., J2_ind]
        domain_data[..., ind] = J1 * np.cos(phi_grid) + J2 * np.sin(phi_grid)
      elif variable == 'Jphi':
        J1_ind, J2_ind = var_list_full.index('J1'), var_list_full.index('J2')
        J1, J2 = domain_hydro[..., J1_ind], domain_hydro[..., J2_ind]
        domain_data[..., ind] = -J1 * np.sin(phi_grid) + J2 * np.cos(phi_grid)
      elif variable == 'Jz':
        J3_ind = var_list_full.index('J3')
        domain_data[..., ind] = domain_hydro[..., J3_ind]
      elif variable in var_list_full:
        var_ind = var_list_full.index(variable)
        domain_data[..., ind] = domain_hydro[..., var_ind]
      elif variable == 'Bmag':
        b1_ind = var_list_full.index('Bcc1')
        for ii in range(3):
          b_ind = b1_ind + ii
          domain_data[..., ind] += domain_hydro[..., b_ind]**2
        domain_data[..., ind] = np.sqrt(domain_data[..., ind])
      elif variable == 'beta':
        b1_ind = var_list_full.index('Bcc1')
        for ii in range(3):
          b_ind = b1_ind + ii
          domain_data[..., ind] += domain_hydro[..., b_ind]**2
        try:
          press_ind = var_list_full.index('press')
          domain_data[..., ind] = domain_hydro[..., press_ind] / (domain_data[..., ind] / 2)
        except Exception as e:
          cs = domain_kwargs.get('cs', 0.05)
          rho_ind = var_list_full.index('rho')
          domain_data[..., ind] = cs**2 * domain_hydro[..., rho_ind] / (domain_data[..., ind] / 2)
      # spherical radial velocity or accretion rate which is just rho * vr (spherical)
      elif variable == 'velrsph' or variable == 'mdotr': 
        # extract velocity components
        vx_ind = var_list_full.index('vel1')
        vy_ind = var_list_full.index('vel2')
        vz_ind = var_list_full.index('vel3')
        vx, vy, vz = domain_hydro[..., vx_ind], domain_hydro[..., vy_ind], domain_hydro[..., vz_ind]
        
        # compute cell-centered coords from faces: (Nmb, mb, 3) -> (Nmb, mb)
        xc = 0.5 * (domain_grid[:, :-1, 0] + domain_grid[:, 1:, 0])
        yc = 0.5 * (domain_grid[:, :-1, 1] + domain_grid[:, 1:, 1])
        zc = 0.5 * (domain_grid[:, :-1, 2] + domain_grid[:, 1:, 2])
        
        # broadcast to 3D grid: (Nmb, mb, mb, mb)
        x = xc[:, :, None, None]
        y = yc[:, None, :, None]
        z = zc[:, None, None, :]
        
        # spherical radial velocity: v_r = (v·r)/|r|
        r_sph = np.sqrt(x**2 + y**2 + z**2)
        vr_sph = (vx*x + vy*y + vz*z) / (r_sph + 1e-20)
        domain_data[..., ind] = vr_sph
        if variable == 'mdotr':
          rho_ind = var_list_full.index('rho')
          domain_data[...,ind] *= domain_hydro[...,rho_ind]

      else:
        print(f"processing {variable} has not yet been implemented! skipping")
      
      if log_flag:
        domain_data[..., ind] = np.log10(np.abs(domain_data[..., ind]))
        
  return domain_data, domain_grid, header

def load_slice_and_variables(ath_file, variables, slice_kwargs):
  with h5py.File(ath_file) as hfp:
    hydro = hfp['hydro']
    xvs = [hfp['x1v'], hfp['x2v'], hfp['x3v']]
    xfs = [hfp['x1f'], hfp['x2f'], hfp['x3f']]

    header = {i: hfp.attrs[i] for i in hfp.attrs.keys()}
    header['VariableNames'] = [i.decode('utf-8') for i in header['VariableNames']]
    
    # load hydro and Jcurrent slices
    slice_hydro, slice_grid = extract_slice(xvs, xfs, hydro, **slice_kwargs)
    
    # extract the variables we wish to plot
    var_list_full = var_list_lookup(header, slice_kwargs['current'])
    slice_data = np.zeros(shape=(*slice_hydro.shape[:-1], len(variables)))
    
    # Determine if we need cylindrical transformations
    cyl_vars = ['velr', 'velphi', 'velz', 'Br', 'Bphi', 'Bz', 'Jr', 'Jphi', 'Jz']
    needs_cyl = any(var.replace('log', '') in cyl_vars for var in variables)
    
    # Compute cylindrical coordinates if needed
    if needs_cyl:
      R_grid, phi_grid = compute_cylindrical_coords(slice_grid, slice_kwargs)
    
    for ind, variableName in enumerate(variables):
      if ('log' in variableName):
        variable = variableName.replace('log', '')
        log_flag = True
      else:
        log_flag = False
        variable = variableName
      
      # Handle cylindrical velocity components
      if variable == 'velr':
        vx_ind, vy_ind = var_list_full.index('vel1'), var_list_full.index('vel2')
        vx, vy = slice_hydro[..., vx_ind], slice_hydro[..., vy_ind]
        slice_data[..., ind] = vx * np.cos(phi_grid) + vy * np.sin(phi_grid)
      elif variable == 'velphi':
        vx_ind, vy_ind = var_list_full.index('vel1'), var_list_full.index('vel2')
        vx, vy = slice_hydro[..., vx_ind], slice_hydro[..., vy_ind]
        slice_data[..., ind] = -vx * np.sin(phi_grid) + vy * np.cos(phi_grid)
      elif variable == 'velz':
        vz_ind = var_list_full.index('vel3')
        slice_data[..., ind] = slice_hydro[..., vz_ind]
      
      # Handle cylindrical magnetic field components
      elif variable == 'Br':
        bx_ind, by_ind = var_list_full.index('Bcc1'), var_list_full.index('Bcc2')
        bx, by = slice_hydro[..., bx_ind], slice_hydro[..., by_ind]
        slice_data[..., ind] = bx * np.cos(phi_grid) + by * np.sin(phi_grid)
      elif variable == 'Bphi':
        bx_ind, by_ind = var_list_full.index('Bcc1'), var_list_full.index('Bcc2')
        bx, by = slice_hydro[..., bx_ind], slice_hydro[..., by_ind]
        slice_data[..., ind] = -bx * np.sin(phi_grid) + by * np.cos(phi_grid)
      elif variable == 'Bz':
        bz_ind = var_list_full.index('Bcc3')
        slice_data[..., ind] = slice_hydro[..., bz_ind]

      # Handle cylindrical current density components
      elif variable == 'Jr':
        J1_ind, J2_ind = var_list_full.index('J1'), var_list_full.index('J2')
        J1, J2 = slice_hydro[..., J1_ind], slice_hydro[..., J2_ind]
        slice_data[..., ind] = J1 * np.cos(phi_grid) + J2 * np.sin(phi_grid)
      elif variable == 'Jphi':
        J1_ind, J2_ind = var_list_full.index('J1'), var_list_full.index('J2')
        J1, J2 = slice_hydro[..., J1_ind], slice_hydro[..., J2_ind]
        slice_data[..., ind] = -J1 * np.sin(phi_grid) + J2 * np.cos(phi_grid)
      elif variable == 'Jz':
        J3_ind = var_list_full.index('J3')
        slice_data[..., ind] = slice_hydro[..., J3_ind]
      
      # Handle Cartesian variables
      elif variable in var_list_full:
        var_ind = var_list_full.index(variable)
        slice_data[..., ind] = slice_hydro[..., var_ind]
      elif variable == 'Bmag':
        b1_ind = var_list_full.index('Bcc1')
        for ii in range(3):
          b_ind = b1_ind + ii
          slice_data[..., ind] += slice_hydro[..., b_ind]**2
        slice_data[..., ind] = np.sqrt(slice_data[..., ind])
      elif variable == 'beta':
        b1_ind = var_list_full.index('Bcc1')
        for ii in range(3):
          b_ind = b1_ind + ii
          slice_data[..., ind] += slice_hydro[..., b_ind]**2
        # adiabatic
        try:
          press_ind = var_list_full.index('press')
          slice_data[..., ind] = slice_hydro[..., press_ind] / (slice_data[..., ind] / 2)
        # isothermal
        except Exception as e:
          cs = slice_kwargs.get('cs', 0.05)
          rho_ind = var_list_full.index('rho')
          slice_data[..., ind] = cs**2 * slice_hydro[..., rho_ind] / (slice_data[..., ind] / 2)
      # spherical radial velocity or accretion rate which is just rho * vr (spherical)
      elif variable == 'velrsph' or variable == 'mdotr': 
        # extract velocity components
        vx_ind = var_list_full.index('vel1')
        vy_ind = var_list_full.index('vel2')
        vz_ind = var_list_full.index('vel3')
        vx, vy, vz = slice_hydro[..., vx_ind], slice_hydro[..., vy_ind], slice_hydro[..., vz_ind]
        
        # compute cell-centered coords from faces: (Nmb, mb+1, 2) -> (Nmb, mb)
        slcdim_1_centers = 0.5 * (slice_grid[:, :-1, 0] + slice_grid[:, 1:, 0])
        slcdim_2_centers = 0.5 * (slice_grid[:, :-1, 1] + slice_grid[:, 1:, 1])
        fixdim_centers = slice_kwargs['loc']
        
        # broadcast to 2D slice: (Nmb, mb, mb)
        slcdim_1 = slcdim_1_centers[:, :, None]
        slcdim_2 = slcdim_2_centers[:, None, :]
        fixdim = fixdim_centers
        
        # spherical radial velocity: v_r = (v·r)/|r|
        r_sph = np.sqrt(slcdim_1**2 + slcdim_2**2 + fixdim**2)
        if slice_kwargs['slice_dim']==0:
          vr_sph = (vx*fixdim + vy*slcdim_1 + vz*slcdim_2) / (r_sph + 1e-20)
        elif slice_kwargs['slice_dim']==1:
          vr_sph = (vx*slcdim_1 + vy*fixdim + vz*slcdim_2) / (r_sph + 1e-20)
        elif slice_kwargs['slice_dim']==2:
          vr_sph = (vx*slcdim_1 + vy*slcdim_2 + vz*fixdim) / (r_sph + 1e-20)
        slice_data[..., ind] = vr_sph
        if variable == 'mdotr':
          rho_ind = var_list_full.index('rho')
          slice_data[...,ind] *= slice_hydro[...,rho_ind]
      else:
        print(f"processing {variable} has not yet been implemented! skipping")
      # Apply log if requested
      if log_flag:
        slice_data[..., ind] = np.log10(np.abs(slice_data[..., ind]))
  
  return slice_data, slice_grid, header

def compute_cylindrical_coords(grid_data, slice_kwargs=None):
  """Compute cylindrical coordinates (R, phi) from Cartesian grid data.
  
  Parameters
  ----------
  grid_data : ndarray
    For slice: (Nmb, mb+1, 2) face coordinates. For domain: (Nmb, mb+1, 3) meshblock faces.
  slice_kwargs : dict, optional
    Contains 'slice_dim' (0=x, 1=y, 2=z) and 'loc' (slice location, default 0.0).
    
  Returns
  -------
  R : ndarray
    Cylindrical radial coordinate. Shape (Nmb, mb, mb) for slice or (Nmb, mb, mb, mb) for domain.
  phi : ndarray
    Azimuthal angle in radians. Same shape as R.
  """
  
  if slice_kwargs is not None and 'slice_dim' in slice_kwargs:
    slice_grid = grid_data
    slice_dim = slice_kwargs['slice_dim']
    loc = slice_kwargs.get('loc', 0.0)
    
    # Compute cell centers from face coordinates
    # slice_grid has shape (Nmb, mb+1, 2) - preserve meshblock dimension!
    coord1_centers = 0.5 * (slice_grid[:, :-1, 0] + slice_grid[:, 1:, 0])  # (Nmb, mb)
    coord2_centers = 0.5 * (slice_grid[:, :-1, 1] + slice_grid[:, 1:, 1])  # (Nmb, mb)
    
    # Create 2D meshgrids for each meshblock
    # Result shapes: (Nmb, mb, mb)
    coord1_2d = coord1_centers[:, :, np.newaxis]  # (Nmb, mb, 1)
    coord2_2d = coord2_centers[:, np.newaxis, :]  # (Nmb, 1, mb)
    coord1_2d = np.broadcast_to(coord1_2d, (coord1_2d.shape[0], coord1_2d.shape[1], coord2_centers.shape[1]))
    coord2_2d = np.broadcast_to(coord2_2d, (coord2_2d.shape[0], coord1_centers.shape[1], coord2_2d.shape[2]))
    
    # Assign Cartesian coordinates based on slice dimension
    if slice_dim == 0:
      # Slice perpendicular to x; coords are (y, z)
      x = loc * np.ones_like(coord1_2d)
      y = coord1_2d
      z = coord2_2d
    elif slice_dim == 1:
      # Slice perpendicular to y; coords are (x, z)
      x = coord1_2d
      y = loc * np.ones_like(coord1_2d)
      z = coord2_2d
    elif slice_dim == 2:
      # Slice perpendicular to z; coords are (x, y)
      x = coord1_2d
      y = coord2_2d
      z = loc * np.ones_like(coord1_2d)
    else:
      raise ValueError(f"Unknown slice_dim: {slice_dim}. Must be 0, 1, or 2.")
    
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return R, phi
    
  else:
    # 3D domain case (unchanged)
    xfs = grid_data
    Nmb = xfs.shape[0]
    mb = xfs.shape[1] - 1
    R_grid = np.zeros((Nmb, mb, mb, mb), dtype=np.float32)
    phi_grid = np.zeros((Nmb, mb, mb, mb), dtype=np.float32)
    
    x1_centers = 0.5 * (xfs[:, :-1, 0] + xfs[:, 1:, 0])
    x2_centers = 0.5 * (xfs[:, :-1, 1] + xfs[:, 1:, 1])
    x = x1_centers[:,:,None,None]
    y = x2_centers[:,None,:,None]
    R_grid = np.sqrt(x**2 + y**2)
    phi_grid = np.arctan2(y, x)
    
    return R_grid, phi_grid

def compute_curl(F1, F2, F3, x1, x2, x3):
    """
    Compute the curl of a 3D vector field F = (F1, F2, F3) given on a grid
    with coordinate arrays x1 (x), x2 (y), x3 (z), where field arrays have shape (nz, ny, nx).
    
    Parameters:
        F1, F2, F3 : ndarray
            Components of the vector field, each of shape (nz, ny, nx).
        x1, x2, x3 : 1D arrays
            Coordinate arrays for x, y, z directions.
            
    Returns:
        (curl_x, curl_y, curl_z) : tuple of ndarrays
            Components of the curl vector field, each of shape (nz, ny, nx).
    """

    # dx = np.gradient(x1)[0]
    # dy = np.gradient(x2)[0]
    # dz = np.gradient(x3)[0]

    # Partial derivatives using correct axis mapping
    dF3_dy = np.gradient(F3, x2, axis=1)
    dF2_dz = np.gradient(F2, x3, axis=0)

    dF1_dz = np.gradient(F1, x3, axis=0)
    dF3_dx = np.gradient(F3, x1, axis=2)

    dF2_dx = np.gradient(F2, x1, axis=2)
    dF1_dy = np.gradient(F1, x2, axis=1)

    
    
    # Curl components (in (nz, ny, nx) order)
    curl_x = dF3_dy - dF2_dz
    curl_y = dF1_dz - dF3_dx
    curl_z = dF2_dx - dF1_dy

    return np.array([curl_x, curl_y, curl_z])

def compute_J(x1v,x2v,x3v,hydro):
    zones = np.array([x1v,x2v,x3v]).transpose((1,2,0))
    Nprims = hydro.shape[0]
    Nmb = hydro.shape[1]
    mb = hydro.shape[-1]
    bx_ind = -3
    by_ind = -2
    bz_ind = -1
    j_data = []
    # iterate through each meshblock and compute j
    for i in range(Nmb):
        j_data.append(compute_curl(hydro[bx_ind,i,...],hydro[by_ind,i,...],hydro[bz_ind,i,...],x1v[i],x2v[i],x3v[i]))
    j_data = np.array(j_data).transpose((1,0,2,3,4))
    return j_data

def compute_equatorial_radial_profile(xfs, field, zmax, Rbins, weights="volume", 
                                     return_std=False, coord_range=None):
    """
    Compute an axisymmetrized equatorial radial profile from Cartesian Athena++ dataset.
    
    Process: (1) select cells near midplane |z| < zmax, (2) compute R = sqrt(x^2 + y^2),
    (3) bin onto radial bins, (4) azimuthally average.

    Parameters
    ----------
    xfs : list of np.ndarray
        Face-centered coordinates [x1f, x2f, x3f], shape (Nmb, mb+1)
        Assumed Cartesian: x1->x, x2->y, x3->z
    field : np.ndarray
        Scalar field to profile, shape (Nmb, mb, mb, mb). Axis order: (z, y, x).
    zmax : float
        Half-thickness of equatorial region. Includes cells with |z| < zmax.
    Rbins : np.ndarray
        Cylindrical radial bin edges, shape (Nr+1,).
    weights : str
        Averaging weights: "volume" (volume-weighted) or "uniform" (simple average).
    return_std : bool
        If True, also return standard deviation profile.
    coord_range : list or None
        Optional spatial restriction [xmin, xmax, ymin, ymax, zmin, zmax].
        Includes meshblocks overlapping this region.

    Returns
    -------
    profile : np.ndarray
        Radial profile, shape (Nr,).
    Rcent : np.ndarray
        Radial bin centers.
    std : np.ndarray, optional
        Standard deviation in each radial bin (if return_std=True).
    """
    if weights not in ["volume", "uniform"]:
        raise ValueError(f"Unknown weighting method: {weights}")
    
    xf, yf, zf = xfs
    xv = 0.5*(xf[:,:-1]+xf[:,1:])
    yv = 0.5*(yf[:,:-1]+yf[:,1:])
    zv = 0.5*(zf[:,:-1]+zf[:,1:])
    R_all, q_all, w_all = [], [], []
    
    for m in range(field.shape[0]):
        # Domain restriction: check meshblock overlap with coord_range
        if coord_range is not None:
            xmin, xmax, ymin, ymax, zmin, zmax_dom = coord_range
            if not ((xf[m,-1] >= xmin and xf[m,0] <= xmax) and 
                   (yf[m,-1] >= ymin and yf[m,0] <= ymax) and 
                   (zf[m,-1] >= zmin and zf[m,0] <= zmax_dom)):
                continue
        
        # Create proper 3D coordinate grids
        Z_3d, Y_3d, X_3d = np.meshgrid(zv[m], yv[m], xv[m], indexing='ij')
        R = np.sqrt(X_3d**2 + Y_3d**2)
        eq_mask = np.abs(Z_3d) < zmax
        
        # Compute weights: volume-weighted or uniform
        if weights == "volume":
            dx, dy, dz = np.diff(xf[m]), np.diff(yf[m]), np.diff(zf[m])
            w = dz[:,None,None] * dy[None,:,None] * dx[None,None,:]
        else:  # uniform
            w = np.ones_like(field[m])
        
        # Flatten selected cells
        R_all.append(R[eq_mask])
        q_all.append(field[m][eq_mask])
        w_all.append(w[eq_mask])
    
    # Concatenate all meshblocks and setup radial bins
    R_all, q_all, w_all = np.concatenate(R_all), np.concatenate(q_all), np.concatenate(w_all)
    Rcent = 0.5 * (Rbins[1:] + Rbins[:-1])
    Nr = len(Rcent)
    profile, std = np.full(Nr, np.nan), (np.full(Nr, np.nan) if return_std else None)
    
    # Radial averaging loop
    for n in range(Nr):
        mask = (R_all >= Rbins[n]) & (R_all < Rbins[n+1])
        if not np.any(mask):
            continue
        qbin, wbin = q_all[mask], w_all[mask]
        qmean = np.sum(qbin * wbin) / np.sum(wbin)  # Weighted mean
        profile[n] = qmean
        if return_std:
            std[n] = np.sqrt(np.sum(wbin * (qbin - qmean)**2) / np.sum(wbin))  # Weighted std
    
    return (profile, Rcent, std) if return_std else (profile,Rcent)

# def compute_origin_centered_profile(xfs, fields, bins, grid_type="spherical", coord_range=None):
#   """
#   Accumulate mass-weighted quantities in bins centered at origin.
  
#   Parameters
#   ----------
#   xfs : list of np.ndarray
#     Face-centered coordinates [x1f, x2f, x3f], shape (Nmb, mb+1).
#   fields : np.ndarray
#     Cell-centered data with rho as first field, shape (Nmb, mb, mb, mb, Nfields).
#   bins : np.ndarray or tuple
#     For "spherical": 1D radial bin edges.
#     For "cylindrical": (R_bins, z_bins).
#     For "spherical_polar": (R_bins, theta_bins) where theta in [0, pi].
#   grid_type : str
#     "spherical" (1D R), "cylindrical" (2D R,z), or "spherical_polar" (2D R,theta).
#   coord_range : list or None
#     Optional [xmin, xmax, ymin, ymax, zmin, zmax] to restrict domain.
  
#   Returns
#   -------
#   mass_accum : np.ndarray
#     Accumulated mass in each bin.
#   quantity_accum : np.ndarray
#     Accumulated ρ*q*dV for each field (excluding rho), shape matches bins.
#   bin_centers : dict
#     Dictionary with 'R' and 'z' or 'theta' depending on grid_type.
#   """
#   if grid_type not in ["spherical", "cylindrical", "spherical_polar"]:
#     raise ValueError(f"grid_type must be 'spherical', 'cylindrical', or 'spherical_polar'")
  
#   xf, yf, zf = xfs
#   xv = 0.5 * (xf[:, :-1] + xf[:, 1:])
#   yv = 0.5 * (yf[:, :-1] + yf[:, 1:])
#   zv = 0.5 * (zf[:, :-1] + zf[:, 1:])
#   Nmb, Nfields = fields.shape[0], fields.shape[-1]
  
#   # setup bins based on grid type
#   if grid_type == "spherical":
#     Rbins = bins
#     Rcent = 0.5 * (Rbins[1:] + Rbins[:-1])
#     Nbins = len(Rcent)
#     mass_accum = np.zeros(Nbins)
#     quantity_accum = np.zeros((Nbins, Nfields - 1))
#     bin_centers = {'R': Rcent}
#   elif grid_type == "cylindrical":
#     Rbins, zbins = bins
#     Rcent = 0.5 * (Rbins[1:] + Rbins[:-1])
#     zcent = 0.5 * (zbins[1:] + zbins[:-1])
#     Nbins = (len(Rcent), len(zcent))
#     mass_accum = np.zeros(Nbins)
#     quantity_accum = np.zeros((*Nbins, Nfields - 1))
#     bin_centers = {'R': Rcent, 'z': zcent}
#   else:  # spherical_polar
#     Rbins, theta_bins = bins
#     Rcent = 0.5 * (Rbins[1:] + Rbins[:-1])
#     theta_cent = 0.5 * (theta_bins[1:] + theta_bins[:-1])
#     Nbins = (len(Rcent), len(theta_cent))
#     mass_accum = np.zeros(Nbins)
#     quantity_accum = np.zeros((*Nbins, Nfields - 1))
#     bin_centers = {'R': Rcent, 'theta': theta_cent}
  
#   # accumulate over meshblocks
#   for m in range(Nmb):
#     # check domain restriction
#     if coord_range is not None:
#       xmin, xmax, ymin, ymax, zmin, zmax = coord_range
#       if not ((xf[m, -1] >= xmin and xf[m, 0] <= xmax) and
#               (yf[m, -1] >= ymin and yf[m, 0] <= ymax) and
#               (zf[m, -1] >= zmin and zf[m, 0] <= zmax)):
#         continue
    
#     # compute 3D coordinate grids
#     Z_3d, Y_3d, X_3d = np.meshgrid(zv[m], yv[m], xv[m], indexing='ij')
    
#     if grid_type == "cylindrical":
#       R_3d = np.sqrt(X_3d**2 + Y_3d**2)
#       coord2_3d = Z_3d
#       bins1, bins2 = Rbins, zbins
#       N1, N2 = len(Rcent), len(zcent)
#     else:  # spherical or spherical_polar
#       R_3d = np.sqrt(X_3d**2 + Y_3d**2 + Z_3d**2)
#       if grid_type == "spherical_polar":
#         coord2_3d = np.arccos(np.clip(Z_3d / (R_3d + 1e-20), -1, 1))  # theta
#         bins1, bins2 = Rbins, theta_bins
#         N1, N2 = len(Rcent), len(theta_cent)
    
#     # compute cell volumes
#     dx, dy, dz = np.diff(xf[m]), np.diff(yf[m]), np.diff(zf[m])
#     dV = dz[:, None, None] * dy[None, :, None] * dx[None, None, :]
    
#     # extract density and compute mass elements
#     rho = fields[m, ..., 0]
#     dm = rho * dV
    
#     # bin cells
#     if grid_type == "spherical":
#       for n in range(len(Rcent)):
#         mask = (R_3d >= Rbins[n]) & (R_3d < Rbins[n + 1])
#         if not np.any(mask):
#           continue
#         mass_accum[n] += np.sum(dm[mask])
#         for f in range(1, Nfields):
#           quantity_accum[n, f - 1] += np.sum(dm[mask] * fields[m, ..., f][mask])
#     else:  # cylindrical or spherical_polar
#       for n1 in range(N1):
#         for n2 in range(N2):
#           mask = ((R_3d >= bins1[n1]) & (R_3d < bins1[n1 + 1]) &
#                   (coord2_3d >= bins2[n2]) & (coord2_3d < bins2[n2 + 1]))
#           if not np.any(mask):
#             continue
#           mass_accum[n1, n2] += np.sum(dm[mask])
#           for f in range(1, Nfields):
#             quantity_accum[n1, n2, f - 1] += np.sum(dm[mask] * fields[m, ..., f][mask])
  
#   return mass_accum, quantity_accum, bin_centers

def compute_spherical_integral(xfs, field, radius, weights="volume", coord_range=None):
  """
  Compute integral of a field within a sphere of given radius from origin.
  
  Parameters
  ----------
  xfs : list of np.ndarray
    Face-centered coordinates [x1f, x2f, x3f], shape (Nmb, mb+1).
  field : np.ndarray
    Scalar field to integrate, shape (Nmb, mb, mb, mb).
  radius : float
    Sphere radius for integration region.
  weights : str
    Integration weights: "volume" (volume-weighted) or "uniform" (count).
  coord_range : list or None
    Optional spatial restriction [xmin, xmax, ymin, ymax, zmin, zmax].
  
  Returns
  -------
  integral : float
    Integrated quantity within the sphere.
  """
  if weights not in ["volume", "uniform"]:
    raise ValueError(f"Unknown weighting method: {weights}")
  
  xf, yf, zf = xfs
  xv = 0.5*(xf[:,:-1]+xf[:,1:])
  yv = 0.5*(yf[:,:-1]+yf[:,1:])
  zv = 0.5*(zf[:,:-1]+zf[:,1:])
  
  integral = 0.0
  
  X_3d = xv[:,:,None,None]
  Y_3d = yv[:,None,:,None]
  Z_3d = zv[:,None,None,:]
  R_3d = np.sqrt(X_3d**2 + Y_3d**2 + Z_3d**2)
  sphere_mask = R_3d <= radius
  
  # Compute weights: volume-weighted or uniform
  if weights == "volume":
    dx, dy, dz = np.diff(xf,axis=-1), np.diff(yf,axis=-1), np.diff(zf,axis=-1)
    w = dz[:,:,None,None] * dy[:,None,:,None] * dx[:,None,None,:]
  else:  # uniform
    w = np.ones_like(field)
  
  # Accumulate integral
  integral += np.sum(field[sphere_mask] * w[sphere_mask])
  
  return integral

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("USAGE: python athena_util_functions.py [files]\nPrints header for each athena file")
        exit()
    
    for i, file in enumerate(sys.argv[1:]):
        with h5py.File(file) as hfp:
            hydro, xvs, xfs = hfp['hydro'], [hfp['x1v'], hfp['x2v'], hfp['x3v']], [hfp['x1f'], hfp['x2f'], hfp['x3f']]
            if i == 0:
                header = {k: hfp.attrs[k] for k in hfp.attrs.keys()}
                header['VariableNames'] = [v.decode('utf-8') for v in header['VariableNames']]
                print(header)
