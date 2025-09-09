import numpy as np
import sys,os,glob
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,script_dir)
import current_computation

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
        currentJ = current_computation.compute_J(xvs[0],xvs[1],xvs[2],hydro)
        hydro = np.concatenate([hydro,currentJ],axis=0)
    faces = np.array(xfs).transpose((1,2,0))
    Nprims = hydro.shape[0]
    Nmb = hydro.shape[1]
    mb = hydro.shape[-1]
    remain_dim = list(set(range(3)) - set([slice_dim]))
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