import numpy as np

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