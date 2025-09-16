import numpy as np
import functools
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
# mpl.use("Agg")
import h5py
import sys,os,glob
athena_dir = "/users/avjoshi2/athena"
sys.path.insert(0, os.path.join(athena_dir,'vis/python'))
import click
import multiprocessing as mp

file_path = os.path.realpath(__file__)
sys.path.insert(0, os.path.basename(file_path))
import athena_slicing

def get_dim(variable="vel2",subvar="vel"):
  if "sq" in variable:
    return '^2'
  else:
    index = variable.find(subvar)
    dim=variable[index + len(subvar)]
    return f'_{dim}'
  
def var_list_lookup(header,current=False):
  var_list = header['VariableNames']  
  if(current):
    var_list_full = var_list+['J1','J2','J3']
    return var_list_full
  else:
    return var_list

def get_label_dictionary(variables):
  label_dict={}
  for variable in variables:
    if "log" in variable:
      log_prepend=r"$\log$ "
    else:
      log_prepend=""
    if 'rho' in variable:
      label_dict[variable]=rf'{log_prepend}$\rho$'
    elif "vel" in variable:
      append_str = get_dim(variable,"vel")
      label_dict[variable]=rf'{log_prepend}$v{append_str}$'
    elif "Bcc" in variable:
      append_str = get_dim(variable,"Bcc")
      label_dict[variable]=rf'{log_prepend}$B{append_str}$'
    elif "Bmag" in variable:
      label_dict[variable]=rf'{log_prepend}$|B|$'
    elif "beta" in variable:
      label_dict[variable]=rf'{log_prepend}$\beta$'
    elif "J" in variable:
      append_str = get_dim(variable,"J")
      label_dict[variable]=rf'{log_prepend}$J{append_str}$'
    else:
      label_dict[variable] = variable
  return label_dict

def generate_plot_kwargs_dict(kwargs):
  # Extract possible per-variable plotting options
  vmin_dict={}
  vmax_dict={}
  cmap_dict={}
  for key in ['vmin', 'vmax', 'cmap']:
    plot_dict = kwargs.pop(key, {})
    if(plot_dict)!={}:
      # if plot kwarg is a dictionary or a number
      try:
        plot_dict = eval(str(plot_dict))
        if type(plot_dict) != dict:
          val = plot_dict
          plot_dict = {var: val for var in variables}
      # if plot kwarg is a string
      except NameError:
        val = plot_dict
        plot_dict = {var: val for var in variables}

      if key == 'vmin':
        vmin_dict = plot_dict
      elif key == 'vmax':
        vmax_dict = plot_dict
      elif key == 'cmap':
        cmap_dict = plot_dict
  return vmin_dict,vmax_dict,cmap_dict

def plot_slice(slice_data,slice_grid,header,ath_file='test',variables="rho", slice_dim=0,loc=0, remain_coord_range=None, length_scale=1, length_scale_label='',output_dir=None,**kwargs):
  """
  Plot a 2D slice of a 3D Athena++ HDF5 dataset using cell interface coordinates.

  Parameters:
  -----------
  slice_data : np.ndarray
      2D slices of hydro variables from meshblocks intersecting the slice location.
      Shape: (Nmb_selected, Nvars, mb, mb).
  slice_grid : np.ndarray
      Corresponding face-centered grid coordinates in the remaining two dimensions.
      Shape: (Nmb_selected, mb+1, 2)  
  ath_file: string
      Name of the dump file for png name
  variables : list
      List of the variables to plot (e.g., "density", "pressure", etc.)
      Should match Nvars from slice_data
  slice_dim : int
      Dimension to slice along: 0 (x1), 1 (x2), or 2 (x3).
  loc : float
      Physical location along `slice_dim` where the slice is taken.
  remain_coord_range : float or None
      Maximum value of the coordinates for the slice
  length_scale : float
      Length scale in code unites to divide the axes values with
  length_scale_label : string
      String for axes labels
  **kwargs : dict
      Additional keyword arguments passed to matplotlib.pyplot.pcolormesh
  """

  
  pid = os.getpid()

  grid_min = np.min(slice_grid,axis=(0,1))
  grid_max = np.max(slice_grid,axis=(0,1))

  xrange = grid_max[0]-grid_min[0]
  yrange = grid_max[1]-grid_min[1]

  aspect_ratio = max(4/9,min(9/4,xrange/yrange))
  if(aspect_ratio<=1):
    width = 5
    height = width/aspect_ratio
  else:
    height = 5
    width = height * aspect_ratio

  # Get simulation time
  time = header['Time']
  # Generate dictionaries of labels and plot kwargs
  label_dict = get_label_dictionary(variables)
  vmin_dict,vmax_dict,cmap_dict = generate_plot_kwargs_dict(kwargs)

  for ind,variableName in enumerate(variables):
    plt.figure(num=pid,clear=True,figsize=(width,height))
    ax = plt.gca()
    if ('log' in variableName):
      variable = variableName.replace('log','')
      log_flag = True
    else:
      log_flag = False
      variable = variableName
    var_ind = variables.index(variableName)
    print(f"{variable} max {np.max(slice_data[...,var_ind]):.2e}, min {np.min(slice_data[...,var_ind]):.2e}")

    # Determine color scale and colormap for this variable
    vmin = vmin_dict.get(variableName, None)
    vmax = vmax_dict.get(variableName, None)
    cmap = cmap_dict.get(variableName, None)

    # Prepare pcolormesh kwargs for this variable
    pcolormesh_kwargs = kwargs.copy()
    if vmin is not None:
      pcolormesh_kwargs['vmin'] = vmin
    if vmax is not None:
      pcolormesh_kwargs['vmax'] = vmax
    if cmap is not None:
      pcolormesh_kwargs['cmap'] = cmap

    for i in range(slice_grid.shape[0]):
        data_toplot=slice_data[i,...,var_ind]
        if(log_flag):
          im=ax.pcolormesh(slice_grid[i,:,0]/length_scale,slice_grid[i,:,1]/length_scale,np.log10(abs(data_toplot)),**pcolormesh_kwargs)
        else:
          im=ax.pcolormesh(slice_grid[i,:,0]/length_scale,slice_grid[i,:,1]/length_scale,data_toplot,**pcolormesh_kwargs)
    
    slice_coord_dims = list(set(range(3))-{slice_dim})
    slice_coords_list = [f"x{i+1}" for i in slice_coord_dims]

    ax.set_xlabel(f"{slice_coords_list[0]}{length_scale_label}")
    ax.set_ylabel(f"{slice_coords_list[1]}{length_scale_label}")
    #plt.colorbar(mesh,label=variableName,pad=0)
    plt.colorbar(im,pad=0)
    remaining_coord_field=f"x{slice_dim+1}"
    #plt.title(f"{label_dict[variables[ind]]} at {remaining_coord_field} t={time:.2f}")
    ax.set_title(f"{label_dict[variables[ind]]} at t={time:.2f}")
    ax.set_xlim(np.array(remain_coord_range[:2])/length_scale)
    ax.set_ylim(np.array(remain_coord_range[2:])/length_scale)
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    if(remain_coord_range==None):
      plt_filename = f"{os.path.splitext(ath_file)[0]}_slice_{variableName}_{str.join('',slice_coords_list)}_{remaining_coord_field}_{loc}.png"
    else:
      plt_filename = f"{os.path.splitext(ath_file)[0]}_slice_{variableName}_{str.join('',slice_coords_list)}_{np.max(remain_coord_range):.1e}_{remaining_coord_field}_{loc}.png"

    if output_dir!=None:
      plt_filename = os.path.join(output_dir,os.path.basename(plt_filename))
    
    plt.savefig(plt_filename,bbox_inches="tight",dpi=200)
    plt.clf()
  plt.close()

def plot_panel_slice(slice_data,slice_grid,header,ath_file='test',variables="rho", slice_dim=0,loc=0, remain_coord_range=None, length_scale=1, length_scale_label='',output_dir=None,**kwargs):
  """
  Plot a 2D slice of a 3D Athena++ HDF5 dataset using cell interface coordinates.

  Parameters:
  -----------
  slice_data : np.ndarray
      2D slices of hydro variables from meshblocks intersecting the slice location.
      Shape: (Nmb_selected, Nvars, mb, mb).
  slice_grid : np.ndarray
      Corresponding face-centered grid coordinates in the remaining two dimensions.
      Shape: (Nmb_selected, mb+1, 2)  
  ath_file: string
      Name of the dump file for png name
  variables : list
      List of the variables to plot (e.g., "density", "pressure", etc.)
      Should match Nvars from slice_data
  slice_dim : int
      Dimension to slice along: 0 (x1), 1 (x2), or 2 (x3).
  loc : float
      Physical location along `slice_dim` where the slice is taken.
  remain_coord_range : float or None
      Maximum value of the coordinates for the slice
  length_scale : float
      Length scale in code unites to divide the axes values with
  length_scale_label : string
      String for axes labels
  **kwargs : dict
      Additional keyword arguments passed to matplotlib.pyplot.pcolormesh
  """

  
  pid = os.getpid()

  grid_min = np.min(slice_grid,axis=(0,1))
  grid_max = np.max(slice_grid,axis=(0,1))

  xrange = grid_max[0]-grid_min[0]
  yrange = grid_max[1]-grid_min[1]

  # aspect ratio of each subplot
  #aspect_ratio = max(4/9,min(9/4,xrange/yrange))
  aspect_ratio = xrange/yrange
  if(aspect_ratio<=1):
    width = 4
    height = width/aspect_ratio
  else:
    height = 4
    width = height * aspect_ratio

  n_vars = len(variables)
  nrows = max(n_vars//4,1)
  ncols = min(n_vars,4)
  fig, axes = plt.subplots(nrows, ncols, figsize=(1.2*(ncols)*width, nrows*height), sharey=True, sharex=True, squeeze=False)

  # Get simulation time
  time = header['Time']
  # Generate dictionaries of labels and plot kwargs
  label_dict = get_label_dictionary(variables)
  vmin_dict,vmax_dict,cmap_dict = generate_plot_kwargs_dict(kwargs)

  for ind,variableName in enumerate(variables):
    ax = axes.flatten()[ind]
    if ('log' in variableName):
      variable = variableName.replace('log','')
      log_flag = True
    else:
      log_flag = False
      variable = variableName
    var_ind = variables.index(variableName)
    print(f"{variable} max {np.max(slice_data[...,var_ind]):.2e}, min {np.min(slice_data[...,var_ind]):.2e}")

    # Determine color scale and colormap for this variable
    vmin = vmin_dict.get(variableName, None)
    vmax = vmax_dict.get(variableName, None)
    cmap = cmap_dict.get(variableName, None)

    # Prepare pcolormesh kwargs for this variable
    pcolormesh_kwargs = kwargs.copy()
    if vmin is not None:
      pcolormesh_kwargs['vmin'] = vmin
    if vmax is not None:
      pcolormesh_kwargs['vmax'] = vmax
    if cmap is not None:
      pcolormesh_kwargs['cmap'] = cmap

    for i in range(slice_grid.shape[0]):
        data_toplot=slice_data[i,...,var_ind]
        if(log_flag):
          im=ax.pcolormesh(slice_grid[i,:,0]/length_scale,slice_grid[i,:,1]/length_scale,np.log10(abs(data_toplot)),**pcolormesh_kwargs)
        else:
          im=ax.pcolormesh(slice_grid[i,:,0]/length_scale,slice_grid[i,:,1]/length_scale,data_toplot,**pcolormesh_kwargs)
    
    slice_coord_dims = list(set(range(3))-{slice_dim})
    slice_coords_list = [f"x{i+1}" for i in slice_coord_dims]

    if(ind//4==nrows-1):
      ax.set_xlabel(f"{slice_coords_list[0]}{length_scale_label}")
    if(ind%4==0):
      ax.set_ylabel(f"{slice_coords_list[1]}{length_scale_label}")
    ax.set_title(f"{label_dict[variables[ind]]}")
    cbar = fig.colorbar(im,ax=ax,pad=0)

    remaining_coord_field=f"x{slice_dim+1}"
    ax.set_xlim(np.array(remain_coord_range[:2])/length_scale)
    ax.set_ylim(np.array(remain_coord_range[2:])/length_scale)
  
  plt.suptitle(f"Panel at t={time:.2f}",y=(1.0+0.02*ncols))
  plt.subplots_adjust(left=0,right=1,bottom=0,top=1)

  variableNames = "-".join(variables)
  if(remain_coord_range==None):
    plt_filename = f"{os.path.splitext(ath_file)[0]}_slice_{variableNames}_{str.join('',slice_coords_list)}_{remaining_coord_field}_{loc}.png"
  else:
    plt_filename = f"{os.path.splitext(ath_file)[0]}_slice_{variableNames}_{str.join('',slice_coords_list)}_{np.max(remain_coord_range):.1e}_{remaining_coord_field}_{loc}.png"
  
  if output_dir!=None:
    plt_filename = os.path.join(output_dir,os.path.basename(plt_filename))
  
  plt.savefig(plt_filename,bbox_inches="tight",dpi=200)
  plt.clf()
  plt.close()
  
def load_slice_and_variables(ath_file,variables,slice_kwargs):
  hfp = h5py.File(ath_file)
  hydro = hfp['hydro']
  xvs = [hfp['x1v'],hfp['x2v'],hfp['x3v']]
  xfs = [hfp['x1f'],hfp['x2f'],hfp['x3f']]

  header={i:hfp.attrs[i] for i in hfp.attrs.keys()}
  header['VariableNames'] = [i.decode('utf-8') for i in header['VariableNames']]
  # load hydro and Jcurrent slices
  slice_hydro,slice_grid = athena_slicing.extract_slice(xvs,xfs,hydro,**slice_kwargs)
  # extract the variables we wish to plot
  var_list_full=var_list_lookup(header,slice_kwargs['current'])
  slice_data = np.zeros(shape=(*slice_hydro.shape[:-1],len(variables)))
  for ind,variableName in enumerate(variables):
    if ('log' in variableName):
      variable = variableName.replace('log','')
      log_flag = True
    else:
      log_flag = False
      variable = variableName
    if variable in var_list_full:
      var_ind = var_list_full.index(variable)
      slice_data[...,ind] = slice_hydro[...,var_ind]
    elif variable=='Bmag':
      b1_ind = var_list_full.index('Bcc1')
      for ii in range(3):
        b_ind = b1_ind+ii
        slice_data[...,ind]+=slice_hydro[...,b_ind]**2
      slice_data[...,ind]=np.sqrt(slice_data[...,ind])
    elif variable=='beta':
      b1_ind = var_list_full.index('Bcc1')
      for ii in range(3):
        b_ind = b1_ind+ii
        slice_data[...,ind]+=slice_hydro[...,b_ind]**2
      # adiabatic
      try:
        press_ind = var_list_full.index('press')
        slice_data[...,ind]=slice_hydro[...,press_ind]/(slice_data[...,ind]/2)
      # isothermal
      except Exception as e:
        if ('cs' in slice_kwargs.keys()):
          cs = slice_kwargs['cs']
        else:
          cs = 0.05
        rho_ind = var_list_full.index('rho')
        slice_data[...,ind]=cs**2 * slice_hydro[...,rho_ind]/(slice_data[...,ind]/2)
    else:
      print(f"processing {variable} has not yet been implemented! skipping")
  return slice_data,slice_grid,header

def process_file(ath_file,plot_type,slice_kwargs,plot_kwargs):
  """
  Loads the slice and grid structures, and computes all the variables requested
  """

  slice_data,slice_grid,header=load_slice_and_variables(ath_file,plot_kwargs['variables'],slice_kwargs)  

  print(f"Processed {ath_file}")
  if(plot_type=='slice'):
    plot_slice(slice_data,slice_grid,header,ath_file,**plot_kwargs)
  elif(plot_type=='panel'):
    plot_panel_slice(slice_data,slice_grid,header,ath_file,**plot_kwargs)
  # # elif(plot_type=='project'):
  # #   print("still need to write projection plot code")  
  print(f"Plotted: {ath_file}")
  del(slice_data);del(slice_grid)

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('files_and_kwargs', nargs=-1,type=click.UNPROCESSED)
@click.option("--plot_type",default='slice',type=str,help="plot type (slice,panel,projection)")
@click.option('--num_cores_multiplier',default=0.95,type=float,help='multiplier to set number of cores')
@click.option('--parallel','-p',is_flag=True,help='flag to process the files in parallel')
@click.option("--variables",'-v',default=['rho'],help='variables to plot',multiple=True)
@click.option("--slice_dim",'-d', default=0,help="dimension (x1=0,x2=1,x3=2) over which to slice")
@click.option("--loc",default=0,type=float,help="Location to slice along slice_dim")
@click.option("--slice_domain",default=None, help='domain over which to plot the slice x1min,x1max,x2min,x2max default over entire domain')
@click.option("--length_scale", default=1, type=float, help="Length scale for scaling slice coordinates")
@click.option("--length_scale_label",default="",type=str, help="label for clarifying length scale in plots")
@click.option("--output_dir",default=None,help="Location to output the plots")
def load_and_plot(files_and_kwargs,plot_type,num_cores_multiplier,parallel,variables,slice_dim,loc,slice_domain,length_scale,length_scale_label,output_dir):
  #separate the files to analyze from the matplotlib kwargs
  files = [i for i in files_and_kwargs if os.path.exists(i)]
  if files == []:
    print("No files found! Exiting")
    exit(1)
  else:
    print(f"Plotting {len(files)} files")
  plot_kwargs_list = [i for i in files_and_kwargs if not os.path.exists(i)]
  plot_kwargs = {}
  i=0
  while i < (len(plot_kwargs_list)):
    st=plot_kwargs_list[i].strip('-')
    if '=' in st:
      st,val = st.split('=')
      plot_kwargs[st]=val
      i+=1
    else:
      plot_kwargs[st] = (plot_kwargs_list[i+1])
      i+=2
    try:
      int(plot_kwargs[st])
      plot_kwargs[st] = eval(plot_kwargs[st])
    except ValueError:
      pass

  #if a specific location for the orthogonal coordinate is given, then extract only that slice in the reader
  # Validate slice_coords
  if(slice_dim not in set(range(3))):
    print(f"Specify a valid axis to slice! slice_dim={slice_dim} valid numbers: x1=0, x2=1, x3=2")
    exit()

  # set the slice bounds if specified
  if (slice_domain != None):
    # Regex to match integers, decimals, and scientific notation with optional sign
    pattern = r'[+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?'
    slice_domain = [eval(i) for i in re.findall(pattern, slice_domain)]

  slice_kwargs={'slice_dim':slice_dim,'loc':loc,'remain_coord_range':slice_domain,'current':any(s.startswith('J') for s in list(variables))}

  plot_kwargs_sub={'variables':list(variables),'slice_dim':slice_dim,'loc':loc,'length_scale':length_scale,'length_scale_label':length_scale_label,'output_dir':output_dir}
  if (slice_domain != None):
    plot_kwargs_sub['remain_coord_range']=slice_domain
  plot_kwargs = plot_kwargs_sub | plot_kwargs

  if parallel:
    num_procs = int(mp.cpu_count()*num_cores_multiplier)
    print(f"launching in parallel across {num_procs} processors")
    with mp.Pool(processes = num_procs) as pool:
      func = functools.partial(process_file,plot_type=plot_type,slice_kwargs=slice_kwargs,plot_kwargs=plot_kwargs)
      pool.map(func,files,chunksize=max(1,int(len(files)//num_procs)))
  else:
    for ath_file in files:
      process_file(ath_file,plot_type,slice_kwargs,plot_kwargs)


if __name__ == "__main__":
    load_and_plot()
