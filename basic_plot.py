import numpy as np
import functools
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
mpl.use("Agg")
import h5py
import sys,os,glob
home = os.environ["HOME"]
try:
  athena_dir = os.path.join(home,"athena")
  sys.path.insert(0, os.path.join(athena_dir,'vis/python'))
  import athena_read # pyright: ignore[reportMissingImports]
except Exception as e:
  athena_dir = os.path.join(home,"Documents/athena")
  sys.path.insert(0, os.path.join(athena_dir,'vis/python'))
  import athena_read # pyright: ignore[reportMissingImports]
import click
import multiprocessing as mp

def get_dim(variable="vel2",subvar="vel"):
  if "sq" in variable:
    return '^2'
  else:
    index = variable.find(subvar)
    dim=variable[index + len(subvar)]
    return f'_{dim}'

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
    else:
      label_dict[variable] = variable
  return label_dict

def plot_slice(data, ath_file='test',variables="density", slice_coords="x1x2",slice_domain=None,remaining_coord=None, length_scale=1, length_scale_label='',output_dir=None,**kwargs):
  """
  Plot a 2D slice of a 3D Athena++ HDF5 dataset using cell interface coordinates.

  Parameters:
  -----------
  data : dict
      Dictionary returned by athena_read.athdf()
  variables : list
      List of the variables to plot (e.g., "density", "pressure", etc.)
  slice_coords : str
      String indicating the 2D plane to slice (e.g., "x1x2", "x1x3", or "x2x3")
  remaining_coord : float or None
      Location to slice along the orthogonal axis to `slice_coords`. If None, average along that axis.
  **kwargs : dict
      Additional keyword arguments passed to matplotlib.pyplot.pcolormesh
  """

  if ('cs' in kwargs):
    cs = kwargs['cs']
  else:
    cs = 0.05
  
  pid = os.getpid()

  # Validate slice_coords
  valid_slices = {"x1x2", "x1x3", "x2x3"}
  if slice_coords not in valid_slices:
      raise ValueError(f"slice_coords must be one of {valid_slices}")
  
  # modify slice_coords to be a list
  slice_coords_list = [slice_coords[:2],slice_coords[2:]]    

  # Map coordinates to axis indices
  axis_map = {'x1': 0, 'x2': 1, 'x3': 2}
  dims = [axis_map[dim] for dim in slice_coords_list]

  # Get coordinate interface arrays
  xfs = [data[f'{dim}f']/length_scale for dim in slice_coords_list]

  xgrids = np.meshgrid(*xfs, indexing='ij')
  
  xrange = xfs[0][-1]-xfs[0][0]
  yrange = xfs[1][-1]-xfs[1][0]

  aspect_ratio = max(4/9,min(9/4,xrange/yrange))
  if(aspect_ratio<=1):
    width = 5
    height = width/aspect_ratio
  else:
    height = 5
    width = height * aspect_ratio


  # Get simulation time
  time = data['Time']

  label_dict = get_label_dictionary(variables)

  # Extract possible per-variable plotting options
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

  for ind,variableName in enumerate(variables):
    plt.figure(num=pid,clear=True,figsize=(width,height))
    if ('log' in variableName):
      variable = variableName.replace('log','')
      log_flag = True
    else:
      log_flag = False
      variable = variableName
      
    if variable == 'Bmag':
      var_data = np.zeros(data['Bcc1'].shape)
      for bfield in ['Bcc1','Bcc2','Bcc3']:
        var_data += data[bfield]**2
      var_data = np.sqrt(var_data)
    elif variable =='beta':
      bsq = np.zeros(data['Bcc1'].shape)
      for bfield in ['Bcc1','Bcc2','Bcc3']:
        bsq += data[bfield]**2
      press = cs**2 * data['rho']
      var_data = press/(bsq/2)
    else:
      # Extract variable data
      var_data = data[variable]
    
    #transpose the data since athena stores it as k,j,i
    var_data = var_data.transpose(2,1,0)

    # Determine slicing or averaging
    axis_to_reduce = list(set(range(3)) - set(dims))[0]  # remaining axis
    #if the remaining coordinate was not meant to be averaged then the reader should have loaded only the slice in that direction
    if remaining_coord is not None:
      if var_data.shape[axis_to_reduce]!=1:
        remaining_coord_idx = np.argmin(abs(data[f'x{axis_to_reduce+1}v']-remaining_coord))
      else:
        remaining_coord_idx = 0
      slice_obj = [slice(None)] * 3
      slice_obj[axis_to_reduce] = remaining_coord_idx
      var_slice = var_data[tuple(slice_obj)]
    else:
      var_slice = np.mean(var_data, axis=axis_to_reduce)

    # Transpose data to match x-y meshgrid orientation
    permute_order = dims
    var_slice = np.transpose(var_slice, axes=np.argsort(permute_order))
    print(f"{variable} max {np.max(var_slice):.2e}, min {np.min(var_slice):.2e}")

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

    # Plot
    if(log_flag):
      mesh = plt.pcolormesh(xgrids[0], xgrids[1], np.log10(np.abs(var_slice)), shading='auto', **pcolormesh_kwargs)
    else:
      mesh = plt.pcolormesh(xgrids[0], xgrids[1], var_slice, shading='auto', **pcolormesh_kwargs)
    plt.xlabel(f"{slice_coords_list[0]}{length_scale_label}")
    plt.ylabel(f"{slice_coords_list[1]}{length_scale_label}")
    #plt.colorbar(mesh,label=variableName,pad=0)
    plt.colorbar(mesh,pad=0)
    remaining_coord_field=f"x{axis_to_reduce+1}_avg" if remaining_coord==None else f"x{axis_to_reduce+1}_{remaining_coord}"
    #plt.title(f"{label_dict[variables[ind]]} at {remaining_coord_field} t={time:.2f}")
    plt.title(f"{label_dict[variables[ind]]} at t={time:.2f}")
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    if(slice_domain==None):
      plt_filename = f"{os.path.splitext(ath_file)[0]}_slice_{variableName}_{str.join('',slice_coords)}_{remaining_coord_field}.png"
    else:
      plt_filename = f"{os.path.splitext(ath_file)[0]}_slice_{variableName}_{str.join('',slice_coords)}_{slice_domain:.1e}_{remaining_coord_field}.png"

    if output_dir!=None:
      plt_filename = os.path.join(output_dir,os.path.basename(plt_filename))
    plt.savefig(plt_filename,bbox_inches="tight",dpi=200)
    plt.clf()
  plt.close()

def plot_panel_slice(data,ath_file='test',variables="density",slice_coords="x1x2",slice_domain=None,remaining_coord=None, length_scale=1, length_scale_label='',output_dir=None,**kwargs):
  """
  Plot a 2D slice of a 3D Athena++ HDF5 dataset using cell interface coordinates.

  Parameters:
  -----------
  data : dict
    Dictionary returned by athena_read.athdf()
  variables : list
    List of the variables to plot (e.g., "density", "pressure", etc.)
  slice_coords : str
    String indicating the 2D plane to slice (e.g., "x1x2", "x1x3", or "x2x3")
  remaining_coord : float or None
    Location to slice along the orthogonal axis to `slice_coords`. If None, average along that axis.
  **kwargs : dict
    Additional keyword arguments for plotting.
    Special keys:
      - vmin: dict mapping variable names to minimum values for color scale
      - vmax: dict mapping variable names to maximum values for color scale
      - cmap: dict mapping variable names to matplotlib colormaps
    Other kwargs are passed to pcolormesh.
  """


  if ('cs' in kwargs):
    cs = kwargs['cs']
  else:
    cs = 0.05

  pid = os.getpid()

  # Validate slice_coords
  valid_slices = {"x1x2", "x1x3", "x2x3"}
  if slice_coords not in valid_slices:
    raise ValueError(f"slice_coords must be one of {valid_slices}")

  # Modify slice_coords to be a list
  slice_coords_list = [slice_coords[:2], slice_coords[2:]]

  # Map coordinates to axis indices
  axis_map = {'x1': 0, 'x2': 1, 'x3': 2}
  dims = [axis_map[dim] for dim in slice_coords_list]

  # Get coordinate interface arrays
  xfs = [data[f'{dim}f']/length_scale for dim in slice_coords_list]
  xgrids = np.meshgrid(*xfs, indexing='ij')

  # Get simulation time
  time = data['Time']

  xrange = (xfs[0][-1]-xfs[0][0])
  yrange = (xfs[1][-1]-xfs[1][0])

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

  # Determine slicing or averaging axis
  axis_to_reduce = list(set(range(3)) - set(dims))[0]  # remaining axis
  remaining_coord_field = f"x{axis_to_reduce + 1}_avg" if remaining_coord is None else f"x{axis_to_reduce + 1}_{remaining_coord}"

  # Extract possible per-variable plotting options
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
  # assign plot filename early on in case we want to skip plotting if it exists already.
  variableNames = "-".join(variables)
  if(slice_domain==None):
    plt_filename = f"{os.path.splitext(ath_file)[0]}_panel_{variableNames}_{str.join('',slice_coords)}_{remaining_coord_field}.png"
  else:
    plt_filename = f"{os.path.splitext(ath_file)[0]}_panel_{variableNames}_{str.join('',slice_coords)}_{slice_domain:.1e}_{remaining_coord_field}.png"
  if output_dir!=None:
    plt_filename = os.path.join(output_dir,os.path.basename(plt_filename))
  
  # if(os.path.exists(plt_filename)):
  #   return
  label_dict = get_label_dictionary(variables)

  for ind,variableName in enumerate(variables):
    ax = axes.flatten()[ind]
    # if(aspect_ratio)
    # Check for log flag in variable name
    if 'log' in variableName:
      variable = variableName.replace('log', '')
      log_flag = True
    else:
      variable = variableName
      log_flag = False

    if variable == 'Bmag':
      var_data = np.zeros(data['Bcc1'].shape)
      for bfield in ['Bcc1','Bcc2','Bcc3']:
        var_data += data[bfield]**2
      var_data = np.sqrt(var_data)
    elif variable =='beta':
      bsq = np.zeros(data['Bcc1'].shape)
      for bfield in ['Bcc1','Bcc2','Bcc3']:
        bsq += data[bfield]**2
      press = cs**2 * data['rho']
      var_data = press/(bsq/2)
    else:
      # Extract variable data
      var_data = data[variable]

    # transpose the data since athena stores it as k,j,i
    var_data = var_data.transpose(2, 1, 0)

    # Slice or average
    if remaining_coord is not None:
      if var_data.shape[axis_to_reduce] != 1:
        remaining_coord_idx = np.argmin(abs(data[f'x{axis_to_reduce + 1}v'] - remaining_coord))
      else:
        remaining_coord_idx = 0
      slice_obj = [slice(None)] * 3
      slice_obj[axis_to_reduce] = remaining_coord_idx
      var_slice = var_data[tuple(slice_obj)]
    else:
      var_slice = np.mean(var_data, axis=axis_to_reduce)

    # Transpose data to match x-y meshgrid orientation
    permute_order = dims
    var_slice = np.transpose(var_slice, axes=np.argsort(permute_order))

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
    
    print(f"{variable} max {np.max(var_slice):.2e}, min {np.min(var_slice):.2e}")
    # Plot
    if log_flag:
      # Use log10(abs(var_slice)) but handle zeros by masking
      safe_data = np.abs(var_slice)
      safe_data[safe_data == 0] = np.nan
      mesh = ax.pcolormesh(xgrids[0], xgrids[1], np.log10(safe_data), shading='auto', **pcolormesh_kwargs)
    else:
      mesh = ax.pcolormesh(xgrids[0], xgrids[1], var_slice, shading='auto', **pcolormesh_kwargs)

    if(ind%4==0):
      ax.set_ylabel(slice_coords_list[1]+length_scale_label)
    if(ind//4==nrows-1):
      ax.set_xlabel(slice_coords_list[0]+length_scale_label)
    ax.set_title(f"{label_dict[variables[ind]]}")

    # Colorbar for each subplot
    cbar = fig.colorbar(mesh, ax=ax, pad=0)
    # cbar.set_label(variableName)

  plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0.1)
  #plt.suptitle(f"Panel at {remaining_coord_field} t={time:.2f}",y=1.02)
  plt.suptitle(f"Panel at t={time:.2f}",y=1.02)
  plt.savefig(plt_filename,bbox_inches="tight",dpi=200)
  plt.close(fig) 

def process_file(ath_file,plot_type,athdf_args,plot_args):
  ath_data = athena_read.athdf(ath_file, **athdf_args)
  print(f"Processed {ath_file}")
  if(plot_type=='slice'):
    plot_slice(ath_data, ath_file=ath_file, **plot_args)
  elif(plot_type=='panel'):
    plot_panel_slice(ath_data, ath_file=ath_file, **plot_args)
  elif(plot_type=='project'):
    print("still need to write projection plot code")  
  print(f"Plotted: {ath_file}")
  del(ath_data)

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('files_and_kwargs', nargs=-1,type=click.UNPROCESSED)
@click.option("--plot_type",default='slice',type=str,help="plot type (slice,panel,projection)")
@click.option('--num_cores_multiplier',default=0.95,type=float,help='multiplier to set number of cores')
@click.option('--parallel','-p',is_flag=True,help='flag to process the files in parallel')
@click.option("--fast_restrict",is_flag=True,help='fast_restrict flag for athena reader, see wiki')
@click.option("--level",'-l',default=0,help='refinement level to generate the dump')
@click.option("--variables",'-v',default=['rho'],help='variables to plot',multiple=True)
@click.option("--slice_coords",default="x1x2",help='one of x1x2, x2x3 or x1x3 to plot the two dimensions')
@click.option("--slice_domain",default=None, help='domain over which to plot the slice x1min,x1max,x2min,x2max default over entire domain')
@click.option("--remaining_coord",default=None,type=float,help="Location to slice along orthogonal axis to 'slice_coords'. If None, average along that axis. NOTE athdf reader has some issues when setting it to exactly 0. Use a very small number instead")
@click.option("--length_scale", default=1, type=float, help="Length scale for scaling slice coordinates")
@click.option("--length_scale_label",default="",type=str, help="label for clarifying length scale in plots")
@click.option("--output_dir",default=None,help="Location to output the plots")
def load_and_plot(files_and_kwargs,plot_type,num_cores_multiplier,parallel,fast_restrict,level,variables,slice_coords,slice_domain,remaining_coord,length_scale,length_scale_label,output_dir):
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

  athdf_args={}
  athdf_args['fast_restrict']=fast_restrict
  athdf_args['quantities']=np.unique([i.replace('log','') for i in variables])
  if('Bmag' in athdf_args['quantities']):
    athdf_args['quantities']=np.delete(athdf_args['quantities'],np.where(athdf_args['quantities']=='Bmag'))
    athdf_args['quantities']=np.append(athdf_args['quantities'],['Bcc1','Bcc2','Bcc3'])
# only works for isothermal
  if('beta' in athdf_args['quantities']):
    athdf_args['quantities']=np.delete(athdf_args['quantities'],np.where(athdf_args['quantities']=='beta'))
    athdf_args['quantities']=np.append(athdf_args['quantities'],['Bcc1','Bcc2','Bcc3','rho'])
  athdf_args['level'] = level
  #if a specific location for the orthogonal coordinate is given, then extract only that slice in the reader
  if(remaining_coord!=None):
    # Validate slice_coords
    valid_slices = {"x1x2", "x1x3", "x2x3"}
    if slice_coords not in valid_slices:
      raise ValueError(f"slice_coords must be one of {valid_slices}")
    
    # modify slice_coords to be a list
    slice_coords_list = [slice_coords[:2],slice_coords[2:]]

    # set the slice bounds if specified
    if (slice_domain != None):
      # Regex to match integers, decimals, and scientific notation with optional sign
      pattern = r'[+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?'
      slice_domain = [eval(i) for i in re.findall(pattern, slice_domain)]
      for i in range(2):
        athdf_args[f"{slice_coords_list[i]}_min"]=slice_domain[2*i]  
        athdf_args[f"{slice_coords_list[i]}_max"]=slice_domain[2*i+1]

    # Map coordinates to axis indices
    axis_map = {'x1': 0, 'x2': 1, 'x3': 2}
    dims = [axis_map[dim] for dim in slice_coords_list]
    axis_to_reduce = list(set(range(3)) - set(dims))[0]  # remaining axis
    if remaining_coord==0:
      print("athena's athdf reader does not return a dataset if a slice is requested at exactly 0. Use a very small number instead")
      exit(1)
    athdf_args[f'x{axis_to_reduce+1}_min']=remaining_coord
    athdf_args[f'x{axis_to_reduce+1}_max']=remaining_coord
  plot_args={'variables':list(variables),'slice_coords':slice_coords,'remaining_coord':remaining_coord,'length_scale':length_scale,'length_scale_label':length_scale_label,'output_dir':output_dir}
  if (slice_domain != None):
    plot_args['slice_domain']=max(slice_domain)
  plot_args = plot_args | plot_kwargs

  if parallel:
    num_procs = int(mp.cpu_count()*num_cores_multiplier)
    print(f"launching in parallel across {num_procs} processors")
    with mp.Pool(processes = num_procs) as pool:
      func = functools.partial(process_file,plot_type=plot_type,athdf_args=athdf_args,plot_args=plot_args)
      pool.map(func,files,chunksize=max(1,int(len(files)//num_procs)))
  else:
    for ath_file in files:
      process_file(ath_file,plot_type,athdf_args,plot_args)


if __name__ == "__main__":
    load_and_plot()
