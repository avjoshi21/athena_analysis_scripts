import numpy as np

def get_subscript(variable="vel2",subvar="vel"):
  if "sq" in variable:
    return '^2'
  # elif
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
      append_str = get_subscript(variable,"vel")
      label_dict[variable]=rf'{log_prepend}$v{append_str}$'
    elif "Bcc" in variable:
      append_str = get_subscript(variable,"Bcc")
      label_dict[variable]=rf'{log_prepend}$B{append_str}$'
    elif "Bmag" in variable:
      label_dict[variable]=rf'{log_prepend}$|B|$'
    elif "beta" in variable:
      label_dict[variable]=rf'{log_prepend}$\beta$'
    elif "J" in variable:
      append_str = get_subscript(variable,"J")
      label_dict[variable]=rf'{log_prepend}$J{append_str}$'
    else:
      label_dict[variable] = variable
  return label_dict

def generate_plot_kwargs_dict(variables,kwargs):
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

def get_vmin_vmax_cmap(vmin_dict,vmax_dict,cmap_dict,variableName,var_ind,slice_data,log_flag):
  # Determine color scale and colormap for this variable
  vmin = vmin_dict.get(variableName, None)
  vmax = vmax_dict.get(variableName, None)
  cmap = cmap_dict.get(variableName, 'turbo')

  # **ADD THIS WARNING AND AUTO-SCALING BLOCK:**
  if vmin is None or vmax is None:
    import warnings
    # Compute global min/max across all meshblocks
    if log_flag:
      global_min = np.min(np.log10(np.abs(slice_data[..., var_ind]) + 1e-50))
      global_max = np.max(np.log10(np.abs(slice_data[..., var_ind]) + 1e-50))
    else:
      global_min = np.min(slice_data[..., var_ind])
      global_max = np.max(slice_data[..., var_ind])
    
    if (vmin is None) and (vmax is None) and (cmap == 'seismic'):
      # since it's a divergent colormap you probably want it centered around zero.
      warnings.warn(f"Using global minimum and maxima.",UserWarning)
      max_val = max(abs(global_max),abs(global_min))
      vmin = -max_val
      vmax = max_val

    elif vmin is None:
      vmin = global_min
      warnings.warn(f"vmin not specified for '{variableName}'. Using global minimum: {vmin:.3e}. "
                    f"Consider setting explicit limits to avoid per-meshblock scaling artifacts.",
                    UserWarning)
    elif vmax is None:
      vmax = global_max
      warnings.warn(f"vmax not specified for '{variableName}'. Using global maximum: {vmax:.3e}. "
                    f"Consider setting explicit limits to avoid per-meshblock scaling artifacts.",
                    UserWarning)
    # print(vmin,vmax,cmap);exit()
  return vmin,vmax,cmap