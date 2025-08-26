import pandas as pd
import matplotlib.pyplot as plt
import math
import click

def read_athena_hst(filename,restart=True):
  # Skip the first line (file info), use second line as header
  
  # Now read the data using pandas with correct headers
  if(restart):
    header_line = "# [1]=time     [2]=dt       [3]=mass     [4]=1-mom    [5]=2-mom    [6]=3-mom    [7]=1-KE     [8]=2-KE     [9]=3-KE     [10]=1-ME    [11]=2-ME    [12]=3-ME    [13]=divB    [14]=JetM    [15]=JetE    [16]=JetKE   [17]=JetBE   [18]=JetThE  [19]=JetPow"
    header_line=header_line.strip().strip('#')
    headers = header_line.split()
    headers = [i.split('=')[-1] for i in headers]
    data = pd.read_csv(filename, sep=r'\s+',names=headers)
  else:
    with open(filename, 'r') as f:
      f.readline()  # skip first line
      header_line = f.readline().strip().strip('#')
    headers = header_line.split()
    headers = [i.split('=')[-1] for i in headers]
    data = pd.read_csv(filename, sep=r'\s+', skiprows=2, names=headers)
  return data

def plot_columns(data, columns, tmin=None, tmax=None, outfile='test.png'):
    
  # Filter data by time range if specified
  if tmin is not None:
    data = data[data['time'] >= tmin]
  if tmax is not None:
    data = data[data['time'] <= tmax]

  ncols = len(columns)
  if list(columns) == ['all']:
    ncols = len(data.columns)-1
    columns = list(data.columns[1:])
  
  if ncols == 0:
    print("No columns to plot.")
    return

  # Layout decision: if <=4 columns, stack vertically (1 column),
  # else use 2 columns and enough rows
  if ncols <= 4:
    nrows = ncols
    ncols_plot = 1
  else:
    nrows = 4
    ncols_plot = math.ceil(ncols/nrows)

  fig, axes = plt.subplots(nrows=nrows, ncols=ncols_plot, sharex=True, figsize=(4*ncols_plot, 4*nrows))

  # Flatten axes array for easy indexing (handles case nrows=1 or ncols=1)
  if nrows == 1 and ncols_plot == 1:
    axes = [axes]
  else:
    axes = axes.T.flatten()

  for i, col in enumerate(columns):
    ax = axes[i]
    if col not in data.columns:
      print(f"Warning: Column '{col}' not found in data.")
      ax.set_visible(False)
      continue
    ax.plot(data['time'], data[col],label=col)
    ax.set_ylabel(col)
    ax.grid(True)
    ax.legend()

  # Hide any unused subplots (if total subplots > ncols)
  for j in range(len(columns), len(axes)):
    axes[j].set_visible(False)

  axes[-1].set_xlabel('time [code]')  # Label x-axis on bottom subplot(s)

  plt.suptitle('Athena HST Data')
  plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
  plt.savefig(outfile,dpi=150,bbox_inches='tight')

@click.command(help='Plot Athena .hst file columns vs time.')
@click.argument('filename', nargs=1)
@click.argument('columns', nargs=-1)
@click.option('--tmin', type=float, default=None, help='Minimum time to plot')
@click.option('--tmax', type=float, default=None, help='Maximum time to plot')
@click.option('--outfile', type=str, default='test.png', help='output file for plot')
@click.option('--restart', type=bool, default=True, help='whether the hst is from a restart or a new sim')
def main(filename,columns,**kwargs):
  data = read_athena_hst(filename,kwargs['restart'])
  plot_columns(data, columns,kwargs['tmin'],kwargs['tmax'],kwargs['outfile'])

if __name__ == '__main__':
  main()

