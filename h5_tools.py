### Tools to use VTP files in Python ###

### IMPORT LIBRARIES ###
import numpy as np
import pandas as pd

def geth5Data(filesID, path = '', lastTimePoint=710, timestep=2):
    """
    This function extracts data from h5 files and store it in a DataFrame, outputted by the function.

    Output/ Extracted data (for each node):
    DataFrame with columns "nodeID, time, x, y, z, atFA".

    Keyword arguments:
    filesID (int/array) -
    path (string) - path to the folder containing all VTP files. If it is not specified, current dir is used.
    """

    import h5py

    # Reading the first results file
    f = h5py.File(path + '001/pstudy.h5', 'r')

    # Storing the parameters (kECM and pFA_rev) in a DataFrame
    params = pd.DataFrame(0.0, index=range(0, 24), columns=['samp_num', 'kECM', 'pFA_rev', 'lt_FA0'])

    params['samp_num'] = range(0, 24)
    params['kECM'] = f['results']['params']['kECM']['data'].value
    params['pFA_rev'] = f['results']['params']['pFA_rev']['data'].value
    params['lt_FA0'] = -1/(np.log(1 - f['results']['params']['pFA_rev']['data'].value))/60

    data = pd.DataFrame(0.0, index=range(0, 710 * 24 * 5),
                        columns=['time', 'sim_num', 'samp_num', 'kECM', 'log_kECM', 'pFA_rev', 'lt_FA0', 'nFA',
                                 'lt_FA', 'multFam', 'rpdFA', 'trac_cell', 'CoM', 'sum_disp', 'diff_disp',
                                 'final_disp', 'abs_disp', 'nFA_back'])

    # Going through all .h5 files to get the results
    for m in filesID:

        f = h5py.File(path + str(m + 1).zfill(3) + '/pstudy.h5', 'r')

        for n in range(0, 24):
            data['time'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = np.arange(0, 710 * timestep, 2)
            data['pFA_rev'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = params['pFA_rev'][n]
            data['kECM'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = params['kECM'][n]
            data['log_kECM'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = np.log10(params['kECM'][n])
            data['lt_FA0'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = params['lt_FA0'][n].round(decimals=1)
            data['sim_num'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = m + 1
            data['samp_num'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = n

            data['nFA'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = f['results']['sim_output']['nFA'][
                                                                                        'data'][n, 0:710]
            data['nFA_back'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = \
            f['results']['sim_output']['nFA_back']['data'][n, 0:710]
            data['lt_FA'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = \
            f['results']['sim_output']['lt_FA']['data'][n, 0:710]
            data['multFam'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = \
            f['results']['sim_output']['multFam']['data'][n, 0:710]
            data['rpdFA'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = \
            f['results']['sim_output']['rpdFA']['data'][n, 0:710]
            data['trac_cell'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = \
            f['results']['sim_output']['trac_cell']['data'][n, 0:710]
            data['CoM'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = f['results']['sim_output']['CoM'][
                                                                                        'data'][:, :, 0][n, 0:710]

            diff_disp = np.diff(f['results']['sim_output']['CoM']['data'][:, :, 0][n, 0:710]) * 10e5
            cum_sum = np.cumsum(np.diff(f['results']['sim_output']['CoM']['data'][:, :, 0][n, 0:710])) * 10e5

            data['diff_disp'][m * 24 * 710 + (n * 710) + 1: m * 24 * 710 + (n * 710) + 710] = diff_disp
            data['sum_disp'][m * 24 * 710 + (n * 710) + 1: m * 24 * 710 + (n * 710) + 710] = cum_sum
            data['abs_disp'][m * 24 * 710 + (n * 710) + 1: m * 24 * 710 + (n * 710) + 710] = cum_sum

            final_disp = np.mean(cum_sum[-5:])

            data['final_disp'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = final_disp

    return data


def plotFinalDisp(h5Data, mode='avg'):
    """
    This function extracts data from h5 files and store it in a DataFrame, outputted by the function.

    Output/ Extracted data (for each node):
    DataFrame with columns "nodeID, time, x, y, z, atFA".

    Keyword arguments:
    filesID (int/array) -
    path (string) - path to the folder containing all VTP files. If it is not specified, current dir is used.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(20, 10))
    sns.set_style("white")
    sns.set_palette('viridis_r', 6)

    my_xticks = np.unique(h5Data['kECM'])

    if mode == 'all':

        g = sns.FacetGrid(h5Data[709::710], col="lt_FA0", hue="kECM", aspect=1.15, size=5, gridspec_kws={"wspace": 0.08})

        g = (g.map(plt.scatter, "log_kECM", "final_disp")
             .set(xticks=np.log10(my_xticks))
             .add_legend()
             .set(ylim=(-5, 26))
             .set_xticklabels(my_xticks)
             .set_titles("Lifetime: {col_name} min")
             .set_ylabels("Displacement [$\mu$m]", labelpad=10)
             .set_xlabels("kECM [N/m]", labelpad=15))

        for i in range(0, 4):
            g.axes.flatten()[i].yaxis.grid(color='lightgray', linestyle='--', linewidth=0.5)
            g.axes.flatten()[i].spines.values()[0].set_edgecolor('white')
            g.axes.flatten()[i].spines.values()[2].set_edgecolor('lightgray')

        plt.subplots_adjust(top=0.8)
        # g.fig.suptitle('Scatter plot of the final displacement of the cell for all 5 simulations', weight = 'bold')

    elif mode == 'avg':

        ax = sns.lmplot(data=h5Data[709::710], x="log_kECM", y="final_disp", col="lt_FA0", hue="kECM", fit_reg=False,
                        x_estimator=np.mean)

        ax.set(xticks=np.log10(my_xticks))
        ax.set_titles("Lifetime: {col_name} min")
        ax.set_xticklabels(my_xticks)
        plt.subplots_adjust(top=0.8)
        ax.set_xlabels('kECM [N/m]', labelpad=15)
        ax.set_ylabels('Displacement [$\mu$m]', labelpad=10)

        for i in range(0, 4):
            ax.axes.flatten()[i].yaxis.grid(color='lightgray', linestyle='--', linewidth=0.5)
            ax.axes.flatten()[i].spines.values()[0].set_edgecolor('white')
            ax.axes.flatten()[i].spines.values()[2].set_edgecolor('lightgray')

        plt.subplots_adjust(wspace=0.08)
        # plt.suptitle('Mean and SEM of the final displacement of the 5 simulations', weight = 'bold')


