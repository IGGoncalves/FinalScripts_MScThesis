### Tools to use VTP files in Python ###

### IMPORT LIBRARIES ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    params['lt_FA0'] = params['lt_FA0'].round(decimals=1)

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
            data['lt_FA0'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = params['lt_FA0'][n]
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
            data['abs_disp'][m * 24 * 710 + (n * 710) + 1: m * 24 * 710 + (n * 710) + 710] = diff_disp

            final_disp = np.mean(cum_sum[-5:])

            data['final_disp'][m * 24 * 710 + (n * 710): m * 24 * 710 + (n * 710) + 710] = final_disp

    return data, params


def plotFinalDisp(h5Data, mode='avg'):
    """
    This function extracts data from h5 files and store it in a DataFrame, outputted by the function.

    Output/ Extracted data (for each node):
    DataFrame with columns "nodeID, time, x, y, z, atFA".

    Keyword arguments:
    filesID (int/array) -
    path (string) - path to the folder containing all VTP files. If it is not specified, current dir is used.
    """

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


def plotMetric3D(metric, h5Data):

    from mpl_toolkits.mplot3d import Axes3D
    plt.rcParams['grid.color'] = "lightgray"
    plt.rcParams['grid.linewidth'] = .5

    ### 3D PLOT ###
    fig = plt.figure(figsize=(14, 8))
    ax = fig.gca(projection='3d')

    y = h5Data['kECM'][h5Data['time'] == 0][h5Data['sim_num'] == 1]
    x = h5Data['lt_FA0'][h5Data['time'] == 0][h5Data['sim_num'] == 1]

    surf = ax.plot_surface(np.log10(x.as_matrix().reshape((6, 4))),
                           np.log10(y.as_matrix().reshape((6, 4))),
                           h5Data.groupby('samp_num')[metric].mean().as_matrix().reshape((6, 4)),
                           cmap='BuGn', linewidths=0)

    yticks = np.unique(h5Data['kECM'])
    xticks = np.unique(h5Data['lt_FA0'])
    ax.set_yticks(np.log10(yticks))
    ax.set_yticklabels(yticks)

    ax.set_xticks(np.log10(xticks))
    ax.set_xticklabels(xticks)

    ax.set_ylabel('kECM [N/m]', labelpad=20)
    ax.set_xlabel('Lifetime [min]', labelpad=20)
    ax.set_zlabel('Displacement [nm]', labelpad=10)
    #plt.gca().invert_xaxis()
    cb = fig.colorbar(surf)
    cb.outline.set_visible(False)
    # Get rid of the panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_xaxis.line.set_color('gray')
    ax.w_yaxis.line.set_color('gray')
    ax.w_zaxis.line.set_color('gray')


def plotMetricHeatMap(metric, h5Data, params):

    fig, ax = plt.subplots(figsize=(9, 7))

    new_metric_df = h5Data.groupby('samp_num')[metric].mean()

    new_metric_df = pd.merge(new_metric_df, params, on='samp_num')

    # Creating the heatmaps
    new_metric_df= new_metric_df.pivot('kECM', 'lt_FA0', metric)

    # Plotting the heatmaps (saving colormap to plot colorbar afterwards)
    im = sns.heatmap(new_metric_df, cmap="BuGn", linewidths=.9, annot=True, ax=ax)
    plt.gca().invert_yaxis()
    ax.set_yticklabels(np.unique(params['kECM']), va='center')
    ax.set_ylabel('kECM [N/m]', labelpad=15)
    ax.set_xlabel('Lifetime [min]', labelpad=15)
    #plt.title('Number of "negative" jumps', weight='bold')


def plotDisp(h5Data, mode='all', disp='sum_disp'):

    sns.set_style("white")
    sns.set_palette('viridis_r', 6)

    if mode == 'all':

        g = sns.FacetGrid(h5Data[h5Data['sim_num'] == 1], col="pFA_rev", hue="kECM", margin_titles=True, size=5)

        g = (g.map(plt.plot, "time", "sum_disp")
             .set(xlim=(0, None))
             .add_legend()
             .set_ylabels("Displacement [$\mu$m]", labelpad=10)
             .set_xlabels("Time [min]", labelpad=15)
             .fig.subplots_adjust(wspace=.1, hspace=.05))

        plt.subplots_adjust(top=.8)
        #plt.suptitle("Displacement of the cell's center of mass (simulation 1)", weight='bold')

    elif mode == 'overlap':

        ### PLOT DISPLACEMENT - CUMSUM (VARIATION) ###
        g = sns.FacetGrid(h5Data, col="pFA_rev", hue="kECM", size=5)

        g = (g.map(sns.lineplot, "time", disp)
             .set(xlim=(0, None))
             .add_legend()
             .set_ylabels("Displacement [$\mu$m]", labelpad=10)
             .set_xlabels("Time [min]", labelpad=15))


def getJumps(h5Data):

    from scipy.signal import find_peaks

    # Set the minimum difference in displacement to be considered a jump
    partialJumpMin = 2e-1
    fullJumpMin = 5e-1

    # Define dictionary to store the time values for the jumps of each sample (for all simulations)
    allSimJumps = {}

    # Define DataFrame to store general information on the jumps for each sample
    jumpsInfo = pd.DataFrame(np.nan, index=range(0, 24 * 5), columns=['sim_num', 'samp_num', 'pFA_rev', 'lt_FA0', 'kECM',
                                                                       'jumps_num', 'full_jump_num', 'partial_jump_num',
                                                                       'first_full_jump', 'jump_time_mean',
                                                                       'jump_time_std', 'neg_jumps_num'])

    for sim in range(1, 6):

        # Define dictionary to store the time values for the jumps of each sample
        specificSimJumps = {}

        # Going through the samples to store (and print) information on jumps
        for samp in range(0, 24):

            sampData = h5Data[h5Data['sim_num'] == sim][h5Data['samp_num'] == samp]
            sampDiffDisp = sampData['diff_disp']

            # Setting a threshold (minimum only, to prevent a small jump appearing next to a big one)
            jumps, _ = find_peaks(abs(sampDiffDisp),distance=15, height=partialJumpMin)

            jumpsDiffDispValues = abs(sampDiffDisp.iloc[jumps])


            # Differentiating small and big jumps
            partialJumps = jumps[jumpsDiffDispValues < fullJumpMin]
            fullJumps = jumps[jumpsDiffDispValues >= fullJumpMin]

            # Small adjusment because of the timestep
            partialJumps = partialJumps * 2
            fullJumps = fullJumps * 2

            negJumps = fullJumps[sampDiffDisp.iloc[fullJumps / 2] < 0]

            # Dictionary with all jumps, partial and full
            jumpValues = {}

            jumpValues['all_jumps'] = jumps * 2
            jumpValues['partial_jumps'] = partialJumps
            jumpValues['full_jumps'] = fullJumps
            jumpValues['neg_jumps'] = negJumps

            # Store information
            jumpsInfo['sim_num'][(sim - 1) * 24 + samp] = sim
            jumpsInfo['samp_num'][(sim - 1) * 24 + samp] = samp
            jumpsInfo['jumps_num'][(sim - 1) * 24 + samp] = np.size(jumps)
            jumpsInfo['full_jump_num'][(sim - 1) * 24 + samp] = np.size(fullJumps)
            jumpsInfo['partial_jump_num'][(sim - 1) * 24 + samp] = np.size(partialJumps)
            jumpsInfo['pFA_rev'][(sim - 1) * 24 + samp] = sampData['pFA_rev'][sampData['time'] == 0]
            jumpsInfo['lt_FA0'][(sim - 1) * 24 + samp] = sampData['lt_FA0'][sampData['time'] == 0]
            jumpsInfo['kECM'][(sim - 1) * 24 + samp] = sampData['kECM'][sampData['time'] == 0]

            if np.size(jumps) == 0:

                jumpsInfo['first_full_jump'][(sim - 1) * 24 + samp] = np.nan
                jumpsInfo['jump_time_mean'][(sim - 1) * 24 + samp] = np.nan
                jumpsInfo['jump_time_std'][(sim - 1) * 24 + samp] = np.nan
                jumpsInfo['neg_jumps_num'][(sim - 1) * 24 + samp] = 0

            else:

                if np.size(fullJumps) == 0:
                    jumpsInfo['first_full_jump'][(sim - 1) * 24 + samp] = np.nan
                    jumpsInfo['neg_jumps_num'][(sim - 1) * 24 + samp] = 0

                else:
                    jumpsInfo['first_full_jump'][(sim - 1) * 24 + samp] = fullJumps[0]
                    jumpsInfo['neg_jumps_num'][(sim - 1) * 24 + samp] = np.size(negJumps)

                jumpsInfo['jump_time_mean'][(sim - 1) * 24 + samp] = np.mean(np.diff(jumps))
                jumpsInfo['jump_time_std'][(sim - 1) * 24 + samp] = np.std(np.diff(jumps))

            # Store the jump information in the dictionary, with the sample number as key
            specificSimJumps[samp] = jumpValues

        allSimJumps[sim] = specificSimJumps

    return allSimJumps, jumpsInfo