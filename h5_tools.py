### Tools to use VTP files in Python ###

### IMPORT LIBRARIES ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def geth5Data(filesID, path, lastTimePoint = 710, sampleNum = 24, timestep = 1):
    """
    Extracts data from  a group of .h5 files and stores it in a DataFrame outputted by the function.

    Output/ Extracted data (for each node):
    DataFrame with columns: simulation number, sample number, ECM stiffness, probability of disassembly,
    initial lifetime, time, number of FAs (total, back and front), mean lifetime, maturation level of the fibers,
    rupture, cell traction, center of mass (x coordinate).

    Keyword arguments:
    ffilesID - (array/int) IDs of the simulation to be analyzed
    path - (string) Path to the "Condition" directory
    lastTimePoint - (int)) Number of time points in the simulation. If not specified, will be set as 710
    sampleNum - (int) Number of samples in the simulation. If not specified, will be set as 24.
    timestep - (int) Factor to match the recording time to the simulation time. If not specified, will be set as 1.
    """

    ### IMPORT LIBRARIES
    import h5py

    ### DEFINE VARIABLES
    simNum = np.size(filesID)
    data = pd.DataFrame(0.0, index=range(0, lastTimePoint*sampleNum*simNum),
                        columns=['sim_num', 'samp_num', 'kECM', 'log_kECM', 'pFA_rev', 'lt_FA0', 'time',
                                 'nFA', 'nFA_back', 'nFA_front', 'lt_FA', 'multFam', 'rpdFA', 'trac_cell', 'CoM'])


    ### EXTRACT AND STORE DATA
    for ind, sim in enumerate(filesID):

        # Read file
        f = h5py.File(path + str(sim).zfill(3) + '/pstudy.h5', 'r')

        for samp in range(0, 24):

            # Variables
            initPoint = ind*sampleNum*lastTimePoint + (samp*lastTimePoint)
            finalPoint = ind*sampleNum*lastTimePoint + (samp*lastTimePoint) + lastTimePoint
            dataLoc = range(initPoint, finalPoint)
            time = np.arange(0, lastTimePoint* timestep, timestep)

            # General outputs (include all metrics)
            params = f['results']['params']
            output = f['results']['sim_output']

            # "Meta" data
            data['sim_num'][dataLoc] = sim
            data['samp_num'][dataLoc] = samp
            data['time'][dataLoc] = time

            # Parameters
            pFA_rev = params['pFA_rev']['data'][samp]
            kECM = params['kECM']['data'][samp]
            lt_FA0 = (-1 / (np.log(1 - pFA_rev)))/60

            data['pFA_rev'][dataLoc] = pFA_rev
            data['kECM'][dataLoc] = kECM
            data['log_kECM'][dataLoc] = np.log10(kECM)
            data['lt_FA0'][dataLoc] = round(lt_FA0, 1)

            # Outputs
            data['nFA'][dataLoc] = output['nFA']['data'][samp, 0:lastTimePoint]
            data['nFA_back'][dataLoc] = output['nFA_back']['data'][samp, 0:lastTimePoint]
            data['nFA_front'][dataLoc] = output['nFA_front']['data'][samp, 0:lastTimePoint]
            data['lt_FA'][dataLoc] = output['lt_FA']['data'][samp, 0:lastTimePoint]
            data['multFam'][dataLoc] = output['multFam']['data'][samp, 0:lastTimePoint]
            data['rpdFA'][dataLoc] = output['rpdFA']['data'][samp, 0:lastTimePoint]
            data['trac_cell'][dataLoc] = output['trac_cell']['data'][samp, 0:lastTimePoint]
            data['CoM'][dataLoc] = output['CoM']['data'][:, :, 0][samp, 0:lastTimePoint]


    return data


def plotFinalDisp(finalMetrics, metric, mode='avg'):
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

    my_xticks = np.unique(finalMetrics['kECM'])

    if mode == 'all':

        g = sns.FacetGrid(finalMetrics, col="lt_FA0", hue="kECM", aspect=1.15, size=5, gridspec_kws={"wspace": 0.08})

        g = (g.map(plt.scatter, "log_kECM", metric)
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

        ax = sns.lmplot(data=finalMetrics, x="log_kECM", y=metric, col="lt_FA0", hue="kECM", fit_reg=False,
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


def plotMetric3D(metric, finalMetrics):

    from mpl_toolkits.mplot3d import Axes3D
    plt.rcParams['grid.color'] = "lightgray"
    plt.rcParams['grid.linewidth'] = .5

    ### 3D PLOT ###
    fig = plt.figure(figsize=(14, 8))
    ax = fig.gca(projection='3d')

    y = finalMetrics['kECM'][finalMetrics['sim_num'] == 1]
    x = finalMetrics['lt_FA0'][finalMetrics['sim_num'] == 1]

    surf = ax.plot_surface(np.log10(x.as_matrix().reshape((6, 4))),
                           np.log10(y.as_matrix().reshape((6, 4))),
                           finalMetrics.groupby('samp_num')[metric].mean().as_matrix().reshape((6, 4)),
                           cmap='BuGn', linewidths=0)

    yticks = np.unique(finalMetrics['kECM'])
    xticks = np.unique(finalMetrics['lt_FA0'])
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


def plotMetricHeatMap(metric, finalMetrics, params):

    fig, ax = plt.subplots(figsize=(9, 7))

    new_metric_df = finalMetrics.groupby('samp_num')[metric].mean()

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


def plotDisp(h5Data, mode='all', disp='sum_disp', sim=1):

    sns.set_style("white")
    sns.set_palette('viridis_r', 6)

    if mode == 'all':

        g = sns.FacetGrid(h5Data[h5Data['sim_num'] == sim], col="lt_FA0", hue="kECM", size=5)

        g = (g.map(plt.plot, "time", disp)
             .set(xlim=(0, None))
             .add_legend()
             .set_ylabels("Displacement [$\mu$m]", labelpad=10)
             .set_xlabels("Time [min]", labelpad=15)
             .set_titles("Lifetime: {col_name} min")
             .fig.subplots_adjust(wspace=.1, hspace=.5))

        plt.subplots_adjust(top=.8)
        #plt.suptitle("Displacement of the cell's center of mass (simulation 1)", weight='bold')

    elif mode == 'overlap':

        ### PLOT DISPLACEMENT - CUMSUM (VARIATION) ###
        g = sns.FacetGrid(h5Data, col="lt_FA0", hue="kECM", size=5)

        g = (g.map(sns.lineplot, "time", disp)
             .set(xlim=(0, None))
             .add_legend()
             .set_titles("Lifetime: {col_name} min")
             .set_ylabels("Displacement [$\mu$m]", labelpad=10)
             .set_xlabels("Time [min]", labelpad=15)
             .fig.subplots_adjust(wspace=.1, hspace=.05))


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


def getAbsDisp(h5Data, jumpsValues):

    simNum = map(int, np.unique(h5Data['sim_num']))

    for sim in simNum:

        for samp in range(0, 24):

            negJumps = jumpsValues[sim][samp]['neg_jumps']
            fullJumps = jumpsValues[sim][samp]['full_jumps']
            sampData = h5Data[h5Data['sim_num'] == sim][h5Data['samp_num'] == samp]
            sampDiffDisp = sampData['diff_disp'].copy()

            if np.size(negJumps) != 0:

                # Invert displacement for negative peaks
                for ind, jump in enumerate(fullJumps):

                    if ind < np.size(fullJumps) - 1:

                        if np.any(negJumps == jump):

                            sampDiffDisp.iloc[jump / 2 - 5: fullJumps[ind + 1] / 2 - 1] = - sampDiffDisp.iloc[
                                                                                            jump / 2 - 5: fullJumps[ind + 1] / 2 - 1]

                    else:

                        if np.any(negJumps == jump):
                            sampDiffDisp.iloc[jump / 2 - 5:] = - sampDiffDisp.iloc[jump / 2 - 5:]

            sampCumDisp = np.cumsum(sampDiffDisp)
            h5Data['abs_disp'][(sim - 1) * 24 * 710 + (samp * 710): (sim - 1) * 24 * 710 + (samp * 710) + 710] = sampCumDisp

    return h5Data


def getFinalMetrics(h5Data, jumpsValues, params):

    finalMetrics = pd.DataFrame(np.nan, index=range(0, 24 * 5),
                                columns=['sim_num', 'samp_num', 'nFA', 'nFA_back', 'nFA_front', 'lt_FA', 'multFam',
                                         'rpdFA', 'trac_cell', 'sum_disp', 'abs_disp'])

    for sim in range(0, 5):

        for samp in range(0, 24):

            criteria1 = h5Data['sim_num'] == (sim + 1)
            criteria2 = h5Data['samp_num'] == samp
            criteria = criteria1 & criteria2
            metricLoc = sim * 24 + samp

            # Independent variables
            finalMetrics['sim_num'][metricLoc] = sim + 1
            finalMetrics['samp_num'][metricLoc] = samp

            # Metrics based on final values
            for metric in ['rpdFA', 'sum_disp', 'abs_disp']:
                finalMetrics[metric][metricLoc] = np.mean(h5Data[criteria][metric].iloc[-5])

            # Metrics based on jumps
            jumpValuesSize = np.size(jumpsValues[sim + 1][samp]['full_jumps'])

            if jumpValuesSize > 0:

                for metric in ['nFA', 'nFA_back', 'nFA_front', 'lt_FA', 'multFam', 'trac_cell']:

                    meanTemp = np.zeros(jumpValuesSize)

                    for ind, jump in enumerate(jumpsValues[sim + 1][samp]['full_jumps']):
                        jumpRange = range(jump / 2 - 5, jump / 2 + 1)
                        meanTemp[ind] = np.nanmax(h5Data[criteria][metric].iloc[jumpRange])

                    finalMetrics[metric][metricLoc] = np.mean(meanTemp)

            else:

                for metric in ['nFA', 'nFA_back', 'nFA_front', 'lt_FA', 'multFam', 'trac_cell']:
                    finalMetrics[metric][metricLoc] = np.nanmean(h5Data[criteria][metric])

    finalMetrics = pd.merge(finalMetrics, params, on='samp_num')
    finalMetrics['log_kECM'] = np.log10(finalMetrics['kECM'])
    finalMetrics['lt_FA'] = finalMetrics['lt_FA']/60
    finalMetrics['nFA_perc'] = finalMetrics['nFA_back']/finalMetrics['nFA']

    return finalMetrics