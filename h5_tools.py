### IMPORT GENERAL LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def geth5Data(filesID, path, lastTimePoint = 710, sampleNum = 24, timestep = 1):
    """
    Extracts data from  a group of .h5 files and stores it in a DataFrame outputted by the function.

    Output:
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

    ### IMPORT SPECIFIC LIBRARIES
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

        for samp in range(0, sampleNum):

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


def getSumDisp(h5Data):
    """
    Calculates the total displacement of the cell, disregarding the cell's position (i.e., how much it moves
    instead of the coordinates of its positions). To do so, the differences between the center of mass in
    consecutive time points are calculated and the results are then summed to get the cumulative displacement.

    Output:
    DataFrame used as input, plus "diff_disp" and "sum_disp" columns.

    Keyword arguments:
    h5Data - DataFrame with the geth5Data() format.
    """

    ### DEFINE VARIABLES
    simValues = h5Data['sim_num'].unique()
    sampValues = h5Data['samp_num'].unique()
    simSize = np.size(simValues)
    sampSize = np.size(sampValues)
    timeSize = np.size(h5Data['time'].unique())

    diffSeries = pd.Series(0.0, index=range(0, simSize*sampSize*timeSize))
    cumsumSeries = pd.Series(0.0, index=range(0, simSize*sampSize*timeSize))

    # GET DIFFERENCES AND CALCULATE CUMULATIVE SUM
    for ind, sim in enumerate(simValues):

        condition1 = h5Data['sim_num'] == sim

        for samp in sampValues:

            condition2 = h5Data['samp_num'] == samp
            condition = condition1 & condition2

            # Variables
            initPoint = int(ind*sampSize*timeSize + (samp*timeSize))
            finalPoint = int(initPoint + timeSize)
            diffDataLoc = range(initPoint + 1, finalPoint)
            cumDataLoc = range(initPoint, finalPoint)

            diffSeries[diffDataLoc] = np.diff(h5Data['CoM'][condition])*10e5
            cumsumSeries[cumDataLoc] = np.cumsum(diffSeries[cumDataLoc])

    # STORE RESULTS
    h5Data['diff_disp'] = diffSeries
    h5Data['sum_disp'] = cumsumSeries

    return h5Data


def getJumps(h5Data, partialJumpMin = 2e-1, fullJumpMin = 5e-1, jumpInterval = 30):
    """
    Identifies the jumps the cell makes while migrating, having

    Output:
    jumpsValues - Dicitonary with the time points at which jumps occur (total, partial, full and negative),
    for all samples and simulations
    jumpsInfo - DataFrame with the summarized data on the jumps (number of jumps, mean time between jumps,
    standard deviation of the time between jumps, time at which the first jump occurs).

    Keyword arguments:
    h5Data - DataFrame with the geth5Data() format.
    partialJumpMin - Threshold for partial jumps. If not defined, will be set as .2 micrometers
    fullJumpMin - Threshold for full jumps. If not defined, will be set as .5 micrometers
    jumpInterval - Time window between jumps, where no jumps should occur. If not defined will be set as 30 min
    """

    ### IMPORT SPECIFIC LIBRARIES
    from scipy.signal import find_peaks

    ### DEFINE VARIABLES
    simValues = h5Data['sim_num'].unique()
    sampValues = h5Data['samp_num'].unique()
    simSize = np.size(simValues)
    sampSize = np.size(sampValues)
    timestep = h5Data['time'].iloc[1] - h5Data['time'].iloc[0]

    allSimJumps = {}

    jumpsInfo = pd.DataFrame(np.nan, index=range(0, sampSize*simSize),
                             columns=['sim_num', 'samp_num', 'jumps_num', 'full_jump_num', 'partial_jump_num',
                                      'neg_jumps_num', 'dettach_jump_num', 'first_full_jump', 'jump_time_mean',
                                      'jump_time_std'])


    for ind, sim in enumerate(simValues):

        # Define dictionary to store the time values for the jumps of each sample
        specificSimJumps = {}

        # Going through the samples to store (and print) information on jumps
        for samp in sampValues:

            sampData = h5Data[h5Data['sim_num'] == sim][h5Data['samp_num'] == samp]
            sampDiffDisp = sampData['diff_disp']
            sampLt = sampData['lt_FA']

            # Setting a threshold (minimum only, to prevent a small jump appearing next to a big one)
            jumps, _ = find_peaks(abs(sampDiffDisp), distance = jumpInterval/timestep, height = partialJumpMin)

            jumpsDiffDispValues = abs(sampDiffDisp.iloc[jumps])

            # Differentiating small and big jumps
            partialJumps = jumps[jumpsDiffDispValues < fullJumpMin]
            fullJumps = jumps[jumpsDiffDispValues >= fullJumpMin]

            # Small adjusment because of the timestep
            partialJumps = partialJumps*timestep
            fullJumps = fullJumps*timestep

            negJumps = fullJumps[sampDiffDisp.iloc[fullJumps/timestep] < 0]

            dettachJumps = np.copy(fullJumps)

            if len(dettachJumps) > 0:

                for dtjump in dettachJumps:

                    if np.isnan(sampLt.iloc[int(dtjump/timestep) - 10: int(dtjump/timestep) + 10]).any() == False:

                        dettachJumps = np.delete(dettachJumps, np.argwhere(dettachJumps==dtjump))

            # Dictionary with all jumps, partial and full
            jumpValues = {}

            jumpValues['all_jumps'] = jumps*timestep
            jumpValues['partial_jumps'] = partialJumps
            jumpValues['full_jumps'] = fullJumps
            jumpValues['neg_jumps'] = negJumps
            jumpValues['dettach_jumps'] = dettachJumps

            dataLoc = ind*sampSize + samp

            # Store information
            jumpsInfo['sim_num'][dataLoc] = sim
            jumpsInfo['samp_num'][dataLoc] = samp
            jumpsInfo['jumps_num'][dataLoc] = np.size(jumps)
            jumpsInfo['full_jump_num'][dataLoc] = np.size(fullJumps)
            jumpsInfo['partial_jump_num'][dataLoc] = np.size(partialJumps)
            jumpsInfo['dettach_jump_num'][dataLoc] = np.size(dettachJumps)

            if np.size(jumps) == 0:

                jumpsInfo['first_full_jump'][dataLoc] = np.nan
                jumpsInfo['jump_time_mean'][dataLoc] = np.nan
                jumpsInfo['jump_time_std'][dataLoc] = np.nan
                jumpsInfo['neg_jumps_num'][dataLoc] = 0

            else:

                if np.size(fullJumps) == 0:
                    jumpsInfo['first_full_jump'][dataLoc] = np.nan
                    jumpsInfo['neg_jumps_num'][dataLoc] = 0

                else:
                    jumpsInfo['first_full_jump'][dataLoc] = fullJumps[0]
                    jumpsInfo['neg_jumps_num'][dataLoc] = np.size(negJumps)

                jumpsInfo['jump_time_mean'][dataLoc] = np.mean(np.diff(jumps))
                jumpsInfo['jump_time_std'][dataLoc] = np.std(np.diff(jumps))

            # Store the jump information in the dictionary, with the sample number as key
            specificSimJumps[samp] = jumpValues

        allSimJumps[sim] = specificSimJumps

    return allSimJumps, jumpsInfo


def getAbsDisp(h5Data, jumpsValues):
    """
    Calculates the total displacement of the cell, adding the jumps as if they all happened in the same
    direction, and not in two opposite directions.

    Output:
    DataFrame used as input, plus "abs_disp" columns.

    Keyword arguments:
    h5Data - DataFrame with the geth5Data() format.
    jumpsValues - Dictionary with the getJumps() dictionary format.
    """

    ### DEFINE VARIABLES
    simValues = h5Data['sim_num'].unique()
    sampValues = h5Data['samp_num'].unique()
    simSize = np.size(simValues)
    sampSize = np.size(sampValues)
    timeSize = np.size(h5Data['time'].unique())
    timestep = h5Data['time'].iloc[1] - h5Data['time'].iloc[0]
    jumpFirstInterval = 5                                        # Number of time points before to jump to be flipped
    jumpSecondInterval = 1                                       # Number of time points before the next jump

    absSeries = pd.Series(0.0, index=range(0, simSize*sampSize*timeSize))

    # GET ABSOLUTE DISPLACEMENT
    for ind, sim in enumerate(simValues):

        for samp in sampValues:

            negJumps = jumpsValues[sim][samp]['neg_jumps']
            fullJumps = jumpsValues[sim][samp]['full_jumps']
            sampData = h5Data[h5Data['sim_num'] == sim][h5Data['samp_num'] == samp]
            sampDiffDisp = sampData['diff_disp'].copy()

            if np.size(negJumps) != 0:

                # Invert displacement for negative peaks
                for indJump, jump in enumerate(fullJumps):

                    initPoint = int(jump/timestep - jumpFirstInterval)

                    if indJump < np.size(fullJumps) - 1:

                        if np.any(negJumps == jump):

                            finalPoint = int(fullJumps[indJump + 1]/timestep - jumpSecondInterval)

                            jumpLoc = range(initPoint, finalPoint)

                            sampDiffDisp.iloc[jumpLoc] = - sampDiffDisp.iloc[jumpLoc]

                    else:

                        if np.any(negJumps == jump):
                            sampDiffDisp.iloc[initPoint:] = - sampDiffDisp.iloc[initPoint:]

            sampCumDisp = np.cumsum(sampDiffDisp)

            # STORE RESULTS
            initDataPoint = int(ind*sampSize*timeSize + samp*timeSize)
            finalDataPoint = int(ind*sampSize*timeSize + samp*timeSize + timeSize)
            dataLoc = range(initDataPoint, finalDataPoint)

            absSeries[dataLoc] = sampCumDisp

    h5Data['abs_disp'] = absSeries

    return h5Data


def getDisp(h5Data):
    """
    Combines getSumDisp(), getJumps() and getAbsDisp() to obtain all data on displacement.

    Output:
    DataFrame used as input, plus "diff_disp", "sum_disp" and "abs_disp" columns.
    jumpsValues - Dicitonary with the time points at which jumps occur (total, partial, full and negative),
    for all samples and simulations
    jumpsInfo - DataFrame with the summarized data on the jumps (number of jumps, mean time between jumps,
    standard deviation of the time between jumps, time at which the first jump occurs)

    Keyword arguments:
    h5Data - DataFrame with the geth5Data() format.
    """

    h5Data = getSumDisp(h5Data)
    jumpsValues, jumpsInfo = getJumps(h5Data)
    h5Data = getAbsDisp(h5Data, jumpsValues)

    return h5Data, jumpsValues, jumpsInfo


def plotDisp(h5Data, disp = 'sum_disp', mode = 'single', sim = 1):
    """
    Plots the displacement in terms of time. The user can choose to plot the cumulative (sum_disp) or absolute
    (abs_disp) displacement, as well as the ID of the simulation to plot. It is also possible to plot the data from
    all simulations, choosing the "overlap" mode.

    Output:
    Lineplot of the displacement in terms of time.

    Keyword arguments:
    h5Data (DataFrame) - DataFrame with the geth5Data() format.
    disp (string) - Type of displacement to plot. Options: sum_disp/abs_disp.
    mode (string) - How to plot data, in terms of the simulations (plot a single simulation or a summarized version
    of the data from all simulations). Options: single/overlap.
    sim (int) - ID of the simulation to be plotted, for the "single" mode.
    """

    sns.set_style("white")
    sns.set_palette('viridis_r', 6)

    if mode == 'single':

        g = sns.FacetGrid(h5Data[h5Data['sim_num'] == sim], col="lt_FA0", hue="kECM", size=5)

        g = (g.map(plt.plot, "time", disp)
             .set(xlim=(0, None))
             .add_legend()
             .set_ylabels("Displacement [$\mu$m]", labelpad=10)
             .set_xlabels("Time [min]", labelpad=15)
             .set_titles("Lifetime of the FAs: {col_name} min")
             .fig.subplots_adjust(wspace=.1, hspace=.5))

        plt.subplots_adjust(top=.8)
        #plt.suptitle("Displacement of the cell's center of mass (simulation 1)", weight='bold')

    elif mode == 'overlap':

        ### PLOT DISPLACEMENT - CUMSUM (VARIATION) ###
        g = sns.FacetGrid(h5Data, col="lt_FA0", hue="kECM", size=5)

        g = (g.map(sns.lineplot, "time", disp)
             .set(xlim=(0, None))
             .add_legend()
             .set_titles("Lifetime of the FAs: {col_name} min")
             .set_ylabels("Displacement [$\mu$m]", labelpad=10)
             .set_xlabels("Time [min]", labelpad=15)
             .fig.subplots_adjust(wspace=.1, hspace=.05))


def getFinalMetrics(h5Data, jumpsValues):
    """
    Calculates a representative value for the metrics of all samples.

    Output:
    DataFrame with the calculated values.

    Keyword arguments:
    h5Data - DataFrame with the geth5Data() format.
    jumpsValues - Dictionary with the getJumps() dictionary format.
    """


    ### DEFINE VARIABLES
    simValues = h5Data['sim_num'].unique()
    sampValues = h5Data['samp_num'].unique()
    simSize = np.size(simValues)
    sampSize = np.size(sampValues)
    timeSize = np.size(h5Data['time'].unique())

    finalMetrics = pd.DataFrame(np.nan, index=range(0, sampSize*simSize),
                                columns=['sim_num', 'samp_num', 'nFA', 'nFA_back', 'nFA_front', 'lt_FA', 'multFam',
                                         'rpdFA', 'trac_cell', 'sum_disp', 'abs_disp'])

    # GET FINAL METRICS
    for ind, sim in enumerate(simValues):

        for samp in range(0, sampSize):

            criteria1 = h5Data['sim_num'] == sim
            criteria2 = h5Data['samp_num'] == samp
            criteria = criteria1 & criteria2
            metricLoc = ind*sampSize + samp

            # Independent variables
            finalMetrics['sim_num'][metricLoc] = sim
            finalMetrics['samp_num'][metricLoc] = samp

            # Metrics based on final values
            for metric in ['rpdFA', 'sum_disp', 'abs_disp']:
                finalMetrics[metric][metricLoc] = np.mean(h5Data[criteria][metric].iloc[-5])

            # Metrics based on jumps
            jumpValuesSize = np.size(jumpsValues[sim][samp]['full_jumps'])

            if jumpValuesSize > 0:

                for metric in ['nFA', 'nFA_back', 'nFA_front', 'lt_FA', 'multFam', 'trac_cell']:

                    meanTemp = np.zeros(jumpValuesSize)

                    for ind, jump in enumerate(jumpsValues[sim][samp]['full_jumps']):
                        jumpRange = range(int(jump/2 - 5), int(jump/2 + 1))
                        meanTemp[ind] = np.nanmax(h5Data[criteria][metric].iloc[jumpRange])

                    finalMetrics[metric][metricLoc] = np.mean(meanTemp)

            else:

                for metric in ['nFA', 'nFA_back', 'nFA_front', 'lt_FA', 'multFam', 'trac_cell']:
                    finalMetrics[metric][metricLoc] = np.nanmean(h5Data[criteria][metric])

    # STORE RESULTS
    mergeColumns = ['kECM', 'log_kECM', 'pFA_rev', 'lt_FA0', 'samp_num']
    paramsLoc = range(0, sampSize*timeSize, timeSize)

    finalMetrics = pd.merge(finalMetrics, h5Data[mergeColumns].iloc[paramsLoc])
    finalMetrics['lt_FA'] = finalMetrics['lt_FA']/60
    finalMetrics['nFA_perc'] = finalMetrics['nFA_back']/finalMetrics['nFA']

    return finalMetrics


def plotFinalValues(finalMetrics, metric, mode = 'errorbar'):
    """
    Plots the final values for a specified metric, inputted by the user. The user can also choose
    to plot the results into a scatterplot, showing the results for all simulations, or into a errorbar, which
    will summarize that information. This choice is controled by the "mode" argument.

    Output:
    Errorbar/Scatterplot (depending on the mode).

    Keyword arguments:
    finalMetrics (DataFrame) - DataFrame with the finalMetrics() format.
    metric (string) - Metric (must be present in the finalMetrics DataFrame) the user chooses to plot.
    mode (string) - How the information should be plotted. Options: errorbar/scatter.
    """

    plt.figure(figsize=(20, 10))
    sns.set_style("white")
    sns.set_palette('viridis_r', 6)

    my_xticks = np.unique(finalMetrics['kECM'])

    if metric == 'sum_disp' or metric == 'abs_disp':
        ylab = "Displacement [$\mu$m]"

    elif metric == 'nFA' or metric == 'nFA_back' or metric == 'nFA_front' or metric == 'rpdFA':
        ylab = "Number of FAs"

    elif metric == 'trac_cell':
        ylab = "Cell's traction [Pa]"

    elif metric == 'multFam':
        ylab = "Maturation level"

    elif metric == 'lt_FA':
        ylab = "Lifetime [min]"


    if mode == 'scatter':

        g = sns.FacetGrid(finalMetrics, col="lt_FA0", hue="kECM", aspect=1.15, size=5, gridspec_kws={"wspace": 0.08})

        g = (g.map(plt.scatter, "log_kECM", metric)
             .set(xticks=np.log10(my_xticks))
             .add_legend()
             .set(ylim=(-5, 26))
             .set_xticklabels(my_xticks)
             .set_titles("Lifetime of the FAs: {col_name} min")
             .set_ylabels(ylab, labelpad=10)
             .set_xlabels("kECM [N/m]", labelpad=15))

        for i in range(0, 4):
            g.axes.flatten()[i].yaxis.grid(color='lightgray', linestyle='--', linewidth=0.5)
            g.axes.flatten()[i].spines.values()[0].set_edgecolor('white')
            g.axes.flatten()[i].spines.values()[2].set_edgecolor('lightgray')

        plt.subplots_adjust(top=0.8)
        # g.fig.suptitle('Scatter plot of the final displacement of the cell for all 5 simulations', weight = 'bold')

    elif mode == 'errorbar':

        ax = sns.lmplot(data=finalMetrics, x="log_kECM", y=metric, col="lt_FA0", hue="kECM", fit_reg=False,
                        x_estimator=np.mean)

        ax.set(xticks=np.log10(my_xticks))
        ax.set_titles("Lifetime of the FAs: {col_name} min")
        ax.set_xticklabels(my_xticks)
        plt.subplots_adjust(top=0.8)
        ax.set_xlabels('kECM [N/m]', labelpad=15)
        ax.set_ylabels(ylab, labelpad=10)

        for i in range(0, 4):
            ax.axes.flatten()[i].yaxis.grid(color='lightgray', linestyle='--', linewidth=0.5)
            ax.axes.flatten()[i].spines.values()[0].set_edgecolor('white')
            ax.axes.flatten()[i].spines.values()[2].set_edgecolor('lightgray')

        plt.subplots_adjust(wspace=0.08)
        # plt.suptitle('Mean and SEM of the final displacement of the 5 simulations', weight = 'bold')


def plotMetric3D(finalMetrics, metric):
    """
    This function plots a surface plot for a specified metric inputted by the user, using the values in
    the finalMetrics DataFrame.

    Output:
    Surface plot.

    Keyword arguments:
    finalMetrics (DataFrame) - DataFrame with the finalMetrics() format.
    metric (string) - Metric (must be present in the finalMetrics DataFrame) the user chooses to plot.
    """

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
    ax.set_xlabel('Lifetime of the FAs [min]', labelpad=20)
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


def plotMetricHeatMap(metric, finalMetrics):
    """
    This function plots a heatmap for a specified metric inputted by the user, using the values in
    the finalMetrics DataFrame.

    Output.:
    Heatmap.

    Keyword arguments:
    finalMetrics (DataFrame) - DataFrame with the finalMetrics() format.
    metric (string) - Metric (must be present in the finalMetrics DataFrame) the user chooses to plot.
    """

    fig, ax = plt.subplots(figsize=(9, 7))


    meanMetric = finalMetrics.groupby('samp_num')[metric].mean()

    kECM = finalMetrics.groupby('samp_num')['kECM'].mean()
    ltFA0 = finalMetrics.groupby('samp_num')['lt_FA0'].mean()

    meanMetric = pd.merge(meanMetric, kECM, on = 'samp_num')
    meanMetric = pd.merge(meanMetric, ltFA0, on='samp_num')

    # Creating the heatmaps
    meanMetric = meanMetric.pivot('kECM', 'lt_FA0', metric)

    # Plotting the heatmaps (saving colormap to plot colorbar afterwards)
    im = sns.heatmap(meanMetric, cmap="BuGn", linewidths=.9, annot=True, ax=ax)
    plt.gca().invert_yaxis()
    ax.set_yticklabels(np.unique(finalMetrics['kECM']), va='center')
    ax.set_ylabel('kECM [N/m]', labelpad=15)
    ax.set_xlabel('Lifetime of the FAs [min]', labelpad=15)
    #plt.title('Number of "negative" jumps', weight='bold')
