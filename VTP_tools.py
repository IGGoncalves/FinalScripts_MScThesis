# Tools to use VTP files in Python

# IMPORT LIBRARIES
import numpy as np
import pandas as pd


def getNodesData(timesteps, path=''):
    """
    This function extracts data from VTP files (namely the cell_cell_triangles files) and store it in a DataFrame,
    outputted by the function. The user should indicate the timesteps to be extracted.

    Output/ Extracted data (for each node):
    DataFrame with columns "nodeID, time, x, y, z, atFA". Each row corresponds to a node, in a specific time point.

    Keyword arguments:
    timesteps (int/array) - time point(s) at which the user wants data to be extracted.
    path (string) - path to the folder containing all VTP files. If it is not specified, current dir is used.
    """

    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    # VARIABLES DEFINITION
    nodesNumber = 2562
    rowNumber = nodesNumber*np.size(timesteps)
    nodesData = pd.DataFrame(np.nan, index=range(0, rowNumber), columns=['nodeID', 'time', 'x', 'y', 'z', 'atFA'])

    ### DATA EXTRACTION AND STORAGE ###
    if type(timesteps) == int:

        filename = path + 'cell_cell_triangles_' + str(timesteps) + '.vtp'

        ### SETTING UP THE READER ###
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()

        ### GET DATA FROM FILE ###
        data = reader.GetOutput()
        points = data.GetPoints()
        coords = vtk_to_numpy(points.GetData())
        atFA = vtk_to_numpy(data.GetPointData().GetArray('atFA'))

        ### STORE DATA IN DATAFRAME ###
        nodesData['nodeID'] = range(0, nodesNumber)
        nodesData['time'] = timesteps
        nodesData['x'] = coords[:, 0] * 10e5
        nodesData['y'] = coords[:, 2] * 10e5
        nodesData['z'] = coords[:, 1] * 10e5
        nodesData['atFA'] = atFA

    else:

        for ind, time in enumerate(timesteps):

            filename = path + 'cell_cell_triangles_' + str(time) + '.vtp'

            ### SETTING UP THE READER ###
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(filename)
            reader.Update()

            ### GET DATA FROM FILE ###
            data = reader.GetOutput()
            points = data.GetPoints()
            coords = vtk_to_numpy(points.GetData())
            atFA = vtk_to_numpy(data.GetPointData().GetArray('atFA'))

            ### STORE DATA IN DATAFRAME ###
            nodesData['nodeID'][ind * nodesNumber : ind * nodesNumber + nodesNumber] = range(0, nodesNumber)
            nodesData['time'][ind * nodesNumber: ind * nodesNumber + nodesNumber] = time
            nodesData['x'][ind * nodesNumber : ind * nodesNumber + nodesNumber] = coords[:, 0] * 10e5
            nodesData['y'][ind * nodesNumber : ind * nodesNumber + nodesNumber] = coords[:, 2] * 10e5
            nodesData['z'][ind * nodesNumber : ind * nodesNumber + nodesNumber] = coords[:, 1] * 10e5
            nodesData['atFA'][ind * nodesNumber: ind * nodesNumber + nodesNumber] = atFA

    return nodesData


def plotNodes2DOverlapped(nodesData, timesteps = 'all', view = 'XY'):
    """
    This function uses the x,y and z data from a nodesData DataFrame to create a scatter plot of the cell's cortex nodes,
    with the nodes from different time points overlapped.

    Keyword arguments:
    nodesData - DataFrame with the nodes' info (see getNodesData)
    timesteps (int/array) - time point(s) at which the user wants data to plotted. if not specified, all the time points
    present in the DataFrame are plotted
    view (string) - view for the plot. Options: 'XY', 'XZ'
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    ### VARIABLE DEFINITION ###
    if timesteps == 'all':

        timesteps = np.unique(nodesData['time'])

    timestepNum = np.size(timesteps)

    sns.set_palette('viridis_r', timestepNum)

    ### SCATTER PLOT ###
    if view == 'XZ':

        sns.set_style('white')

        plt.figure(figsize=(10, 2.5))

        if type(timesteps) == int:

            # Scatter with only the front nodes (y <= 0), for more clarity
            sns.scatterplot(nodesData[nodesData['y'] <= 0][nodesData['time'] == timesteps]['x'],
                            nodesData[nodesData['y'] <= 0][nodesData['time'] == timesteps]['z'], label=timesteps * 2, s=12)

        else:

            for time in timesteps:

                # Scatter with only the front nodes (y <= 0), for more clarity
                sns.scatterplot(nodesData[nodesData['y'] <= 0][nodesData['time'] == time]['x'],
                        nodesData[nodesData['y'] <= 0][nodesData['time'] == time]['z'], label = time*2, s = 12)

        # Figure aesthetics
        plt.xlim(-75, -20)
        plt.ylim(-.1, 7)

    elif view == 'XY':

        sns.set_style('darkgrid')

        plt.figure(figsize = (10, 5))

        if type(timesteps) == int:

            # Scatter with only the top nodes (z > .1), for more clarity
            sns.scatterplot(nodesData[nodesData['z'] > 1e-1][nodesData['time'] == timesteps]['x'],
                            nodesData[nodesData['z'] > 1e-1][nodesData['time'] == timesteps]['y'], label=timesteps* 2, s=12)

        else:

            for time in timesteps:

                # Scatter with only the top nodes (z > .1), for more clarity
                sns.scatterplot(nodesData[nodesData['z'] > 1e-1][nodesData['time'] == time]['x'],
                        nodesData[nodesData['z'] > 1e-1][nodesData['time'] == time]['y'], label = time*2, s = 12)

        # Figure aesthetics
        plt.xlim(-75, -20)
        plt.ylim(-14, 14)

    sns.despine(left = True)
    plt.yticks([])
    plt.ylabel(' ')
    plt.xlabel('Position [$\mu$m]', labelpad = 5)
    plt.legend(title = 'Timestep [min]')


def plotNodes2DSubplots(nodesData, timesteps = 'all', view = 'XY'):
    """
    This function uses the x,y and z data from a nodesData DataFrame to create a scatter plot of the cell's cortex nodes,
    with the nodes from different time points in different subplots.

    Keyword arguments:
    nodesData - DataFrame with the nodes' info (see getNodesData)
    timesteps (int/array) - time point(s) at which the user wants data to plotted. if not specified, all the time points
    present in the DataFrame are plotted
    view (string) - view for the plot. Options: 'XY', 'XZ'
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    ### VARIABLE DEFINITION ###
    if timesteps == 'all':

        timesteps = np.unique(nodesData['time'])

    sns.set_palette('viridis_r', 3)

    ### SCATTER PLOT ###
    if view == 'XZ':

        sns.set_style('white')

        g = sns.FacetGrid(nodesData[nodesData['y'] <= 0], col = "time", margin_titles = True, aspect = 2.5)

        g = (g.map(sns.scatterplot, "x", "z", s = 12)
             .set(xlim = (-75, -20), ylim = (-.1, 7))
             .set_axis_labels("Displacement [$\mu$m]", ""))


    elif view == 'XY':

        sns.set_style('darkgrid')

        g = sns.FacetGrid(nodesData[nodesData['z'] > 1e-1], col = "time", margin_titles = True, aspect = 1.5, height=5)

        g = (g.map(sns.scatterplot, "x", "y", s = 12)
             .set(xlim = (-75, -20), ylim = (-14, 14))
             .set_axis_labels("Displacement [$\mu$m]", ""))

    sns.despine(left = True)
    plt.yticks([])


def plotNodes3DOverlapped(nodesData, timesteps='all'):
    """
    This function uses the x,y and z data from a nodesData DataFrame to create a scatter plot of the cell's cortex nodes,
    with the nodes from different time points in different subplots.

    Keyword arguments:
    nodesData - DataFrame with the nodes' info (see getNodesData)
    timesteps (int/array) - time point(s) at which the user wants data to plotted. if not specified, all the time points
    present in the DataFrame are plotted
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    # VARIABLE DEFINITION
    if timesteps == 'all':
        timesteps = np.unique(nodesData['time'])

    # FIGURE DEFINITION
    sns.set_style('white')
    sns.set_palette("viridis_r", 3)
    fig = plt.figure(figsize=(15, 4))
    ax = fig.gca(projection='3d')
    ax._axis3don = True

    # 3D SCATTER PLOT
    if type(timesteps) == int:

        ax.scatter(nodesData[nodesData['time'] == timesteps]['x'], nodesData[nodesData['time'] == timesteps]['y'],
                   nodesData[nodesData['time'] == timesteps]['z'], s=11)

    else:
        for time in timesteps:

            ax.scatter(nodesData[nodesData['time'] == time]['x'], nodesData[nodesData['time'] == time]['y'],
                       nodesData[nodesData['time'] == time]['z'], s=11, label=time*2)

    # FIGURE AESTHETICCS
    ax.grid(False)
    ax.legend(title='Timesteps: ')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_zlim(-.1, 8)
    ax.set_xlim(-75, -20)
    # Get rid of the panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((0.917, 0.917, 0.949, 1.0))
    # Get rid of the spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))


def plotFinalDisp(nodesData, h5data, sample_num):
    """
    This function uses the x,y and z data from a nodesData DataFrame to create a scatter plot of the cell's cortex nodes,
    with the nodes from different time points in different subplots.

    Keyword arguments:
    nodesData - DataFrame with the nodes' info (see getNodesData)
    timesteps (int/array) - time point(s) at which the user wants data to plotted. if not specified, all the time points
    present in the DataFrame are plotted
    """

    import matplotlib.pyplot as plt

    newTime = [nodesData['time'].iloc[0], nodesData['time'].iloc[-1]]

    x1 = h5data[h5data['sim_num'] == 1][h5data['samp_num'] == sample_num][h5data['time'] == newTime[0]*2]['CoM'].values*10e5
    x2 = h5data[h5data['sim_num'] == 1][h5data['samp_num'] == sample_num][h5data['time'] == newTime[1]*2]['CoM'].values*10e5

    plotNodes2DOverlapped(nodesData, newTime, view='XZ')
    plt.plot([float(x1), float(x2)], [6.5, 6.5], marker='o', markersize=4, color='black')
    plt.text((x2 + x1) / 2, 6.8,
             'Final displacement: ' + str(round((float(x2) - float(x1)), 2)) + ' $\mu$m', fontsize=12, ha='center')