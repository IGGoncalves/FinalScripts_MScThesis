### Tools to use VTP files in Python ###

### IMPORT LIBRARIES ###
import vtk
import numpy as np
import pandas as pd


def getNodesData(timesteps, path = ''):
    """
    This function extracts data from VTP files (namely the cell_cell_triangles files) and store it in a DataFrame,
    outputted by the function. The user should indicate the timesteps to be extracted.

    Output/ Extracted data (for each node):
    DataFrame with columns "nodeID, time, x, y, z, atFA". Each row corresponds to a node, in a specific time point.

    Keyword arguments:
    timesteps (int/array) - time point(s) at which the user wants data to be extracted.
    path (string) - path to the folder containing all VTP files. If it is not specified, current dir is used.
    """

    from vtk.util.numpy_support import vtk_to_numpy

    ### VARIABLES DEFINITION ###
    nodesNumber = 2562
    rowNumber = nodesNumber*np.size(timesteps)
    nodesData = pd.DataFrame(np.nan, index = range(0, rowNumber), columns = ['nodeID', 'time', 'x', 'y', 'z', 'atFA'])

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
            nodesData['time'][ind * nodesNumber : ind * nodesNumber + nodesNumber] = time
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

    sns.set_palette('viridis_r')

    ### SCATTER PLOT ###
    if view == 'XZ':

        sns.set_style('white')

        g = sns.FacetGrid(nodesData[nodesData['y'] <= 0], col = "time", margin_titles = True, aspect = 2.5)

        g = (g.map(sns.scatterplot, "x", "z", s = 12)
             .set(xlim = (-75, -20), ylim = (-.1, 7))
             .set_axis_labels("Displacement [$\mu$m]", ""))


    elif view == 'XY':

        sns.set_style('darkgrid')

        g = sns.FacetGrid(nodesData[nodesData['z'] > 1e-1], col = "time", margin_titles = True, aspect = 1.5)

        g = (g.map(sns.scatterplot, "x", "y", s = 7)
             .set(xlim = (-75, -20), ylim = (-14, 14))
             .set_axis_labels("Displacement [$\mu$m]", ""))

    sns.despine(left = True)
    plt.yticks([])

