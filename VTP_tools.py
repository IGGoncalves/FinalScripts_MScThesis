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

