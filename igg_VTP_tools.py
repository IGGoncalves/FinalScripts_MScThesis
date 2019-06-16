### Module to extract and plot information present in VTP files ###

### IMPORT LIBRARIES ###
import vtk
import numpy as np
import pandas as pd


def getNodesData(timesteps, path):
    """
    This function extracts data from VTP files (namely the cell_cell_triangles files) and store it in a DataFrame,
    outputted by the function. The user should indicate the timesteps to be extracted.

    Extracted data:
    coords - xyz coordinates of the nodes

    Keyword arguments:
    timesteps - time points at which the user wants data to be extracted
    path - path to the folder containing all VTP files
    """

    from vtk.util.numpy_support import vtk_to_numpy

    filename = path + 'cell_cell_triangles_' + str(timesteps) + '.vtp'

    ### SETTING UP THE READER ###
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()

    ### GET DATA FROM FILE ###
    data = reader.GetOutput()
    points = data.GetPoints()
    coords = vtk_to_numpy(points.GetData())

    return coords

coords = getNodesData(20, 0, 'AON_FON/extract_files/')
