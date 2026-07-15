import vtk

# Function to inspect the contents of the .vtk file
def inspect_vtk_file(file_path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()

    if not isinstance(polydata, vtk.vtkPolyData):
        print("The file is not of type vtkPolyData.")
        return

    print(f"Number of Points: {polydata.GetNumberOfPoints()}")
    print(f"Number of Cells: {polydata.GetNumberOfCells()}")

    scalars = polydata.GetPointData().GetScalars()
    if scalars:
        print(f"Scalars Name: {scalars.GetName()}")
        print(f"Number of Scalars: {scalars.GetNumberOfTuples()}")
        print(f"Scalars Range: {scalars.GetRange()}")
    else:
        print("No scalars found in the file.")

# Exemple d'utilisation
vtk_file = r"D:\3_INGE\ETS S9\SYS818\projet\vtk\OAS1_0002_MR1_mpr_nn_anon_111_t88_masked_gfc.key.vtk"
inspect_vtk_file(vtk_file)
