import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np


def vtk_polydata_to_mha(vtk_file_path, mha_file_path):
    """
    Convertit un fichier vtkPolyData en .mha en alignant les scalaires avec un volume 3D.

    Args:
        vtk_file_path (str): Chemin vers le fichier .vtk.
        mha_file_path (str): Chemin pour sauvegarder le fichier .mha.
    """
    # Lecture du fichier .vtk
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    polydata = reader.GetOutput()

    if not isinstance(polydata, vtk.vtkPolyData):
        raise ValueError("Le fichier n'est pas du type vtkPolyData.")

    # Récupérer les points et les scalaires
    points = vtk_to_numpy(polydata.GetPoints().GetData())
    scalars = vtk_to_numpy(polydata.GetPointData().GetScalars())

    # Déterminer les dimensions et la plage spatiale
    bounds = polydata.GetBounds()
    spacing = [
        (bounds[1] - bounds[0]) / 100,  # Diviser la plage en 100 intervalles (ajustable)
        (bounds[3] - bounds[2]) / 100,
        (bounds[5] - bounds[4]) / 100,
    ]
    dimensions = [101, 101, 101]  # Ajusté pour couvrir les données (points + 1)

    # Créer un volume vide
    image = vtk.vtkImageData()
    image.SetDimensions(dimensions)
    image.SetSpacing(spacing)
    image.SetOrigin(bounds[0], bounds[2], bounds[4])
    image.AllocateScalars(vtk.VTK_FLOAT, 1)

    # Remplir le volume avec des valeurs par défaut (zéro)
    volume_data = vtk_to_numpy(image.GetPointData().GetScalars())
    volume_data[:] = 0

    # Mapper les scalaires sur le volume
    for i, point in enumerate(points):
        x_idx = int((point[0] - bounds[0]) / spacing[0])
        y_idx = int((point[1] - bounds[2]) / spacing[1])
        z_idx = int((point[2] - bounds[4]) / spacing[2])
        index = z_idx * dimensions[0] * dimensions[1] + y_idx * dimensions[0] + x_idx
        if 0 <= index < len(volume_data):
            volume_data[index] = scalars[i]

    # Remettre les données scalaires dans l'image
    image.GetPointData().SetScalars(numpy_to_vtk(volume_data, deep=True, array_type=vtk.VTK_FLOAT))

    # Sauvegarder le volume au format .mha
    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName(mha_file_path)
    writer.SetInputData(image)
    writer.Write()

    print(f"Fichier .mha sauvegardé à : {mha_file_path}")


# Exemple d'utilisation
vtk_file = r"D:\3_INGE\ETS S9\SYS818\projet\vtk\OAS1_0002_MR1_mpr_nn_anon_111_t88_masked_gfc.key.vtk"
mha_file = r"D:\3_INGE\ETS S9\SYS818\projet\vtk\OAS1_0002_MR1_mpr_nn_anon_111_t88_masked_gfc.key.mha"

# Convertir et générer le fichier
vtk_polydata_to_mha(vtk_file, mha_file)
