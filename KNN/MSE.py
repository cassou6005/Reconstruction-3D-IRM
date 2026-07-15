import SimpleITK as sitk
import numpy as np

# Charger l'image .mhd avec SimpleITK
image1 = sitk.ReadImage(r"C:\Users\alici\Desktop\Polytech\cours\ETS S1\IA\projet\image_58\OAS1_0454_MR1_mpr_nn_anon_111_t88_masked_gfc.hdr")

# Convertir l'image SimpleITK en un tableau NumPy
image1_array = sitk.GetArrayFromImage(image1)

# Si tu as une deuxième image, répète la procédure pour `image2`
image2 = sitk.ReadImage(r"C:\Users\alici\Desktop\Polytech\cours\ETS S1\IA\projet\reconstructed_volume_454_2.mhd")
image2_array = sitk.GetArrayFromImage(image2)

# Calculer l'erreur quadratique moyenne (MSE)
mse = np.mean((image1_array - image2_array) ** 2)
print(f"L'erreur quadratique moyenne (MSE) entre les deux images est : {mse}")
