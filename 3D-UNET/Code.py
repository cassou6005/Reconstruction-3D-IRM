import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Concatenate, Input, Conv3DTranspose
from tensorflow.keras.models import Model
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanSquaredError


# Fonction pour redimensionner un volume 3D
def resize_3d_volume(volume, target_shape):
    zoom_factors = [target_shape[i] / volume.shape[i] for i in range(3)]
    resized_volume = zoom(volume, zoom_factors, order=1)
    return resized_volume


# Fonction pour redimensionner un volume 3D avec descripteurs
def resize_3d_volume_with_descriptor(volume, target_shape):
    zoom_factors = [target_shape[i] / volume.shape[i] for i in range(3)] + [1]
    resized_volume = zoom(volume, zoom_factors, order=1)
    return resized_volume


# Fonction pour charger les keypoints et descripteurs depuis un fichier
def load_keypoints_with_descriptors(keypoint_file, target_shape=(64, 64, 64)):
    try:
        keypoint_file = keypoint_file.numpy().decode('utf-8') #remplac�e par:
        #keypoint_file = keypoint_file.decode('utf-8') if isinstance(keypoint_file, bytes) else keypoint_file

        points, descriptors = [], []
        with open(keypoint_file, 'r') as file:
            lines = file.readlines()
            for line in lines[6:]:  # Ignorer les 6 premi�res lignes
                try:
                    x, y, z = np.array(line.split(sep="\t")[:3], dtype=float)
                    descriptor = np.array(line.split()[17:17+64], dtype=int)
                    points.append((x, y, z))
                    descriptors.append(descriptor)
                except ValueError:
                    continue
        
        points = np.array(points)
        descriptors = np.array(descriptors)
        
        # Cr�er une grille 3D pour les descripteurs
        descriptor_grid = np.zeros((128, 128, 128, 64))  # Grille des descripteurs
        
        for i, (x, y, z) in enumerate(points):
            x_idx = int(x * (descriptor_grid.shape[0] - 1) / 176)
            y_idx = int(y * (descriptor_grid.shape[1] - 1) / 208)
            z_idx = int(z * (descriptor_grid.shape[2] - 1) / 176)
            descriptor_grid[x_idx, y_idx, z_idx, :] = descriptors[i]
        
        descriptor_grid_resized = resize_3d_volume_with_descriptor(descriptor_grid, target_shape[:3])
        
        return descriptor_grid_resized.astype(np.float32)

    except Exception as e:
        print(f"Erreur lors du chargement des keypoints depuis {keypoint_file}: {e}")
        return np.zeros((*target_shape, 64), dtype=np.float32)


# Fonction pour charger une image NIfTI (.nii)
def load_image(image_file, img_shape=(64, 64, 64)):
    try:
        img_data = nib.load(image_file.numpy().decode("utf-8")).get_fdata()
        image_resized = np.expand_dims(resize_3d_volume(img_data[:,:,:,0], img_shape[:3]), axis=-1)
        return image_resized.astype(np.float32)
    except Exception as e:
        print(f"Erreur lors du chargement de l'image depuis {image_file}: {e}")
        return np.zeros(img_shape, dtype=np.float32)


# Fonction pour cr�er un dataset � partir des keypoints
def create_keypoint_dataset(keypoint_dir, target_shape=(64, 64, 64, 64)):
    keypoint_files = [
        os.path.join(keypoint_dir, filename)
        for filename in os.listdir(keypoint_dir)
        if filename.endswith(".key") and os.path.isfile(os.path.join(keypoint_dir, filename))
    ]
    
    def load_keypoints_wrapper(keypoint_file):
        def decode_and_load_keypoints(file):
            return load_keypoints_with_descriptors(file)
        
        descriptor_grid = tf.py_function(func=decode_and_load_keypoints, inp=[keypoint_file], Tout=tf.float32)
        descriptor_grid.set_shape(target_shape)
        return descriptor_grid

    dataset = tf.data.Dataset.from_tensor_slices(keypoint_files)
    dataset = dataset.map(load_keypoints_wrapper)
    
    return dataset


# Fonction pour cr�er un dataset � partir des images
def create_image_dataset(image_dir, img_shape=(64, 64, 64, 1)):
    image_files = [
        os.path.join(image_dir, filename)
        for filename in os.listdir(image_dir)
        if filename.endswith(".img") and os.path.isfile(os.path.join(image_dir, filename))
    ]
    
    def load_image_wrapper(image_file):
        def decode_and_load_image(file):
            return load_image(file)
        
        image_data = tf.py_function(func=decode_and_load_image, inp=[image_file], Tout=tf.float32)
        image_data.set_shape(img_shape)
        return image_data

    return tf.data.Dataset.from_tensor_slices(image_files).map(load_image_wrapper)


# D�finir le mod�le UNet
def unet_model(input_shape=(64, 64, 64, 1), descriptor_shape=(64, 64, 64, 64)):
    keypoint_input = Input(input_shape, name='keypoint_input')
    descriptor_input = Input(descriptor_shape, name='descriptor_input')

    x = Concatenate()([keypoint_input, descriptor_input])

    c1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    c1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(p1)
    c2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(p2)
    c3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(p3)
    c4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(c4)

    u5 = Conv3DTranspose(128, (3, 3, 3), activation='relu', padding='same', strides=(2,2,2))(c4)
    u5 = Concatenate()([u5, c3])
    c5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(u5)
    c5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(c5)

    u6 = Conv3DTranspose(64, (3, 3, 3), activation='relu', padding='same', strides=(2,2,2))(c5)
    u6 = Concatenate()([u6, c2])
    c6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(u6)
    c6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c6)

    u7 = Conv3DTranspose(32, (3, 3, 3), activation='relu', padding='same', strides=(2,2,2))(c6)
    u7 = Concatenate()([u7, c1])
    c7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(u7)
    c7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c7)

    outputs = Conv3D(1, (1, 1, 1), activation='linear')(c7)
    model = Model(inputs=[keypoint_input, descriptor_input], outputs=outputs)

    return model


# Cr�er le dataset combin�
def create_combined_dataset(keypoint_dataset, image_dataset):
    def map_to_model_inputs(keypoint_data, image_data):
        return (keypoint_data, image_data), image_data

    combined_dataset = tf.data.Dataset.zip((keypoint_dataset, image_dataset)).map(map_to_model_inputs)
    return combined_dataset


# Charger les datasets de keypoints et images
keypoint_dir = r"C:\Users\alici\Desktop\Polytech\cours\ETS S1\IA\projet\key"
image_dir = r"C:\Users\alici\Desktop\Polytech\cours\ETS S1\IA\projet\img"

keypoint_dataset = create_keypoint_dataset(keypoint_dir)
image_dataset = create_image_dataset(image_dir)

train_dataset = create_combined_dataset(keypoint_dataset, image_dataset).batch(1).prefetch(tf.data.AUTOTUNE)

# Ajout de SSIM comme metrique
def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

#----------------------------------------------------------------
# Cr�er et compiler le mod�le
#model = unet_model()
#model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', ssim_metric], run_eagerly=True)

# Entra�ner le mod�le
#history = model.fit(train_dataset, epochs=10)

def plot_training_history(history):
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Tracage de la courbe de la perte (loss)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history.history['loss'], label='Perte d\'entrainement', color='b')
    plt.title('Perte au cours de l\'entrainement')
    plt.xlabel('�poques')
    plt.ylabel('Perte')
    plt.legend()
    
    # Tracage de  la courbe de MSE (Mean Squared Error)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history.history['mse'], label='MSE (Mean Squared Error)', color='blue')
    plt.title('MSE au cours de l\'entrainement')
    plt.xlabel('�poques')
    plt.ylabel('MSE')
    plt.legend()

    # Tracage de la courbe de MSE (Mean Squared Error) a partir de l'epoque 2
    epochs_filtered = epochs[1:]  # Filtrage pour ignorer l'epoque 1
    mse_filtered = history.history['mse'][1:]  # Filtrage pour ignorer le MSE de l'epoque 1
    plt.subplot(2, 2, 3)
    plt.plot(epochs_filtered, mse_filtered, label='MSE d\'entrainement', color='b')
    plt.title("MSE au cours de l\'entrainement � partir de l'epoque 2")
    plt.xlabel('�poques')
    plt.ylabel('MSE')
    plt.legend()

    # Tracage de la courbe de SSIM
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history.history['ssim_metric'], label='SSIM d\'entrainement', color='b')
    plt.title('SSIM au cours de l\'entrainement')
    plt.xlabel('�poques')
    plt.ylabel('SSIM')
    plt.legend()

    # Affichage de tous les graphiques
    plt.tight_layout()
    plt.show()

# Appel de la fonction de trace apres l'entrainement
#plot_training_history(history)

# Sauvegarder le mod�le
#model.save('unet_model.h5')
#----------------------------------------------------------------


#----------------------------------------------------------------
# Chargement du mod�le U-Net avec correction si d�ja enregistr�
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)  # Ajouter mse dans custom_objects
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', ssim_metric], run_eagerly=True)# Sp�cifier la m�trique explicitement
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du mod�le: {e}")
        return None

model_path = "unet_model.h5"

# Charger le mod�le U-Net et le compiler avec la m�trique
model = load_model(model_path)
#---------------------------------------------------------------


# # Reconstruction et sauvegarde d'une image a partir des keypoints et des descripteurs
for (inputs, target) in train_dataset.take(1):  
    
    keypoints, descripteurs = inputs  
    
    # Predire l'image reconstruite avec le modele UNet
    almost_reconstructed_image = model.predict([keypoints, descripteurs])
    
    # Reorganiser les axes apres la generation de l'image 3D
    almost_reconstructed_image1 = np.transpose(almost_reconstructed_image, (0, 1 , 2, 3 ,4))
    almost_reconstructed_image2 = np.flip(almost_reconstructed_image1, axis=2)
    
    # Sauvegarde de l'image predite au format NIfTI
    reconstructed_image = nib.Nifti1Pair(almost_reconstructed_image2[0], np.eye(4))
    nib.save(reconstructed_image, 'image_reconstruite_finale.img')
    
    # Assurez-vous d'extraire les donn�es du NIfTI
    reconstructed_data = reconstructed_image.get_fdata()

    # Tranche que vous voulez visualiser
    tranche = 30

    # Visualisation de la coupe s�lectionn�e
    plt.imshow(reconstructed_data[:, :, tranche], cmap='gray')
    plt.title("Image reconstruite brute - Coupe axiale")
    plt.colorbar()
    plt.show()

# Visualisation de la coupe s�lectionn�e
    plt.imshow(reconstructed_data[:, tranche,: ], cmap='gray')
    plt.title("Image reconstruite brute - Coupe coronale")
    plt.colorbar()
    plt.show()

    plt.imshow(reconstructed_data[tranche, : , :], cmap='gray')
    plt.title("Image reconstruite brute - Coupe sagittal")
    plt.colorbar()
    plt.show()
  
    # Chargement de l'image originale
    ground_truth = nib.load(r"C:\Users\alici\Desktop\Polytech\cours\ETS S1\IA\projet\key_img_hdr\OAS1_0400_MR1_mpr_nn_anon_111_t88_masked_gfc.img")
    reconstructed_image_2 = nib.load('image_reconstruite_finale.img')

    # Conversion en tableaux NumPy
    ground_truth = ground_truth.get_fdata()
    ground_truth = np.flip(ground_truth, axis=1) 
    reconstructed = reconstructed_image_2.get_fdata()

    # Calcul du facteur d'echelle pour chaque dimension
    scale_factors = (
        ground_truth.shape[0] / reconstructed.shape[0],
        ground_truth.shape[1] / reconstructed.shape[1],
        ground_truth.shape[2] / reconstructed.shape[2],
        ground_truth.shape[3] / reconstructed.shape[3]
    )

    # Appliquer le redimensionnement avec interpolation
    reconstructed_resized = zoom(reconstructed, scale_factors, order=1)  

    # Calcul de la difference absolue
    difference_absolute = np.abs(ground_truth - reconstructed_resized)

    

    # Affichage de la difference
    plt.imshow(difference_absolute[:, tranche, :], cmap='gray')
    plt.title("Diff�rence absolue - Coupe coronale")
    plt.colorbar() 
    plt.show()

    plt.imshow(difference_absolute[tranche, : , :], cmap='gray')
    plt.title("Diff�rence absolue - Coupe sagittal")
    plt.colorbar()
    plt.show()

    plt.imshow(difference_absolute[:, :, tranche], cmap='gray')
    plt.title("Diff�rence absolue - Coupe axiale")
    plt.colorbar()
    plt.show()


#----------------------------------------------------
#NE FONCTIONNE PAS POUR 1 CAS PRECIS
# for layer in model.layers:
#     print(layer.name, layer.get_weights())

# # Sp�cifiez le chemin vers le fichier keypoint pr�cis
# keypoint_file_path = r"C:\Users\alici\Desktop\Polytech\cours\ETS S1\IA\projet\key_img_hdr\OAS1_0457_MR1_mpr_nn_anon_111_t88_masked_gfc.key"

# # Charger les keypoints et descripteurs depuis le fichier
# keypoints_with_descriptors = load_keypoints_with_descriptors(keypoint_file_path, target_shape=(64, 64, 64))

# # Ajouter une dimension batch pour compatibilit� avec TensorFlow
# keypoints_with_descriptors = np.expand_dims(keypoints_with_descriptors, axis=0)

# # Cr�er un tensor placeholder pour l'image
# image_placeholder = np.zeros((1, 64, 64, 64, 1), dtype=np.float32)  # Ajustez la taille si n�cessaire

# # �tape 1 : Pr�diction avec le mod�le
# print("D�but de la pr�diction avec le mod�le...")
# almost_reconstructed_image = model.predict([keypoints_with_descriptors,image_placeholder])
# print("Pr�diction termin�e.")

# # �tape 2 : V�rification de la sortie brute
# print("V�rification de la sortie brute du mod�le :")
# print("Dimensions de l'image reconstruite :", almost_reconstructed_image.shape)
# print("Valeurs min/max de l'image reconstruite :", np.min(almost_reconstructed_image), np.max(almost_reconstructed_image))
# print("Valeurs uniques de l'image reconstruite :", np.unique(almost_reconstructed_image))

# tranche = 30  # Tranche au milieu pour visualisation
# # Visualisation de l'image normalisee
# reconstructed_image = (almost_reconstructed_image - np.min(almost_reconstructed_image)) / (np.max(almost_reconstructed_image) - np.min(almost_reconstructed_image))

# plt.imshow(reconstructed_image[0, :, :, tranche, 0], cmap='gray')
# plt.title("Image reconstruite (normalis�e)")
# plt.colorbar()
# plt.show()

# # �tape 3 : Transformation - np.transpose
# print("Application de np.transpose...")
# almost_reconstructed_image1 = np.transpose(almost_reconstructed_image, (0, 1, 2, 3, 4))
# print("Dimensions apr�s np.transpose :", almost_reconstructed_image1.shape)

# # Visualisation apr�s np.transpose
# plt.imshow(almost_reconstructed_image1[0, :, :, tranche, 0], cmap='gray')
# plt.title("Image reconstruite apr�s np.transpose - Coupe axiale")
# plt.colorbar()
# plt.show()

# # �tape 4 : Transformation - np.flip
# print("Application de np.flip...")
# almost_reconstructed_image2 = np.flip(almost_reconstructed_image1, axis=2)
# print("Dimensions apr�s np.flip :", almost_reconstructed_image2.shape)

# # Visualisation apr�s np.flip
# plt.imshow(almost_reconstructed_image2[0, :, :, tranche, 0], cmap='gray')
# plt.title("Image reconstruite apr�s np.flip - Coupe axiale")
# plt.colorbar()
# plt.show()

# # �tape 5 : Sauvegarde en tant qu'image NIfTI
# print("Sauvegarde de l'image reconstruite au format NIfTI...")
# reconstructed_image = nib.Nifti1Pair(almost_reconstructed_image2[0], np.eye(4))
# nib.save(reconstructed_image, 'image_reconstruite_finale.img')
# print("Image reconstruite sauvegard�e sous le nom 'image_reconstruite_finale.img'.")

# # �tape 6 : V�rification de l'image sauvegard�e
# print("Chargement et v�rification de l'image sauvegard�e...")
# reconstructed_loaded = nib.load('image_reconstruite_finale.img').get_fdata()
# print("Dimensions de l'image reconstruite charg�e :", reconstructed_loaded.shape)
# print("Valeurs min/max de l'image reconstruite charg�e :", np.min(reconstructed_loaded), np.max(reconstructed_loaded))

# # Visualisation de l'image sauvegard�e
# plt.imshow(reconstructed_loaded[:, :, tranche], cmap='gray')
# plt.title("Image reconstruite sauvegard�e - Coupe axiale")
# plt.colorbar()
# plt.show()

