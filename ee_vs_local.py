# #########
# ## GEE ##
# #########

# from cubexpress import RasterTransform, RasterTransformSet
# import cubexpress
# import pandas as pd

# import ee
# ee.Initialize()

# # Load the table of images positions
# table = pd.read_csv("tables/neon_end_equigrid_geodata.csv")

# bands_neon = [f"B{str(i).zfill(3)}" for i in range(1, 427)]
# bands_s2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

# # Load the table of weights
# table_path_norm = "tables/S2toAVIRIS_norm.csv"
# df_s2_aviris_norm = pd.read_csv(table_path_norm)
# band_indices = {band: df_s2_aviris_norm[df_s2_aviris_norm[band].notnull()].index.to_list()
#                 for band in bands_s2}

# table = table.iloc[20:21]

# for i, row in table.iterrows():

#     neon_img = ee.Image(row["image_id_neon"])
#     s2_alike_bands = {} 

#     for band in bands_s2:
#         indices = band_indices[band] 
#         weights = df_s2_aviris_norm.loc[indices, band].values  # Pesos de convolución
#         neon_band_names = [f"B{str(idw + 1).zfill(3)}" for idw in indices]
#         expression = " + ".join([f"{w} * b{idx+1}" for idx, w in zip(indices, weights)])
#         s2_alike_band = neon_img.expression(expression, {
#             f'b{idv+1}': neon_img.select(f"B{str(idv + 1).zfill(3)}") for idv in indices
#         })
#         s2_alike_bands[band] = s2_alike_band

#     s2_alike_image = ee.Image(list(s2_alike_bands.values())).rename(list(bands_s2))

#     xmin = row["utm_x"] - 5160/2
#     ymax = row["utm_y"] + 5160/2
    
#     raster_transform = RasterTransform(
#         crs=row["utm"],
#         geotransform = dict(
#             scaleX=1,
#             shearX=0,
#             translateX=xmin,
#             scaleY=-1,
#             shearY=0,
#             translateY=ymax
#         ), 
#         width=128, 
#         height=128
#     )

#     raster_transform_set = RasterTransformSet(rastertransformset = [raster_transform])

#     table_manifest = cubexpress.dataframe_manifest(
#         geometadatas=raster_transform_set, 
#         bands=bands_s2, 
#         image=s2_alike_image,
#     )

#     table_manifest["outname"] = row["ID"] + ".tif"

#     cubexpress.getCube(table_manifest, nworkers=4, deep_level=5, output_path="vs")


#############################
## Only download the image ##
#############################

#########
## GEE ##
#########

from cubexpress import RasterTransform, RasterTransformSet
import cubexpress
import pandas as pd

import ee
ee.Initialize()

# Load the table of images positions
table = pd.read_csv("tables/neon_end_equigrid_geodata.csv")

bands_neon = [f"B{str(i).zfill(3)}" for i in range(1, 427)]
bands_s2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

# Load the table of weights
table_path_norm = "tables/S2toAVIRIS_norm.csv"
df_s2_aviris_norm = pd.read_csv(table_path_norm)
band_indices = {band: df_s2_aviris_norm[df_s2_aviris_norm[band].notnull()].index.to_list()
                for band in bands_s2}

table = table.iloc[20:21]

for i, row in table.iterrows():

    # neon_img = ee.Image(row["image_id_neon"])
    # s2_alike_bands = {} 

    # for band in bands_s2:
    #     indices = band_indices[band] 
    #     weights = df_s2_aviris_norm.loc[indices, band].values  # Pesos de convolución
    #     neon_band_names = [f"B{str(idw + 1).zfill(3)}" for idw in indices]
    #     expression = " + ".join([f"{w} * b{idx+1}" for idx, w in zip(indices, weights)])
    #     s2_alike_band = neon_img.expression(expression, {
    #         f'b{idv+1}': neon_img.select(f"B{str(idv + 1).zfill(3)}") for idv in indices
    #     })
    #     s2_alike_bands[band] = s2_alike_band

    # s2_alike_image = ee.Image(list(s2_alike_bands.values())).rename(list(bands_s2))

    xmin = row["utm_x"] - 5160/2
    ymax = row["utm_y"] + 5160/2
    
    raster_transform = RasterTransform(
        crs=row["utm"],
        geotransform = dict(
            scaleX=1,
            shearX=0,
            translateX=xmin,
            scaleY=-1,
            shearY=0,
            translateY=ymax
        ), 
        width=128, 
        height=128
    )

    raster_transform_set = RasterTransformSet(rastertransformset = [raster_transform])

    table_manifest = cubexpress.dataframe_manifest(
        geometadatas=raster_transform_set, 
        bands=bands_neon, 
        image=row["image_id_neon"],
    )

    table_manifest["outname"] = row["ID"] + "_local.tif"

    cubexpress.getCube(table_manifest, nworkers=4, deep_level=5, output_path="vs")


###########
## Local ##
###########


import rasterio as rio
import pandas as pd
import numpy as np
import glob

# Rutas de archivos

table_path = "/home/contreras/Documents/GitHub/NEON/tables/S2toAVIRIS_norm.csv"
df_s2_aviris_norm = pd.read_csv(table_path)
sentinel_bands = df_s2_aviris_norm.columns[2:] # Delete index and band columns

# Crear un diccionario con los índices de bandas NEON que corresponden a cada banda Sentinel-2
band_indices = {band: df_s2_aviris_norm[df_s2_aviris_norm[band].notnull()].index.to_list()
                for band in sentinel_bands}

# Listar tif glob 
neon_images = glob.glob("vs/*local.tif")
# neon_images_s2 = [f[:-4] + "_s2.tif" for f in neon_images]

for j, neon_image_path in enumerate(neon_images):
    # Cargar la imagen NEON hiperespectral
    with rio.open(neon_image_path) as src:
        neon_img = src.read()  # Leer todas las bandas (426, H, W)
        profile = src.profile  # Guardar metadatos

    # Obtener dimensiones de la imagen
    _, height, width = neon_img.shape

    # Crear un array para la imagen "S2-alike" con 12 bandas
    s2_alike_img = np.zeros((len(sentinel_bands), height, width), dtype=np.float32)

    # Aplicar la convolución para generar cada banda de Sentinel-2
    for i, band in enumerate(sentinel_bands):

        indices = band_indices[band]  # Bandas de NEON a usar
        weights = df_s2_aviris_norm.loc[indices, band].values  # Pesos de convolución

        # Asegurar que los pesos no tengan NaN
        weights = np.nan_to_num(weights)

        # Multiplicar las bandas seleccionadas por sus pesos y sumarlas
        s2_alike_img[i] = np.tensordot(weights, neon_img[indices, :, :], axes=([0], [0]))

    # Actualizar el perfil para la nueva imagen con 12 bandas
    profile.update(count=s2_alike_img.shape[0], dtype='float32')

    # Guardar la imagen resultante
    with rio.open("vs/NEON_S2__0021_local_procces.tif", "w", **profile) as dst:
        dst.write(s2_alike_img)

    print(f"Imagen Sentinel-2 generada: NEON_S2__0021_local_procces.tif")





