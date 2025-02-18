from cubexpress import RasterTransform, RasterTransformSet
import cubexpress
import pandas as pd
import geopandas as gpd
import io
from PIL import Image

import ee
ee.Initialize()

# Load the table of images positions
table_path = "tables/neon_equigrid_geodata.gpkg"
table = gpd.read_file(table_path)
table["ID"] = ["NEON_S2__" + str(i + 1).zfill(4) for i in range(len(table))]

table.to_csv("tables/neon_equigrid_geodata.csv", index=False)

bands_neon = [f"B{str(i).zfill(3)}" for i in range(1, 427)]
bands_s2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]


# Load the table of weights
table_path_norm = "tables/S2toAVIRIS_norm.csv"
df_s2_aviris_norm = pd.read_csv(table_path_norm)
band_indices = {band: df_s2_aviris_norm[df_s2_aviris_norm[band].notnull()].index.to_list()
                for band in bands_s2}

###################
## Processing S2 ##
###################

# table = table.iloc[130:131]
# # Download the images
# for i, row in table.iterrows():
#     # ###########################
#     # ## Processing image neon ##
#     # ###########################
#     # neon_id = row["image_id_neon"]
#     # neon_img = ee.Image(neon_id)
#     # s2_alike_bands = {} # Crear diccionario para almacenar las bandas generadas
#     # for band in bands_s2:

#     #     indices = band_indices[band] 
#     #     weights = df_s2_aviris_norm.loc[indices, band].values  # Pesos de convolución
#     #     neon_band_names = [f"B{str(i + 1).zfill(3)}" for i in indices]
#     #     expression = " + ".join([f"{w} * b{i+1}" for i, w in zip(indices, weights)])

#     #     s2_alike_band = neon_img.expression(expression, {
#     #         f'b{i+1}': neon_img.select(f"B{str(i + 1).zfill(3)}") for i in indices
#     #     })
#     #     s2_alike_bands[band] = s2_alike_band

#     # s2_alike_image = ee.Image(list(s2_alike_bands.values())).rename(list(bands_s2))

#     x_centroid = row["utm_x"]
#     y_centroid = row["utm_y"]
#     xmin = x_centroid - 5120/2
#     ymax = y_centroid + 5120/2
#     raster_transform = RasterTransform(
#         crs=row["utm"],
#         geotransform = dict(
#             scaleX=10,
#             shearX=0,
#             translateX=xmin,
#             scaleY=-10,
#             shearY=0,
#             translateY=ymax
#         ), 
#         width=512, 
#         height=512
#     )
#     raster_transform_set = RasterTransformSet(rastertransformset = [raster_transform])

#     table_manifest = cubexpress.dataframe_manifest(
#         geometadatas=raster_transform_set, 
#         bands=bands_s2, 
#         image=row["image_id_sentinel2"],
#         # image=s2_alike_image,
#     )
#     # table_manifest["outname"] = row["ID"] + ".tif"

#     cubexpress.getCube(table_manifest, nworkers=4, deep_level=5, output_path="s2_try")
#     break



#####################
## Processing NEON ##
#####################

# Download the images
for i, row in table.iterrows():
    ###########################
    ## Processing image neon ##
    ###########################
    neon_id = row["image_id_neon"]
    neon_img = ee.Image(neon_id)
    s2_alike_bands = {} # Crear diccionario para almacenar las bandas generadas
    for band in bands_s2:

        indices = band_indices[band] 
        weights = df_s2_aviris_norm.loc[indices, band].values  # Pesos de convolución
        neon_band_names = [f"B{str(i + 1).zfill(3)}" for i in indices]
        expression = " + ".join([f"{w} * b{i+1}" for i, w in zip(indices, weights)])

        s2_alike_band = neon_img.expression(expression, {
            f'b{i+1}': neon_img.select(f"B{str(i + 1).zfill(3)}") for i in indices
        })
        s2_alike_bands[band] = s2_alike_band

    s2_alike_image = ee.Image(list(s2_alike_bands.values())).rename(list(bands_s2))

    x_centroid = row["utm_x"]
    y_centroid = row["utm_y"]
    xmin = x_centroid - 5120/2
    ymax = y_centroid + 5120/2
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
        width=5120, 
        height=5120
    )
    raster_transform_set = RasterTransformSet(rastertransformset = [raster_transform])

    table_manifest = cubexpress.dataframe_manifest(
        geometadatas=raster_transform_set, 
        bands=bands_s2, 
        image=s2_alike_image,
    )
    table_manifest["outname"] = row["ID"] + ".tif"

    cubexpress.getCube(table_manifest, nworkers=64, deep_level=5, output_path="s2_try")

