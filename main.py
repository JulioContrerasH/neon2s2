import pandas as pd
import ee
import json
import geopandas as gpd


ee.Initialize()

#################################################
## Generate the intersection of the geometries ##
#################################################

# Load the data
df = pd.read_csv('tables/neon002_sentinel_matches.csv')
df.columns
df["image_id_neon"] = df["Collection_NEON"] + "/" + df["ID_NEON"]
df["image_id_sentinel2"] = df["Collection_Sentinel"] + "/" + df["ID_SENTINEL2"]
df = df.reset_index(drop=True)

geom_list = []

for i, row in df.iterrows():
    neon_image = ee.Image(row["image_id_neon"])
    sentinel_image = ee.Image(row["image_id_sentinel2"])

    intersection = neon_image.geometry().intersection(sentinel_image.geometry())
    geom_list.append(intersection)

# Intersection of the geometries
intersection = ee.FeatureCollection(geom_list).getInfo()

# To GeoJSON
with open("geometries/neon002_sentinel_matches.geojson", "w") as f:
    json.dump(intersection, f)

###################################################
## Add the centroid of the intersection to table ##
###################################################

# Load the GeoJSON file
with open("geometries/neon001_sentinel_matches.geojson", "r") as f:
    data = json.load(f)

table = pd.read_csv('tables/neon001_sentinel_matches.csv')

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
table["x"] = None
table["y"] = None
table["crs"] = None

list_centroid = []

for i, row in gdf.iterrows():
    geom = row.geometry  # Extraer la geometría de la fila

    if geom.is_empty:
        list_centroid.append(None)
        continue

    geom_series = gpd.GeoSeries([geom], crs="EPSG:4326")
    utm_crs = geom_series.estimate_utm_crs()
    geom_utm = geom_series.to_crs(utm_crs).iloc[0]
    centroid_utm = geom_utm.centroid
    buffer_3km = centroid_utm.buffer(3000)

    if geom_utm.contains(buffer_3km):
        table.loc[i, "x"] = centroid_utm.x
        table.loc[i, "y"] = centroid_utm.y
        table.loc[i, "crs"] = utm_crs.to_authority()[1]
        list_centroid.append((centroid_utm.x, centroid_utm.y, utm_crs.to_authority()[1])) 
    else:
        list_centroid.append(None)

table["inf_centroid"] = list_centroid


# Delete the rows with NaN values
table = table.dropna(subset=["inf_centroid"])
table.reset_index(drop=True)
table = table.drop(columns=["inf_centroid"])

# Save the table to a CSV file
table.to_csv('tables/neon001_s2.csv', index=False)


######################################
## Download the 512x512 s2 and neon ##
######################################

import cubexpress
import pandas as pd
import ee
import re
import pathlib
ee.Initialize()

# Load the table
table1 = pd.read_csv('tables/neon001_s2.csv')
table2 = pd.read_csv('tables/neon002_s2.csv')

# Join the tables
table = pd.concat([table1, table2], axis=0)
table = table.reset_index(drop=True)
table["sentinel2_id"] = table["Collection_Sentinel"] + "/" + table["ID_SENTINEL2"]
table["neon_id"] = table["Collection_NEON"] + "/" + table["ID_NEON"]

path_good = pathlib.Path("neon_images/good")
pattern = re.compile(r'COPERNICUS_S2_HARMONIZED_(\d{8}T\d{6}_\d{8}T\d{6}_T\d{2}[A-Z]{3})')
extracted_names = [match.group(1) for file in path_good.glob("*.tif") if (match := pattern.search(file.stem))]
filtered_table = table[table["ID_SENTINEL2"].isin(extracted_names)]

# select 5 random rows
table = filtered_table.sample(5)

bands_neon = [f"B{str(i).zfill(3)}" for i in range(1, 427)]
bands_s2 = ["B2", "B3", "B4"]
# bands_s2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

# Download the images

for i, row in table.iterrows():
    x_centroid = row["x"]
    y_centroid = row["y"]
    xmin = x_centroid - 256 * 10
    ymax = y_centroid + 256 * 10
    raster_transform = cubexpress.RasterTransform(
        crs=f"EPSG:{row['crs']}", 
        geotransform = dict(
            scaleX=1,
            shearX=0,
            translateX=xmin,
            scaleY=-1,
            shearY=0,
            translateY=ymax
        ), 
        width=1024, 
        height=1024
    )
    raster_transform_set = cubexpress.RasterTransformSet(rastertransformset = [raster_transform])

    table_manifest = cubexpress.dataframe_manifest(
        geometadatas=raster_transform_set, 
        bands=bands_neon, 
        image=row["neon_id"],
    )
    cubexpress.getCube(table_manifest, nworkers=4, deep_level=5, output_path="neon_images")


###############################
## Convolution of NEON to S2 ##
###############################

import rasterio as rio
import pandas as pd
import numpy as np

# Rutas de archivos
neon_image_path = "neon_images/projects_neon-prod-earthengine_assets_HSI_REFL_001_2016_HARV_3__0000.tif"
table_path = "tables/S2toAVIRIS_norm.csv"
output_path = "neon_images/S2_alike_NEON.tif"

df_s2_aviris_norm = pd.read_csv("tables/S2toAVIRIS_norm.csv")

sentinel_bands = df_s2_aviris_norm.columns[2:] 

# Crear un diccionario con los índices de bandas NEON que corresponden a cada banda Sentinel-2
band_indices = {band: df_s2_aviris_norm[df_s2_aviris_norm[band].notnull()].index.to_list()
                for band in sentinel_bands}

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
with rio.open(output_path, "w", **profile) as dst:
    dst.write(s2_alike_img)

print(f"Imagen Sentinel-2 generada: {output_path}")


