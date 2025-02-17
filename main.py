import pandas as pd
import ee
import json
import geopandas as gpd


ee.Initialize()

#################################################
## Generate the intersection of the geometries ##
#################################################
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
# Load the data
# df = pd.read_csv('tables/neon002_sentinel_matches.csv')
# df.columns
# df["image_id_neon"] = df["Collection_NEON"] + "/" + df["ID_NEON"]
# df["image_id_sentinel2"] = df["Collection_Sentinel"] + "/" + df["ID_SENTINEL2"]
# df = df.reset_index(drop=True)
df = pd.read_csv('tables/neon_s2_union.csv')
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

features = intersection["features"]  # Lista de Features

# Verificamos que coincida la longitud
print(len(features), len(df))

# Creamos listas para almacenar las geometrías
list_polygons = []
list_centroids = []

for feat in features:
    # Convertimos la geometría GeoJSON a un shapely.geometry
    geom_shapely = shape(feat["geometry"])
    
    # Polígono completo
    list_polygons.append(geom_shapely)
    
    # Centroide (un shapely Point)
    list_centroids.append(geom_shapely.centroid)

# Ahora, asignamos esas geometrías a nuevas columnas del df
# Ojo: asumimos que el orden de `features` es el mismo que en df
df["polygono"] = list_polygons
df["centroide"] = list_centroids

# Si nuestra intención final es tener un GeoDataFrame con PUNTOS,
# usamos la columna "centroide" como la geometría principal:
gdf_puntos = gpd.GeoDataFrame(df, geometry="centroide", crs="EPSG:4326")

# En cambio, si quisiéramos un GeoDataFrame con los POLÍGONOS:
gdf_poligonos = gpd.GeoDataFrame(df, geometry="polygono", crs="EPSG:4326")

# Finalmente, exportamos según convenga:

# 1) A GeoJSON
gdf_puntos.to_file("salida_puntos.geojson", driver="GeoJSON")
gdf_poligonos.to_file("salida_poligonos.geojson", driver="GeoJSON")

# 2) A GeoPackage (solo se puede guardar una capa por archivo,
#    pero puedes guardar múltiples capas indicando un layer distinto).
gdf_puntos.to_file("intersection.gpkg", layer="centroides", driver="GPKG")
gdf_poligonos.to_file("intersection.gpkg", layer="poligonos", driver="GPKG")









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

from cubexpress import RasterTransform, RasterTransformSet
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

# write the table to a CSV file
table.to_csv('tables/neon_s2_union.csv', index=False)

path_good = pathlib.Path("neon_images/good")
pattern = re.compile(r'COPERNICUS_S2_HARMONIZED_(\d{8}T\d{6}_\d{8}T\d{6}_T\d{2}[A-Z]{3})')
extracted_names = [match.group(1) for file in path_good.glob("*.tif") if (match := pattern.search(file.stem))]
filtered_table = table[table["ID_SENTINEL2"].isin(extracted_names)]

# select 5 random rows
table = filtered_table.sample(5)

bands_neon = [f"B{str(i).zfill(3)}" for i in range(1, 427)]
# bands_s2 = ["B2", "B3", "B4"]
bands_s2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

# Download the images

for i, row in table.iterrows():
    x_centroid = row["x"]
    y_centroid = row["y"]
    xmin = x_centroid - 256 * 10
    ymax = y_centroid + 256 * 10
    raster_transform = RasterTransform(
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
    raster_transform_set = RasterTransformSet(rastertransformset = [raster_transform])

    table_manifest = cubexpress.dataframe_manifest(
        geometadatas=raster_transform_set, 
        bands=bands_s2, 
        image=row["sentinel2_id"],
    )
    cubexpress.getCube(table_manifest, nworkers=4, deep_level=5, output_path="neon_images")



###################################
## Download images with equigrid ##
###################################
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import utm

def get_utm_epsg(lat, lon):
    x, y, zone, _ = utm.from_latlon(lat, lon)
    epsg_code = f"326{zone:02d}" if lat >= 0 else f"327{zone:02d}"
    return int(epsg_code)

polygons = gpd.read_file("tables/intersection.gpkg", layer="poligonos")
points = gpd.read_file("equigrid/NA.gpkg")
columns_to_drop = ["utm", "lon", "lat", "utm_x", "utm_y", "mgrs_tiles", "ntiles"]
points = points.drop(columns=columns_to_drop)

if polygons.crs != points.crs:
    points = points.to_crs(polygons.crs)

tables = []

for i in range(len(polygons)):
    polygon = polygons.iloc[[i]]
    points_within = points[points.within(polygon.union_all())]
    buff_points = []
    for _, point in points_within.iterrows():
        lat, lon = point.geometry.y, point.geometry.x
        epsg_code = get_utm_epsg(lat, lon)
        x, y, _, _ = utm.from_latlon(lat, lon)
        point_utm = Point(x, y)
        buffer_geom = point_utm.buffer(2600)
        buffer_geographic = gpd.GeoSeries([buffer_geom], crs=f"EPSG:{epsg_code}").to_crs(points.crs)
        buff_points.append(buffer_geographic.iloc[0])

    buffers = gpd.GeoDataFrame(geometry=buff_points, crs=points.crs)
    points_within_end = points_within.loc[buffers.geometry.within(polygon.geometry.iloc[0]).values]

    if len(points_within_end) == 0:
        table_end = None
    else:
        polygon_repeated = pd.concat([polygon] * len(points_within_end), ignore_index=True)
        points_within_end = points_within_end.reset_index(drop=True).drop(columns=["geometry"])
        table_end = pd.concat([polygon_repeated, points_within_end.reset_index(drop=True)], axis=1)

    tables.append(table_end)
    print(f"Número de puntos dentro del polígono: {len(points_within)}")
    print(f"Número de buffers dentro del polígono: {len(points_within_end)}")

final_table = pd.concat(tables, ignore_index=True)

final_table.to_file("tables/neon_equigrid_geodata.gpkg", layer="neon_equigrid_geodata", driver="GPKG")


###############################
## Convolution of NEON to S2 ##
###############################

import rasterio as rio
import pandas as pd
import numpy as np
import glob

# Rutas de archivos


table_path = "/home/contreras/Documents/GitHub/NEON/tables/S2toAVIRIS_norm.csv"

df_s2_aviris_norm = pd.read_csv(table_path)

sentinel_bands = df_s2_aviris_norm.columns[2:] 

# Crear un diccionario con los índices de bandas NEON que corresponden a cada banda Sentinel-2
band_indices = {band: df_s2_aviris_norm[df_s2_aviris_norm[band].notnull()].index.to_list()
                for band in sentinel_bands}


# Listar tif glob 
neon_images = glob.glob("neon_images/intersections/neon/*.tif")
neon_images_s2 = [f[:-4] + "_s2.tif" for f in neon_images]

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
    with rio.open(neon_images_s2[j], "w", **profile) as dst:
        dst.write(s2_alike_img)

    print(f"Imagen Sentinel-2 generada: {neon_images_s2[j]}")


