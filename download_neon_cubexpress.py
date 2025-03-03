from cubexpress import RasterTransform, RasterTransformSet
import cubexpress
import pandas as pd

import ee
ee.Initialize(project="ee-julius013199")

# Load the table of images positions
table = pd.read_csv("tables/neon_end_equigrid_geodata.csv")
table = table.iloc[388:]


bands_neon = [f"B{str(i).zfill(3)}" for i in range(1, 427)]
bands_s2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

# Load the table of weights
table_path_norm = "tables/S2toAVIRIS_norm.csv"
df_s2_aviris_norm = pd.read_csv(table_path_norm)
band_indices = {band: df_s2_aviris_norm[df_s2_aviris_norm[band].notnull()].index.to_list()
                for band in bands_s2}

for i, row in table.iterrows():

    neon_img = ee.Image(row["image_id_neon"])
    s2_alike_bands = {} 

    for band in bands_s2:
        indices = band_indices[band] 
        weights = df_s2_aviris_norm.loc[indices, band].values  # Pesos de convolución
        neon_band_names = [f"B{str(idw + 1).zfill(3)}" for idw in indices]
        expression = " + ".join([f"{w} * b{idx+1}" for idx, w in zip(indices, weights)])
        s2_alike_band = neon_img.expression(expression, {
            f'b{idv+1}': neon_img.select(f"B{str(idv + 1).zfill(3)}") for idv in indices
        })
        s2_alike_bands[band] = s2_alike_band

    s2_alike_image = ee.Image(list(s2_alike_bands.values())).rename(list(bands_s2))

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
        width=5160, 
        height=5160
    )

    raster_transform_set = RasterTransformSet(rastertransformset = [raster_transform])

    table_manifest = cubexpress.dataframe_manifest(
        geometadatas=raster_transform_set, 
        bands=bands_s2, 
        image=s2_alike_image,
    )

    table_manifest["outname"] = row["ID"] + ".tif"

    cubexpress.getCube(table_manifest, nworkers=4, deep_level=5, output_path="/media/contreras/LaCie")
