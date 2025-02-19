

import pandas as pd
import ee

ee.Initialize()

# Load normalization weights table
df_s2_aviris_norm = pd.read_csv("tables/S2_SRF_spectral_responses_2024_4.0.csv")
bands_2 = df_s2_aviris_norm.columns[1:]



band_dict = {
    band.split("_")[-1]: df_s2_aviris_norm.loc[df_s2_aviris_norm["S2A_SR_AV_B1"] != 0, band].tolist()
    for band in bands_2
}


ee.Image("projects/neon-prod-earthengine/assets/HSI_REFL/001")





# Define band lists
bands_neon = [f"B{str(i).zfill(3)}" for i in range(1, 427)] # NEON Bands
bands_s2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"] # S2 Bands






# Process each row in the table
table = table.iloc[0:6]
for i, row in table.iterrows():

    neon_img = ee.Image(row["image_id_neon"])  # Load NEON image
    s2_alike_bands = {}

    # Compute Sentinel-2 like bands
    for band in bands_s2:
        indices = band_indices[band]
        weights = df_s2_aviris_norm.loc[indices, band].values  # Convolution weights
        expression = " + ".join([f"{w} * b{idx+1}" for idx, w in zip(indices, weights)])
        s2_alike_band = neon_img.expression(expression, {
            f'b{idv+1}': neon_img.select(f"B{str(idv + 1).zfill(3)}") for idv in indices
        })
        s2_alike_bands[band] = s2_alike_band

    # Merge bands into a single image
    s2_alike_image = ee.Image(list(s2_alike_bands.values())).rename(list(bands_s2))

    # Define bounding box parameters
    xmin = row["utm_x"] - 5160 / 2
    ymax = row["utm_y"] + 5160 / 2

    # Define raster transform
    raster_transform = RasterTransform(
        crs=row["utm"],
        geotransform={
            "scaleX": 10,
            "shearX": 0,
            "translateX": xmin,
            "scaleY": -10,
            "shearY": 0,
            "translateY": ymax
        },
        width=516,
        height=516
    )

    # Create raster transform set
    raster_transform_set = RasterTransformSet(rastertransformset=[raster_transform])

    # Generate manifest for data extraction
    table_manifest = cubexpress.dataframe_manifest(
        geometadatas=raster_transform_set,
        bands=bands_s2,
        image=s2_alike_image,
    )

    # Set output filename
    table_manifest["outname"] = row["ID"] + ".tif"

    # Extract and save image
    cubexpress.getCube(table_manifest, nworkers=4, deep_level=5, output_path="/content/NEON")