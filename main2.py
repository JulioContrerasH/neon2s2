import numpy as np
from typing import Callable
import pandas as pd
import ee
from dataclasses import dataclass

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project="ee-julius013199")

# --------------------------------------
# SpectralData class to manage everything
# --------------------------------------

@dataclass
class SpectralData:
    image: ee.Image
    s2_table: pd.DataFrame
    bands_s2: list

    def __init__(self, image: ee.Image, s2_table: pd.DataFrame):
        self.image = image
        self.s2_table = s2_table
        self.bands_s2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
        
        # Prepare spectral bands and metadata for NEON
        self.bands_neon_select = [f"B{i:03d}" for i in range(1, 427)]
        self.band_metadata_neon = [f"WL_FWHM_{band}" for band in self.bands_neon_select]
        self.bands_neon_ee_select = ee.List(self.bands_neon_select)
        
        # Extract wavelengths for NEON bands from the image
        self.wavelength = ee.List([
            ee.Number.parse(
                ee.String(self.image.get(bandName)).split(",").get(0)
            ) for bandName in self.band_metadata_neon
        ])

    def get_wavelengths(self) -> ee.List:
        return self.wavelength

# --------------------------------------
# 3) Interpolation Functions
# --------------------------------------

def _make_segment_dict(pair: ee.List) -> ee.Dictionary:
    pair = ee.List(pair)
    xvals = ee.List(pair.get(0))  # [x0, x1]
    yvals = ee.List(pair.get(1))  # [y0, y1]
    x0 = ee.Number(xvals.get(0))
    x1 = ee.Number(xvals.get(1))
    y0 = ee.Number(yvals.get(0))
    y1 = ee.Number(yvals.get(1))
    m = y1.subtract(y0).divide(x1.subtract(x0))
    return ee.Dictionary({'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 'm': m})

def prepare_segments(x_values: ee.List, y_values: ee.List) -> ee.List:
    pairs = x_values.slice(0, -1).zip(x_values.slice(1))
    pairs = pairs.zip(y_values.slice(0, -1).zip(y_values.slice(1)))
    segments = pairs.map(lambda pair: _make_segment_dict(pair))
    return segments

def interpolate_one_x(segments: ee.List, xq: ee.Number) -> ee.Number:
    candidates = segments.map(
        lambda seg: ee.Algorithms.If(
            ee.Number(ee.Dictionary(seg).get('x0')).lte(xq).And(
                ee.Number(ee.Dictionary(seg).get('x1')).gte(xq)
            ),
            ee.Number(ee.Dictionary(seg).get('y0')).add(
                ee.Number(ee.Dictionary(seg).get('m'))
                  .multiply(xq.subtract(ee.Number(ee.Dictionary(seg).get('x0'))))
            ),
            None
        )
    )
    return ee.List(candidates).removeAll([None]).get(0)

def linear_interpolation(x_values: ee.List, y_values: ee.List, x_query: ee.List) -> ee.List:
    segs = prepare_segments(x_values, y_values)
    result = x_query.map(lambda xq: interpolate_one_x(segs, ee.Number(xq)))
    return result

def create_interp1d(
    x_values: ee.List, 
    y_values: ee.List, 
    kind: str = 'linear'
) -> Callable[[ee.List], ee.List]:
    if kind != 'linear':
        raise ValueError("Only 'linear' supported.")
    def _interpolator(x_query: ee.List) -> ee.List:
        return linear_interpolation(x_values, y_values, x_query)
    return _interpolator

# --------------------------------------
# 4) Generalized function to generate S2 band from NEON
# --------------------------------------

def generate_s2_band_from_neon(image_neon: ee.Image, s2_table: pd.DataFrame, band_name_s2: str, wave_neon: ee.List, bands_neon_ee_select: ee.List) -> ee.Image:
    """
    Generate one band from NEON using interpolation and Sentinel-2 SRF from the table.
    """
    col_name = 'S2A_SR_AV_' + band_name_s2
    mask = s2_table[col_name] != 0
    xvals = s2_table['SR_WL'][mask].astype(float).tolist()
    yvals = s2_table[col_name][mask].astype(float).tolist()
    s2_srfx = ee.List(xvals)
    s2_srfy = ee.List(yvals)
    x_min = min(xvals)
    x_max = max(xvals)
    widx = wave_neon.map(lambda w: ee.Number(w).gte(x_min).And(ee.Number(w).lte(x_max)))
    neon_srfx = (wave_neon.zip(widx)
                 .map(lambda pair: ee.Algorithms.If(
                     ee.List(pair).get(1), ee.List(pair).get(0), None
                 ))
                 .removeAll([None])
                )
    interp_fun = create_interp1d(s2_srfx, s2_srfy, 'linear')
    neon_srfx_interp = interp_fun(neon_srfx)
    sum_val = neon_srfx_interp.reduce(ee.Reducer.sum())
    neon_srfx_norm = neon_srfx_interp.map(lambda elem: ee.Number(elem).divide(sum_val))
    bands_filt = (bands_neon_ee_select.zip(widx)
                  .map(lambda pair: ee.Algorithms.If(
                      ee.List(pair).get(1), ee.List(pair).get(0), None
                  ))
                  .removeAll([None])
                 )
    selected_bands_img = image_neon.select(bands_filt)
    weights_img = ee.Image.constant(neon_srfx_norm)
    weighted_img = selected_bands_img.multiply(weights_img)
    final_single_band = weighted_img.reduce(ee.Reducer.sum()).rename(band_name_s2)
    return final_single_band

# --------------------------------------
# 5) Generate all S2 bands and combine them into a single image
# --------------------------------------

def generate_s2_image_from_neon(neon_id_image: str, s2_id_image: str) -> ee.Image:
    """
    Generates an image with 13 Sentinel-2 bands from NEON.
    """

    image = ee.Image(neon_id_image)
    image_s2 = ee.Image(s2_id_image)

    # Get spacecraft name to determine which Sentinel-2 table to use
    spacecraft_name = image_s2.get("SPACECRAFT_NAME")
    result = ee.Algorithms.If(
        ee.String(spacecraft_name).equals("Sentinel-2A"),
        "Sentinel-2A",
        ee.Algorithms.If(
            ee.String(spacecraft_name).equals("Sentinel-2B"),
            "Sentinel-2B",
            "Unknown"
        )
    )

    # Select appropriate Sentinel-2 SRF table
    type_s2 = result.getInfo() 

    if type_s2 == "Sentinel-2A":
        s2_table_selected = pd.read_csv("tables/srf_s2a.csv")
    elif type_s2 == "Sentinel-2B":
        s2_table_selected = pd.read_csv("tables/srf_s2b.csv")
    else:
        s2_table_selected = None

    # Create SpectralData instance
    spectral_data = SpectralData(image=image, s2_table=s2_table_selected)
    print(spectral_data.bands_s2)
    
    final_bands = []
    for band in spectral_data.bands_s2:
        one_band_img = generate_s2_band_from_neon(spectral_data.image, spectral_data.s2_table, band, spectral_data.get_wavelengths(), spectral_data.bands_neon_ee_select)
        final_bands.append(one_band_img)
        print(f"Generated band {band}")
    # print(final_bands)

    final_s2_like_image = ee.Image(final_bands).rename(list(spectral_data.bands_s2))
    
    # final_s2_like_image = ee.Image.cat(final_bands)

    return final_s2_like_image

# --------------------------------------
# Example Usage
# --------------------------------------

# neon_id_image = 'projects/neon-prod-earthengine/assets/HSI_REFL/001/2016_HARV_3'
# s2_id_image = 'COPERNICUS/S2_HARMONIZED/20160816T153912_20160816T154443_T18TYN'

# # Generate the Sentinel-2-like image from NEON
# final_s2_like_image = generate_s2_image_from_neon(neon_id_image, s2_id_image)

# # Print the band names of the new image
# print("Band names of the new Sentinel-2 like image =>", final_s2_like_image.bandNames().getInfo())


# --------------------------------------
# Other Example Usage
# --------------------------------------
from cubexpress import RasterTransform, RasterTransformSet
import cubexpress
import pandas as pd

# Metadata
bands_neon = [f"B{str(i).zfill(3)}" for i in range(1, 427)] # NEON Bands
bands_s2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"] # S2 Bands

# Load normalization weights table
df_s2_aviris_norm = pd.read_csv("https://raw.githubusercontent.com/JulioContrerasH/neon2s2/main/tables/S2toAVIRIS_norm.csv")

# Map Sentinel-2 bands to AVIRIS indices
band_indices = {
    band: df_s2_aviris_norm[df_s2_aviris_norm[band].notnull()].index.to_list()
    for band in bands_s2
}

# Table
table = pd.read_csv("tables/neon_end_equigrid_geodata.csv") 


# Process each row in the table
table = table.iloc[21:23] # 22 y 23


# --------------------------------------
# Proccess NEON with the first way
# --------------------------------------

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
    xmin = row["utm_x"] - 5120 / 2
    ymax = row["utm_y"] + 5120 / 2

    # Define raster transform
    raster_transform = RasterTransform(
        crs=row["utm"],
        geotransform={
            "scaleX": 1,
            "shearX": 0,
            "translateX": xmin,
            "scaleY": -1,
            "shearY": 0,
            "translateY": ymax
        },
        width=512,
        height=512
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
    cubexpress.getCube(table_manifest, nworkers=4, deep_level=5, output_path="comparative/luis/")


# --------------------------------------
# Download S2 
# --------------------------------------

for i, row in table.iterrows():

    s2_img = ee.Image(row["image_id_sentinel2"])  # Load NEON image

    # Define bounding box parameters
    xmin = row["utm_x"] - 5120 / 2
    ymax = row["utm_y"] + 5120 / 2

    # Define raster transform
    raster_transform = RasterTransform(
        crs=row["utm"],
        geotransform={
            "scaleX": 1,
            "shearX": 0,
            "translateX": xmin,
            "scaleY": -1,
            "shearY": 0,
            "translateY": ymax
        },
        width=512,
        height=512
    )

    # Create raster transform set
    raster_transform_set = RasterTransformSet(rastertransformset=[raster_transform])

    # Generate manifest for data extraction
    table_manifest = cubexpress.dataframe_manifest(
        geometadatas=raster_transform_set,
        bands=bands_s2,
        image=s2_img,
    )

    # Set output filename
    table_manifest["outname"] = row["ID"] + ".tif"

    # Extract and save image
    cubexpress.getCube(table_manifest, nworkers=4, deep_level=5, output_path="comparative/s2")


# --------------------------------------
# Proccess NEON with the second way
# --------------------------------------

# neon_id_image = 'projects/neon-prod-earthengine/assets/HSI_REFL/001/2016_HARV_3'
# s2_id_image = 'COPERNICUS/S2_HARMONIZED/20160816T153912_20160816T154443_T18TYN'

# # Generate the Sentinel-2-like image from NEON
# final_s2_like_image = generate_s2_image_from_neon(neon_id_image, s2_id_image)

for i, row in table.iterrows():

    neon_id_image = row["image_id_neon"]
    s2_id_image = row["image_id_sentinel2"]

    # Generate the Sentinel-2-like image from NEON
    final_s2_like_image = generate_s2_image_from_neon(neon_id_image, s2_id_image)
  
    # Define bounding box parameters
    xmin = row["utm_x"] - 5120 / 2
    ymax = row["utm_y"] + 5120 / 2

    # Define raster transform
    raster_transform = RasterTransform(
        crs=row["utm"],
        geotransform={
            "scaleX": 1,
            "shearX": 0,
            "translateX": xmin,
            "scaleY": -1,
            "shearY": 0,
            "translateY": ymax
        },
        width=512,
        height=512
    )

    # Create raster transform set
    raster_transform_set = RasterTransformSet(rastertransformset=[raster_transform])

    # Generate manifest for data extraction
    table_manifest = cubexpress.dataframe_manifest(
        geometadatas=raster_transform_set,
        bands=bands_s2,
        image=final_s2_like_image,
    )

    # Set output filename
    table_manifest["outname"] = row["ID"] + ".tif"

    # Extract and save image
    cubexpress.getCube(table_manifest, nworkers=4, deep_level=5, output_path="comparative/neon_second")



# --------------------------------------
# Testing errors
# --------------------------------------

from cubexpress.download import getCube_batch
from cubexpress.download_utils import image_from_manifest
from cubexpress.download_utils import computePixels_np
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import json


table = table_manifest

nworkers=4
deep_level=5
output_path="comparative/neon_second"

quiet = False

results = []
with ThreadPoolExecutor(max_workers=nworkers) as executor:
    futures = {
        executor.submit(getCube_batch, row, output_path, deep_level, quiet): row
        for _, row in table.iterrows()
    }
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            if result:
                results.append(result)
        except Exception as e:
            print(f"sfsf")
row["outname"]


for _, row in table.iterrows():
    row = row

manifest_dict = json.loads(row.manifest) if isinstance(row.manifest, str) else row.manifest

data_np = image_from_manifest(
    manifest_dict=manifest_dict,
    max_deep_level=5,
    quiet=quiet
)

if 'expression' in manifest_dict:
    print("Hola")


manifest_dict = eval(str(manifest_dict))
manifest_dict["expression"] = ee.deserializer.decode(
    json.loads(manifest_dict["expression"])
)

computePixels_np(manifest_dict, 5, quiet)


if 'assetId' in manifest_dict:
    return getPixels_np(manifest_dict, max_deep_level, quiet)
elif 'expression' in manifest_dict:

else:
    raise ValueError("Manifest does not contain 'assetId' or 'expression'")







row.manifest["expression"]

a = row.manifest
a.iloc[0]['expression']

print(a["expression"])