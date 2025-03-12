import ee
import pandas as pd
from typing import Callable
import cubexpress
from dataclasses import dataclass 

try:
    ee.Initialize(project="ee-julius013199")
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project="ee-julius013199")


#################################
## Download NEON and S2 images ##
#################################

# ---------------------------------------------------------------------------------------------
# SpectralData class to manage everything
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
    
# Interpolation Functions
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

def linear_interpolation(
        x_values: ee.List, 
        y_values: ee.List, 
        x_query: ee.List
    ) -> ee.List:
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


# Generalized function to generate S2 band from NEON
def generate_s2_band_from_neon(
        image_neon: ee.Image, 
        s2_table: pd.DataFrame, 
        band_name_s2: str, 
        wave_neon: ee.List, 
        bands_neon_ee_select: ee.List
    ) -> ee.Image:
    """
    Generate one band from NEON using interpolation and Sentinel-2 SRF from the table.
    """
    col_name = s2_table.columns[1][:-2] + band_name_s2
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

# Generate all S2 bands and combine them into a single image
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
        s2_table_selected = pd.read_csv("https://raw.githubusercontent.com/JulioContrerasH/neon2s2/refs/heads/main/tables/srf_s2a.csv")
    elif type_s2 == "Sentinel-2B":
        s2_table_selected = pd.read_csv("https://raw.githubusercontent.com/JulioContrerasH/neon2s2/refs/heads/main/tables/srf_s2b.csv")
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

    final_s2_like_image = ee.Image(final_bands).rename(list(spectral_data.bands_s2))

    return final_s2_like_image
# -----------------------------------------------------------------------------------------

table = pd.read_csv("tables/images_neon_s2_final_pairs.csv")
table = table.iloc[67:]

for i, row in table.iterrows():

    neon_id_image = row["neon_id"]
    s2_id_image = row["sentinel2_id"]

    # Generate the Sentinel-2-like image from NEON
    final_s2_like_image = generate_s2_image_from_neon(neon_id_image, s2_id_image)

    # Define bounding box parameters
    xmin = row["utm_x"] - 5160 / 2
    ymax = row["utm_y"] + 5160 / 2

    metadata = cubexpress.RasterTransform(
        crs=row.crs,
        geotransform={
            'scaleX': 1, 
            'shearX': 0, 
            'translateX': xmin,
            'scaleY': -1, 
            'shearY': 0, 
            'translateY': ymax
        },
        width=5160,
        height=5160
    )

    request = cubexpress.Request(
        id=row.id,
        raster_transform=metadata,
        bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
        image=final_s2_like_image
    )

    cube_requests = cubexpress.RequestSet(requestset=[request])

    cubexpress.getcube(
        request=cube_requests,
        output_path="/media/contreras/LaCie/output_neon",
        nworkers=4,
        max_deep_level=5
    )
