import ee
import pandas as pd
import utm
from typing import Tuple, Callable
import matplotlib.pyplot as plt
import cubexpress
from shapely.geometry import shape, Point, box
import geopandas as gpd
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass 

try:
    ee.Initialize(project="ee-julius013199")
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project="ee-julius013199")

def image_to_feature(img):
    ring = ee.Geometry(img.get("system:footprint"))
    poly = ee.Geometry.Polygon([ring.coordinates()])
    ft = ee.Feature(poly, img.toDictionary())
    ft = ft.set({
        "id_geom_array": ee.List([
            img.get("system:id"),
            img.get("system:time_start"),
            poly
        ])
    })
    return ft

def get_utm_epsg(lat, lon):
    x, y, zone, _ = utm.from_latlon(lat, lon)
    epsg_code = f"326{zone:02d}" if lat >= 0 else f"327{zone:02d}"
    return int(epsg_code)

def square_around_point(point_utm, side=2565):
    x_cen, y_cen = point_utm.x, point_utm.y
    half_side = side / 2.0
    return box(x_cen - half_side, y_cen - half_side,
                    x_cen + half_side, y_cen + half_side)

def create_image_with_null_property(row):

    point = ee.Geometry.Point(
        [float(row["lon"]), 
         float(row["lat"])]
    )
    square_5120m = (point
        .transform(row["utm"], 1)
        .buffer(2560)
        .bounds()
        .transform("EPSG:4326", 1)
    )

    base_img = ee.Image(row["neon_id"])
    two_band = ee.Image.constant(1) \
                 .rename("constant") \
                 .addBands(base_img.select("B001"))
    
    count_dict = two_band.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=square_5120m,
        scale=100,
        bestEffort=True
    )
    total_pixels = ee.Number(
        count_dict.get("constant")
    )
    valid_pixels = ee.Number(
        count_dict.get("B001")
    )
    
    null_percent = total_pixels.subtract(valid_pixels) \
                               .divide(total_pixels) \
                               .multiply(100)
    
    return base_img.set({
        "nullPercent": null_percent
    })

def query_utm_crs_info(lon: float, lat: float) -> Tuple[float, float, str]:
    """Converts a pair of lat, lon to UTM coordinates."""
    x, y, zone, _ = utm.from_latlon(lat, lon)
    zone_epsg = f"326{zone:02d}" if lat >= 0 else f"327{zone:02d}"
    return x, y, "EPSG:" + zone_epsg


def image_to_feature(img):
    ring = ee.Geometry(img.get("system:footprint"))   
    poly = ee.Geometry.Polygon([ring.coordinates()])  
    ft = ee.Feature(poly, img.toDictionary())
    return ft

def to_image(id_str):
    """Función 'server-side' que convierte un string en ee.Image."""
    return ee.Image(id_str)



######################
## Get all neon ids ##
######################

refl001 = ee.ImageCollection("projects/neon-prod-earthengine/assets/HSI_REFL/001")
refl002 = ee.ImageCollection("projects/neon-prod-earthengine/assets/HSI_REFL/002")
combined_collection = refl001.merge(refl002)
foot_fc = combined_collection.map(image_to_feature)
results = foot_fc.aggregate_array("id_geom_array").getInfo()

features_list = []

for feat in results:

    features_list.append({
        "neon_id": feat[0], 
        "neon_date": datetime.fromtimestamp(feat[1] / 1000, tz=timezone.utc).strftime('%Y-%m-%d'),
        "geometry": shape(feat[2])
    })

# "2015-06-27" # Start TOA
# "2017-03-28" # Start BOA

polygons = gpd.GeoDataFrame(features_list, crs="EPSG:4326")
polygons["neon_date"] = pd.to_datetime(polygons["neon_date"])
filtered_polygons = polygons[polygons["neon_date"] > "2015-06-27"]


filtered_polygons.to_file("tables/neon_footprints.gpkg", driver="GPKG")
filtered_polygons.drop(columns=["geometry"]).to_csv("tables/neon_footprints.csv", index=False)

############################################
## Get equigrid coordinate per neon image ##
############################################

points = gpd.read_file("equigrid/NA.gpkg")
polygons = gpd.read_file("tables/neon_footprints.gpkg")

tables = []
for i in range(len(polygons)):
    
    polygon = polygons.iloc[[i]]
    points_within = points[points.within(polygon.union_all())].copy()
    points_within["neon_id"] = polygon["neon_id"].iloc[0]
    points_within["neon_date"] = polygon["neon_date"].iloc[0]
    if len(points_within) == 0:
        continue
    else:
        tables.append(points_within)

table_f = pd.concat(tables, ignore_index=True)
table_f.to_file("tables/neon_foot_points.gpkg", driver="GPKG")
table_f.drop(columns=["geometry"]).to_csv("tables/neon_foot_points.csv", index=False)


###############################
## Final ids neon (not null) ##
###############################

table_f = gpd.read_file("tables/neon_foot_points.gpkg")

images_list = []
for i, row in table_f.iterrows():
    image_ee = create_image_with_null_property(row)
    images_list.append(image_ee)

image_collection = ee.ImageCollection(images_list)
null_percent_list = image_collection.aggregate_array("nullPercent")
null_percent_list_ = null_percent_list.getInfo()
table_f["val_null"] = null_percent_list_

table_f.to_file("tables/neon_all_null_values.gpkg", driver="GPKG")
table_f.drop(columns=["geometry"]).to_csv("tables/neon_all_null_values.csv", index=False)

dataf = table_f[table_f["val_null"] < 9]
dataf.to_file("tables/neon_all_filter_ids.gpkg", driver="GPKG")
dataf.drop(columns=["geometry"]).to_csv("tables/neon_all_filter_ids.csv", index=False)


######################
## Find S2 per NEON ##
######################

table = pd.read_csv("tables/neon_all_filter_ids.csv")

dfs_list = []

for i, row in table.iterrows():
    
    image_test = ee.Image(row.neon_id)
    base_date = datetime.strptime(row.neon_date, '%Y-%m-%d')

    start_date = (base_date - timedelta(days=15)).strftime('%Y-%m-%d')
    end_date = (base_date + timedelta(days=15)).strftime('%Y-%m-%d')

    center = ee.Geometry.Point([row.lon, row.lat])
    square_5120m = (
        center
        .transform(ee.Projection(row.utm), 1)  
        .buffer(2560)                     
        .bounds()                         
        .transform(ee.Projection('EPSG:4326'), 1)
    )
    
    cloud_ic = (
        ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(center)
        .select("cs_cdf")
    )

    foot_fc = cloud_ic.map(image_to_feature)

    filtered_fc = foot_fc.filter( # Filter S2 Bounds too
        ee.Filter.contains(
            leftField='.geo',
            rightValue=square_5120m
        )
    )

    ids_server_side = filtered_fc.aggregate_array("SOURCE_PRODUCT_ID")
    
    cloud_ic_filtered = cloud_ic.filter(
        ee.Filter.inList('SOURCE_PRODUCT_ID', ids_server_side)
    )

    try:
        s2cc_list = cloud_ic_filtered.getRegion(
            geometry=center,
            scale=5120
        ).getInfo() 

        df_raw = pd.DataFrame(s2cc_list[1:], columns=s2cc_list[0])

        df = df_raw[df_raw["cs_cdf"] > 0.9].copy()
        if df.empty:
            pass
        else:
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df["base_date"] = base_date  # Agregamos la misma fecha base en cada fila
            df["days_diff"] = (df["time"] - df["base_date"]).dt.days
            df["time"] = df["time"].dt.strftime("%Y-%m-%d")
            df["base_date"] = df["base_date"].dt.strftime("%Y-%m-%d")

            df.rename(columns={'id': 's2_id'}, inplace=True)
            df["neon_id"] = row.neon_id
            df["utm_x"] = row.utm_x
            df["utm_y"] = row.utm_y
            df["crs"] = row.utm
            df["tile_id"] = row.tile_id
            
            dfs_list.append(df)
            
    except Exception as e:
        dfs_list.append(None)
    print(i)

df_final = pd.concat(dfs_list, ignore_index=True)

df_final["abs_days_diff"] = df_final["days_diff"].abs()

counts = df_final["abs_days_diff"].value_counts()
counts = counts.sort_index() 
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

bars = ax.bar(counts.index,
              counts.values,
              color='steelblue',
              edgecolor='black')

ax.set_title("Distribution of the difference days",
             fontsize=20, fontweight='bold')
ax.set_xlabel("Difference days", fontsize=16)
ax.set_ylabel("Frequency", fontsize=16)
ax.set_xticks(counts.index)
ax.set_xticklabels(counts.index, rotation=0, fontsize=14)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 1),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("days_diff_distribution.png", dpi=300, bbox_inches='tight')
# plt.show()

df_final["id"] = ["NEON_S2__" + str(i).zfill(4) for i in range(1, len(df_final) + 1)]

df_final.to_csv("tables/images_neon_s2_preview.csv", index=False)




#############################
## S2 Collection BOA y TOA ##
#############################


df_final = pd.read_csv("tables/images_neon_s2_preview.csv")


filter_ids = df_final["s2_id"].unique().tolist()

ic_sr = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filter(ee.Filter.inList("system:index", filter_ids)))

valid_sr_ids = ic_sr.aggregate_array("system:index").getInfo()
valid_ids = [item for item in filter_ids if item not in valid_sr_ids]

def build_sentinel2_path(s2_id):
    if s2_id in valid_sr_ids:
        return f"COPERNICUS/S2_SR_HARMONIZED/{s2_id}"
    elif s2_id in valid_ids:
        return f"COPERNICUS/S2_HARMONIZED/{s2_id}"
    else:
        return f"UNKNOWN/{s2_id}"

df_final["sentinel2_id"] = df_final["s2_id"].apply(build_sentinel2_path)
df_final_sorted = df_final.sort_values(by=["abs_days_diff", "cs_cdf"])
df_filtered = df_final_sorted.groupby("neon_id", as_index=False).first()


counts = df_filtered["abs_days_diff"].value_counts()
counts = counts.sort_index() 

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

bars = ax.bar(counts.index,
              counts.values,
              color='firebrick',
              edgecolor='black')

ax.set_title("Distribution of days for the final 111 pairs",
             fontsize=20, fontweight='bold')
ax.set_xlabel("Difference days", fontsize=16)
ax.set_ylabel("Frequency", fontsize=16)
ax.set_xticks(counts.index)
ax.set_xticklabels(counts.index, rotation=0, fontsize=14)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 1),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("days_diff_pairs.png", dpi=300, bbox_inches='tight')
plt.show()

df_filtered.to_csv("tables/images_neon_s2_final_pairs.csv", index=False)



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


# Generalized function to generate S2 band from NEON
def generate_s2_band_from_neon(image_neon: ee.Image, s2_table: pd.DataFrame, band_name_s2: str, wave_neon: ee.List, bands_neon_ee_select: ee.List) -> ee.Image:
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
table = table.iloc[5:]

for i, row in table.iterrows():

    neon_id_image = row["neon_id"]
    s2_id_image = row["sentinel2_id"]

    # Generate the Sentinel-2-like image from NEON
    final_s2_like_image = generate_s2_image_from_neon(neon_id_image, s2_id_image)

    # Define bounding box parameters
    xmin = row["utm_x"] - 5140 / 2
    ymax = row["utm_y"] + 5140 / 2

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
        width=5140,
        height=5140
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
        output_path="output_neon_like_s2",
        nworkers=4,
        max_deep_level=5
    )




table = pd.read_csv("tables/images_neon_s2_final_pairs.csv")

for i, row in table.iterrows():

    xmin = row["utm_x"] - 5140 / 2
    ymax = row["utm_y"] + 5140 / 2

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
        width=5140,
        height=5140
    )

    request = cubexpress.Request(
        id=row.id,
        raster_transform=metadata,
        bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
        image=row["sentinel2_id"]
    )

    cube_requests = cubexpress.RequestSet(requestset=[request])

    cubexpress.getcube(
        request=cube_requests,
        output_path="output_s2",
        nworkers=4,
        max_deep_level=5
    )
