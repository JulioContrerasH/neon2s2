import ee
import pandas as pd
import utm
from typing import Tuple
import matplotlib.pyplot as plt
import cubexpress


try:
    ee.Initialize(project="ee-julius013199")
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project="ee-julius013199")


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


table = pd.read_csv("tables/neon_end_equigrid_geodata.csv")

dfs_list = []

for i, row in table.iterrows():

    
    lon = row.lon
    lat = row.lat
    neon_id = row.image_id_neon

    image_test = ee.Image(neon_id)

    date_im = ee.Date(image_test.get("system:time_start")) 
    base_date = pd.to_datetime(date_im.getInfo()["value"], unit="ms")
    start_date = date_im.advance(-30, 'day')
    end_date   = date_im.advance(30, 'day')

    center = ee.Geometry.Point([lon, lat])
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

    filtered_fc = foot_fc.filter(
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
        
        if len(s2cc_list) < 1:
            print("\nNo se encontraron valores de 'cs_cdf' en esa región y rango de fechas.")
        else:
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
                df["neon_id"] = neon_id
                df["utm_x"] = row.utm_x
                df["utm_y"] = row.utm_y
                df["crs"] = row.utm
                df["id"] = row.ID
                
                dfs_list.append(df)
            
    except Exception as e:
        print("Ocurrió un error durante getRegion:", e)
    print(i)


df_final = pd.concat(dfs_list, ignore_index=True)

len(df_final)

df_final.to_csv("tables/images_neon_s2_preview.csv", index=False)


##############
## Bar plot ##
##############

counts = df_final["days_diff"].value_counts()

# Ordenar por el valor numérico (índice) en lugar de la frecuencia
counts = counts.sort_index()  # Ordena de menor a mayor
# counts = counts.sort_index(ascending=False)  # De mayor a menor

# Plotear el diagrama de barras con el eje X ordenado
counts.plot(kind='bar', figsize=(10, 5))
plt.xlabel("days_diff")
plt.ylabel("Frecuencia")
plt.title("Distribución de days_diff (barras ordenadas por valor)")
plt.show()

###############
## BOA exist ##
###############

df_final = pd.read_csv("tables/images_neon_s2_preview.csv")
len(df_final)

df_final_0 = df_final[df_final["days_diff"] == 0]
df0_copy = df_final_0.copy()
unique_ids = df0_copy["s2_id"].unique().tolist()
unique_ids
ic = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filter(ee.Filter.inList("system:index", unique_ids)))

valid_ids = ic.aggregate_array("system:index").getInfo() 

df0_copy["exists_boa"] = df0_copy["s2_id"].isin(valid_ids)
df_valid_0 = df0_copy[df0_copy["exists_boa"]].copy()

df_final_0_uni = (
    df_valid_0
    .loc[df_valid_0.groupby("id")["cs_cdf"].idxmax()]
    .copy()
)

df_final_0_uni.to_csv("tables/images_neon_s2_preview_0_better.csv", index=False)
#################
## Download S2 ##
#################

for x, row_i in df_final_0_uni.iterrows():

    lon = float(row_i.longitude)
    lat = float(row_i.latitude)

    x = row_i.utm_x - 5160/2
    y = row_i.utm_y + 5160/2

    metadata = cubexpress.RasterTransform(
        crs=row_i.crs,
        geotransform={
            'scaleX': 10, 
            'shearX': 0, 
            'translateX': x,
            'scaleY': -10, 
            'shearY': 0, 
            'translateY': y
        },
        width=516,
        height=516
    )

    request = cubexpress.Request(
        id=row_i.id,
        raster_transform=metadata,
        bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
        image="COPERNICUS/S2_SR_HARMONIZED/" + row_i.s2_id  # Note: you can wrap with ee.Image(image_id).divide(10000) if needed
    )


    # Create the RequestSet
    cube_requests = cubexpress.RequestSet(requestset=[request])

    # Download
    cubexpress.getcube(
        request=cube_requests,
        output_path="output_sentinel",
        nworkers=4,
        max_deep_level=5
    )

