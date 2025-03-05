import ee
import pandas as pd
import utm
from typing import Tuple
import matplotlib.pyplot as plt
import cubexpress
from shapely.geometry import shape, Point, box
import geopandas as gpd


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
            poly
        ])
    })
    return ft


def get_utm_epsg(lat, lon):
    x, y, zone, _ = utm.from_latlon(lat, lon)
    epsg_code = f"326{zone:02d}" if lat >= 0 else f"327{zone:02d}"
    return int(epsg_code)


def square_around_point(point_utm, side=2565):
    """
    Crea un polígono cuadrado centrado en 'point_utm'
    con longitud de lado = side (en las mismas unidades que la capa).
    """
    x_cen, y_cen = point_utm.x, point_utm.y
    half_side = side / 2.0
    # box(xmin, ymin, xmax, ymax)
    return box(x_cen - half_side, y_cen - half_side,
                    x_cen + half_side, y_cen + half_side)


refl001 = ee.ImageCollection("projects/neon-prod-earthengine/assets/HSI_REFL/001")
refl002 = ee.ImageCollection("projects/neon-prod-earthengine/assets/HSI_REFL/002")
combined_collection = refl001.merge(refl002)
foot_fc = combined_collection.map(image_to_feature)
results = foot_fc.aggregate_array("id_geom_array").getInfo()

features_list = []

for feat in results:
    feat_id = feat[0]
    shapely_geom = shape(feat[1])
    features_list.append({
        "neon_id": feat_id, 
        "geometry": shapely_geom
    })

# polygons = gpd.GeoDataFrame(features_list, crs="EPSG:4326")
# polygons.to_file("tables/neon_footprints.gpkg", driver="GPKG")
points = gpd.read_file("equigrid/NA.gpkg")
polygons = gpd.read_file("tables/neon_footprints.gpkg")

tables = []
for i in range(len(polygons)):
    polygon = polygons.iloc[[i]]
    points_within = points[points.within(polygon.union_all())]
    # points_within.to_file("geometries/points_within1.gpkg")
    # polygon.to_file("geometries/polygon1.gpkg")

    buff_points = []
    for _, point in points_within.iterrows():
        lat, lon = point.geometry.y, point.geometry.x
        epsg_code = get_utm_epsg(lat, lon)
        x, y, _, _ = utm.from_latlon(lat, lon)
        point_utm = Point(x, y)
        # buffer_geom = point_utm.buffer(2565)
        # buffer_geographic = gpd.GeoSeries([buffer_geom], crs=f"EPSG:{epsg_code}").to_crs(points.crs)
        # buff_points.append(buffer_geographic.iloc[0])
        box_geom = square_around_point(point_utm, side=5120)
    
        # Convertimos de nuevo a la proyección original (por ejemplo EPSG:4326).
        box_geographic = (gpd.GeoSeries([box_geom], crs=f"EPSG:{epsg_code}")
                        .to_crs(points.crs))
        
        buff_points.append(box_geographic.iloc[0])
        
    buffers = gpd.GeoDataFrame(geometry=buff_points, crs=points.crs)

    buffers.to_file("geometries/buffers1.gpkg")