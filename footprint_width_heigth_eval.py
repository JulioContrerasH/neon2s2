
import geopandas as gpd

polygons = gpd.read_file("geometries/neon001_sentinel_matches.geojson")

polygon = polygons.iloc[0]

coords = list(polygon.geometry.exterior.coords)

xmin = min(coord[0] for coord in coords)
xmax = max(coord[0] for coord in coords)
ymin = min(coord[1] for coord in coords)
ymax = max(coord[1] for coord in coords)