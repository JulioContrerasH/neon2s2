import ee
import pandas as pd
import cubexpress

try:
    ee.Initialize(project="ee-julius013199")
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project="ee-julius013199")


table = pd.read_csv("tables/images_neon_s2_final_pairs.csv")

table.iloc[0]["sentinel2_id"]

for i, row in table.iterrows():

    # band_options = {
    #     True: ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9",  "B11", "B12"],  # Si startswith("COPERNICUS/S2_HARMONIZED/") es True
    #     False: ["B2"]  # Si no, se usa esta lista
    # }

    # bands = band_options[table_entry.startswith("COPERNICUS/S2_HARMONIZED/")]

    xmin = row["utm_x"] - 5160 / 2
    ymax = row["utm_y"] + 5160 / 2

    metadata = cubexpress.RasterTransform(
        crs=row.crs,
        geotransform={
            'scaleX': 10, 
            'shearX': 0, 
            'translateX': xmin,
            'scaleY': -10, 
            'shearY': 0, 
            'translateY': ymax
        },
        width=516,
        height=516
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
        output_path="/media/contreras/LaCie/output_s2",
        nworkers=4,
        max_deep_level=5
    )
