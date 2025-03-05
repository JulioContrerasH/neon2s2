import ee
import pandas as pd
import cubexpress

try:
    ee.Initialize(project="ee-julius013199")
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project="ee-julius013199")


table = pd.read_csv("tables/images_neon_s2_final_pairs.csv")

for i, row in table.iterrows():

    xmin = row["utm_x"] - 5140 / 2
    ymax = row["utm_y"] + 5140 / 2

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
        width=514,
        height=514
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
