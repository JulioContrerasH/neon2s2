import ee
import pandas as pd
import cubexpress

try:
    ee.Initialize(project="ee-julius013199")
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project="ee-julius013199")


table = pd.read_csv("tables/methane_experiment.csv")
filtered_table = table[table["tile"].str.startswith("S2", na=False)]


for i, row in filtered_table.iterrows():

    image = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
           .filter(ee.Filter.eq("PRODUCT_ID", row["tile"])) \
           .first()
    
    metadata = cubexpress.RasterTransform(
        crs=row.crs,
        geotransform={
            'scaleX': float(row["transform_a"]), 
            'shearX': float(row["transform_b"]), 
            'translateX': float(row["transform_c"]),
            'scaleY': float(row["transform_e"]), 
            'shearY': float(row["transform_d"]), 
            'translateY': float(row["transform_f"])
        },
        width=int(row["width"]),
        height=int(row["height"])
    )

    request = cubexpress.Request(
        id=row.id_loc_image,
        raster_transform=metadata,
        bands=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
        image=image
    )

    cube_requests = cubexpress.RequestSet(requestset=[request])

    cubexpress.getcube(
        request=cube_requests,
        output_path="/media/contreras/LaCie/cesar_s2",
        nworkers=4,
        max_deep_level=5
    )
    print(i)