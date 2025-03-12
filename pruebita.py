import glob
import os
from pathlib import Path
from osgeo import gdal

def partial_merge_bands_to_tiff(input_tifs, output_tif, block_size=1024):
    """
    Fusiona múltiples TIFF (todas del mismo tamaño/proyección) en un solo multibanda,
    leyendo y escribiendo por bloques para no cargar todo en memoria.

    - Usa compresión DEFLATE + PREDICTOR=3 (adecuado para flotantes).
    - TILED=YES, BLOCKXSIZE=1024, BLOCKYSIZE=1024, INTERLEAVE=PIXEL.
    - Emplea argumentos posicionales en ReadAsArray(col, row, cols_to_read, rows_to_read).
    """

    # Abrimos todos los TIFF de entrada en modo lectura
    datasets = [gdal.Open(str(tif), gdal.GA_ReadOnly) for tif in input_tifs]
    if not datasets:
        raise RuntimeError("No hay TIFFs de entrada para fusionar.")

    # Parámetros espaciales (tomados del primero)
    x_size = datasets[0].RasterXSize
    y_size = datasets[0].RasterYSize
    geotrans = datasets[0].GetGeoTransform()
    proj = datasets[0].GetProjection()

    # Tipo de dato de la primera banda
    first_band = datasets[0].GetRasterBand(1)
    band_type = first_band.DataType  # p.e. gdal.GDT_Float32

    # Creamos el archivo de salida con tantas bandas como TIFFs
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        str(output_tif),
        x_size,
        y_size,
        len(datasets),
        band_type,
        options=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "BLOCKXSIZE=1024",
            "BLOCKYSIZE=1024",
            "INTERLEAVE=PIXEL",
            
        ],
    )
    out_ds.SetGeoTransform(geotrans)
    out_ds.SetProjection(proj)

    # Para cada TIFF de entrada, copiamos su banda 1 como banda i+1 del multibanda
    for i, ds_in in enumerate(datasets):
        in_band = ds_in.GetRasterBand(1)
        out_band = out_ds.GetRasterBand(i + 1)

        # Nombre de la banda = nombre del archivo sin extensión
        band_name = Path(input_tifs[i]).stem
        out_band.SetDescription(band_name)

        # Leer/escribir en bloques (posicionales en vez de xoff=,yoff=,...)
        for row in range(0, y_size, block_size):
            rows_to_read = min(block_size, y_size - row)
            for col in range(0, x_size, block_size):
                cols_to_read = min(block_size, x_size - col)

                # Leer un bloque del TIFF de entrada (posicional)
                data_block = in_band.ReadAsArray(col, row, cols_to_read, rows_to_read)
                # Escribir el bloque
                out_band.WriteArray(data_block, col, row)

    # Cerrar todo
    out_ds = None
    for ds_in in datasets:
        ds_in = None


# ------------------- LÓGICA PRINCIPAL -------------------
# 1) Carpeta donde ya tienes todos los TIFF recortados:
crops_dir = Path("/home/contreras/Documents/GitHub/download_20m/bio/tiff2/try/crops/try")

# 2) Buscamos los .tif en esa carpeta
tif_files = sorted(glob.glob(str(crops_dir / "*.tif")))

# 3) Evitar mezclar un TIFF de salida anterior si existe
input_tifs = [t for t in tif_files if not t.endswith("merged_output.tif")]

# 4) Ejecutar la fusión
if not input_tifs:
    print("No se encontraron TIFFs de entrada para fusionar.")
else:
    output_tif = crops_dir / "merged_output2.tif"
    print(f"Creando multibanda: {output_tif}")
    partial_merge_bands_to_tiff(input_tifs, output_tif, block_size=1024)
    print(f"¡Listo! Multibanda guardado en: {output_tif}")
