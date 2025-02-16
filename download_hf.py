#########################################
## 1) How to download from huggingface ##
#########################################

# https://huggingface.co/datasets/satellogic/EarthView/tree/main/neon
# https://huggingface.co/datasets/satellogic/EarthView/resolve/main/neon/train/29_609628957997028N_82_0258062601623W/neon.h5py
# Loguearte 
# pip install --upgrade huggingface_hub
# huggingface-cli login
# pegar token: https://huggingface.co/settings/tokens

from huggingface_hub import list_repo_files, hf_hub_download
import os

repo_id = "satellogic/EarthView"

# 1) Listar todos los archivos en el repo (repo_type="dataset" para datasets):
all_files = list_repo_files(repo_id, repo_type="dataset")

# 2) Filter paths to get only val and train files
files = [f for f in all_files if f.startswith("neon/val/") and f.endswith(".h5py")]

# val_files = [f for f in all_files if f.startswith("neon/val/")]
# len(val_files)

# 3) Descargar en un bucle
for file_path in files:
    # Crear carpetas locales si fuese necesario
    local_dir = os.path.dirname(file_path)  # la ruta donde guardarlo
    os.makedirs(local_dir, exist_ok=True)

    # Descargar
    hf_hub_download(
        repo_id=repo_id,
        filename=file_path,
        repo_type="dataset",
        local_dir="./",
        local_dir_use_symlinks=False  # para forzar descarga completa sin symlinks
    )

print("Descarga completa.")




#################
## 2) Creation ##
#################


import os
import h5py
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import glob

# Ruta base de la carpeta 'train'
ruta_train = "neon/val"
filenames = glob.glob(os.path.join(ruta_train, "**/*.h5py"), recursive=True)

param = "rgb"

for filename in filenames:
    with h5py.File(filename, "r") as f:
        # Atributos del H5
        print("Info General: ", list(f))
        print("Attributes: ", list(f.attrs))
        
        # Bounds y EPSG
        bounds = f.attrs["bounds"]         # (xmin, ymin, xmax, ymax)
        epsg_attr = f.attrs["epsg"]
        timestamps = f.attrs["timestamp"]  # Suponiendo que sea un array con fechas o índices

        xmin, ymin, xmax, ymax = bounds
        
        # Dataset con la forma [T, B, H, W], suponiendo:
        #  T = número de fechas
        #  B = número de bandas para 'chm' (puede ser 1 o más)
        #  H, W = alto y ancho de cada ráster
        data_1m = f[param]

        # Iteramos sobre cada fecha
        for t in range(data_1m.shape[0]):
            # Extraemos los datos para la fecha 't'
            arr_t = data_1m[t, :, :, :]  # => forma (B, H, W)

            # Transformación geográfica
            B, H, W = arr_t.shape
            transform = from_bounds(xmin, ymin, xmax, ymax, width=W, height=H)

            # Para nombrar el archivo, tomamos el timestamp correspondiente
            # (ajusta según el formato real de "timestamp"; aquí se usa tal cual)
            fecha_str = str(timestamps[t]).split('-')[0]  # O usa un formateo si es necesario
   
            # Nombre de salida (quitamos .h5py usando os.path.splitext)
            base_name = os.path.splitext(filename)[0]
            output_tif = f"{base_name}_{param}_{fecha_str}.tif"
            
            # Creamos el archivo GeoTIFF
            with rasterio.open(
                output_tif,
                "w",
                driver="GTiff",
                height=H,
                width=W,
                count=B,  # número de bandas en la salida
                dtype=arr_t.dtype,
                transform=transform,
                crs="EPSG:4326"
            ) as dst:
                # Escribimos cada banda al archivo
                for b in range(B):
                    dst.write(arr_t[b, :, :], b + 1)

            print(f"¡Listo! Archivo generado: {output_tif}")



# plot image
import matplotlib.pyplot as plt
import rasterio.plot

with rasterio.open(output_tif) as src:
    rasterio.plot.show(src)
    plt.show()

################
## 3) Jupyter ##
################



import numpy as np
import earthview as ev

data = ev.load_dataset("neon", shards=[100])  # shard is optional

sample = next(iter(data))

print(sample.keys())
print(np.array(sample['rgb']).shape)       # RGB Data
print(np.array(sample['chm']).shape)       # Canopy Height
print(np.array(sample['1m']).shape)        # Hyperspectral





from huggingface_hub import list_repo_files
import os

# ID del repositorio en Hugging Face (tipo "dataset")
repo_id = "satellogic/EarthView"

# 1) Listar todos los archivos del repositorio:
all_files = list_repo_files(repo_id, repo_type="dataset")

# 2) Filtrar archivos que estén en neon/train/ y neon/val/ y terminen en .h5py
train_files = [f for f in all_files if f.startswith("neon/train/") and f.endswith(".h5py")]
val_files = [f for f in all_files if f.startswith("neon/val/") and f.endswith(".h5py")]

# 3) Mostrar resultados
print(f"Total de archivos en neon/train/ con extensión .h5py: {len(train_files)}")
print("Ejemplos (train):", train_files[:5], "...\n")

print(f"Total de archivos en neon/val/ con extensión .h5py: {len(val_files)}")
print("Ejemplos (val):", val_files[:5], "...")
