######################
## Generar la tabla ##
######################

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np
import ee

ee.Initialize()

refl001 = ee.ImageCollection("projects/neon-prod-earthengine/assets/HSI_REFL/001").first().getInfo()

def compute_std_dev(band_name):
    refl001 = ee.ImageCollection("projects/neon-prod-earthengine/assets/HSI_REFL/001")
    numeric_values_0 = ee.List(refl001.aggregate_array(band_name)).map(
        lambda x: ee.Number.parse(ee.String(x).split(",").get(0))
    )
    numeric_values_1 = ee.List(refl001.aggregate_array(band_name)).map(
        lambda x: ee.Number.parse(ee.String(x).split(",").get(1))
    )
    return ee.List([numeric_values_0, numeric_values_1])

band_names = ["WL_FWHM_B" + "{:03d}".format(i) for i in range(1, 427)]
data = {}
for band_name in band_names:
    values = compute_std_dev(band_name).getInfo()
    
    data[f"{band_name}_Wavelength"] = values[0]
    data[f"{band_name}_Bandwidth"] = values[1]
    print(band_name)

df = pd.DataFrame(data)
df.to_csv("Wave_Width.csv", index=False)


############################
## Código para el gráfico ##
############################

# Cargar datos
df = pd.read_csv("Wave_Width.csv")
band_names = ["WL_FWHM_B" + "{:03d}".format(i) + "_Wavelength" for i in range(1, 427)]

# Cálculo del intervalo de confianza
confianza = 0.95
z = stats.norm.ppf((1 + confianza) / 2) 

resultados = []
for band in band_names:
    valores = df[band]
    media = np.mean(valores)
    std = np.std(valores, ddof=1)
    n = len(valores)
    margen_error = z * (std / np.sqrt(n))
    ci_lower, ci_upper = media - margen_error, media + margen_error
    resultados.append({"Banda": band, "Media": media, "IC_Lower": ci_lower, "IC_Upper": ci_upper})

df_resultados = pd.DataFrame(resultados)


# Crear una nueva columna que indique si la media está dentro del intervalo de confianza
df_resultados["Dentro_IC"] = ((df_resultados["Media"] >= df_resultados["IC_Lower"]) & 
                              (df_resultados["Media"] <= df_resultados["IC_Upper"]))


df_resultados.to_csv("Wave_Width_End.csv", index=False)

# Verificar si todas las medias están dentro del intervalo de confianza
if sum(df_resultados["Dentro_IC"]) == len(df_resultados):
    print("Todas las medias están dentro del intervalo de confianza")
else:
    print("Al menos una media está fuera del intervalo de confianza")


# Gráfico principal
fig, ax = plt.subplots(figsize=(12, 6))

# Convertir el índice en valores numéricos para el eje X
x_vals = np.arange(len(df_resultados))

# Graficar la media en el gráfico principal
ax.plot(x_vals, df_resultados["Media"], label="Media", color='blue', linewidth=2)
ax.fill_between(x_vals, df_resultados["IC_Lower"], df_resultados["IC_Upper"], 
                color='blue', alpha=0.2, label="Intervalo de Confianza")

# Configuración del eje X: Mostrar solo algunos valores
ax.set_xticks([0, 100, 200, 300, 425])
ax.set_xticklabels(["1", "100", "200", "300", "426"])

ax.set_xlabel("Banda")
ax.set_ylabel("Longitud de onda (nm)")
ax.set_title("Intervalo de Confianza para las Bandas")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.6)

# -------------------- ZOOM --------------------
# Definir el rango de zoom (ejemplo: bandas de 200 a 250)
zoom_start, zoom_end = 245, 247

# Crear un eje de zoom en la esquina superior derecha
ax_zoom = inset_axes(ax, width="30%", height="30%", loc="upper right")  # Tamaño del zoom

# Graficar en el eje de zoom
ax_zoom.plot(x_vals[zoom_start:zoom_end], df_resultados["Media"][zoom_start:zoom_end], color='blue', linewidth=2)
ax_zoom.fill_between(x_vals[zoom_start:zoom_end], df_resultados["IC_Lower"][zoom_start:zoom_end], 
                     df_resultados["IC_Upper"][zoom_start:zoom_end], color='blue', alpha=0.3)

# Estilo del zoom
ax_zoom.set_xticks([])
ax_zoom.set_yticks([])
ax_zoom.set_title("Zoom en bandas 200-250", fontsize=10)

# Conectar el zoom con el gráfico principal
mark_inset(ax, ax_zoom, loc1=2, loc2=4, fc="none", ec="red", linestyle="--")

plt.show()



####################
## Ver si son GAO ##
####################
import ee

# Inicializar Earth Engine
ee.Initialize()

# Crear una lista para almacenar las imágenes
images = []

# Iterar sobre cada fila del DataFrame
for i, row in gao_table.iterrows():
    image = ee.Image(row["image_id_neon"])
    images.append(image)

# Crear una ImageCollection a partir de la lista de imágenes
image_collection = ee.ImageCollection(images)

# Extraer la propiedad SENSOR_ID para cada imagen
sensor_ids = image_collection.aggregate_array("SENSOR_ID")

# Obtener la lista de SENSOR_ID
sensor_ids_list = sensor_ids.getInfo()

# Imprimir los resultados
print("Lista de SENSOR_IDs:", sensor_ids_list)
