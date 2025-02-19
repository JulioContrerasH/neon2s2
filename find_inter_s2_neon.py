import ee
import pandas as pd

# Inicializar Google Earth Engine
ee.Initialize()


####################################################
## Encontrar concidencias entre NEON y Sentinel 2 ##
####################################################

collection_neon = "projects/neon-prod-earthengine/assets/HSI_REFL/001"
collection_s2 = "COPERNICUS/S2_HARMONIZED"
refl002 = ee.ImageCollection(collection_neon)

# Función para extraer fecha y geometría
def extract_info(image):
    date = image.get('system:time_start')
    geometry = image.geometry()
    return image.set('date', ee.Date(date).format('YYYY-MM-dd')).set('geometry', geometry)

# Aplicar la función a la colección
refL002WithInfo = refl002.map(extract_info)

# Obtener listas de fechas y geometrías
dates = refL002WithInfo.aggregate_array('date')

dates.getInfo()



geometries = refL002WithInfo.aggregate_array('geometry')
neon_ids = refL002WithInfo.aggregate_array('system:index')



# Función para buscar coincidencias en Sentinel 2
def find_sentinel_matches(index):
    neon_id = ee.String(neon_ids.get(index))
    date = ee.Date(dates.get(index))
    geometry = ee.Geometry(geometries.get(index))
    
    # Filtrar Sentinel 2 por fecha y geometría
    sentinel2_matches = ee.ImageCollection(collection_s2) \
        .filterBounds(geometry) \
        .filterDate(date.advance(-1, 'day'), date.advance(1, 'day')) \
        .aggregate_array('system:index')  # Obtener IDs de las imágenes coincidentes

    # Si hay coincidencias, guardar los IDs
    return ee.Feature(None, {
        'ID_NEON': neon_id,
        'ID_SENTINEL2': sentinel2_matches
    })

# Crear una tabla con las coincidencias (solo si hay imágenes Sentinel 2)
matches = ee.List.sequence(0, neon_ids.size().subtract(1)) \
    .map(find_sentinel_matches) \
    .filter(ee.Filter.notNull(['ID_SENTINEL2']))  # Filtra los casos donde no hay coincidencias

# Convertir a FeatureCollection
results_table = ee.FeatureCollection(matches)

# Obtener la tabla como JSON para visualizar en Python
results_json = results_table.getInfo()


########################################
## Generar la tabla con coincidencias ##
########################################


# Crear una nueva lista para almacenar los datos formateados
data = []

# Procesar cada fila en los resultados obtenidos
for feature in results_json['features']:
    properties = feature['properties']
    neon_id = properties['ID_NEON'] # collection_neon + "/" + 
    sentinel_ids = properties['ID_SENTINEL2']
    
    # Si hay múltiples imágenes Sentinel, crear una fila por cada una
    if sentinel_ids:
        for sentinel_id in sentinel_ids:
            data.append([neon_id, sentinel_id])
    else:
        # Si no hay coincidencias, no agregamos la fila (evitando NaN)
        continue



# Convertir la lista en un DataFrame
df_final = pd.DataFrame(data, columns=['ID_NEON', 'ID_SENTINEL2'])
df_final["Collection_NEON"] = collection_neon
df_final["Collection_Sentinel"] = collection_s2

# Save the DataFrame to a CSV file
df_final.to_csv('tables/neon001_sentinel_matches.csv', index=False)