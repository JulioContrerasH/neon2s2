import ee
ee.Initialize()

from ee_plugin import Map

# Imagen de ejemplo
image = ee.Image("projects/neon-prod-earthengine/assets/HSI_REFL/001/2013_CPER_1")

# Corregir: usar comillas para las claves del diccionario
vis_params = {
    'min': 340,
    'max': 2150,
    'bands': ['B053','B035','B019'],
    'gamma': 2
}

Map.addLayer(image, vis_params, "NEON HSI 2013 CPER 1")
