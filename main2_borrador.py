


import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import ee

ee.Initialize(project = "ee-julius013199")


###########################################
## Get the wavelengths of the NEON image ##
###########################################

# Bands of NEON
bands_neon = [f"WL_FWHM_B{i:03d}" for i in range(1, 427)]

# Select a hyperspectral image (NEON)
image = ee.Image(
    "projects/neon-prod-earthengine/assets/HSI_REFL/001/2016_HARV_3"
)

ee_dict = ee.Dictionary(
    {
        band_neon: ee.String(image.get(band_neon)).split(",").get(0) for band_neon in bands_neon
    }
)

ee_dict.getInfo()


###########################################
## Get the SRF wavelengths of Sentinel 2 ##
##############################################

# Load the spectral response functions of Sentinel 2
s2 = pd.read_csv("tables/S2_SRF_spectral_responses_2024_4.0.csv")

# Get the bands of Sentinel 2
bands_s2 = s2.columns[1:]
band = bands_s2[0] # B1

# Get index of wavelengths in the band
index = s2[band] != 0
df = pd.DataFrame(
    {
        "w": s2['SR_WL'][index],
        "rf": s2[band][index]
    }
)


properties = ee_dict.getInfo()

string_list = list(properties.values())
aviris_ = list(properties.values())
WAViris = np.array([float(item) for item in string_list])


max = float(df["w"].max())
min = float(df["w"].min())

idx = (WAViris <= max) & (WAViris >= min)
len(idx)

filter_aviris = WAViris[idx]

x = df["w"]
y = df["rf"]

len(df["w"])

interpolador = interp1d(x, y, kind='linear')

y_nuevo = interpolador(filter_aviris)

[3.75833894e-03, 1.65813802e-03, 7.15914464e-04, 3.63825558e-03,
       6.77135987e-01, 8.20776729e-01, 9.68288976e-01, 8.59824690e-01,
       1.72324197e-01]


norm_ynew = y_nuevo / sum(y_nuevo)


[1.07132527e-03, 4.72656991e-04, 2.04073468e-04, 1.03709517e-03,
       1.93019552e-01, 2.33964757e-01, 2.76013545e-01, 2.45095490e-01,
       4.91215057e-02]

bands_neon = [f"B{str(i).zfill(3)}" for i in range(1, 427)]

selected_bands = [band for band, flag in zip(bands_neon, idx) if flag]


result_band = ee.Image(0)

for band, factor in zip(selected_bands, norm_ynew):
    result_band = result_band.add(image.select(band).multiply(factor))

result_band.getInfo()









x_list = x.astype(float).tolist()
y_list = y.astype(float).tolist()
x_ee   = ee.List(x_list)
y_ee   = ee.List(y_list)

prepare_segments(x_ee, y_ee).getInfo()

{'m': 0.00229732, 'x0': 412, 'x1': 413, 'y0': 0.00177574, 'y1': 0.00407306}

(0.00177574 - 0.00407306)/ (412 - 413)
