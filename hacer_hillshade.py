# -*- coding: utf-8 -*-
import numpy as np
import json
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as Res
from rasterio.transform import from_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_G = r"G:\Unidades compartidas\ISA_Etapa_4_ variables_regional\VARIABLES_MODELO_REGIONAL"
SLOPE_TIF = f"{BASE_G}\\Pendientes\\Pendientes.tif"
ASPECT_TIF = f"{BASE_G}\\Aspecto\\Aspecto.tif"
OUTDIR = r"C:\GEOHAZARDS\ISA_4\MODELO_FINAL\modelo_pixel_regional\graficas\webmap"

# bbox nativo (EPSG:32619) de susceptibilidad_regional_v2_recortada.tif -- la
# misma huella que ya se uso para susceptibilidad_gam.png / bounds_ours
UTM_LEFT, UTM_BOTTOM, UTM_RIGHT, UTM_TOP = -469934.6291542904, 88240.1928596812, 267033.57134525373, 962721.3005308529
SRC_CRS = "EPSG:32619"

with open(f"{OUTDIR}/meta.json") as f:
    meta = json.load(f)
b = meta["bounds_ours"]  # lat/lon: south, west, north, east
OUT_W, OUT_H = 5915, 7000  # mismas dimensiones que susceptibilidad_gam.png

dst_crs = "EPSG:4326"
dst_transform = from_bounds(b["west"], b["south"], b["east"], b["north"], OUT_W, OUT_H)


def leer_ventana_decimada(path, resampling):
    with rasterio.open(path) as src:
        win = window_from_bounds(UTM_LEFT, UTM_BOTTOM, UTM_RIGHT, UTM_TOP, transform=src.transform)
        print(f"  {path.split(chr(92))[-1]}: ventana {win.width:.0f}x{win.height:.0f} -> leyendo diezmado a {OUT_W}x{OUT_H}", flush=True)
        data = src.read(1, window=win, out_shape=(OUT_H, OUT_W), resampling=resampling)
        win_transform = src.window_transform(win)
        # transform de la salida diezmada (mismo bbox de la ventana, nuevo tamano)
        from rasterio.transform import Affine
        scale_x = win.width / OUT_W
        scale_y = win.height / OUT_H
        out_transform = win_transform * Affine.scale(scale_x, scale_y)
        nodata = src.nodata
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    return data.astype("float32"), out_transform


print("=== Leyendo ventana diezmada de slope/aspect (recorte local, sin reproyectar el archivo completo) ===", flush=True)
slope_utm, transform_utm = leer_ventana_decimada(SLOPE_TIF, Res.bilinear)
aspect_utm, _ = leer_ventana_decimada(ASPECT_TIF, Res.nearest)
print(f"  slope: min={np.nanmin(slope_utm):.2f} media={np.nanmean(slope_utm):.2f} max={np.nanmax(slope_utm):.2f}", flush=True)

print("=== Reproyectando el recorte (ya pequeno) a EPSG:4326 ===", flush=True)
slope = np.full((OUT_H, OUT_W), np.nan, dtype="float32")
aspect = np.full((OUT_H, OUT_W), np.nan, dtype="float32")
reproject(source=slope_utm, destination=slope, src_transform=transform_utm, src_crs=SRC_CRS,
          dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan, resampling=Res.bilinear)
reproject(source=aspect_utm, destination=aspect, src_transform=transform_utm, src_crs=SRC_CRS,
          dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan, resampling=Res.nearest)

print("=== Calculando hillshade (formula analitica, azimuth=315 altitude=45) ===", flush=True)
AZIMUTH = 315.0
ALTITUDE = 45.0
zenith_rad = np.radians(90.0 - ALTITUDE)
azimuth_rad = np.radians(AZIMUTH)

slope_rad = np.radians(slope)
es_plano = aspect < 0  # sentinel del proyecto: -1 = terreno plano
aspect_seguro = np.where(es_plano, 0.0, aspect)
aspect_rad = np.radians(aspect_seguro)

hs = (np.cos(zenith_rad) * np.cos(slope_rad) +
      np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad))
hs = np.where(es_plano, np.cos(zenith_rad), hs)
hs = np.clip(hs, 0, 1)

valido = ~np.isnan(slope) & ~np.isnan(aspect)
print(f"  hillshade: min={np.nanmin(hs[valido]):.3f} media={np.nanmean(hs[valido]):.3f} max={np.nanmax(hs[valido]):.3f}", flush=True)

print("=== Guardando PNG (gris, opaco donde hay dato) ===", flush=True)
gris = np.clip(hs, 0, 1)
rgba = np.zeros((OUT_H, OUT_W, 4), dtype="float32")
rgba[..., 0] = gris
rgba[..., 1] = gris
rgba[..., 2] = gris
rgba[..., 3] = np.where(valido, 1.0, 0.0)
plt.imsave(f"{OUTDIR}/hillshade.png", rgba)
print(f"  guardado {OUTDIR}/hillshade.png", flush=True)
print("=== HILLSHADE OK ===", flush=True)
