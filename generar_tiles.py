# -*- coding: utf-8 -*-
"""Genera una piramide de teselas XYZ (Web Mercator) para el mapa GAM,
en vez de una sola imagen plana diezmada -- asi el zoom muestra detalle
real en vez de verse borroso. Solo genera teselas donde realmente hay
datos (interseccion con la huella de geologia), no sobre todo el bbox."""
import os
import math
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import box
from shapely.strtree import STRtree
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

SRC = r"C:\GEOHAZARDS\ISA_4\MODELO_FINAL\modelo_pixel_regional\susceptibilidad_regional_v2_recortada.tif"
GEO_SHP = r"G:\Unidades compartidas\ISA_Etapa_4_ variables_regional\VARIABLES_MODELO_REGIONAL\Geología_regional\Geologia_consolidada_ISA-4.shp"
OUTDIR = r"C:\GEOHAZARDS\ISA_4\MODELO_FINAL\modelo_pixel_regional\graficas\webmap\tiles_gam"
ZMIN, ZMAX = 6, 14
TS = 256  # tamano de tesela en pixeles
VMAX = 0.85
DST_CRS = "EPSG:3857"
R = 6378137.0
ORIGIN = math.pi * R  # 20037508.342789244

ramp_colors = ["#1a9850", "#66bd63", "#a6d96a", "#fee08b", "#fdae61", "#f46d43", "#d73027", "#a50026"]
CMAP = LinearSegmentedColormap.from_list("riesgo_verde_rojo", ramp_colors, N=256)

os.makedirs(OUTDIR, exist_ok=True)


def tile_bounds_3857(z, x, y):
    tile_size = 2 * ORIGIN / (2 ** z)
    xmin = -ORIGIN + x * tile_size
    xmax = xmin + tile_size
    ymax = ORIGIN - y * tile_size
    ymin = ymax - tile_size
    return xmin, ymin, xmax, ymax


print("=== Cargando huella de geologia (para saber donde SI generar teselas) ===", flush=True)
geo = gpd.read_file(GEO_SHP).to_crs(DST_CRS)
geoms = list(geo.geometry)
tree = STRtree(geoms)
total_bounds = geo.total_bounds  # xmin, ymin, xmax, ymax en 3857
print(f"  bounds totales (3857): {total_bounds}", flush=True)

print("=== Abriendo raster fuente (local, deberia ser rapido) ===", flush=True)
src_ds = rasterio.open(SRC)
src_nodata = src_ds.nodata

total_generadas = 0
total_candidatas = 0
for z in range(ZMIN, ZMAX + 1):
    tile_size_m = 2 * ORIGIN / (2 ** z)
    x_ini = max(0, int((total_bounds[0] + ORIGIN) / tile_size_m))
    x_fin = min(2 ** z - 1, int((total_bounds[2] + ORIGIN) / tile_size_m))
    y_ini = max(0, int((ORIGIN - total_bounds[3]) / tile_size_m))
    y_fin = min(2 ** z - 1, int((ORIGIN - total_bounds[1]) / tile_size_m))
    n_generadas_zoom = 0
    for x in range(x_ini, x_fin + 1):
        for y in range(y_ini, y_fin + 1):
            xmin, ymin, xmax, ymax = tile_bounds_3857(z, x, y)
            tb = box(xmin, ymin, xmax, ymax)
            candidatos = tree.query(tb)
            if len(candidatos) == 0:
                continue
            intersecta = any(geoms[i].intersects(tb) for i in candidatos)
            if not intersecta:
                continue
            total_candidatas += 1
            dst_transform = from_bounds(xmin, ymin, xmax, ymax, TS, TS)
            dst = np.full((TS, TS), np.nan, dtype="float32")
            reproject(
                source=rasterio.band(src_ds, 1), destination=dst,
                src_transform=src_ds.transform, src_crs=src_ds.crs, src_nodata=src_nodata,
                dst_transform=dst_transform, dst_crs=DST_CRS, dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
            if np.all(np.isnan(dst)):
                continue
            norm = np.clip(dst / VMAX, 0, 1)
            rgba = CMAP(norm)
            rgba[..., 3] = np.where(np.isnan(dst), 0.0, 0.85)
            dirpath = f"{OUTDIR}/{z}/{x}"
            os.makedirs(dirpath, exist_ok=True)
            plt.imsave(f"{dirpath}/{y}.png", rgba)
            n_generadas_zoom += 1
            total_generadas += 1
    print(f"  zoom {z}: rango x[{x_ini},{x_fin}] y[{y_ini},{y_fin}] -> {n_generadas_zoom} teselas con datos", flush=True)

src_ds.close()
print(f"\nTotal teselas candidatas (interseccion bbox): {total_candidatas}", flush=True)
print(f"Total teselas generadas (con datos reales): {total_generadas}", flush=True)
print("=== TILES OK ===", flush=True)
