# -*- coding: utf-8 -*-
import numpy as np
import json
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as Res
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd

OURS = r"C:\GEOHAZARDS\ISA_4\MODELO_FINAL\modelo_pixel_regional\susceptibilidad_regional_v2_recortada.tif"
RF = r"G:\Unidades compartidas\ISA_Etapa4_2026\Resultados_Modelado\susceptibilidad_RF.tif"
GEO_SHP = r"G:\Unidades compartidas\ISA_Etapa_4_ variables_regional\VARIABLES_MODELO_REGIONAL\Geología_regional\Geologia_consolidada_ISA-4.shp"
OUTDIR = r"C:\GEOHAZARDS\ISA_4\MODELO_FINAL\modelo_pixel_regional\graficas\webmap"

# rampa verde -> amarillo -> rojo (convencion estandar de mapas de amenaza)
ramp_colors = ["#1a9850", "#66bd63", "#a6d96a", "#fee08b", "#fdae61", "#f46d43", "#d73027", "#a50026"]
CMAP = LinearSegmentedColormap.from_list("riesgo_verde_rojo", ramp_colors, N=256)


def leer_diezmado_4326(path, max_dim, vmax, resampling=Res.average):
    with rasterio.open(path) as src:
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        # limitar tamano de salida para que el PNG sea manejable, conservando los limites geograficos
        escala = max(width, height) / max_dim
        if escala > 1:
            from rasterio.transform import array_bounds, from_bounds
            b0 = array_bounds(height, width, transform)  # left, bottom, right, top
            width = max(1, int(width / escala))
            height = max(1, int(height / escala))
            transform = from_bounds(b0[0], b0[1], b0[2], b0[3], width, height)
        print(f"  {path.split(chr(92))[-1]}: reproyectando a EPSG:4326, salida {width}x{height}", flush=True)
        dst = np.full((height, width), np.nan, dtype="float32")
        src_nodata = src.nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        # limites geograficos (lat/lon) de la salida
        from rasterio.transform import array_bounds
        bounds = array_bounds(height, width, transform)  # (left, bottom, right, top)
    return dst, bounds


def guardar_png_overlay(arr, vmax, out_png):
    norm = np.clip(arr / vmax, 0, 1)
    rgba = CMAP(norm)
    rgba[..., 3] = np.where(np.isnan(arr), 0.0, 0.88)
    plt.imsave(out_png, rgba)
    print(f"  guardado {out_png}", flush=True)


print("=== 1. NUESTRO mapa -> EPSG:4326 ===", flush=True)
# nearest: OURS es una capa dispersa/lineal (corredores) sobre un fondo casi
# todo NaN -- "average" puede mezclar mal celdas validas con vecinos NaN al
# diezmar tanto; nearest conserva el valor real de la linea sin sesgo
arr_ours, bounds_ours = leer_diezmado_4326(OURS, max_dim=7000, vmax=0.85, resampling=Res.nearest)
print(f"  rango de valores: min={np.nanmin(arr_ours):.4f} media={np.nanmean(arr_ours):.4f} max={np.nanmax(arr_ours):.4f}", flush=True)
guardar_png_overlay(arr_ours, 0.85, f"{OUTDIR}/susceptibilidad_gam.png")

print("=== 2. RF -> EPSG:4326 ===", flush=True)
arr_rf, bounds_rf = leer_diezmado_4326(RF, max_dim=3500, vmax=0.85, resampling=Res.average)
print(f"  rango de valores: min={np.nanmin(arr_rf):.4f} media={np.nanmean(arr_rf):.4f} max={np.nanmax(arr_rf):.4f}", flush=True)
guardar_png_overlay(arr_rf, 0.85, f"{OUTDIR}/susceptibilidad_rf.png")

print("=== 3. Contorno de la zona de estudio (union de poligonos de geologia) ===", flush=True)
geo = gpd.read_file(GEO_SHP)
geo = geo.to_crs("EPSG:4326")
union = geo.geometry.union_all()
union_simpl = union.simplify(0.0003, preserve_topology=True)
gdf_out = gpd.GeoDataFrame(geometry=[union_simpl], crs="EPSG:4326")
gdf_out.to_file(f"{OUTDIR}/zona_estudio.geojson", driver="GeoJSON")
print(f"  guardado zona_estudio.geojson, bounds: {union_simpl.bounds}", flush=True)

meta = {
    "bounds_ours": {"south": bounds_ours[1], "west": bounds_ours[0], "north": bounds_ours[3], "east": bounds_ours[2]},
    "bounds_rf": {"south": bounds_rf[1], "west": bounds_rf[0], "north": bounds_rf[3], "east": bounds_rf[2]},
    "vmax": 0.85,
}
with open(f"{OUTDIR}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("=== WEBMAP DATA OK ===", flush=True)
