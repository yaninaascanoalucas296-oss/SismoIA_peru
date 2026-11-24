"""
Genera data/sismos.csv  ->  DATASET SIMULADO (uso educativo).

Cada fila es un registro evento-localidad: un sismo observado desde una localidad.
La intensidad (Mercalli modificada) se simula con una ley de atenuación simple
+ ruido + efecto de sitio. NO son datos reales del IGP / CENSIS / USGS.
Para usar datos reales, reemplaza data/sismos.csv respetando las columnas.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OBJETIVO_REGISTROS = 2486
rng = np.random.default_rng(2025)

from regiones import DEPTOS

nombres = list(DEPTOS)
lat_loc = np.array([DEPTOS[d][1] for d in nombres])
lon_loc = np.array([DEPTOS[d][2] for d in nombres])
pesos = np.array([DEPTOS[d][3] for d in nombres]); pesos = pesos / pesos.sum()
efecto_sitio = dict(zip(nombres, rng.normal(0, 0.25, len(nombres))))


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def magnitud():
    u = rng.random()
    if u < 0.563:   return 2.6 + min(rng.exponential(0.45), 1.35)       # < 4.0
    if u < 0.911:   return 4.0 + min(rng.exponential(0.6), 1.9)         # 4.0 - 5.9
    return 6.0 + min(rng.exponential(0.5), 1.9)                         # >= 6.0


def profundidad():
    u = rng.random()
    if u < 0.90: return rng.uniform(5, 70)
    if u < 0.99: return rng.uniform(70, 200)
    return rng.uniform(200, 600)


def clase(mmi):
    return "I–IV" if mmi <= 4 else ("V–VI" if mmi <= 6 else "VII–XII")


filas, n_ev = [], 0
fechas = pd.date_range("2010-01-01", "2025-11-20", freq="h")
while len(filas) < OBJETIVO_REGISTROS:
    n_ev += 1
    d_epi = rng.choice(nombres, p=pesos)
    i = nombres.index(d_epi)
    lat = lat_loc[i] + rng.normal(0, 0.45)
    lon = lon_loc[i] + rng.normal(0, 0.45)
    mag, prof = round(magnitud(), 1), round(profundidad())
    fecha = str(rng.choice(fechas))[:16].replace("T", " ")
    dist_all = haversine(lat, lon, lat_loc, lon_loc)
    cercanas = np.argsort(dist_all)[:6]
    k = rng.choice([1, 2, 3, 4], p=[.15, .35, .3, .2])
    for j in rng.choice(cercanas, size=k, replace=False):
        d = max(round(float(dist_all[j])), 1)
        R = np.sqrt(d ** 2 + prof ** 2)
        mmi = 2.4 * mag - 2.4 * np.log10(R + 1) + 0.8 + efecto_sitio[nombres[j]] + rng.normal(0, 0.55)
        mmi = int(np.clip(round(mmi), 1, 12))
        filas.append(dict(evento_id=f"EV{n_ev:04d}", fecha=fecha, latitud=round(lat, 3),
                          longitud=round(lon, 3), departamento_epicentro=d_epi,
                          localidad=DEPTOS[nombres[j]][0], departamento=nombres[j],
                          magnitud=mag, escala="Mw", profundidad_km=prof, distancia_km=d,
                          intensidad_mmi=mmi, intensidad=clase(mmi), dato_simulado=1))

df = pd.DataFrame(filas).iloc[:OBJETIVO_REGISTROS]
Path("data").mkdir(exist_ok=True)
df.to_csv("data/sismos.csv", index=False, encoding="utf-8")
print(f"Dataset SIMULADO: {len(df)} registros, {df.evento_id.nunique()} eventos")
print((df.intensidad.value_counts(normalize=True) * 100).round(1).to_string())
print(df.groupby("evento_id").first().departamento_epicentro.value_counts().head(6).to_string())
