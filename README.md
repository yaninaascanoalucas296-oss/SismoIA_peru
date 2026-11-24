# SISMO AI PERÚ · SENATI / Inteligencia Artificial / Reto individual
Estimación de la categoría de intensidad sísmica (Mercalli modificada: I–IV, V–VI, VII–XII)
con KNN, integrada en una aplicación web Flask.

> Proyecto educativo. NO predice fecha, ubicación ni magnitud de terremotos y NO es una alerta oficial.
> El dataset incluido es **SIMULADO** (columna `dato_simulado = 1`).

## Ejecutar
```bash
pip install -r requirements.txt
python app.py
```
Abrir en el navegador: **http://localhost:5000**
(el mapa Leaflet requiere internet; sin internet se muestra un mapa alternativo).
Detener el servidor: `Ctrl + C`.

## Comandos opcionales
```bash
python generar_dataset.py    # regenera data/sismos.csv (2486 registros simulados)
python generar_informe.py    # regenera reportes/informe_sismo_ai_peru.pdf
```

## Estructura
| Archivo | Función |
|---|---|
| app.py | Servidor Flask + API (valida entradas, ejecuta el modelo) |
| modelo.py | Limpieza (Pandas/NumPy), GroupShuffleSplit 80/20, Pipeline(StandardScaler + KNN), K = 3,5,7,9 con GroupKFold, métricas |
| generar_dataset.py / regiones.py | Dataset simulado y departamentos |
| generar_informe.py | Informe PDF (matplotlib + reportlab) |
| templates/, static/ | Interfaz web (dashboard, mapa, simulador, evaluación, dataset, reportes, ayuda) |
| data/sismos.csv | Dataset |
| reportes/ | Informe PDF, figuras y capturas |

## Usar datos reales
Reemplaza `data/sismos.csv` (IGP / CENSIS / USGS) manteniendo estas columnas:
evento_id, fecha, latitud, longitud, departamento_epicentro, localidad, departamento,
magnitud, profundidad_km, distancia_km, intensidad (I–IV, V–VI, VII–XII).
