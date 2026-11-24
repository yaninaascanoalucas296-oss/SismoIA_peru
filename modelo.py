"""
Modelo KNN de SISMO AI PERÚ.
Pipeline(StandardScaler + KNeighborsClassifier), partición 80/20 agrupada por evento
y comparación de K = 3, 5, 7, 9 con validación cruzada agrupada (GroupKFold).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from regiones import DEPTOS

BASE = Path(__file__).parent
CSV = BASE / "data" / "sismos.csv"
FEATURES = ["magnitud", "profundidad_km", "distancia_km"]
CLASES = ["I–IV", "V–VI", "VII–XII"]
NOMBRE = {"I–IV": "Baja", "V–VI": "Moderada", "VII–XII": "Fuerte"}
KS = [3, 5, 7, 9]
LIMITES = {"magnitud": (2.0, 9.5), "profundidad_km": (0, 700), "distancia_km": (0, 1500)}


def hacer_pipeline(k):
    return Pipeline([("scaler", StandardScaler()),
                     ("knn", KNeighborsClassifier(n_neighbors=k))])


class SismoModelo:
    def __init__(self):
        self.cargar_y_limpiar()
        self.entrenar()

    # ---------- 1. Datos ----------
    def cargar_y_limpiar(self):
        df = pd.read_csv(CSV)
        self.filas_originales = len(df)
        df = df.drop_duplicates()
        df = df.dropna(subset=FEATURES + ["intensidad", "evento_id"])
        for c, (lo, hi) in LIMITES.items():
            df = df[(df[c] >= lo) & (df[c] <= hi)]
        df = df[df["intensidad"].isin(CLASES)].reset_index(drop=True)
        self.df = df
        self.filas_limpias = len(df)

    # ---------- 2. Entrenamiento ----------
    def entrenar(self):
        df = self.df
        X, y, g = df[FEATURES], df["intensidad"], df["evento_id"]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        itr, ite = next(gss.split(X, y, groups=g))
        self.Xtr, self.ytr, self.gtr = X.iloc[itr], y.iloc[itr], g.iloc[itr]
        self.Xte, self.yte = X.iloc[ite], y.iloc[ite]
        self.meta_tr = df.iloc[itr].reset_index(drop=True)
        self.eventos_solapados = len(set(g.iloc[itr]) & set(g.iloc[ite]))  # debe ser 0

        gkf = GroupKFold(n_splits=5)
        self.cv, self.modelos, self.por_k = {}, {}, {}
        for k in KS:
            pipe = hacer_pipeline(k)
            sc = cross_val_score(pipe, self.Xtr, self.ytr, groups=self.gtr, cv=gkf, scoring="accuracy")
            self.cv[k] = {"media": float(sc.mean()), "desv": float(sc.std())}
            pipe.fit(self.Xtr, self.ytr)
            self.modelos[k] = pipe
            self.por_k[k] = self.evaluar(pipe)
        self.k_optimo = max(KS, key=lambda k: (round(self.cv[k]["media"], 6), -k))

    def evaluar(self, pipe):
        pred = pipe.predict(self.Xte)
        p, r, f, _ = precision_recall_fscore_support(self.yte, pred, labels=CLASES,
                                                     average="macro", zero_division=0)
        pc, rc, fc, sc = precision_recall_fscore_support(self.yte, pred, labels=CLASES, zero_division=0)
        return {
            "exactitud": float(accuracy_score(self.yte, pred)),
            "precision": float(p), "recall": float(r), "f1": float(f),
            "matriz": confusion_matrix(self.yte, pred, labels=CLASES).tolist(),
            "por_clase": [{"clase": c, "nombre": NOMBRE[c], "precision": float(pc[i]),
                           "recall": float(rc[i]), "f1": float(fc[i]), "soporte": int(sc[i])}
                          for i, c in enumerate(CLASES)],
        }

    # ---------- 3. Predicción ----------
    def predecir(self, magnitud, profundidad, distancia, k):
        pipe = self.modelos[k]
        x = pd.DataFrame([[magnitud, profundidad, distancia]], columns=FEATURES)
        proba = pipe.predict_proba(x)[0]
        clase = str(pipe.predict(x)[0])
        xs = pipe.named_steps["scaler"].transform(x)
        dist, idx = pipe.named_steps["knn"].kneighbors(xs, n_neighbors=k)
        vecinos = []
        for n, (d, j) in enumerate(zip(dist[0], idx[0]), 1):
            fila = self.meta_tr.iloc[j]
            vecinos.append({"n": n, "evento_id": fila.evento_id, "localidad": fila.localidad,
                            "magnitud": float(fila.magnitud), "profundidad_km": float(fila.profundidad_km),
                            "distancia_km": float(fila.distancia_km), "intensidad": fila.intensidad,
                            "nombre": NOMBRE[fila.intensidad], "distancia": round(float(d), 3)})
        votos = {c: sum(v["intensidad"] == c for v in vecinos) for c in CLASES}
        return {"clase": clase, "nombre": NOMBRE[clase], "k": k,
                "probabilidades": {c: round(float(p) * 100, 1) for c, p in zip(pipe.classes_, proba)},
                "votos": votos, "vecinos": vecinos}

    def escenarios(self):
        casos = [("Escenario 1 · Evento leve y lejano", 3.2, 35, 100),
                 ("Escenario 2 · Evento moderado", 4.0, 30, 60),
                 ("Escenario 3 · Evento fuerte y superficial", 6.8, 20, 25)]
        out = []
        for nombre, m, p, d in casos:
            r = self.predecir(m, p, d, self.k_optimo)
            out.append({"nombre": nombre, "magnitud": m, "profundidad_km": p, "distancia_km": d,
                        "clase": r["clase"], "categoria": r["nombre"], "votos": r["votos"],
                        "probabilidades": r["probabilidades"]})
        return out

    # ---------- 4. Resúmenes para el dashboard ----------
    def resumen(self):
        df = self.df
        ev = df.drop_duplicates("evento_id")
        mag = pd.cut(df.magnitud, [0, 3.999, 5.999, 99], labels=["lt4", "mid", "ge6"]).value_counts()
        top = ev.departamento_epicentro.value_counts()
        h_prof, e_prof = np.histogram(df.profundidad_km, bins=[0, 20, 40, 60, 100, 200, 700])
        h_mag, e_mag = np.histogram(df.magnitud, bins=[2, 3, 4, 5, 6, 7, 8])
        return {
            "registros": int(len(df)), "eventos": int(ev.evento_id.nunique()),
            "mag_prom": round(float(df.magnitud.mean()), 1),
            "prof_prom": round(float(df.profundidad_km.mean())),
            "regiones": int(df.departamento_epicentro.nunique()),
            "dist_magnitud": {k: int(mag.get(k, 0)) for k in ["lt4", "mid", "ge6"]},
            "dist_intensidad": {c: int((df.intensidad == c).sum()) for c in CLASES},
            "top_deptos": [[d, int(n)] for d, n in top.head(5).items()],
            "todos_deptos": [[d, int(n)] for d, n in top.items()],
            "hist_prof": {"etq": ["0-20", "20-40", "40-60", "60-100", "100-200", "200+"], "n": h_prof.tolist()},
            "hist_mag": {"etq": ["2-3", "3-4", "4-5", "5-6", "6-7", "7+"], "n": h_mag.tolist()},
            "limpieza": {"originales": self.filas_originales, "limpias": self.filas_limpias},
            "split": {"train": int(len(self.Xtr)), "test": int(len(self.Xte)),
                      "eventos_train": int(self.gtr.nunique()),
                      "eventos_test": int(self.df.loc[self.Xte.index, "evento_id"].nunique()),
                      "eventos_solapados": self.eventos_solapados},
        }

    def info_modelo(self):
        return {"k_optimo": self.k_optimo, "ks": KS, "cv": {str(k): v for k, v in self.cv.items()},
                "por_k": {str(k): v for k, v in self.por_k.items()},
                "clases": [{"clase": c, "nombre": NOMBRE[c]} for c in CLASES]}

    def epicentros(self):
        ev = self.df.drop_duplicates("evento_id")
        return [[r.evento_id, r.latitud, r.longitud, r.magnitud, r.profundidad_km,
                 r.fecha, r.departamento_epicentro] for r in ev.itertuples()]

    def tabla(self, pagina=1, por_pagina=15, q="", intensidad=""):
        df = self.df
        if q:
            m = df.evento_id.str.contains(q, case=False) | df.localidad.str.contains(q, case=False) \
                | df.departamento_epicentro.str.contains(q, case=False)
            df = df[m]
        if intensidad in CLASES:
            df = df[df.intensidad == intensidad]
        total = len(df)
        ini = (pagina - 1) * por_pagina
        cols = ["evento_id", "fecha", "departamento_epicentro", "localidad", "magnitud",
                "profundidad_km", "distancia_km", "intensidad_mmi", "intensidad"]
        return {"total": total, "pagina": pagina, "por_pagina": por_pagina,
                "filas": df.iloc[ini:ini + por_pagina][cols].values.tolist(), "cols": cols}

    @staticmethod
    def coords_depto(depto):
        d = DEPTOS.get(depto)
        return (d[1], d[2], d[0]) if d else None
