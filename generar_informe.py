"""Genera reportes/informe_sismo_ai_peru.pdf (figuras con matplotlib, documento con reportlab)."""
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (KeepTogether, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

from modelo import CLASES, KS, NOMBRE, SismoModelo

REP = Path(__file__).parent / "reportes"
AZUL, GRIS = colors.HexColor("#1b57c7"), colors.HexColor("#555555")


def figuras(m):
    REP.mkdir(exist_ok=True)
    df, k = m.df, m.k_optimo
    # 1. Matriz de confusión
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    mat = np.array(m.por_k[k]["matriz"])
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(3), [NOMBRE[c] for c in CLASES]); ax.set_yticks(range(3), [NOMBRE[c] for c in CLASES])
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(f"Matriz de confusión (K = {k})")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, mat[i, j], ha="center", va="center", color="white" if mat[i, j] > mat.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(REP / "fig_matriz.png", dpi=150); plt.close(fig)
    # 2. Comparación de K
    fig, ax = plt.subplots(figsize=(5, 3.4))
    x = np.arange(len(KS))
    ax.bar(x - 0.2, [m.cv[q]["media"] for q in KS], 0.4, yerr=[m.cv[q]["desv"] for q in KS], label="Validación cruzada (GroupKFold)", color="#2f7bff", capsize=3)
    ax.bar(x + 0.2, [m.por_k[q]["exactitud"] for q in KS], 0.4, label="Prueba (20 %)", color="#3ddc97")
    ax.set_xticks(x, [f"K={q}" for q in KS]); ax.set_ylim(0.6, 1); ax.set_ylabel("Exactitud"); ax.legend(fontsize=8)
    ax.set_title("Comparación de K"); fig.tight_layout(); fig.savefig(REP / "fig_k.png", dpi=150); plt.close(fig)
    # 3. Distribuciones
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].hist(df.magnitud, bins=12, color="#f59e3b"); ax[0].set_title("Magnitud (Mw)")
    ax[1].hist(df.profundidad_km, bins=20, color="#2f7bff"); ax[1].set_title("Profundidad (km)")
    top = df.drop_duplicates("evento_id").departamento_epicentro.value_counts().head(8)
    ax[2].barh(top.index[::-1], top.values[::-1], color="#7a5cff"); ax[2].set_title("Eventos por departamento (Top 8)")
    fig.tight_layout(); fig.savefig(REP / "fig_dist.png", dpi=150); plt.close(fig)
    # 4. Mapa de epicentros
    ev = df.drop_duplicates("evento_id")
    fig, ax = plt.subplots(figsize=(4.6, 5.2))
    col = np.where(ev.magnitud >= 6, "#e5534b", np.where(ev.magnitud >= 4, "#f59e3b", "#d4b800"))
    ax.scatter(ev.longitud, ev.latitud, s=ev.magnitud ** 2 * 1.5, c=col, alpha=.6, edgecolors="none")
    ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud"); ax.set_title("Epicentros simulados (rojo >= 6.0 | naranja 4.0-5.9 | amarillo < 4.0)", fontsize=7)
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(REP / "fig_mapa.png", dpi=150); plt.close(fig)


def main():
    m = SismoModelo()
    figuras(m)
    r, k = m.resumen(), m.k_optimo
    met = m.por_k[k]
    est = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=est["Heading1"], textColor=AZUL, fontSize=15, spaceBefore=10)
    P = ParagraphStyle("P", parent=est["Normal"], fontSize=10, leading=14, alignment=4)
    PC = ParagraphStyle("PC", parent=P, alignment=1, textColor=GRIS, fontSize=9)
    T = ParagraphStyle("T", parent=est["Title"], fontSize=30, textColor=AZUL)

    def tabla(datos, anchos=None):
        t = Table(datos, colWidths=anchos, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), AZUL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), .4, colors.grey),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3fb")])]))
        return t

    def img(nombre, ancho, maxh=None):
        f = REP / nombre
        if not f.exists():
            return Spacer(1, 1)
        from PIL import Image as PI
        w, h = PI.open(f).size
        alto = ancho * h / w
        if maxh and alto > maxh:
            ancho, alto = ancho * maxh / alto, maxh
        return Image(str(f), width=ancho, height=alto)

    s = []
    s += [Spacer(1, 3 * cm), Paragraph("SENATI · INTELIGENCIA ARTIFICIAL · RETO INDIVIDUAL", PC), Spacer(1, 12),
          Paragraph("SISMO AI PERÚ", T),
          Paragraph("Machine Learning aplicado a la estimación de intensidad sísmica", est["Heading2"]),
          Paragraph("HUÁNUCO · PERÚ | Algoritmo KNN | 20 puntos", PC), Spacer(1, 30),
          Paragraph("Informe técnico con evidencias y conclusiones", est["Heading3"]),
          Paragraph("Alumno: ______________________________<br/>Fecha: " + date.today().strftime("%d/%m/%Y"), P),
          Spacer(1, 20), Paragraph("<b>Aviso:</b> este proyecto es educativo. No predice la fecha, ubicación ni magnitud de futuros "
          "terremotos y no constituye una alerta ni una herramienta oficial de seguridad. Los datos utilizados son <b>simulados</b>.", P),
          PageBreak()]

    s += [Paragraph("1. El reto y el objetivo", H1),
          Paragraph("Se construyó una aplicación web (Flask) que estima la categoría de intensidad sísmica (escala Mercalli modificada "
                    "agrupada en I–IV, V–VI y VII–XII) de un evento hipotético o registrado, usando un clasificador KNN entrenado en Python. "
                    "La aplicación ejecuta el modelo entrenado en cada consulta; no se usan reglas if/else ni resultados estáticos.", P),
          Paragraph("2. Datos", H1),
          Paragraph(f"<b>Origen: DATOS SIMULADOS.</b> No se contó con un catálogo de intensidades observadas, por lo que se generó "
                    f"(script <i>generar_dataset.py</i>) un dataset de {r['registros']:,} registros evento-localidad correspondientes a "
                    f"{r['eventos']} eventos y {r['regiones']} departamentos. La intensidad se simuló con una ley de atenuación simple más ruido "
                    "y efecto de sitio. Las conclusiones se limitan a la práctica educativa. Para datos reales se recomienda IGP, CENSIS y USGS, "
                    "verificando unidades, escala y procedencia.", P), Spacer(1, 6),
          tabla([["Campo", "Descripción"], ["evento_id, fecha", "Identificación y fecha del evento"],
                 ["latitud, longitud", "Epicentro (para el mapa)"], ["localidad / departamento", "Localidad donde se observa el evento"],
                 ["magnitud (Mw)", "Magnitud reportada (escala Mw)"], ["profundidad_km", "Profundidad focal"],
                 ["distancia_km", "Distancia epicentral a la localidad"],
                 ["intensidad", "Mercalli modificada agrupada: I–IV, V–VI, VII–XII"], ["dato_simulado", "1 = dato simulado"]],
                [5 * cm, 11 * cm]), Spacer(1, 8),
          Paragraph(f"<b>Limpieza (Pandas/NumPy):</b> {r['limpieza']['originales']:,} registros originales; se eliminaron duplicados, nulos y "
                    f"valores fuera de rango (magnitud 2–9.5, profundidad 0–700 km, distancia 0–1500 km); quedaron {r['limpieza']['limpias']:,}. "
                    f"Distribución de clases: Baja {r['dist_intensidad']['I–IV']:,} · Moderada {r['dist_intensidad']['V–VI']:,} · "
                    f"Fuerte {r['dist_intensidad']['VII–XII']:,}.", P), img("fig_dist.png", 16.5 * cm), PageBreak()]

    sp = r["split"]
    s += [Paragraph("3. Metodología", H1),
          Paragraph(f"<b>Partición 80/20 agrupada por evento:</b> <i>GroupShuffleSplit</i> con evento_id como grupo: entrenamiento "
                    f"{sp['train']:,} registros ({sp['eventos_train']} eventos) y prueba {sp['test']:,} registros ({sp['eventos_test']} eventos). "
                    f"Eventos presentes en ambos conjuntos: {sp['eventos_solapados']} (sin fuga de información).", P),
          Paragraph("<b>Pipeline:</b> <i>StandardScaler</i> + <i>KNeighborsClassifier</i> (distancia euclidiana). Variables: magnitud, profundidad_km "
                    "y distancia_km. El escalador se ajusta solo con los datos de entrenamiento.", P),
          Paragraph("<b>Selección de K:</b> se compararon K = 3, 5, 7 y 9 con <i>GroupKFold</i> (5 pliegues) únicamente sobre el entrenamiento.", P),
          Spacer(1, 6),
          tabla([["K", "Exactitud CV (media ± desv.)", "Exactitud en prueba", "F1 macro en prueba"]] +
                [[f"{q}{' *' if q == k else ''}", f"{m.cv[q]['media']:.3f} ± {m.cv[q]['desv']:.3f}",
                  f"{m.por_k[q]['exactitud']:.3f}", f"{m.por_k[q]['f1']:.3f}"] for q in KS], [2 * cm, 5.5 * cm, 4.5 * cm, 4 * cm]),
          Paragraph(f"* K seleccionado = {k} (mayor exactitud media en validación cruzada).", PC), Spacer(1, 6),
          Paragraph("4. Resultados en el conjunto de prueba", H1),
          Paragraph(f"Con K = {k}: exactitud {met['exactitud']:.3f}, precisión macro {met['precision']:.3f}, recall macro {met['recall']:.3f} y "
                    f"F1 macro {met['f1']:.3f}. La mayor confusión ocurre entre clases contiguas (Baja–Moderada y Moderada–Fuerte), lo que es "
                    "coherente con el carácter continuo de la intensidad.", P), Spacer(1, 6),
          tabla([["Clase", "Precisión", "Recall", "F1", "Soporte"]] +
                [[f"{c['nombre']} ({c['clase']})", f"{c['precision']:.2f}", f"{c['recall']:.2f}", f"{c['f1']:.2f}", c["soporte"]] for c in met["por_clase"]],
                [5 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm]), Spacer(1, 8),
          Table([[img("fig_matriz.png", 8 * cm), img("fig_k.png", 8.5 * cm)]]), PageBreak()]

    s += [Paragraph("5. Interfaz web y evidencias", H1),
          Paragraph("La aplicación incluye: dashboard con indicadores, mapa de epicentros, simulador conectado al modelo (con validación de entradas, "
                    "selección de K y votos de los vecinos), matriz de confusión, métricas, gráficos de magnitud, profundidad y distribución regional, "
                    "tabla del dataset y reportes. Capturas de la aplicación en ejecución:", P), Spacer(1, 6),
          img("captura_dashboard.png", 16.5 * cm, 10 * cm), Spacer(1, 6), img("captura_simulador.png", 16.5 * cm, 11 * cm), PageBreak()]

    esc = m.escenarios()
    s += [Paragraph("6. Tres escenarios hipotéticos", H1),
          tabla([["Escenario", "Mw", "Prof. (km)", "Dist. (km)", "Resultado", "Votos I–IV / V–VI / VII–XII"]] +
                [[e["nombre"].split("·")[1].strip(), e["magnitud"], e["profundidad_km"], e["distancia_km"], e["categoria"],
                  " / ".join(str(e["votos"][c]) for c in CLASES)] for e in esc], [4.6 * cm, 1.5 * cm, 2.2 * cm, 2.2 * cm, 2.5 * cm, 4 * cm]),
          Spacer(1, 6),
          Paragraph(f"Interpretación (K = {k}): un evento pequeño y lejano produce una intensidad baja; al aumentar la magnitud y reducir la "
                    "distancia, la mayoría de los vecinos pertenece a clases de mayor intensidad. Los resultados reflejan los datos simulados "
                    "y no deben interpretarse como estimaciones reales.", P),
          Paragraph("7. Limitaciones", H1),
          Paragraph("• Datos simulados: el modelo aprende la regla con la que se generaron. • Solo tres variables; no se considera el tipo de suelo, "
                    "la construcción ni la geología local. • KNN es sensible a la escala y a la densidad de datos y no extrapola fuera del rango visto. • "
                    "El sistema no predice terremotos ni sirve como alerta.", P),
          Paragraph("8. Preguntas de sustentación", H1)]
    qa = [("¿Por qué escalar las variables?", "KNN usa distancias; sin escalar, la profundidad y la distancia (cientos de km) dominarían sobre la magnitud (2–8). StandardScaler pone todas en la misma escala y, dentro del Pipeline, se ajusta solo con el entrenamiento."),
          ("¿Cómo influye K?", "K pequeño sobreajusta (sensible al ruido); K grande suaviza demasiado y mezcla clases. Se eligió K con validación cruzada agrupada."),
          ("¿Por qué separar por evento?", "Un evento aparece en varias localidades con la misma magnitud y profundidad; repartirlas entre entrenamiento y prueba filtra información e infla la exactitud."),
          ("¿Diferencia entre magnitud e intensidad?", "La magnitud mide la energía liberada en la fuente (una por evento); la intensidad mide los efectos observados en un lugar y varía con distancia, profundidad y suelo."),
          ("¿Por qué no puede anticipar un terremoto?", "Solo recibe las características de un evento ya definido; no hay señales precursoras confiables y la ciencia actual no puede predecir cuándo, dónde ni con qué magnitud ocurrirá un sismo.")]
    for q, a in qa:
        s += [Paragraph(f"<b>{q}</b> {a}", P), Spacer(1, 4)]
    s += [Paragraph("9. Conclusiones", H1),
          Paragraph(f"El pipeline StandardScaler + KNN (K = {k}) alcanzó una exactitud de {met['exactitud']:.2f} en un conjunto de prueba separado por eventos, "
                    "mostrando que la magnitud, la profundidad y la distancia permiten estimar la categoría de intensidad en el conjunto simulado. "
                    "La partición agrupada evita resultados optimistas por fuga de información. Como siguiente paso se recomienda reemplazar el "
                    "dataset por registros reales de IGP/CENSIS/USGS con intensidades observadas.", P),
          Spacer(1, 8), KeepTogether([Paragraph("Mapa de epicentros simulados", est["Heading4"]), img("fig_mapa.png", 7.5 * cm)])]

    def pie(c, d):
        c.setFont("Helvetica", 8); c.setFillColor(GRIS)
        c.drawString(2 * cm, 1.2 * cm, "SENATI · SISMO AI PERÚ · Proyecto educativo (datos simulados)")
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Página {d.page}")

    out = REP / "informe_sismo_ai_peru.pdf"
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                      title="SISMO AI PERÚ - Informe", author="SENATI").build(s, onFirstPage=pie, onLaterPages=pie)
    print("Informe generado:", out)


if __name__ == "__main__":
    main()
