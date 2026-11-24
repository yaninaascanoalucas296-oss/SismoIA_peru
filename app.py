"""SISMO AI PERÚ - servidor Flask. Ejecuta el modelo KNN entrenado (sin reglas if/else)."""
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from modelo import CLASES, FEATURES, KS, LIMITES, NOMBRE, SismoModelo
from regiones import DEPTOS

BASE = Path(__file__).parent
app = Flask(__name__)
M = SismoModelo()


@app.route("/")
def index():
    deptos = sorted(DEPTOS)
    return render_template("index.html", deptos=deptos, ks=KS, k_optimo=M.k_optimo)


@app.route("/api/resumen")
def resumen():
    return jsonify(M.resumen())


@app.route("/api/modelo")
def modelo():
    return jsonify(M.info_modelo())


@app.route("/api/mapa")X
def mapa():
    return jsonify(M.epicentros())


@app.route("/api/dataset")
def dataset():
    try:
        pagina = max(int(request.args.get("pagina", 1)), 1)
    except ValueError:
        pagina = 1
    return jsonify(M.tabla(pagina, 15, request.args.get("q", "").strip(),
                           request.args.get("intensidad", "")))


@app.route("/api/escenarios")
def escenarios():
    return jsonify(M.escenarios())


@app.route("/api/predecir", methods=["POST"])
def predecir():
    d = request.get_json(silent=True) or {}
    try:
        mag = float(d.get("magnitud"))
        prof = float(d.get("profundidad"))
        dist = float(d.get("distancia"))
        k = int(d.get("k", M.k_optimo))
    except (TypeError, ValueError):
        return jsonify(error="Ingresa valores numéricos válidos."), 400
    for nombre, v, (lo, hi) in (("Magnitud", mag, LIMITES["magnitud"]),
                                ("Profundidad (km)", prof, LIMITES["profundidad_km"]),
                                ("Distancia (km)", dist, LIMITES["distancia_km"])):
        if not (lo <= v <= hi):
            return jsonify(error=f"{nombre} debe estar entre {lo} y {hi}."), 400
    if k not in KS:
        return jsonify(error=f"K debe ser uno de {KS}."), 400
    r = M.predecir(mag, prof, dist, k)
    c = M.coords_depto(d.get("departamento"))
    r["ubicacion"] = {"lat": c[0], "lon": c[1], "localidad": c[2]} if c else None
    return jsonify(r)


@app.route("/descargar/dataset.csv")
def dl_dataset():
    return send_file(BASE / "data" / "sismos.csv", as_attachment=True, download_name="sismos.csv")


@app.route("/descargar/informe.pdf")
def dl_informe():
    pdf = BASE / "reportes" / "informe_sismo_ai_peru.pdf"
    if not pdf.exists():
        import generar_informe
        generar_informe.main()
    return send_file(pdf, as_attachment=True, download_name="informe_sismo_ai_peru.pdf")


if __name__ == "__main__":
    print("\n  SISMO AI PERÚ listo ->  http://localhost:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
