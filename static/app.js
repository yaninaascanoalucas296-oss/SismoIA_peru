const $ = id => document.getElementById(id);
const CL = ["I–IV", "V–VI", "VII–XII"];
const NOM = {"I–IV": "Baja", "V–VI": "Moderada", "VII–XII": "Fuerte"};
const COL = {"I–IV": "#3ddc97", "V–VI": "#f5b942", "VII–XII": "#e5534b"};
const fmt = n => Number(n).toLocaleString("es-PE");
const pct = x => (x * 100).toFixed(0) + " %";
const get = async u => (await fetch(u)).json();
let resumen, modelo, escenarios, K, mapa, capaSim, pag = 1, totalPag = 1;

/* ---------------- Vistas ---------------- */
function vista(v) {
  document.body.dataset.vista = v;
  document.querySelectorAll("[data-v]").forEach(el => {
    if (el.tagName === "BUTTON") return;
    el.hidden = !el.dataset.v.split(" ").includes(v);
  });
  document.querySelectorAll("#nav button").forEach(b => b.classList.toggle("activo", b.dataset.v === v));
  if (mapa) setTimeout(() => mapa.invalidateSize(), 80);
  window.scrollTo({top: 0, behavior: "smooth"});
}

/* ---------------- Dashboard ---------------- */
function pintarStats() {
  $("stReg").textContent = fmt(resumen.registros);
  $("stEv").textContent = fmt(resumen.eventos) + " eventos · registros históricos (simulados)";
  $("stMag").textContent = resumen.mag_prom;
  $("stProf").textContent = resumen.prof_prom;
  $("stReg2").textContent = resumen.regiones;
}

function pintarDonut() {
  const d = resumen.dist_magnitud, tot = resumen.registros;
  const partes = [["lt4", "< 4.0", "#f7e04a"], ["mid", "4.0 – 5.9", "#f59e3b"], ["ge6", "≥ 6.0", "#e5534b"]];
  let acum = 0, grad = [];
  partes.forEach(([k, , c]) => { const p = d[k] / tot * 100; grad.push(`${c} ${acum}% ${acum + p}%`); acum += p; });
  $("donut").style.background = `conic-gradient(${grad.join(",")})`;
  $("donutTxt").innerHTML = `${fmt(tot)}<br><small style="font-weight:400;color:#8fa9bd">registros</small>`;
  $("donutLeyenda").innerHTML = partes.map(([k, t, c]) =>
    `<li><i style="background:${c}"></i>${t} (${(d[k] / tot * 100).toFixed(1)} %)</li>`).join("");
}

function barras(lista, max) {
  return lista.map(([n, v]) => `<div class="fila"><span>${n}</span><div><i style="width:${v / max * 100}%"></i></div><b>${v}</b></div>`).join("");
}
function pintarTop() { $("top5").innerHTML = barras(resumen.top_deptos, resumen.top_deptos[0][1]); }

function pintarModelo() {
  const m = modelo.por_k[K];
  const filas = m.matriz;
  let h = `<div class="mtx"><span></span>${CL.map(c => `<span class="h">${NOM[c]}<br>(${c})</span>`).join("")}`;
  filas.forEach((f, i) => {
    const tot = f.reduce((a, b) => a + b, 0) || 1;
    h += `<span class="h">${NOM[CL[i]]}</span>`;
    f.forEach((v, j) => {
      const a = (0.12 + 0.75 * v / tot).toFixed(2);
      const rgb = i === j ? "61,220,151" : "245,185,66";
      h += `<span class="c" style="background:rgba(${rgb},${i === j ? a : (0.08 + 0.6 * v / tot).toFixed(2)})">${v}</span>`;
    });
  });
  $("matriz").innerHTML = h + `</div><p class="nota" style="text-align:center">Filas: real · Columnas: predicho · K = ${K}</p>`;
  $("rend").innerHTML = [["Exactitud", `${m.exactitud.toFixed(2)} (${pct(m.exactitud)})`],
    ["Precisión (macro)", m.precision.toFixed(2)], ["Recall (macro)", m.recall.toFixed(2)],
    ["F1-Score (macro)", m.f1.toFixed(2)]].map(([a, b]) => `<li><span>${a}</span><b>${b}</b></li>`).join("");
  $("kNota").textContent = `K = ${K}${K === modelo.k_optimo ? " (óptimo por validación cruzada agrupada)" : ""}. Evaluado en ${fmt(resumen.split.test)} registros de prueba.`;
  $("porClase").innerHTML = `<table><tr><th>Clase</th><th>Precisión</th><th>Recall</th><th>F1</th><th>Soporte</th></tr>` +
    m.por_clase.map(c => `<tr><td><span class="badge" style="background:${COL[c.clase]}">${c.nombre} · ${c.clase}</span></td>
      <td>${c.precision.toFixed(2)}</td><td>${c.recall.toFixed(2)}</td><td>${c.f1.toFixed(2)}</td><td>${c.soporte}</td></tr>`).join("") + `</table>`;
  pintarKTabla();
}

function pintarKTabla() {
  $("kTabla").innerHTML = `<table><tr><th>K</th><th>CV (media ± desv.)</th><th>Exactitud prueba</th><th>F1</th></tr>` +
    modelo.ks.map(k => {
      const cv = modelo.cv[k], t = modelo.por_k[k];
      return `<tr class="sel ${k === modelo.k_optimo ? "opt" : ""}" onclick="elegirK(${k})"><td>${k}${k === modelo.k_optimo ? " ★" : ""}</td>
        <td>${cv.media.toFixed(3)} ± ${cv.desv.toFixed(3)}</td><td>${t.exactitud.toFixed(3)}</td><td>${t.f1.toFixed(3)}</td></tr>`;
    }).join("") + `</table><p class="nota">★ K seleccionado con la mayor exactitud media en validación cruzada (GroupKFold, 5 pliegues, solo entrenamiento). Clic en una fila para ver sus métricas.</p>`;
}
function elegirK(k) { K = k; $("kSel").value = k; pintarModelo(); }

function pintarAnalisis() {
  const s = resumen.split, l = resumen.limpieza;
  $("splitInfo").innerHTML = `<div class="info-split">
    <p>Registros originales: <b>${fmt(l.originales)}</b> · tras limpieza: <b>${fmt(l.limpias)}</b></p>
    <p>Entrenamiento: <b>${fmt(s.train)}</b> registros (${fmt(s.eventos_train)} eventos)</p>
    <p>Prueba: <b>${fmt(s.test)}</b> registros (${fmt(s.eventos_test)} eventos)</p>
    <p>Eventos presentes en ambos conjuntos: <b>${s.eventos_solapados}</b> ✔ (sin fuga de información)</p>
    <p>Clases: ${CL.map(c => `<span class="badge" style="background:${COL[c]}">${NOM[c]} ${fmt(resumen.dist_intensidad[c])}</span>`).join(" ")}</p></div>`;
  const hist = (h, id) => { const mx = Math.max(...h.n);
    $(id).innerHTML = h.n.map((n, i) => `<div><i style="height:${n / mx * 120}px"></i>${n}<br>${h.etq[i]}</div>`).join(""); };
  hist(resumen.hist_mag, "hMag"); hist(resumen.hist_prof, "hProf");
  $("todos").innerHTML = barras(resumen.todos_deptos, resumen.todos_deptos[0][1]);
}

/* ---------------- Mapa (Leaflet) ---------------- */
function initMapa() {
  if (typeof L === "undefined") { mapaOffline(); return; }
  mapa = L.map("map", {preferCanvas: true, zoomControl: true}).setView([-9.5, -75.5], 5);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    {attribution: "© OpenStreetMap · © CARTO", maxZoom: 12}).addTo(mapa);
  get("/api/mapa").then(ev => ev.forEach(([id, lat, lon, mag, prof, fecha, dep]) => {
    const c = mag >= 6 ? "#e5534b" : mag >= 4 ? "#f59e3b" : "#f7e04a";
    L.circleMarker([lat, lon], {radius: 2 + mag * 0.9, color: c, weight: 1, fillColor: c, fillOpacity: .6})
      .bindPopup(`<b>${dep}</b> · ${id}<br>Magnitud: ${mag} Mw<br>Profundidad: ${prof} km<br>Fecha: ${fecha}`).addTo(mapa);
  }));
}

/* Mapa alternativo (sin internet): epicentros dibujados en un canvas lat/lon */
let epiOffline = [], puntoSim = null;
async function mapaOffline() {
  epiOffline = await get("/api/mapa");
  const cont = $("map"); cont.innerHTML = '<canvas id="cv" style="width:100%;height:100%"></canvas>';
  dibujarOffline();
  window.addEventListener("resize", dibujarOffline);
}
function dibujarOffline() {
  const cv = $("cv"); if (!cv) return;
  const w = cv.clientWidth, h = cv.clientHeight; cv.width = w; cv.height = h;
  const g = cv.getContext("2d"), LON = [-82, -68], LAT = [-19, 0];
  const X = lon => (lon - LON[0]) / (LON[1] - LON[0]) * w, Y = lat => (LAT[1] - lat) / (LAT[1] - LAT[0]) * h;
  g.fillStyle = "#0a1a28"; g.fillRect(0, 0, w, h);
  g.strokeStyle = "#16324a"; g.fillStyle = "#58758c"; g.font = "11px sans-serif";
  for (let lo = -80; lo <= -70; lo += 2) { g.beginPath(); g.moveTo(X(lo), 0); g.lineTo(X(lo), h); g.stroke(); g.fillText(lo + "°", X(lo) + 3, h - 4); }
  for (let la = -2; la >= -18; la -= 4) { g.beginPath(); g.moveTo(0, Y(la)); g.lineTo(w, Y(la)); g.stroke(); g.fillText(la + "°", 4, Y(la) - 3); }
  g.fillText("OCÉANO PACÍFICO", X(-81.5), Y(-9));
  epiOffline.forEach(([id, lat, lon, mag]) => {
    g.fillStyle = mag >= 6 ? "rgba(229,83,75,.7)" : mag >= 4 ? "rgba(245,158,59,.65)" : "rgba(247,224,74,.55)";
    g.beginPath(); g.arc(X(lon), Y(lat), 2 + mag * 0.8, 0, 7); g.fill();
  });
  if (puntoSim) { g.fillStyle = "#5aa9ff"; g.strokeStyle = "#fff"; g.lineWidth = 2;
    g.beginPath(); g.arc(X(puntoSim.lon), Y(puntoSim.lat), 8, 0, 7); g.fill(); g.stroke();
    g.fillStyle = "#fff"; g.font = "bold 12px sans-serif"; g.fillText(puntoSim.localidad, X(puntoSim.lon) + 11, Y(puntoSim.lat) + 4); }
}

/* ---------------- Simulador ---------------- */
function barraProb(p) {
  return CL.map(c => `<div class="prob"><span>${NOM[c]}</span><div><i style="width:${p[c] || 0}%;background:${COL[c]}"></i></div><b>${(p[c] || 0).toFixed(0)} %</b></div>`).join("");
}
async function estimar() {
  $("err").textContent = "";
  const body = {departamento: $("dep").value, magnitud: $("mag").value, profundidad: $("prof").value,
                distancia: $("dist").value, k: $("kSel").value};
  const r = await fetch("/api/predecir", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const d = await r.json();
  if (!r.ok) { $("err").textContent = d.error; return; }
  K = d.k; pintarModelo();
  const caja = $("resCaja");
  caja.className = "caja " + (d.clase === "I–IV" ? "baja" : d.clase === "V–VI" ? "moderada" : "");
  caja.innerHTML = `<small>Intensidad estimada:</small><b>${d.nombre}</b> <small style="display:inline">(MMI ${d.clase})</small>`;
  $("probs").innerHTML = barraProb(d.probabilidades);
  $("votos").textContent = "Votos de los " + d.k + " vecinos → " + CL.map(c => `${c}: ${d.votos[c]}`).join(" | ") + ". Resultado del modelo KNN entrenado con datos simulados.";
  $("vecinos").innerHTML = `<div class="scroll"><table><tr><th>#</th><th>Evento</th><th>Localidad</th><th>Mw</th><th>Prof. (km)</th><th>Dist. (km)</th><th>Intensidad</th><th>Distancia (estandarizada)</th></tr>` +
    d.vecinos.map(v => `<tr><td>${v.n}</td><td>${v.evento_id}</td><td>${v.localidad}</td><td>${v.magnitud}</td><td>${v.profundidad_km}</td><td>${v.distancia_km}</td>
      <td><span class="badge" style="background:${COL[v.intensidad]}">${v.nombre} · ${v.intensidad}</span></td><td>${v.distancia}</td></tr>`).join("") +
    `</table></div><p class="nota">Las variables se estandarizan antes de calcular la distancia euclidiana.</p>`;
  if (!mapa && d.ubicacion) { puntoSim = d.ubicacion; dibujarOffline(); }
  if (mapa && d.ubicacion) {
    if (capaSim) mapa.removeLayer(capaSim);
    capaSim = L.marker([d.ubicacion.lat, d.ubicacion.lon]).addTo(mapa)
      .bindPopup(`<b>${d.ubicacion.localidad}</b><br>Mw ${$("mag").value} · ${$("prof").value} km<br>Intensidad estimada: ${d.nombre}`).openPopup();
  }
}

function cargarEsc(e) {
  $("mag").value = e.magnitud; $("prof").value = e.profundidad_km; $("dist").value = e.distancia_km; estimar();
}
async function pintarEscenarios() {
  escenarios = await get("/api/escenarios");
  $("escBotones").innerHTML = escenarios.map((e, i) => `<button class="btn" onclick="cargarEsc(escenarios[${i}])">${e.nombre}</button>`).join("");
  $("escTabla").innerHTML = `<table><tr><th>Escenario</th><th>Mw</th><th>Prof.</th><th>Dist.</th><th>Resultado</th><th>Votos</th></tr>` +
    escenarios.map(e => `<tr><td>${e.nombre}</td><td>${e.magnitud}</td><td>${e.profundidad_km}</td><td>${e.distancia_km}</td>
      <td><span class="badge" style="background:${COL[e.clase]}">${e.categoria}</span></td><td>${CL.map(c => e.votos[c]).join(" / ")}</td></tr>`).join("") +
    `</table><p class="nota">Votos ordenados como I–IV / V–VI / VII–XII, con K = ${modelo.k_optimo}.</p>`;
}

/* ---------------- Dataset ---------------- */
async function cargarDataset() {
  const q = encodeURIComponent($("q").value), f = encodeURIComponent($("fInt").value);
  const d = await get(`/api/dataset?pagina=${pag}&q=${q}&intensidad=${f}`);
  totalPag = Math.max(Math.ceil(d.total / d.por_pagina), 1);
  $("tabla").innerHTML = `<tr>${d.cols.map(c => `<th>${c}</th>`).join("")}</tr>` +
    d.filas.map(r => `<tr>${r.map((v, i) => i === 8 ? `<td><span class="badge" style="background:${COL[v]}">${v}</span></td>` : `<td>${v}</td>`).join("")}</tr>`).join("");
  $("pagInfo").textContent = `Página ${pag} de ${totalPag} · ${fmt(d.total)} registros`;
}

/* ---------------- Ayuda ---------------- */
const FAQ = [
  ["¿Por qué escalar las variables?", "KNN compara registros con la distancia euclidiana. Sin escalar, la profundidad (0–600 km) y la distancia (0–1500 km) dominarían sobre la magnitud (2–8 Mw) solo por su unidad. StandardScaler lleva cada variable a media 0 y desviación 1, y se ajusta únicamente con el entrenamiento (dentro del Pipeline) para no filtrar información de prueba."],
  ["¿Cómo influye K?", "Con K pequeño el modelo es muy sensible al ruido y a casos aislados (sobreajuste). Con K grande suaviza las fronteras pero puede mezclar clases vecinas (subajuste). Por eso se compararon K = 3, 5, 7 y 9 con validación cruzada agrupada y se eligió el de mayor exactitud media."],
  ["¿Por qué separar por evento?", "Un mismo sismo aparece en varias localidades con idéntica magnitud y profundidad. Si filas del mismo evento caen en entrenamiento y en prueba, el modelo ya 'conoce' el evento y la exactitud se infla (fuga de información). GroupShuffleSplit y GroupKFold por evento_id lo evitan."],
  ["¿Qué diferencia existe entre magnitud e intensidad?", "La magnitud (Mw) mide la energía liberada en la fuente y es única por evento. La intensidad (escala Mercalli modificada, I–XII) describe los efectos observados en un lugar concreto y cambia con la distancia, la profundidad y el tipo de suelo. Este proyecto estima la intensidad a partir de la magnitud, la profundidad y la distancia."],
  ["¿Por qué el modelo no puede anticipar un terremoto?", "Porque solo recibe las características de un evento ya definido (real o hipotético). No recibe señales precursoras ni predice cuándo, dónde o con qué magnitud ocurrirá un sismo, algo que la ciencia actual no puede hacer. Además, este prototipo se entrena con datos simulados y es solo educativo."]
];
$("faq") && ($("faq").className = "faq");
$("faq").innerHTML = FAQ.map(([q, a]) => `<details><summary>${q}</summary><p>${a}</p></details>`).join("");

/* ---------------- Inicio ---------------- */
async function init() {
  [resumen, modelo] = await Promise.all([get("/api/resumen"), get("/api/modelo")]);
  K = modelo.k_optimo; $("kSel").value = K;
  pintarStats(); pintarDonut(); pintarTop(); pintarAnalisis(); pintarModelo();
  initMapa(); pintarEscenarios(); cargarDataset();
  vista("dashboard");
}
document.querySelectorAll("#nav button").forEach(b => b.onclick = () => vista(b.dataset.v));
$("btnEst").onclick = estimar;
$("kSel").onchange = () => { K = +$("kSel").value; pintarModelo(); };
$("q").oninput = () => { pag = 1; cargarDataset(); };
$("fInt").onchange = () => { pag = 1; cargarDataset(); };
$("prev").onclick = () => { if (pag > 1) { pag--; cargarDataset(); } };
$("next").onclick = () => { if (pag < totalPag) { pag++; cargarDataset(); } };
init();
