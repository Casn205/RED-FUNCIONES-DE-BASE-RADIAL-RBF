"""
app.py — Backend Flask + Lógica RBF
Red Neuronal de Funciones de Base Radial
"""

import json
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ════════════════════════════════════════════
# LÓGICA RBF
# ════════════════════════════════════════════

def distancia_euclidiana(x, centro):
    return np.sqrt(np.sum((x - centro) ** 2))

def funcion_activacion(d, tipo):
    if tipo == "logaritmica":
        return 0.0 if d == 0 else (d**2) * np.log(d)
    elif tipo == "gaussiana":
        return np.exp(-(d**2))
    elif tipo == "multicuadrica":
        return np.sqrt(d**2 + 1)
    elif tipo == "inversa":
        return 1.0 / np.sqrt(d**2 + 1)
    return 0.0

def construir_matriz_A(X, centros, tipo_fa):
    m, n = len(X), len(centros)
    A = np.ones((m, n + 1))
    for i in range(m):
        for j in range(n):
            d = distancia_euclidiana(X[i], centros[j])
            A[i, j + 1] = funcion_activacion(d, tipo_fa)
    return A

def simular(X, centros, W, tipo_fa):
    A = construir_matriz_A(X, centros, tipo_fa)
    return A @ W

def clasificar(yr, clases=None):
    """Redondea al entero más cercano y recorta al rango de clases conocidas."""
    yc = np.rint(yr).astype(int)
    if clases is not None:
        mn, mx = min(clases), max(clases)
        yc = np.clip(yc, mn, mx)
    return yc

def error_general(yd, yr):
    return np.sum(np.abs(yd - yr)) / len(yd)

def calcular_metricas(yd, yc):
    clases = sorted(np.unique(np.concatenate([yd, yc])).astype(int).tolist())
    n = len(yd)

    # Matriz de confusión N×N
    cm = [[int(np.sum((yc == pc) & (yd == rc))) for pc in clases] for rc in clases]

    exactitud = float(np.sum(yc == yd) / n)

    # Métricas por clase (macro)
    per_clase = {}
    sens_list, prec_list, f1_list = [], [], []
    for c in clases:
        TP = int(np.sum((yc == c) & (yd == c)))
        FP = int(np.sum((yc == c) & (yd != c)))
        FN = int(np.sum((yc != c) & (yd == c)))
        TN = int(np.sum((yc != c) & (yd != c)))
        sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        f1   = (2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
        per_clase[int(c)] = dict(TP=TP, TN=TN, FP=FP, FN=FN,
                                 sensibilidad=sens, precision=prec, f1=f1)
        sens_list.append(sens); prec_list.append(prec); f1_list.append(f1)

    return dict(
        clases=clases,
        cm=cm,
        exactitud=exactitud,
        sensibilidad=float(np.mean(sens_list)),
        precision=float(np.mean(prec_list)),
        f1=float(np.mean(f1_list)),
        per_clase=per_clase,
        # Compatibilidad legado 2 clases
        TP=per_clase[clases[-1]].get('TP', 0) if len(clases) == 2 else 0,
        TN=per_clase[clases[0]].get('TN', 0)  if len(clases) == 2 else 0,
        FP=per_clase[clases[-1]].get('FP', 0) if len(clases) == 2 else 0,
        FN=per_clase[clases[-1]].get('FN', 0) if len(clases) == 2 else 0,
    )

def estadistica_descriptiva(X, features):
    return [
        dict(nombre=features[i],
             min=float(X[:, i].min()),
             max=float(X[:, i].max()),
             media=float(X[:, i].mean()),
             std=float(X[:, i].std()))
        for i in range(X.shape[1])
    ]

def entrenar_rbf(config, X_train, Y_train):
    """
    Entrena la red RBF aumentando centros hasta converger.
    Devuelve centros, pesos W, historial de EG y número de centros.
    """
    n_entradas    = X_train.shape[1]
    n_centros     = config["n_centros"]
    error_optimo  = config["error_optimo"]
    max_iter      = config["max_iter"]
    tipo_fa       = config["tipo_fa"]
    semilla       = config["semilla"]
    centros_manuales = config.get("centros_manuales")

    np.random.seed(semilla)
    x_min, x_max = X_train.min(), X_train.max()

    historial_eg = []
    historial_nc = []
    centros_final = None
    W_final       = None
    eg_final      = None
    log           = []

    nc = n_centros
    for iteracion in range(1, max_iter + 1):

        # Definir centros
        if centros_manuales and iteracion == 1:
            centros = np.array(centros_manuales, dtype=float)
            nc = len(centros)
            log.append(f"Iter {iteracion} | Usando {nc} centros manuales")
        else:
            centros = np.random.uniform(x_min, x_max, size=(nc, n_entradas))

        # Construir A y resolver W = A\Y
        A = construir_matriz_A(X_train, centros, tipo_fa)
        W, _, _, _ = np.linalg.lstsq(A, Y_train, rcond=None)

        # Calcular error general
        YR = simular(X_train, centros, W, tipo_fa)
        EG = error_general(Y_train, YR)

        historial_eg.append(float(EG))
        historial_nc.append(int(nc))
        centros_final = centros
        W_final       = W
        eg_final      = EG

        converge = EG <= error_optimo
        log.append(
            f"Iter {iteracion:2d} | Centros: {nc:3d} | "
            f"FA: {tipo_fa} | EG = {EG:.4f} | "
            f"{'✓ CONVERGE' if converge else '✗ no converge'}"
        )

        if converge:
            break

        nc += 2

    return centros_final, W_final, eg_final, historial_eg, historial_nc, log

# ════════════════════════════════════════════
# RUTAS FLASK
# ════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/cargar", methods=["POST"])
def cargar_dataset():
    """Recibe el JSON del dataset y devuelve info básica."""
    try:
        archivo = request.files["archivo"]
        raw = json.load(archivo)

        X = np.array([d["input"]  for d in raw["data"]], dtype=float)
        Y = np.array([d["output"] for d in raw["data"]], dtype=float)

        clases, conteos = np.unique(Y.astype(int), return_counts=True)

        return jsonify({
            "ok": True,
            "dataset":   raw.get("dataset", "sin nombre"),
            "patrones":  len(X),
            "features":  raw.get("features", [f"x{i+1}" for i in range(X.shape[1])]),
            "n_entradas": int(X.shape[1]),
            "clases":    clases.tolist(),
            "conteos":   conteos.tolist(),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/entrenar", methods=["POST"])
def entrenar():
    """Recibe configuración, entrena la red y devuelve todos los resultados."""
    try:
        # ── Leer dataset y configuración
        archivo = request.files["archivo"]
        config  = json.loads(request.form["config"])
        raw     = json.load(archivo)

        X = np.array([d["input"]  for d in raw["data"]], dtype=float)
        Y = np.array([d["output"] for d in raw["data"]], dtype=float)
        features = raw.get("features", [f"x{i+1}" for i in range(X.shape[1])])

        # ── Partición  (sin errores de redondeo flotante)
        n         = len(X)
        train_pct = config["train_pct"]

        np.random.seed(config["semilla"])
        idx     = np.random.permutation(n)
        n_train = round(n * train_pct)           # round en lugar de int
        n_resto = n - n_train                    # lo que queda (val + test)
        n_val   = n_resto // 2                   # mitad exacta del resto
        n_test  = n_resto - n_val                # el resto restante (siempre cierra a n)

        X_train = X[idx[:n_train]]
        Y_train = Y[idx[:n_train]]
        X_test  = X[idx[n_train + n_val:]]
        Y_test  = Y[idx[n_train + n_val:]]

        # ── Estadística descriptiva (sobre todos los datos)
        stats = estadistica_descriptiva(X, features)

        # ── Entrenar
        centros, W, eg_train, hist_eg, hist_nc, log_lines = \
            entrenar_rbf(config, X_train, Y_train)

        # ── Simular con datos de prueba
        clases_unicas = sorted(np.unique(Y.astype(int)).tolist())
        YR_test = simular(X_test, centros, W, config["tipo_fa"])
        YC_test = clasificar(YR_test, clases_unicas)
        eg_test = float(error_general(Y_test, YR_test))

        # ── Métricas
        metricas = calcular_metricas(Y_test.astype(int), YC_test)

        # ── Preparar respuesta
        return jsonify({
            "ok": True,

            # Consola
            "log": log_lines,

            # Métricas resumen
            "eg_train":    float(eg_train),
            "eg_test":     eg_test,
            "convergio":   bool(eg_train <= config["error_optimo"]),
            "metricas":    metricas,

            # Historial de entrenamiento
            "historial_eg": hist_eg,
            "historial_nc": hist_nc,

            # Conteos reales sobre TODO el conjunto de prueba (para gráfico de clases)
            "conteos_test": {
                "yd": [int(np.sum(Y_test.astype(int) == c)) for c in clases_unicas],
                "yc": [int(np.sum(YC_test == c))            for c in clases_unicas],
            },

            # Simulación (primeros 30 patrones de prueba, para la tabla)
            "simulacion": {
                "xd": X_test[:30].tolist(),
                "yd": Y_test[:30].tolist(),
                "yr": YR_test[:30].tolist(),
                "yc": YC_test[:30].tolist(),
            },

            # Estadística descriptiva
            "estadistica": stats,

            # Arquitectura final de la red
            "red": {
                "n_entradas":  int(X.shape[1]),
                "n_ocultas":   len(centros),
                "tipo_fa":     config["tipo_fa"],
                "centros":     centros.tolist(),
                "pesos":       W.tolist(),
                "features":    features,
            },

            # Partición usada
            "particion": {
                "n_train": n_train,
                "n_val":   n_val,
                "n_test":  n_test,
            },
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)