"""
Sistema Predictivo de Apoyo a la Toma de Decisiones Editoriales, NewsHub MX.
Se carga el modelo ya entrenado en notebook_analisis.ipynb (modelo_riesgo.joblib).
No se reentrena nada en este archivo.
"""

import datetime as dt
import html as html_lib
import json
import os
import re
import time
from pathlib import Path

import feedparser
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "modelo_riesgo.joblib"
HIST_PATH = BASE_DIR / "datos" / "historico_categoria.csv"
HORA_PATH = BASE_DIR / "datos" / "mejor_hora_categoria.json"
AUTORES_PATH = BASE_DIR / "datos" / "autores.json"
RSS_LOCAL_PATH = BASE_DIR / "rss" / "noticias.xml"
LOGO_PATH = BASE_DIR / "logo" / "newshub.png"

FEATURES_NUMERICAS = [
    "caracteres_titulo", "palabras_titulo", "palabras_cuerpo", "parrafos_cuerpo",
    "longitud_promedio_parrafo", "h1", "h2", "link_build", "imagen", "video",
    "twitter", "facebook", "hora", "Audio",
]
FEATURES_CATEGORICAS = ["categoria", "nombre", "dia_semana", "mes", "tipo_contenido", "horario"]
FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS

# Se usa 0.40 porque en un sistema de alerta un falso negativo pesa más que un
# falso positivo: no avisar de una nota que sí va a fallar cuesta más que revisar
# de más. Con este umbral se obtiene recall de 0.68 y precision de 0.56 (ver
# notebook, sección 8.6). Es la frontera única para la clasificación binaria, el
# badge de riesgo y la confianza que se muestran en el dashboard.
UMBRAL_DECISION = 0.40

# Distancia mínima a UMBRAL_DECISION (ver calcular_confianza) para considerar que
# el modelo dejó de adivinar.
CORTE_CONFIANZA_MEDIO = 0.34


# Fronteras del gauge: se pinta rojo/amarillo/verde exactamente donde el badge de
# riesgo dice Alto/Medio/Bajo.
GAUGE_CORTE_ROJO = round((1 - (UMBRAL_DECISION + (1 - UMBRAL_DECISION) * CORTE_CONFIANZA_MEDIO)) * 100, 1)
GAUGE_CORTE_VERDE = round((1 - UMBRAL_DECISION * (1 - CORTE_CONFIANZA_MEDIO)) * 100, 1)

FACTOR_TEMPLATES = {
    "parrafos_cuerpo": {
        "riesgo": "Muy pocos párrafos; divide el texto en más bloques.",
        "protector": "La cantidad de párrafos es adecuada.",
    },
}

# Frases naturales para los factores categóricos: categoría, día, mes, tipo de
# contenido, horario.
SUJETOS_CATEGORICOS = {
    "categoria": lambda v: f'Las notas de la categoría "{v}"',
    "dia_semana": lambda v: f'Las notas publicadas en {v}',
    "mes": lambda v: f'Las notas publicadas en {v}',
    "tipo_contenido": lambda v: f'Las notas de tipo "{v}"',
    "horario": lambda v: f'Las notas publicadas en horario de {v}',
}


DIAS_ES = {
    "Monday": "lunes", "Tuesday": "martes", "Wednesday": "miércoles",
    "Thursday": "jueves", "Friday": "viernes", "Saturday": "sábado", "Sunday": "domingo",
}

st.set_page_config(page_title="NewsHub MX — Asistente Predictivo", page_icon="📰", layout="wide")


# ---------- Carga de artefactos (con caché, no se reentrena nada) ----------

@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)


@st.cache_data
def cargar_historico():
    return pd.read_csv(HIST_PATH)


@st.cache_data
def cargar_mejor_hora():
    with open(HORA_PATH, encoding="utf-8") as f:
        datos = json.load(f)
    return {k: int(v) for k, v in datos.items()}


@st.cache_data
def cargar_autores():
    with open(AUTORES_PATH, encoding="utf-8") as f:
        return json.load(f)


modelo = cargar_modelo()
tabla_historica = cargar_historico()
mejor_hora_categoria = cargar_mejor_hora()
CATEGORIAS = sorted(tabla_historica["categoria"].unique().tolist())
AUTORES = cargar_autores()

preprocesador = modelo.named_steps["preprocesador"]
modelo_xgb = modelo.named_steps["modelo"]
booster = modelo_xgb.get_booster()
nombres_features = preprocesador.get_feature_names_out()


def cargar_rss(url):
    """Trae las notas de un feed RSS para elegir una y analizarla.
    Si no hay URL o la URL falla, se usa el archivo local de ejemplo.

    Devuelve (notas, uso_fallback). uso_fallback en True indica que se cargó el
    feed de ejemplo en vez del que pidió el editor.
    """
    uso_fallback = not (url and url.strip())
    fuente = url.strip() if not uso_fallback else str(RSS_LOCAL_PATH)
    feed = feedparser.parse(fuente)
    if not feed.entries:
        feed = feedparser.parse(str(RSS_LOCAL_PATH))
        uso_fallback = True

    notas = []
    for entrada in feed.entries:
        categoria_rss = entrada.tags[0].term if entrada.get("tags") else None
        autor_rss = entrada.get("author")
        tiene_audio = any(m.get("medium") == "audio" for m in entrada.get("media_content", []))

        imagenes = re.findall(r'<img[^>]+src="([^"]+)"', entrada.get("description", ""))
        fecha_parseada = entrada.get("published_parsed")
        fecha_corta = time.strftime("%d/%m %H:%M", fecha_parseada) if fecha_parseada else entrada.get("published", "")

        notas.append({
            "titulo": entrada.get("title", ""),
            # Si la categoría o el autor no están en las listas conocidas, quedan en
            # None: se prefiere que el editor los elija a mano antes que mandar al
            # modelo un valor que no vio en entrenamiento.
            "categoria": categoria_rss if categoria_rss in CATEGORIAS else None,
            "autor": autor_rss if autor_rss in AUTORES else None,
            "fecha": entrada.get("published", ""),
            "fecha_corta": fecha_corta,
            "contenido": entrada.get("description", ""),
            "imagen_portada": imagenes[0] if imagenes else None,
            "audio": tiene_audio,
        })
    return notas, uso_fallback


# ---------- Lógica de negocio (misma que se valida en el notebook) ----------

def referencia_contenido(categoria):
    """Se calcula el punto de referencia de 'cuánto texto es óptimo': la mediana
    histórica de palabras_cuerpo, longitud_promedio_parrafo, caracteres_titulo y
    palabras_titulo entre las notas de esta categoría con buen desempeño
    (bajo_desempeno=0). Los datos se toman de historico_categoria.csv. Si la
    categoría no tiene datos, se usa el promedio global."""
    fila = tabla_historica[(tabla_historica["categoria"] == categoria) & (tabla_historica["bajo_desempeno"] == 0)]
    if fila.empty:
        fila = tabla_historica[tabla_historica["bajo_desempeno"] == 0]
    return {
        "palabras": round(fila["palabras_mediana"].mean()),
        "longitud_parrafo": round(fila["longitud_parrafo_mediana"].mean()),
        "caracteres_titulo": round(fila["caracteres_titulo_mediana"].mean()),
        "palabras_titulo": round(fila["palabras_titulo_mediana"].mean()),
    }


def clasificar_horario(hora):
    if 6 <= hora <= 11:
        return "Mañana"
    elif 12 <= hora <= 18:
        return "Tarde"
    return "Noche"


def quitar_embeds(texto_html: str) -> str:
    """Se quita el contenido de <blockquote> (incrustaciones de Twitter/X) antes de
    contar palabras y párrafos, porque ese texto es el tuit real, no redacción del
    editor. Se usa la misma lógica que quitar_embeds() en notebook_analisis.ipynb.
    No se filtra <iframe> (Facebook, YouTube) porque casi nunca trae texto adentro."""
    return re.sub(r"<blockquote.*?</blockquote>", " ", str(texto_html), flags=re.DOTALL | re.IGNORECASE)


def contar_palabras_cuerpo(texto_html: str) -> int:
    """Se cuentan las palabras reales del cuerpo, excluyendo tuits incrustados
    (quitar_embeds). Se usa la misma lógica que contar_palabras() en
    notebook_analisis.ipynb, con la que se entrenó modelo_riesgo.joblib."""
    texto = quitar_embeds(texto_html)
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = html_lib.unescape(texto)
    return len(texto.split())


def contar_parrafos_reales(html_texto: str) -> int:
    """Se cuentan los <p> que tienen texto real adentro, ignorando los vacíos
    (<p>&nbsp;</p>) que dejan pegados algunos widgets. También se excluyen tuits
    incrustados (quitar_embeds), para no contar el <p> del tuit como párrafo de la
    nota. Se usa la misma lógica que contar_parrafos_reales() en
    notebook_analisis.ipynb, con la que se entrenó modelo_riesgo.joblib."""
    texto = quitar_embeds(html_texto)
    bloques = re.findall(r"<p[^>]*>(.*?)</p>", texto, flags=re.DOTALL | re.IGNORECASE)
    contador = 0
    for bloque in bloques:
        texto_plano_bloque = re.sub(r"<[^>]+>", " ", bloque)
        texto_plano_bloque = html_lib.unescape(texto_plano_bloque)
        if texto_plano_bloque.strip():
            contador += 1
    return contador


def calcular_features_texto(titulo: str, cuerpo: str) -> dict:
    caracteres_titulo = len(titulo)
    palabras_titulo = len(titulo.split())

    # El cuerpo puede venir en HTML o en texto plano. contar_palabras_cuerpo() quita
    # etiquetas, decodifica entidades y excluye el texto de tuits incrustados
    # (quitar_embeds).
    palabras_cuerpo = contar_palabras_cuerpo(cuerpo)

    # Se cuentan párrafos con contar_parrafos_reales(), que ignora los <p> vacíos y
    # el <p> propio de tuits incrustados.
    parrafos_cuerpo = contar_parrafos_reales(cuerpo)

    longitud_promedio_parrafo = (
        palabras_cuerpo / parrafos_cuerpo if parrafos_cuerpo > 0 else palabras_cuerpo
    )

    return {
        "caracteres_titulo": caracteres_titulo,
        "palabras_titulo": palabras_titulo,
        "palabras_cuerpo": palabras_cuerpo,
        "parrafos_cuerpo": parrafos_cuerpo,
        "longitud_promedio_parrafo": longitud_promedio_parrafo,
    }


def _contar_embeds(cuerpo: str, patron_dominio: str) -> int:
    """Se cuentan elementos <iframe> o <blockquote> cuya etiqueta de apertura
    coincide con el patrón de dominio dado, para no contar de más cuando el dominio
    aparece repetido dentro de un href codificado en la URL del embed."""
    tags = re.findall(r"<iframe\b[^>]*>", cuerpo, flags=re.IGNORECASE)
    tags += re.findall(r"<blockquote\b[^>]*>", cuerpo, flags=re.IGNORECASE)
    return sum(1 for tag in tags if re.search(patron_dominio, tag, flags=re.IGNORECASE))


def analizar_html_contenido(cuerpo: str) -> dict:
    """Se detecta en el HTML pegado: encabezados, links, imágenes, embeds de
    Twitter y Facebook (iframes/blockquotes), y video de YouTube. Son conteos
    aproximados por patrones de texto, no un parser HTML completo, pero alcanzan
    para las variables que necesita el modelo."""
    h1_n = len(re.findall(r"<h1[\s>]", cuerpo, flags=re.IGNORECASE))
    h2_n = len(re.findall(r"<h2[\s>]", cuerpo, flags=re.IGNORECASE))
    link_n = len(re.findall(r"<a\s+[^>]*href\s*=", cuerpo, flags=re.IGNORECASE))
    imagen_n = len(re.findall(r"<img[\s>]", cuerpo, flags=re.IGNORECASE))
    twitter_n = _contar_embeds(cuerpo, r"twitter\.com|x\.com|twitter-tweet")
    facebook_n = _contar_embeds(cuerpo, r"facebook\.com|fb-post|fb-video")
    tiene_youtube = bool(re.search(r"youtube\.com|youtu\.be", cuerpo, flags=re.IGNORECASE))

    return {
        "h1": h1_n, "h2": h2_n, "link_build": link_n, "imagen": imagen_n,
        "twitter": twitter_n, "facebook": facebook_n, "video": 1 if tiene_youtube else 0,
    }


def formatear_embeds_para_lectura(html_contenido: str) -> str:
    """Streamlit no ejecuta el widget real de Twitter/Facebook. Se reemplaza el
    <blockquote>/<div> crudo por una tarjeta simple que avisa que ahí había un
    embed, con link a la publicación original cuando se puede sacar del HTML."""

    def _tarjeta_twitter(match):
        bloque = match.group(0)
        enlaces = re.findall(r'href="([^"]+)"', bloque)
        url_status = next((u for u in enlaces if "status" in u), (enlaces[0] if enlaces else "#"))
        return f'<div class="embed-card">🐦 <b>Publicación de Twitter/X incrustada</b> — <a href="{url_status}" target="_blank">ver original</a></div>'

    html_contenido = re.sub(
        r'<blockquote[^>]*class="[^"]*twitter-tweet[^"]*".*?</blockquote>',
        _tarjeta_twitter, html_contenido, flags=re.IGNORECASE | re.DOTALL,
    )
    html_contenido = re.sub(
        r'<(div|iframe)[^>]*(?:class="[^"]*fb-(?:post|video)[^"]*"|src="[^"]*facebook\.com[^"]*")[^>]*>.*?</\1>',
        '<div class="embed-card">📘 <b>Publicación de Facebook incrustada</b></div>',
        html_contenido, flags=re.IGNORECASE | re.DOTALL,
    )
    return html_contenido


def _texto_presencia(riesgo_ausente, riesgo_presente, protector_presente, protector_ausente, valor, es_riesgo):
    """Se redacta el texto para features de presencia/conteo (imagen, video,
    twitter, facebook, Audio, link_build). El signo de SHAP (es_riesgo) por sí solo
    no dice si el elemento está presente en la nota, así que se revisa el valor
    real antes de describirlo."""
    presente = bool(valor) and valor > 0
    if es_riesgo:
        return riesgo_presente if presente else riesgo_ausente
    return protector_presente if presente else protector_ausente


def factores_de_riesgo(shap_fila, nombres_cols, categoria, hora, feats_texto, valores_detectados, top_n=5):
    ref = referencia_contenido(categoria)
    mejor_hora = mejor_hora_categoria.get(categoria)

    # Se usa el ranking completo (con autor incluido) solo para saber si el autor
    # pesa lo suficiente como para mencionarlo. Nunca se muestra su
    # valor, por diseño ético del proyecto.
    pares_todos = sorted(zip(nombres_cols, shap_fila), key=lambda t: abs(t[1]), reverse=True)
    autor_es_relevante = any(
        nombre.startswith("cat__nombre_") and abs(valor) > 1e-6
        for nombre, valor in pares_todos[:top_n]
    )

    pares = [
        (nombre, valor) for nombre, valor in zip(nombres_cols, shap_fila)
        if not nombre.startswith("cat__nombre_")
    ]
    pares.sort(key=lambda t: abs(t[1]), reverse=True)

    factores = []
    for nombre_col, shap_val in pares[:top_n]:
        if abs(shap_val) < 1e-6:
            continue
        es_riesgo = shap_val > 0

        if nombre_col.startswith("num__"):
            base = nombre_col.replace("num__", "")
            # Para palabras_cuerpo y longitud_promedio_parrafo se usa el número real
            # de referencia de la categoría en vez de una plantilla fija, ligado
            # directamente al factor que detectó el modelo.
            if base == "palabras_cuerpo":
                # RIESGO no siempre significa "muy largo": un cuerpo demasiado corto
                # también puede ser riesgoso. Se compara el valor real contra el
                # punto de referencia en vez de asumir la dirección.
                actual = feats_texto["palabras_cuerpo"]
                cualitativo = "más largo" if actual > ref["palabras"] else "más corto"
                if es_riesgo:
                    texto = f"El cuerpo tiene {actual} palabras, {cualitativo} de lo recomendado; en {categoria}, las notas con buen desempeño rondan {ref['palabras']} palabras."
                else:
                    texto = f"La extensión del cuerpo es adecuada ({actual} palabras; las notas con buen desempeño en {categoria} rondan {ref['palabras']} palabras)."
            elif base == "caracteres_titulo":
                # RIESGO no siempre significa "título largo": puede ser al revés.
                # La longitud del título casi no correlaciona con el desempeño
                # (r=-0.058), así que se avisa en el texto para no sonar más seguro
                # de lo que el modelo está.
                actual = feats_texto["caracteres_titulo"]
                cualitativo = "más largo" if actual > ref["caracteres_titulo"] else "más corto"
                if es_riesgo:
                    texto = f"El título tiene {actual} caracteres, {cualitativo} de lo habitual; en {categoria}, los títulos con buen desempeño rondan {ref['caracteres_titulo']} caracteres (señal débil: la longitud del título influye poco, pesa más de qué habla)."
                else:
                    texto = f"La longitud del título es adecuada ({actual} caracteres, cerca de los {ref['caracteres_titulo']} habituales en {categoria})."
            elif base == "palabras_titulo":
                actual = feats_texto["palabras_titulo"]
                cualitativo = "más palabras" if actual > ref["palabras_titulo"] else "menos palabras"
                if es_riesgo:
                    texto = f"El título tiene {actual} palabras, {cualitativo} de lo habitual; en {categoria}, los títulos con buen desempeño rondan {ref['palabras_titulo']} palabras (señal débil)."
                else:
                    texto = f"El número de palabras del título es adecuado ({actual}, cerca de las {ref['palabras_titulo']} habituales en {categoria})."
            elif base == "longitud_promedio_parrafo":
                actual = round(feats_texto["longitud_promedio_parrafo"])
                cualitativo = "largos" if actual > ref["longitud_parrafo"] else "cortos"
                if es_riesgo:
                    texto = f"Los párrafos son {cualitativo} en promedio (~{actual} palabras); en {categoria}, las notas con buen desempeño usan párrafos de ~{ref['longitud_parrafo']} palabras."
                else:
                    texto = f"La longitud de los párrafos es adecuada (~{actual} palabras, como las notas con buen desempeño en {categoria}, ~{ref['longitud_parrafo']} palabras)."
            elif base == "hora":
                # Se dice la hora real elegida y se compara contra la mejor hora
                # histórica de la categoría.
                if mejor_hora is not None:
                    if es_riesgo:
                        texto = f"Publicar a las {hora:02d}:00 está asociado a menor desempeño en {categoria}; el mejor horario histórico es las {mejor_hora:02d}:00."
                    elif hora == mejor_hora:
                        texto = f"Las {hora:02d}:00 es la mejor hora histórica para publicar en {categoria}."
                    else:
                        texto = f"Publicar a las {hora:02d}:00 no penaliza el desempeño en {categoria} (mejor hora histórica: {mejor_hora:02d}:00)."
                else:
                    texto = (
                        f"Publicar a las {hora:02d}:00 está asociado a menor desempeño para este tipo de nota; considera otro horario."
                        if es_riesgo else
                        f"Publicar a las {hora:02d}:00 no está penalizando el desempeño de esta nota en particular."
                    )
            elif base == "twitter":
                texto = _texto_presencia(
                    "No se detectó una incrustación de Twitter/X en el contenido; agrégala si viene al caso.",
                    "Se detectó una incrustación de Twitter/X, pero no está asociada a mejor desempeño en este caso (señal débil).",
                    "Buena incrustación de Twitter/X en el contenido.",
                    "No tener una incrustación de Twitter/X no está penalizando el desempeño de esta nota en particular (señal débil).",
                    valores_detectados.get("twitter"), es_riesgo,
                )
            elif base == "facebook":
                texto = _texto_presencia(
                    "No se detectó una incrustación de Facebook en el contenido; agrégala si viene al caso.",
                    "Se detectó una incrustación de Facebook, pero no está asociada a mejor desempeño en este caso (señal débil).",
                    "Buena incrustación de Facebook en el contenido.",
                    "No tener una incrustación de Facebook no está penalizando el desempeño de esta nota en particular (señal débil).",
                    valores_detectados.get("facebook"), es_riesgo,
                )
            elif base == "video":
                texto = _texto_presencia(
                    "No se detectó video en el contenido (señal débil: no siempre es necesario, revisa caso por caso).",
                    "Se detectó video en el contenido, pero no está asociado a mejor desempeño en esta categoría (señal débil, revisar caso por caso).",
                    "La incrustación de video parece favorable aquí (señal débil).",
                    "No tener video no está penalizando el desempeño de esta nota en particular (señal débil).",
                    valores_detectados.get("video"), es_riesgo,
                )
            elif base == "Audio":
                texto = _texto_presencia(
                    "No incluir audio está asociado a menor desempeño en este caso (señal débil, los datos históricos no muestran una relación consistente).",
                    "El audio no está asociado a mejor desempeño en los datos históricos (señal débil).",
                    "El uso de audio parece favorable aquí (señal débil).",
                    "No incluir audio no está penalizando el desempeño de esta nota en particular (señal débil).",
                    valores_detectados.get("Audio"), es_riesgo,
                )
            elif base == "imagen":
                texto = _texto_presencia(
                    "Pocas o ninguna imagen; agrega al menos una.",
                    "Ya tiene imagen, pero el modelo no lo está favoreciendo en este caso (señal débil); revisa la calidad o cantidad.",
                    "Tiene suficientes imágenes.",
                    "No tener imagen no está penalizando el desempeño de esta nota en particular (señal débil).",
                    valores_detectados.get("imagen"), es_riesgo,
                )
            elif base == "link_build":
                texto = _texto_presencia(
                    "Falta link-building (enlaces internos/externos); agrega algunos.",
                    "Ya tiene enlaces, pero el modelo no lo está favoreciendo en este caso (señal débil).",
                    "El link-building es adecuado.",
                    "No tener enlaces no está penalizando el desempeño de esta nota en particular (señal débil).",
                    valores_detectados.get("link_build"), es_riesgo,
                )
            else:
                plantilla = FACTOR_TEMPLATES.get(base)
                if plantilla is None:
                    continue
                texto = plantilla["riesgo"] if es_riesgo else plantilla["protector"]
        elif nombre_col.startswith("cat__"):
            sin_prefijo = nombre_col.replace("cat__", "")
            # No se puede cortar en el primer "_": "dia_semana" y "tipo_contenido" ya
            # traen guion bajo en el nombre. Se busca cuál de las variables
            # categóricas conocidas es el prefijo real.
            base = next((c for c in FEATURES_CATEGORICAS if sin_prefijo.startswith(c + "_")), None)
            if base is None:
                continue
            valor_categoria = sin_prefijo[len(base) + 1:]
            if base == "dia_semana":
                valor_categoria = DIAS_ES.get(valor_categoria, valor_categoria)
            sujeto = SUJETOS_CATEGORICOS.get(base, lambda v: f'Las notas con "{base}" = "{v}"')(valor_categoria)
            # Se usa marco positivo, igual que el resto del dashboard: se dice
            # directamente si sube o baja la posibilidad de éxito, sin doble
            # negativo.
            texto = f'{sujeto} tienden a tener {"menor" if es_riesgo else "mayor"} posibilidad de éxito que el resto de las notas.'
        else:
            continue

        factores.append({"variable": nombre_col, "shap": float(shap_val), "es_riesgo": es_riesgo, "texto": texto})

    # Se calcula el "peso" relativo de cada factor (Alto/Medio/Bajo) según su SHAP
    # frente al más fuerte de los mostrados, para que se vea que no todos pesan
    # igual.
    valores_abs = [abs(f["shap"]) for f in factores]
    max_shap = max(valores_abs) if valores_abs else 0
    for f in factores:
        if max_shap <= 0:
            f["peso"] = "Bajo"
            continue
        ratio = abs(f["shap"]) / max_shap
        f["peso"] = "Alto" if ratio >= 0.66 else "Medio" if ratio >= 0.33 else "Bajo"

    if autor_es_relevante:
        # Se menciona al autor de forma agregada, sin dirección ni valor, para no
        # señalar a ningún periodista en particular.
        factores.append({
            "variable": "cat__nombre_agregado",
            "shap": None,
            "es_riesgo": None,
            "texto": "El autor es uno de los factores de mayor peso en esta predicción, pero no se muestra el detalle por diseño ético del proyecto.",
            "peso": "Alto",
        })

    return factores


def analizar_nota(
    titulo, cuerpo, categoria, nombre, fecha_publicacion,
    imagen=0, video=0, twitter=0, facebook=0, audio=0, h1=0, h2=0, link_build=0,
):
    feats_texto = calcular_features_texto(titulo, cuerpo)

    hora = fecha_publicacion.hour
    dia_semana = fecha_publicacion.strftime("%A")
    mes = fecha_publicacion.strftime("%Y-%m")
    tiene_multimedia = (imagen > 0) or (video > 0) or (audio == 1)
    tipo_contenido = "Multimedia" if tiene_multimedia else "Texto"
    horario = clasificar_horario(hora)

    fila = pd.DataFrame([{
        **feats_texto,
        "h1": h1, "h2": h2, "link_build": link_build,
        "imagen": imagen, "video": video, "twitter": twitter, "facebook": facebook,
        "hora": hora, "Audio": audio,
        "categoria": categoria, "nombre": nombre, "dia_semana": dia_semana,
        "mes": mes, "tipo_contenido": tipo_contenido, "horario": horario,
    }])[FEATURES]

    probabilidad_riesgo = float(modelo.predict_proba(fila)[0, 1])
    es_bajo_desempeno = probabilidad_riesgo >= UMBRAL_DECISION

    fila_transformada = preprocesador.transform(fila)
    if hasattr(fila_transformada, "toarray"):
        fila_transformada = fila_transformada.toarray()
    dmatrix_fila = xgb.DMatrix(fila_transformada, feature_names=list(nombres_features))
    shap_fila = booster.predict(dmatrix_fila, pred_contribs=True)[0, :-1]

    valores_detectados = {
        "imagen": imagen, "video": video, "twitter": twitter, "facebook": facebook,
        "Audio": audio, "link_build": link_build,
    }
    factores = factores_de_riesgo(shap_fila, nombres_features, categoria, hora, feats_texto, valores_detectados, top_n=5)

    # Se calcula el inverso directo del riesgo ((1 - riesgo) * 100), para que el
    # número sea legible sin traducción mental: más alto siempre es mejor.
    probabilidad_buen_desempeno = round((1 - probabilidad_riesgo) * 100)

    return {
        "riesgo_bajo_desempeno": es_bajo_desempeno,
        "probabilidad_riesgo": round(probabilidad_riesgo, 3),
        "probabilidad_buen_desempeno": probabilidad_buen_desempeno,
        "factores_riesgo": factores,
    }


def generar_recomendaciones_fallback(resultado, titulo):
    # Sin LLM no se redacta nada nuevo: se reusa el texto que SHAP generó para cada
    # factor de riesgo, y el "porque" se basa en qué tan arriba salió ese factor en
    # el ranking real del modelo.
    riesgos = [f for f in resultado["factores_riesgo"] if f["es_riesgo"]][:4]
    peso_por_posicion = ["el factor de mayor peso que detectó el modelo", "un factor de peso medio", "un factor de menor peso"]
    acciones = [
        {
            "recomendacion": f["texto"],
            "porque": f"Es {peso_por_posicion[min(i, len(peso_por_posicion) - 1)]} para esta nota (según SHAP).",
        }
        for i, f in enumerate(riesgos)
    ]

   

    sugerencias = [{"titulo": titulo, "tip": "Título original, sin cambios."}]

    palabras = titulo.split()
    if len(titulo) > 60 and len(palabras) > 8:
        sugerencias.append({
            "titulo": " ".join(palabras[:8]) + "...",
            "tip": "Más corto y directo: mejor para vistas previas en buscadores y redes.",
        })
    else:
        sugerencias.append({
            "titulo": titulo.rstrip(".") + ": lo que debes saber",
            "tip": "Agrega un gancho de curiosidad sin perder la palabra clave principal.",
        })

    if not titulo.strip().endswith("?"):
        sugerencias.append({
            "titulo": "¿" + titulo[0].lower() + titulo[1:].rstrip(".") + "?",
            "tip": "Formato de pregunta: genera curiosidad y suele funcionar bien en redes.",
        })
    else:
        sugerencias.append({
            "titulo": titulo.rstrip("?") + " hoy",
            "tip": "Añade actualidad con 'hoy', útil para notas de coyuntura.",
        })

    return {
        "fuente": "reglas (fallback)",
        "acciones_recomendadas": acciones,
        "sugerencias_titulos": sugerencias[:3],
    }


def _get_secret(nombre):
    valor = os.environ.get(nombre)
    if valor:
        return valor
    try:
        return st.secrets[nombre]
    except Exception:
        return None


def _cliente_llm():
    from openai import OpenAI

    groq_key = _get_secret("GROQ_API_KEY")
    if groq_key:
        return OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1", timeout=10), "llama-3.3-70b-versatile"

    raise RuntimeError("No hay GROQ_API_KEY configurada.")


def generar_recomendaciones_llm(resultado, titulo, cuerpo):
    cliente, modelo_llm = _cliente_llm()

    # Se le pasa al LLM solo los factores que son riesgo, ordenados por SHAP de
    # mayor a menor peso. El orden lo decide el modelo, no el LLM, así el badge
    # Alto/Medio/Bajo queda ligado al peso real que calculó XGBoost.
    riesgos = [f for f in resultado["factores_riesgo"] if f["es_riesgo"]][:4]
    factores_texto = "\n".join(f"{i}. {f['texto']}" for i, f in enumerate(riesgos, start=1))
    resumen_cuerpo = " ".join(cuerpo.split()[:40])

    prompt = f"""Eres un asistente editorial de un medio de noticias. Con base en los siguientes
factores de riesgo detectados por un modelo (ya vienen ordenados de mayor a menor peso real
según SHAP -- no los reordenes, no los cuestiones, no agregues factores que no estén en la
lista), escribe una recomendación por cada uno, y 3 títulos alternativos para la misma nota.

Factores de riesgo (de mayor a menor peso):
{factores_texto}

Título actual: {titulo}
Inicio del cuerpo: {resumen_cuerpo}

Para cada factor de riesgo escribe un objeto con:
- "recomendacion": una acción concreta que el editor puede hacer en ESTA nota antes de publicar
  (título, redacción, longitud, párrafos, imágenes, video, enlaces).
- "porque": una frase corta (máximo ~15 palabras) explicando el motivo, basada en el factor
  que se te dio -- no inventes datos que no estén ahí.
Cuando el factor hable de Twitter, Facebook o video, llámalos "incrustación" o "incrustaciones"
(son elementos embebidos en el HTML de la nota, no promoción en redes sociales).
NUNCA sugieras cambiar, reescribir ni quitar la categoría, el autor ni la fecha — esas son
señales de contexto, no decisiones editoriales prácticas. La categoría es un campo fijo del
sistema (un valor de un menú), NO es texto de la nota: si el nombre de la categoría (ej.
"Boca del Río") también aparece como palabra dentro del título o el cuerpo, IGNÓRALO por
completo -- nunca sugieras reemplazar, reescribir o quitar esa palabra del texto.

Para "sugerencias_titulos": ajusta el título pensando en SEO actual (palabras clave relevantes
al inicio, claridad sobre qué trata la nota, longitud razonable para buscadores) y agrégale
un toque de clickbait (curiosidad, urgencia o un gancho emocional) sin caer en engaño ni
exagerar algo que la nota no dice — el titular debe seguir siendo honesto sobre el contenido.
Para cada título, agrega un "tip" de una sola frase corta (máximo ~12 palabras) explicando
en términos de SEO/clics por qué ESE título específico es mejor (ej. "palabra clave al inicio",
"genera curiosidad con una pregunta", "más corto para vista previa en buscadores").

Ordena "sugerencias_titulos" de mayor a menor potencial de clics (el primero es el mejor).
"acciones_recomendadas" debe tener EXACTAMENTE un objeto por cada factor de riesgo listado
arriba, en el mismo orden -- si la lista de factores viene vacía, respóndelo como lista vacía.

Después de esos, agrega 1 o 2 objetos más con recomendaciones de SEO generales para esta nota
(no atadas a ningún factor de riesgo específico): uso de la palabra clave principal en el primer
párrafo, longitud/claridad del primer párrafo como "meta descripción" implícita, uso de subtítulos
o enlaces internos si aplica, texto alternativo de imágenes. Mismas reglas que arriba: solo sobre
título/redacción/estructura de ESTA nota, nunca sobre categoría, autor o fecha.

Responde SOLO con un JSON con esta forma exacta:
{{"acciones_recomendadas": [{{"recomendacion": "...", "porque": "..."}}], "sugerencias_titulos": [{{"titulo": "...", "tip": "..."}}, {{"titulo": "...", "tip": "..."}}, {{"titulo": "...", "tip": "..."}}]}}"""

    respuesta = cliente.chat.completions.create(
        model=modelo_llm,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    contenido = json.loads(respuesta.choices[0].message.content)

    # El prompt le pide al LLM que nunca sugiera cambiar la categoría, pero no
    # siempre lo respeta. Si de todos modos la menciona, esa acción se sustituye
    # por el texto real del factor, sin quitarla de la lista, para no desalinear
    # el badge Alto/Medio/Bajo (depende de la posición).
    acciones = contenido.get("acciones_recomendadas", [])
    for i, accion in enumerate(acciones):
        texto_combinado = f"{accion.get('recomendacion', '')} {accion.get('porque', '')}".lower()
        if "categoría" in texto_combinado or "categoria" in texto_combinado:
            if i < len(riesgos):
                acciones[i] = {
                    "recomendacion": riesgos[i]["texto"],
                    "porque": "Basado directamente en el factor detectado por el modelo.",
                }
    contenido["acciones_recomendadas"] = acciones

    contenido["fuente"] = f"LLM ({modelo_llm})"
    return contenido


def generar_recomendaciones(resultado, titulo, cuerpo):
    try:
        return generar_recomendaciones_llm(resultado, titulo, cuerpo)
    except Exception:
        return generar_recomendaciones_fallback(resultado, titulo)


def calcular_confianza(prob, umbral):
    """Se calcula qué tan lejos está la predicción de UMBRAL_DECISION: 0 significa
    que está justo en la frontera, 1 que la predicción es lo más clara posible
    (prob = 0 o prob = 1). Se normaliza por separado a cada lado del umbral, porque
    este no está a la misma distancia de 0 que de 1."""
    if prob < umbral:
        return (umbral - prob) / umbral
    return (prob - umbral) / (1 - umbral)


def nivel_confianza_texto(confianza):
    """Se traduce el número crudo de confianza (0-1) a una palabra clave para el
    editor, cortando en tercios."""
    if confianza < CORTE_CONFIANZA_MEDIO:
        return "Baja"
    elif confianza < 0.67:
        return "Media"
    return "Alta"


def nivel_riesgo_texto(prob, confianza, umbral):
    """Se devuelve "Medio" cuando la confianza es baja: la predicción está justo
    en la frontera de decisión, sin importar de qué lado cae. "Bajo"/"Alto" son
    los casos con confianza hacia un lado u otro. "Riesgo Medio" siempre implica
    "Confianza Baja" (ver nota en ACCIONES_EDITOR)."""
    if confianza < CORTE_CONFIANZA_MEDIO:
        return "Medio"
    return "Bajo" if prob < umbral else "Alto"


def nivel_desempeno_texto(nivel_riesgo):
    """Se traduce el nivel de riesgo interno a su equivalente en marco positivo
    para mostrar al editor: Riesgo Bajo equivale a Posibilidad Alta de buen
    desempeño. El cálculo interno (modelo, SHAP, ACCIONES_EDITOR) sigue en
    términos de riesgo; esto solo traduce el texto en pantalla."""
    return {"Bajo": "Alta", "Medio": "Media", "Alto": "Baja"}[nivel_riesgo]


# Matriz Riesgo x Confianza que define la acción concreta para el editor.
# "Riesgo Medio" solo puede darse con "Confianza Baja", porque nivel_riesgo_texto
# deriva "Medio" directamente de confianza < 0.34. Esas otras combinaciones no
# aparecen en la tabla.
ACCIONES_EDITOR = {
    ("Bajo", "Alta"): {"accion": "PUBLICAR", "detalle": "certeza alta de buen desempeño, no requiere revisión adicional", "estilo": "ok"},
    ("Bajo", "Media"): {"accion": "PUBLICAR", "detalle": "buena señal, sin acciones pendientes", "estilo": "ok"},
    ("Bajo", "Baja"): {"accion": "PUBLICAR", "detalle": "corrige el factor que más reduce la posibilidad de éxito antes de salir", "estilo": "ok"},
    ("Medio", "Baja"): {"accion": "DETENER", "detalle": "no publicar hasta que un editor revise manualmente", "estilo": "warn"},
    ("Alto", "Baja"): {"accion": "CORRIGE ANTES DE PUBLICAR", "detalle": "aplica al menos 2 acciones recomendadas", "estilo": "warn"},
    ("Alto", "Media"): {"accion": "NO PUBLICAR SIN CAMBIOS", "detalle": "aplica todas las acciones recomendadas", "estilo": "danger"},
    ("Alto", "Alta"): {"accion": "BLOQUEAR", "detalle": "requiere reescritura y aprobación del jefe de editores", "estilo": "danger"},
}


def accion_editorial(nivel_riesgo, nivel_confianza):
    # Se usa .get() con un valor por defecto: si aparece una combinación fuera de
    # la matriz, se trata como el caso ambiguo más cercano en vez de dejar caer la
    # app.
    return ACCIONES_EDITOR.get((nivel_riesgo, nivel_confianza), ACCIONES_EDITOR[("Medio", "Baja")])


def estimar_impacto_y_proyeccion(categoria, prediccion_bajo_desempeno):
    grupo = tabla_historica[
        (tabla_historica["categoria"] == categoria)
        & (tabla_historica["bajo_desempeno"] == int(prediccion_bajo_desempeno))
    ]
    grupo_bueno = tabla_historica[(tabla_historica["categoria"] == categoria) & (tabla_historica["bajo_desempeno"] == 0)]
    grupo_malo = tabla_historica[(tabla_historica["categoria"] == categoria) & (tabla_historica["bajo_desempeno"] == 1)]

    if grupo.empty:
        return None

    fila = grupo.iloc[0]

    return {
        "vistas_estimadas": round(fila["vistas_mediana"]),
        "vistas_rango": (round(fila["vistas_p25"]), round(fila["vistas_p75"])),
        "vistas_mediana_buena": round(grupo_bueno.iloc[0]["vistas_mediana"]) if not grupo_bueno.empty else None,
        "vistas_mediana_mala": round(grupo_malo.iloc[0]["vistas_mediana"]) if not grupo_malo.empty else None,
    }


def construir_reporte_dashboard(resultado, recomendaciones, categoria, facebook, twitter):
    prob = resultado["probabilidad_riesgo"]
    # nivel_riesgo/nivel_confianza deciden sobre el valor exacto, sin redondear, para
    # coincidir siempre con las bandas de color del gauge. confianza_modelo
    # (redondeado) es solo para mostrar en pantalla.
    confianza_exacta = calcular_confianza(prob, UMBRAL_DECISION)
    nivel_riesgo = nivel_riesgo_texto(prob, confianza_exacta, UMBRAL_DECISION)
    nivel_confianza = nivel_confianza_texto(confianza_exacta)
    confianza_modelo = round(confianza_exacta, 2)

    # No se predice de dónde vendrá el tráfico, eso requeriría datos de campañas
    # que no se tienen. Solo se reporta un hecho del contenido: si la nota trae
    # elementos embebidos de redes sociales.
    contenido_social_embebido = "Sí" if (facebook > 0 or twitter > 0) else "No"

    accion = accion_editorial(nivel_riesgo, nivel_confianza)
    prioridad_por_estilo = {"ok": "Baja", "warn": "Media", "danger": "Alta"}
    prioridad = prioridad_por_estilo[accion["estilo"]]

    hora_optima = mejor_hora_categoria.get(categoria)
    proyeccion = estimar_impacto_y_proyeccion(categoria, resultado["riesgo_bajo_desempeno"])

    etiquetas_acciones = ["Alto", "Medio", "Bajo"]

    # Cada sugerencia trae {"titulo": ..., "tip": ...} directo del LLM o del
    # fallback de reglas; el tip explica el porqué real de cada título.
    sugerencias_titulos = recomendaciones["sugerencias_titulos"]
    # El orden de "acciones_recomendadas" viene fijado por el ranking real de SHAP,
    # así que se le pone la etiqueta Alto/Medio/Bajo según la posición. Para
    # "es_seo": las primeras num_riesgos acciones están atadas 1 a 1 a un factor de
    # SHAP; lo que sigue son las recomendaciones SEO generales que se agregan
    # siempre al final.
    num_riesgos = len([f for f in resultado["factores_riesgo"] if f["es_riesgo"]][:4])
    acciones_recomendadas = [
        {
            "recomendacion": a["recomendacion"],
            "porque": a["porque"],
            "impacto": etiquetas_acciones[min(i, len(etiquetas_acciones) - 1)],
            "es_seo": i >= num_riesgos,
        }
        for i, a in enumerate(recomendaciones["acciones_recomendadas"])
    ]
    if not acciones_recomendadas:
        acciones_recomendadas = [{
            "recomendacion": "No se detectaron factores que reduzcan la posibilidad de éxito; la nota luce bien tal como está.",
            "porque": "Ningún factor negativo superó el umbral mínimo de importancia del modelo.",
            "impacto": "Bajo",
            "es_seo": False,
        }]

    return {
        "probabilidad_riesgo": prob,
        "nivel_riesgo": nivel_riesgo,
        "probabilidad_buen_desempeno": resultado["probabilidad_buen_desempeno"],
        "confianza_modelo": confianza_modelo,
        "nivel_confianza": nivel_confianza,
        "contenido_social_embebido": contenido_social_embebido,
        "accion_editorial": accion,
        "prioridad": prioridad,
        "hora_optima_publicacion": hora_optima,
        "proyeccion_visitas": proyeccion,
        "factores_riesgo": resultado["factores_riesgo"],
        "sugerencias_titulos": sugerencias_titulos,
        "acciones_recomendadas": acciones_recomendadas,
        "fuente_recomendaciones": recomendaciones["fuente"],
        "fecha_evaluacion": dt.datetime.now().strftime("%d/%m/%Y %I:%M %p"),
    }


# ---------- Estilos: tema oscuro, tarjetas negras, acentos amarillo/verde ----------

st.markdown("""
<style>
:root {
    --font-main: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif;
    --fs-display: 34px;
    --fs-h1: 25px;
    --fs-h2: 17px;
    --fs-h3: 20px;
    --fs-body: 15px;
    --fs-small: 13px;
    --fs-label: 13px;
    --fs-badge: 12px;
    --bg-card: #2A2A2D;
    --bg-card-alt: #303034;
    --border-subtle: #3C3C42;
    --border-faint: #38383D;
    --bg-input: #39393D;
}

html, body, [class*="css"] {font-family: var(--font-main);}
[data-testid="stWidgetLabel"] p {font-size: var(--fs-h2) !important;}
[data-testid="stCaptionContainer"] p {font-size: var(--fs-small) !important;}
[data-testid="stMarkdownContainer"] p {font-size: var(--fs-body);}
.block-container {padding-top: 1.2rem; max-width: 1400px;}
/* Se oculta el menú hamburguesa, el botón "Deploy" y el footer, pero no el
   <header> completo: ahí vive el control para reabrir el sidebar una vez que se
   colapsa. */
#MainMenu, footer, [data-testid="stToolbarActions"], [data-testid="stAppDeployButton"] {visibility: hidden;}
/* Se hace transparente el fondo del header para que no tape el logo/fecha, y se
   deja visible para conservar el botón de reabrir el sidebar. */
[data-testid="stHeader"] {background:transparent !important;}

.app-header {display:flex; align-items:center; gap:18px; padding: 4px 0 18px 0; border-bottom: 1px solid var(--border-subtle); margin-bottom: 18px;}
.brand {font-size: var(--fs-h1); font-weight: 700; color: #E6E8EC;}
.brand-mx {color: #F5C518;}
.header-divider {width:1px; height:34px; background:var(--border-subtle);}
.header-title-main {font-weight:600; font-size:var(--fs-h2); color:#E6E8EC;}
.header-title-sub {font-size:var(--fs-small); color:#8A8F98;}
.header-date {margin-left:auto; font-size:var(--fs-body); color:#B7BCC4; background:var(--bg-card); border:1px solid var(--border-subtle); padding:6px 12px; border-radius:8px;}

.panel-title {font-size:var(--fs-h2); font-weight:600; letter-spacing:0.02em; color:#E6E8EC; text-transform:uppercase; margin-bottom:2px;}

.card {background:var(--bg-card); border:1px solid var(--border-subtle); border-radius:14px; padding:18px 20px; margin-bottom:16px;}
.card-title {font-size:var(--fs-label); font-weight:600; letter-spacing:0.02em; color:#9AA0A8; text-transform:uppercase; margin-bottom:10px;}

.st-key-card_riesgo, .st-key-card_info, .st-key-card_decision,
.st-key-card_proyeccion,
.st-key-card_titulos, .st-key-card_acciones, .st-key-card_factores {
    background:var(--bg-card); border:1px solid var(--border-subtle); border-radius:14px;
    padding:18px 20px; margin-bottom:16px;
}

.badge {display:inline-block; padding:3px 10px; border-radius:20px; font-size:var(--fs-badge); font-weight:600; letter-spacing:0.01em;}
.badge-green {background:#123321; color:#3DDC84;}
.badge-yellow {background:#332B12; color:#F5C518;}
.badge-red {background:#331414; color:#F16565;}
.badge-blue {background:#12232F; color:#4FA3F7;}
.badge-gray {background:#262A33; color:#B7BCC4;}
.badge-black {background:#111111; color:#E6E8EC;}

.lectura-titulo {
    font-size:var(--fs-h1); font-weight:800; color:#E6E8EC; line-height:1.25;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}

.kv-row {display:flex; justify-content:space-between; align-items:center; padding:7px 0; border-bottom:1px solid var(--border-faint);}
.kv-row:last-child {border-bottom:none;}
.kv-label {color:#9AA0A8; font-size:var(--fs-body);}
.kv-value {font-weight:600; font-size:var(--fs-body); color:#E6E8EC;}

.big-number {font-size:var(--fs-display); font-weight:700; color:#E6E8EC;}
.sub-text {font-size:var(--fs-small); color:#8A8F98; margin-top:2px;}

.decision-ok {color:#3DDC84; font-size:var(--fs-h3); font-weight:700;}
.decision-warn {color:#F5C518; font-size:var(--fs-h3); font-weight:700;}
.decision-danger {color:#F16565; font-size:var(--fs-h3); font-weight:700;}

.factor-row {display:flex; gap:10px; align-items:flex-start; padding:8px 0; border-bottom:1px solid var(--border-faint); font-size:var(--fs-body);}
.factor-row:last-child {border-bottom:none;}
/* Se usa un ancho fijo porque "RIESGO"/"POSITIVO"/"INFO" tienen distinto largo, y
   sin un ancho común el texto de cada fila arranca en una columna distinta. */
.factor-tag {flex-shrink:0; width:74px; text-align:center; font-size:var(--fs-badge); font-weight:600; padding:2px 8px; border-radius:6px; margin-top:1px;}

.title-row {padding:10px 0; border-bottom:1px solid var(--border-faint);}
.title-row:last-child {border-bottom:none;}
.title-rank {display:inline-flex; align-items:center; justify-content:center; width:20px; height:20px; border-radius:50%; background:var(--bg-card-alt); color:#9AA0A8; font-size:var(--fs-badge); font-weight:600; margin-right:8px;}
.title-tip {margin-left:28px; margin-top:4px; font-size:var(--fs-small); color:#9AA0A8; font-style:italic;}

.action-row {padding:10px 0; border-bottom:1px solid var(--border-faint); font-size:var(--fs-body);}
.action-row:last-child {border-bottom:none;}
.action-row-top {display:flex; justify-content:space-between; align-items:center; gap:10px;}
.action-tip {margin-top:4px; font-size:var(--fs-small); color:#9AA0A8; font-style:italic;}
/* Punto azul, mismo color del badge "IA" del disclaimer de arriba. Marca las
   recomendaciones SEO generales, para distinguirlas de las atadas a un factor
   real de SHAP. */
.seo-dot {color:#4FA3F7; margin-right:6px; font-size:10px; vertical-align:middle;}

.empty-state {text-align:center; padding: 50px 20px;}
.empty-icon {width:90px; height:90px; border-radius:50%; border:2px solid var(--border-subtle); display:flex; align-items:center; justify-content:center; margin:0 auto 22px auto; font-size:var(--fs-display);}
.empty-title {font-size:var(--fs-h3); font-weight:700; color:#E6E8EC; margin-bottom:8px;}
.empty-sub {font-size:var(--fs-body); color:#9AA0A8; margin-bottom:34px;}
.step-flow {display:flex; align-items:flex-start; justify-content:center; gap:26px;}
.step-item {width:120px; text-align:center;}
.step-circle {width:44px; height:44px; border-radius:50%; border:2px solid var(--border-subtle); display:flex; align-items:center; justify-content:center; margin:0 auto 8px auto; font-size:var(--fs-h3);}
.step-num {display:inline-block; width:18px; height:18px; border-radius:50%; background:#6B7078; color:#FFFFFF; font-size:var(--fs-badge); font-weight:600; line-height:18px; margin-bottom:6px;}
.step-text {font-size:var(--fs-small); color:#9AA0A8;}

.deteccion-box {background:var(--bg-card-alt); border:1px solid var(--border-subtle); border-radius:12px; padding:16px 18px; margin:10px 0 6px 0;}
.deteccion-label {font-size:15px; font-weight:700; color:#F5C518; text-transform:uppercase; letter-spacing:0.02em; margin-bottom:12px;}
.deteccion-badges {display:flex; flex-wrap:wrap; gap:10px;}
.deteccion-badges .badge {font-size:14px; font-weight:700; padding:6px 14px;}

.st-key-form_panel, .st-key-rss_lista_panel, .st-key-rss_lectura_panel {
    border:1px solid var(--border-subtle); border-radius:14px; padding:20px 22px;
}
/* Cada nota del feed es un solo contenedor (imagen + texto juntos). El fondo/
   borde va aquí; el botón de adentro queda transparente. */
[class*="st-key-rss_row_"] {
    border-radius:10px; padding:7px 12px 10px 12px; margin-bottom:10px;
    background:var(--bg-card-alt); border:1px solid var(--border-subtle);
}
[class*="st-key-rss_row_sel_"] {background:#F5C518; border-color:#F5C518;}

[class*="st-key-rss_row_"] div.stButton > button {
    background:transparent !important; border:none !important; box-shadow:none !important;
    text-align:left; justify-content:flex-start; white-space:normal; height:auto;
    padding:0 !important; color:#E6E8EC !important; font-weight:600;
    /* Se ajusta el line-height para que el texto empiece a la misma altura que la
       imagen de al lado. */
    line-height:1.3 !important;
    /* El botón es inline-flex dentro de un contenedor block, así que el navegador
       lo alinea por "baseline". Se fuerza vertical-align:top para quitar ese
       espacio fantasma. */
    vertical-align:top !important;
    display:block !important;
}
[class*="st-key-rss_row_"] div.stButton > button p {margin:0 !important; line-height:1.3 !important;}
[class*="st-key-rss_row_sel_"] div.stButton > button {color:#111 !important;}

.embed-card {
    background:var(--bg-card-alt); border:1px solid var(--border-subtle); border-left:3px solid #4FA3F7;
    border-radius:8px; padding:10px 14px; margin:12px 0; font-size:var(--fs-body);
}

.rango-track {position:relative; height:8px; border-radius:4px; background:linear-gradient(to right, #B23B3B 0%, #C9971F 50%, #3DDC84 100%);}
.rango-punto {position:absolute; top:50%; width:14px; height:14px; border-radius:50%; background:#F5C518; border:2px solid #1C1C1E; transform:translate(-50%, -50%);}
.rango-marcador-label {position:absolute; bottom:0; transform:translateX(-50%); font-size:var(--fs-small); font-weight:700; color:#F5C518; white-space:nowrap;}
.rango-extremos {display:flex; justify-content:space-between; margin-top:10px; font-size:var(--fs-small);}
.rango-extremo-bajo {color:#F16565; font-weight:600;}
.rango-extremo-bueno {color:#3DDC84; font-weight:600; text-align:right;}

.stTextInput input, .stTextArea textarea, .stNumberInput input,
.stDateInput input, .stTimeInput input,
div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {
    background-color: var(--bg-input) !important;
    border-color: var(--border-subtle) !important;
    border-radius: 8px !important;
}

div.stButton > button[kind="primary"] {
    background:#F5C518; color:#111; font-weight:700; letter-spacing:0.01em;
    border:none; border-radius:10px; padding:12px 0; font-size:var(--fs-body);
}
div.stButton > button[kind="primary"]:hover {background:#FFD84A; color:#111;}

/* Se usa "secondary" para el nav de la sidebar y las filas no elegidas del feed
   RSS, con un estilo discreto para que se note cuál está activo. */
div.stButton > button[kind="secondary"] {
    background:var(--bg-card-alt); color:#B7BCC4; border:1px solid var(--border-subtle);
    border-radius:10px; font-size:var(--fs-body);
}
div.stButton > button[kind="secondary"]:hover {border-color:#F5C518; color:#E6E8EC;}

/* Botones del menú (Notas / Validación) con el texto a la izquierda, 20px. */
[data-testid="stSidebar"] div.stButton > button {
    text-align:left; justify-content:flex-start; padding-left:20px !important;
}

/* Se recorta el logo del sidebar en círculo, con su fondo blanco, y se centra
   horizontalmente. */
[data-testid="stSidebar"] [data-testid="stImage"] img {
    border-radius:50%; object-fit:cover; aspect-ratio:1/1;
    display:block; margin:0 auto;
}
/* Se centra en stFullScreenFrame, el primer ancestro que mide el ancho completo
   de la columna del sidebar. stImage/stImageContainer se encogen al ancho exacto
   de la imagen (135px), así que ahí "justify-content:center" no tiene espacio
   para centrar. */
[data-testid="stSidebar"] [data-testid="stFullScreenFrame"] {
    display:flex; justify-content:center;
}

/* El sidebar reserva un espacio de encabezado pensado para st.logo(). Se achica
   porque el logo se pone más abajo, para que quede pegado arriba. */
[data-testid="stSidebarHeader"] {min-height:0 !important; padding-bottom:0 !important;}
[data-testid="stLogoSpacer"] {display:none;}
/* Los 60px de ese encabezado son del botón para colapsar el sidebar y no se
   pueden quitar. Se sube el logo con un margen negativo para que quede pegado
   arriba. */
[data-testid="stSidebar"] [data-testid="stImage"] {margin-top:-18px;}
</style>
""", unsafe_allow_html=True)


# ---------- Header ----------

st.markdown("""
<div class="app-header">
  <div class="brand">newshub<span class="brand-mx">mx</span></div>
  <div class="header-divider"></div>
  <div>
    <div class="header-title-main">Asistente Predictivo</div>
    <div class="header-title-sub">Decisiones editoriales inteligentes</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ---------- Navegación: Notas (feed RSS) / Validación (análisis) ----------

if "vista" not in st.session_state:
    st.session_state["vista"] = "notas"

with st.sidebar:
    st.image(str(LOGO_PATH), width=135)
    if st.button("📋 Notas", use_container_width=True, type=("primary" if st.session_state["vista"] == "notas" else "secondary")):
        st.session_state["vista"] = "notas"
        st.rerun()
    if st.button("✅ Validación", use_container_width=True, type=("primary" if st.session_state["vista"] == "validacion" else "secondary")):
        st.session_state["vista"] = "validacion"
        st.rerun()

vista = st.session_state["vista"]


# ---------- Vista "Notas": feed RSS -> elegir una nota para analizar ----------

if vista == "notas":
    st.markdown('<div class="panel-title">📡 Notas desde RSS</div>', unsafe_allow_html=True)
    st.caption("Elige una nota del feed para analizarla. También puedes ir directo a Validación y escribir una a mano.")

    col_url, col_btn = st.columns([5, 1.2])
    with col_url:
        rss_url = st.text_input(
            "URL del RSS", key="rss_url", label_visibility="collapsed",
            placeholder="Pega la URL de un RSS (o deja vacío para usar el feed de ejemplo del proyecto)",
        )
    with col_btn:
        actualizar_rss = st.button("🔄 Actualizar RSS", use_container_width=True)

    if actualizar_rss:
        # Se carga el feed solo cuando el editor le da clic a "Actualizar RSS". Si
        # nunca le dio clic, el panel de lectura queda en blanco.
        with st.spinner("Cargando feed RSS..."):
            notas_cargadas, uso_fallback = cargar_rss(rss_url)
            st.session_state["rss_notas"] = notas_cargadas
            st.session_state["rss_uso_fallback"] = uso_fallback
            st.session_state["rss_seleccionada"] = 0

    notas_rss = st.session_state.get("rss_notas")

    if notas_rss:
        if st.session_state.get("rss_uso_fallback"):
            st.caption("ℹ️ No se pudo usar la URL indicada (o se dejó vacía) . Se cargó el feed de ejemplo del proyecto.")
        else:
            st.caption("✅ Feed cargado correctamente.")

    if not notas_rss:
        # No se muestran cajas con borde mientras no hay ningún feed cargado.
        st.markdown(
            '<div class="sub-text" style="text-align:center; padding:40px 0;">'
            '📭 Todavía no cargas ningún feed — pega una URL y da clic en "Actualizar RSS", '
            'o solo da clic para usar el feed de ejemplo incluido en el proyecto.</div>',
            unsafe_allow_html=True,
        )
    else:
        col_lista, col_lectura = st.columns([1, 1], gap="large")

        with col_lista, st.container(key="rss_lista_panel"):
            st.markdown(f'<div class="card-title">Feed RSS ({len(notas_rss)} notas)</div>', unsafe_allow_html=True)
            for i, nota in enumerate(notas_rss):
                es_la_seleccionada = i == st.session_state.get("rss_seleccionada", 0)
                # Se ponen imagen y texto dentro del mismo contenedor con borde. El
                # color de selección lo pone el contenedor, por eso el botón queda
                # transparente.
                fila_key = f"rss_row_sel_{i}" if es_la_seleccionada else f"rss_row_{i}"
                with st.container(key=fila_key):
                    # Se alinea arriba, no al centro, porque si se centra la imagen
                    # se corre hacia abajo cuando el texto de al lado tiene más
                    # líneas.
                    col_img, col_txt = st.columns([1, 3], vertical_alignment="top")
                    with col_img:
                        if nota["imagen_portada"]:
                            st.image(nota["imagen_portada"], use_container_width=True)
                    with col_txt:
                        texto_boton = f"{i + 1}. {nota['titulo']} ({nota['categoria'] or 'Sin categoría'})"
                        if st.button(texto_boton, key=f"rss_item_{i}", use_container_width=True):
                            st.session_state["rss_seleccionada"] = i
                            st.rerun()

        with col_lectura, st.container(key="rss_lectura_panel"):
            nota = notas_rss[st.session_state.get("rss_seleccionada", 0)]
            # Se muestra el título en una sola línea (con "..." si no cabe), el
            # autor debajo, y la sección/hora en el mismo renglón. No se repite el
            # <h1> que trae el HTML del feed.
            st.markdown(f"""
            <div class="lectura-titulo" title="{nota['titulo']}">{nota['titulo']}</div>
            <br>
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <span class="sub-text">✍️ {nota['autor'] or 'Sin autor'} · 📅 {nota['fecha_corta']}</span>
              <span class="badge badge-black">{nota['categoria'] or 'Sin categoría'}</span>
            </div>
            """, unsafe_allow_html=True)
            contenido_sin_h1 = re.sub(r"<h1[^>]*>.*?</h1>", "", nota["contenido"], count=1, flags=re.IGNORECASE | re.DOTALL)
            st.markdown(formatear_embeds_para_lectura(contenido_sin_h1), unsafe_allow_html=True)
            st.markdown("---")
            if st.button("✨ Validar esta nota", type="primary", use_container_width=True, key="btn_analizar_desde_rss"):
                # Solo se pasan los datos al formulario de Validación. El análisis
                # ocurre allá, cuando el editor le da clic a "ANALIZAR NOTA".
                st.session_state["form_titulo"] = nota["titulo"]
                st.session_state["form_categoria"] = nota["categoria"] or "Selecciona una categoría"
                st.session_state["form_autor"] = nota["autor"] or "Selecciona un autor"
                st.session_state["form_cuerpo"] = nota["contenido"]
                st.session_state["form_audio"] = nota["audio"]
                st.session_state["vista"] = "validacion"
                st.rerun()


# ---------- Vista "Validación": el formulario + dashboard que ya existían ----------

elif vista == "validacion":
    col_izq, col_der = st.columns([5, 7], gap="large")

    with col_izq:
        with st.container(key="form_panel"):
            st.markdown('<div class="panel-title">📝 Entradas de la nota</div>', unsafe_allow_html=True)
            st.caption("Completa la información de la noticia para generar la predicción")

            # No se usa max_chars porque con un título largo Streamlit vacía el
            # campo en vez de recortarlo. Se prefiere solo advertir, sin borrar lo
            # que escribió el editor. Las keys explícitas (form_*) permiten que la
            # vista "Notas" precargue estos campos desde una nota del RSS.
            titulo = st.text_input("Título de la noticia *", key="form_titulo", placeholder="Escribe un título atractivo y claro")
            if len(titulo) > 60:
                st.caption(f"⚠️ {len(titulo)} / 60 — más largo de lo recomendado")
            else:
                st.caption(f"{len(titulo)} / 60")

            categoria = st.selectbox("Categoría *", key="form_categoria", options=["Selecciona una categoría"] + CATEGORIAS)
            autor = st.selectbox("Autor *", key="form_autor", options=["Selecciona un autor"] + AUTORES)

            cuerpo = st.text_area(
                "Contenido de la noticia *", key="form_cuerpo", height=220,
                placeholder="Escribe texto plano, o pega el HTML de la nota (con <p>, <h2>, <img>, enlaces, embeds de Twitter/Facebook/YouTube, etc.)",
            )
            detectado = analizar_html_contenido(cuerpo)
            feats_previa = calcular_features_texto(titulo or "", cuerpo)
            ref_form = referencia_contenido(categoria)
            st.caption(
                f"{feats_previa['palabras_cuerpo']} palabras — en **{categoria}**, las notas con buen desempeño "
                f"rondan **{ref_form['palabras']} palabras** (párrafos de ~{ref_form['longitud_parrafo']} palabras)."
                if categoria != "Selecciona una categoría" else
                f"{feats_previa['palabras_cuerpo']} palabras — nuestros datos muestran que notas más **cortas** "
                f"(≈{ref_form['palabras']} palabras, párrafos de ≈{ref_form['longitud_parrafo']} palabras) tienden a rendir mejor que las largas."
            )

            def _badge_deteccion(etiqueta, valor, positivo):
                clase = "badge-green" if positivo else "badge-gray"
                return f'<span class="badge {clase}">{etiqueta}: {valor}</span>'

            badges_deteccion = "".join([
                _badge_deteccion("🖼️ Imagen", detectado["imagen"], detectado["imagen"] > 0),
                _badge_deteccion("🎬 YouTube", "Sí" if detectado["video"] else "No", detectado["video"] > 0),
                _badge_deteccion("🔗 Enlaces", detectado["link_build"], detectado["link_build"] > 0),
                _badge_deteccion("H1", detectado["h1"], detectado["h1"] > 0),
                _badge_deteccion("H2", detectado["h2"], detectado["h2"] > 0),
                _badge_deteccion("🐦 Twitter", detectado["twitter"], detectado["twitter"] > 0),
                _badge_deteccion("📘 Facebook", detectado["facebook"], detectado["facebook"] > 0),
            ])
            st.markdown(f"""
            <div class="deteccion-box">
              <div class="deteccion-label">🔎 Detectado automáticamente en el contenido</div>
              <div class="deteccion-badges">{badges_deteccion}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="panel-title" style="font-size:var(--fs-body); margin-top:6px;">⚙️ Opciones avanzadas</div>', unsafe_allow_html=True)
            audio_flag = st.checkbox("🔊 Incluye audio", key="form_audio")

            programar = st.checkbox("🗓️ Programar fecha/hora de publicación distinta a ahora")
            if programar:
                fecha_in = st.date_input("Fecha de publicación", dt.date.today())
                hora_in = st.time_input("Hora de publicación", dt.datetime.now().time())
                fecha_publicacion = dt.datetime.combine(fecha_in, hora_in)
            else:
                fecha_publicacion = dt.datetime.now()

            analizar = st.button("✨ ANALIZAR NOTA", use_container_width=True, type="primary")

            st.caption("💡 Nota: Las predicciones se basan en datos históricos y pueden variar según el comportamiento real de los usuarios.")

            if analizar:
                errores = []
                if not titulo.strip():
                    errores.append("el título")
                if categoria == "Selecciona una categoría":
                    errores.append("la categoría")
                if autor == "Selecciona un autor":
                    errores.append("el autor")
                if not cuerpo.strip():
                    errores.append("el contenido")

                if errores:
                    st.error("Falta completar: " + ", ".join(errores))
                else:
                    with st.spinner("Analizando nota..."):
                        resultado = analizar_nota(
                            titulo=titulo, cuerpo=cuerpo, categoria=categoria, nombre=autor,
                            fecha_publicacion=fecha_publicacion,
                            imagen=detectado["imagen"], video=detectado["video"],
                            twitter=detectado["twitter"], facebook=detectado["facebook"],
                            audio=1 if audio_flag else 0,
                            h1=detectado["h1"], h2=detectado["h2"], link_build=detectado["link_build"],
                        )
                        recomendaciones = generar_recomendaciones(resultado, titulo, cuerpo)
                        reporte = construir_reporte_dashboard(resultado, recomendaciones, categoria, detectado["facebook"], detectado["twitter"])
                        st.session_state["reporte"] = reporte


    with col_der:
        st.markdown('<div class="panel-title">📊 Dashboard de resultados</div>', unsafe_allow_html=True)
        st.caption("Análisis de desempeño potencial de la noticia")

        if "reporte" not in st.session_state:
            st.markdown("""
            <div class="card empty-state">
              <div class="empty-icon">📊</div>
              <div class="empty-title">Tu dashboard aún no se ha generado</div>
              <div class="empty-sub">Completa la información de la nota y haz clic en <b>"Analizar nota"</b> para ver los resultados.</div>
              <div class="step-flow">
                <div class="step-item"><div class="step-circle">📝</div><div class="step-num">1</div><div class="step-text">Completa los campos de entrada</div></div>
                <div class="step-item"><div class="step-circle">✨</div><div class="step-num">2</div><div class="step-text">Haz clic en "Analizar nota"</div></div>
                <div class="step-item"><div class="step-circle">📈</div><div class="step-num">3</div><div class="step-text">Revisa tu dashboard con insights y recomendaciones</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            r = st.session_state["reporte"]

            badge_color = {"Bajo": "badge-green", "Medio": "badge-yellow", "Alto": "badge-red"}[r["nivel_riesgo"]]
            # "estilo" viene de la matriz ACCIONES_EDITOR (ok/warn/danger) y define
            # la clase CSS y el color según riesgo y confianza juntos.
            decision_class = {"ok": "decision-ok", "warn": "decision-warn", "danger": "decision-danger"}[r["accion_editorial"]["estilo"]]
            decision_icon = {"ok": "✅", "warn": "🟡", "danger": "🔴"}[r["accion_editorial"]["estilo"]]
            confianza_badge = {"Baja": "badge-yellow", "Media": "badge-blue", "Alta": "badge-green"}[r["nivel_confianza"]]

            col1, col2, col3 = st.columns([1.1, 1, 1])

            nivel_desempeno = nivel_desempeno_texto(r["nivel_riesgo"])

            with col1, st.container(key="card_riesgo"):
                st.markdown('<div class="card-title">Posibilidad de buen desempeño</div>', unsafe_allow_html=True)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    # Se usa el mismo número que "Probabilidad de buen desempeño" en
                    # Detalle: (1 - riesgo) * 100. El gauge muestra posibilidad de
                    # buen desempeño, no riesgo.
                    value=r["probabilidad_buen_desempeno"],
                    number={"suffix": "%", "font": {"size": 36, "color": "#E6E8EC"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#8A8F98", "tickwidth": 1},
                        "bar": {"color": "#FFFFFF", "thickness": 0.28},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        # Se invierten las bandas respecto al riesgo: aquí más alto es mejor,
                        # así que el verde queda del lado derecho. Las fronteras se calculan a
                        # partir de UMBRAL_DECISION/CORTE_CONFIANZA_MEDIO para que coincidan con
                        # el badge de riesgo Alto/Medio/Bajo.
                        "steps": [
                            {"range": [0, GAUGE_CORTE_ROJO], "color": "#B23B3B"},
                            {"range": [GAUGE_CORTE_ROJO, GAUGE_CORTE_VERDE], "color": "#C9971F"},
                            {"range": [GAUGE_CORTE_VERDE, 100], "color": "#1F8B4C"},
                        ],
                    },
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=200, margin=dict(l=10, r=10, t=10, b=0), font={"color": "#E6E8EC"})
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(f'<div style="text-align:center;"><span class="badge {badge_color}">Posibilidad {nivel_desempeno.lower()}</span></div>', unsafe_allow_html=True)

            with col2, st.container(key="card_info"):
                st.markdown(f"""
                <div class="card-title">Detalle</div>
                <div class="kv-row"><span class="kv-label">Posibilidad de buen desempeño</span><span class="badge {badge_color}">{nivel_desempeno.upper()}</span></div>
                <div class="kv-row"><span class="kv-label">Probabilidad estimada</span><span class="kv-value">{r['probabilidad_buen_desempeno']} / 100</span></div>
                <div class="kv-row"><span class="kv-label">Confianza del modelo</span><span class="badge {confianza_badge}">{r['nivel_confianza'].upper()}</span></div>
                <div class="kv-row"><span class="kv-label">Contenido social embebido</span><span class="badge {'badge-blue' if r['contenido_social_embebido'] == 'Sí' else 'badge-gray'}">{r['contenido_social_embebido']}</span></div>
                <div class="kv-row"><span class="kv-label">Fecha de evaluación</span><span class="kv-value" style="font-size:11px;">{r['fecha_evaluacion']}</span></div>
                """, unsafe_allow_html=True)

            with col3, st.container(key="card_decision"):
                hora_opt = r["hora_optima_publicacion"]
                hora_txt = f"Hoy {hora_opt:02d}:00" if hora_opt is not None else "Sin dato suficiente"
                prioridad_badge = {"Baja": "badge-green", "Media": "badge-yellow", "Alta": "badge-red"}[r["prioridad"]]
                st.markdown(f"""
                <div class="card-title">Decisión editorial</div>
                <div class="{decision_class}">{decision_icon} {r['accion_editorial']['accion']}</div>
                <div class="sub-text" style="margin-bottom:14px;">{r['accion_editorial']['detalle']}</div>
                <div class="kv-row"><span class="kv-label">Requiere atención editorial</span><span class="badge {prioridad_badge}">{r['prioridad'].upper()}</span></div>
                <div class="kv-row"><span class="kv-label">Mejor hora histórica</span><span class="kv-value">{hora_txt}</span></div>
                """, unsafe_allow_html=True)

            proy = r["proyeccion_visitas"]

            with st.container(key="card_proyeccion"):
                st.markdown('<div class="card-title">Proyección de visitas</div>', unsafe_allow_html=True)
                if proy:
                    lo, hi = proy["vistas_rango"]
                    st.markdown(f"""
                    <div class="big-number">{proy['vistas_estimadas']:,}</div>
                    <div class="sub-text">Rango histórico de notas similares: {lo:,} – {hi:,} visitas</div>
                    <div class="sub-text">Comparado contra el histórico de <b>{categoria}</b>, no contra otras categorías — cada categoría tiene su propio techo de tráfico.</div>
                    """, unsafe_allow_html=True)

                    marca = proy["vistas_estimadas"]
                    mala = proy["vistas_mediana_mala"]
                    buena = proy["vistas_mediana_buena"]

                    if mala is not None and buena is not None:
                        # Se construye la barra en HTML/CSS puro: las etiquetas de los extremos
                        # quedan en posición fija, así nunca se superponen aunque los dos
                        # valores estén muy cerca.
                        extremo_bajo, extremo_alto = sorted([mala, buena])
                        rango = extremo_alto - extremo_bajo
                        pct = 50.0 if rango <= 0 else max(0, min(100, (marca - extremo_bajo) / rango * 100))

                        st.markdown(f"""
                        <div style="position:relative; height:20px; margin-top:18px;">
                          <div class="rango-marcador-label" style="left:{pct}%;">Tu nota: {marca:,.0f}</div>
                        </div>
                        <div class="rango-track">
                          <div class="rango-punto" style="left:{pct}%;"></div>
                        </div>
                        <div class="rango-extremos">
                          <span class="rango-extremo-bajo">Bajo desempeño<br>{mala:,.0f}</span>
                          <span class="rango-extremo-bueno">Buen desempeño<br>{buena:,.0f}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('<div class="sub-text" style="margin-top:14px;">Basado en el histórico real de la categoría (mediana p25–p75), no en una proyección diaria simulada.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="sub-text">Sin datos históricos suficientes para esta categoría.</div>', unsafe_allow_html=True)

            # Se muestra un solo disclaimer para las dos tarjetas de abajo, que
            # comparten la misma fuente: el editor necesita saber si está viendo
            # texto generado por IA o el plan de respaldo basado en reglas fijas.
            fuente = r["fuente_recomendaciones"]
            es_llm = fuente.startswith("LLM")
            # fuente tiene la forma "LLM (llama-3.3-70b-versatile)". Se extrae solo
            # el nombre del modelo para no repetir "LLM" dos veces en la misma
            # frase.
            modelo_llm_nombre = fuente.split("(", 1)[1].rstrip(")") if "(" in fuente else fuente
            disclaimer_badge = "badge-blue" if es_llm else "badge-yellow"
            disclaimer_icono = "🤖" if es_llm else "⚙️"
            disclaimer_texto = (
                f"Recomendaciones y títulos generados con IA (modelo {modelo_llm_nombre})."
                if es_llm else
                "Modo de respaldo: IA no disponible en este momento, recomendaciones generadas con reglas fijas."
            )
            st.markdown(
                f'<div class="sub-text" style="margin-bottom:12px;">'
                f'<span class="badge {disclaimer_badge}">{disclaimer_icono} {"IA" if es_llm else "RESPALDO"}</span> '
                f'{disclaimer_texto}</div>',
                unsafe_allow_html=True,
            )

            # Cada tarjeta ocupa el ancho completo, porque con el "porque"/"tip"
            # el contenido queda apretado a la mitad.
            with st.container(key="card_titulos"):
                st.markdown('<div class="card-title">Sugerencias de títulos</div>', unsafe_allow_html=True)
                filas = "".join(
                    f"""<div class="title-row">
                      <div><span class="title-rank">{i}</span>{t['titulo']}</div>
                      <div class="title-tip">💡 {t['tip']}</div>
                    </div>"""
                    for i, t in enumerate(r["sugerencias_titulos"], start=1)
                )
                st.markdown(filas, unsafe_allow_html=True)

            with st.container(key="card_acciones"):
                st.markdown('<div class="card-title">Acciones recomendadas</div>', unsafe_allow_html=True)
                imp_badge = {"Alto": "badge-red", "Medio": "badge-yellow", "Bajo": "badge-gray"}
                # "impacto" refleja el ranking real de SHAP que le paso al LLM o al
                # fallback. Cada acción trae su "porque" además del texto.
                filas = "".join(
                    f"""<div class="action-row">
                      <div class="action-row-top">
                        <span>{'<span class="seo-dot" title="Recomendación SEO general, no atada a un factor detectado por el modelo">●</span>' if a['es_seo'] else ''}{a['recomendacion']}</span>
                        <span class="badge {imp_badge.get(a['impacto'], 'badge-gray')}">{a['impacto']}</span>
                      </div>
                      <div class="action-tip">💡 {a['porque']}</div>
                    </div>"""
                    for a in r["acciones_recomendadas"]
                )
                st.markdown(filas, unsafe_allow_html=True)

            with st.container(key="card_factores"):
                st.markdown('<div class="card-title">Factores que influyen en el desempeño (explicabilidad SHAP)</div>', unsafe_allow_html=True)

                def _badge_factor(es_riesgo):
                    # es_riesgo es None para la mención agregada del autor. Se usa
                    # un badge neutral, no RIESGO ni POSITIVO.
                    if es_riesgo is None:
                        return "badge-gray", "INFO"
                    return ("badge-red", "RIESGO") if es_riesgo else ("badge-green", "POSITIVO")

                # Se muestra el "peso" en texto gris neutral, no en badge de color,
                # para no confundirlo con el badge de RIESGO/POSITIVO (son
                # dimensiones distintas: dirección y fuerza del efecto).
                filas = "".join(
                    f"""<div class="factor-row">
                      <span class="factor-tag badge {_badge_factor(f['es_riesgo'])[0]}">{_badge_factor(f['es_riesgo'])[1]}</span>
                      <span>
                        {f['texto']}
                        <span class="sub-text" style="display:block;">Peso: {f.get('peso', 'Bajo')}</span>
                      </span>
                    </div>"""
                    for f in r["factores_riesgo"]
                )
                st.markdown(filas, unsafe_allow_html=True)

            st.caption("Nota: el autor nunca se muestra como factor individual de desempeño, por diseño ético del proyecto.")
