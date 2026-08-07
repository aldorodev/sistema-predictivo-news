<p align="center">
  <img src="logo/newshub.png" alt="newshubmx" width="220">
</p>

<h1 align="center">newshubmx</h1>

<h1 align="center">Sistema Predictivo de Apoyo a la Toma de Decisiones Editoriales</h1>

<p align="center">
Hoy, un editor solo sabe si una nota funcionó <b>después</b> de publicarla, cuando ya
no hay nada que hacer. newshubmx mueve esa respuesta al momento de redactar: con un
modelo de ciencia de datos entrenado con el histórico real del medio, predice, antes
de publicar, si una nota va a tener bajo desempeño de tráfico, explica por qué con
factores concretos en lenguaje editorial (no técnico), y entrega recomendaciones
puntuales para corregirla a tiempo, comparándola siempre contra el comportamiento
histórico real de notas similares del propio medio. El resultado: menos notas que
salen sin revisión, prácticas editoriales más consistentes en todo el equipo, y un
camino directo entre lo que se escribe y el tráfico que ese medio necesita.
</p>

<p align="center"><b>Autor:</b> Aldo Alberto Rodríguez Flores</p>

## Prueba de la demo

https://sistema-predictivo-news-2cq3rxqzqtpaxhdm49rf7d.streamlit.app/

Para probar la vista de "Notas" (importar una nota real desde un feed RSS en vez de
escribirla a mano), puedes usar este feed de ejemplo:
`http://jarochilandia-com.ntc4-p2stl.ezhostingserver.com/rss/noticias.xml`

## Beneficio del proyecto

Hoy, un editor solo sabe si una nota funcionó **después** de publicarla, cuando ya no
hay nada que hacer. newshubmx mueve esa retroalimentación al momento de redactar,
convirtiendo el histórico de tráfico del propio medio en una señal accionable antes de
publicar: menos notas sin revisar, prácticas más consistentes entre todo el equipo, y
una vía directa para conectar el trabajo editorial con el impacto real de negocio.

## Estructura del repositorio

```
app.py                     Dashboard de Streamlit (la app en producción)
notebook_analisis.ipynb    Pipeline completo: EDA, procesamiento, modelación, SHAP
modelo_riesgo.joblib       Modelo final ya entrenado (XGBoost) que carga app.py
requirements.txt           Dependencias de Python con versión fija
runtime.txt                Versión de Python para el despliegue

datos/
├── autores.json              Lista de autores que reconoce el modelo (para el
│                              selector de la app) — generado desde las
│                              categorías con las que se entrenó el modelo
├── historico_categoria.csv   Medianas históricas reales por categoría (vistas,
│                              engagement, longitud de nota, etc.) — generado
│                              por el notebook a partir de los datos crudos
└── mejor_hora_categoria.json Mejor hora histórica de publicación por
                               categoría — generado por el notebook

logo/newshub.png           Logo del proyecto
rss/noticias.xml           Feed RSS de ejemplo (respaldo si no hay conexión)
.streamlit/config.toml     Tema visual de la app
```

## Los datos

El proyecto cruza dos fuentes reales de un medio de noticias digital, año 2025
completo: el **CMS** del medio (34,734 notas: título, categoría, autor, fecha,
contenido HTML, multimedia) y **Google Analytics 4** (vistas, engagement, rebote,
ingresos por nota).

Esos dos CSV crudos pesan más de 110 MB juntos, así que **no están en este
repositorio**; se comparten en el Drive del Classroom, junto con `clave groq.txt` (ver
más abajo). Los agregados pequeños que sí necesita la app en producción
(`datos/*.json`, `datos/historico_categoria.csv`) ya están incluidos: son generados
por `notebook_analisis.ipynb` a partir de esos CSV crudos, no hay que crearlos a mano.

## Ciencia de datos en la aplicación

Se entrenaron y compararon cuatro modelos de clasificación binaria (Regresión
Logística, Random Forest, Gradient Boosting y XGBoost). Se eligió **XGBoost** por su
mejor equilibrio entre `recall` y `precision` frente al costo real de negocio (es más
caro no avisar de una nota riesgosa que avisar de más), confirmado con validación
cruzada estratificada de 5 particiones. Se usa un umbral de decisión de 0.40, más
sensible a notas de riesgo real, y cada predicción se explica con SHAP en lenguaje
editorial. El desarrollo completo, con cifras y gráficas reales, está en
`notebook_analisis.ipynb`.

## Flujo de la aplicación

1. El editor escribe una nota (vista **Validación**) o la importa desde RSS (vista
   **Notas**).
2. La app calcula las mismas variables con las que se entrenó el modelo y obtiene la
   probabilidad de bajo desempeño.
3. SHAP explica qué factores empujan esa probabilidad, traducidos a lenguaje
   editorial, siempre contra la mediana histórica real de la categoría.
4. Groq redacta recomendaciones y sugerencias de título (con respaldo por reglas si no
   está disponible).
5. El dashboard muestra todo junto: probabilidad, decisión editorial, proyección de
   visitas y acciones a seguir.

## Cómo está construida la app

`app.py` es una app de Streamlit de un solo archivo: carga el modelo entrenado
(`modelo_riesgo.joblib`) una vez y lo mantiene en caché, calcula las variables de cada
nota en tiempo real, corre SHAP sobre esa predicción puntual, y llama a Groq (con su
respaldo por reglas) para redactar las recomendaciones finales.

## La API de Groq

Redacta las recomendaciones y sugerencias de título a partir de los factores que ya
calculó SHAP (nunca predice el riesgo). Es opcional: sin configurarla, la app cae
automáticamente a un generador basado en reglas, sin errores.

Para activarla: consigue la clave (`clave groq.txt` del Classroom) y pásala como
`GROQ_API_KEY` — variable de entorno en local, o en **Settings → Secrets** en
Streamlit Cloud.

## Instalación y versiones

Python **3.10** (`runtime.txt`).

```bash
pip install -r requirements.txt
```

| Paquete | Versión |
|---|---|
| streamlit | 1.51.0 |
| pandas | 2.2.3 |
| numpy | 1.26.4 |
| scikit-learn | 1.7.2 |
| xgboost | 3.2.0 |
| joblib | 1.5.3 |
| plotly | 6.3.1 |
| openai (cliente para Groq) | 1.102.0 |
| feedparser | 6.0.14 |

Para correr `notebook_analisis.ipynb` también se necesitan `shap`,
`sentence-transformers`, `matplotlib` y `seaborn` (no están en `requirements.txt`
porque la app en producción no los usa). Instálalos con:

```bash
pip install shap sentence-transformers matplotlib seaborn
```

## Cómo correrlo en local

```bash
pip install -r requirements.txt
GROQ_API_KEY=tu_api_key streamlit run app.py
```

## Cómo correr el notebook en Google Colab

**⚠️ Importante**: los datos crudos deben cargarse **directamente desde Google Drive**,
no subirse a mano a la carpeta temporal de la sesión. Un archivo de más de 90 MB
subido por el panel del navegador se puede truncar sin avisar, y después revienta la
lectura del CSV con un error de "tokenizing". Montar Drive evita ese problema.

1. Sube `cms-2025.csv` y `google-analytics-2025.csv` a tu Google Drive, en
   `Mi unidad/proyecto_final/datos/`.
2. Abre `notebook_analisis.ipynb` en [colab.research.google.com](https://colab.research.google.com).
3. **Entorno de ejecución → Ejecutar todas.** El notebook detecta que está en Colab,
   instala `shap`/`sentence-transformers` solo, y monta tu Drive (autorízalo con la
   misma cuenta del paso 1) para leer los CSV desde ahí.
4. Si usaste otra ruta de Drive, ajusta `DATA_DIR = Path(...)` en la primera celda.
5. Opcional, para probar recomendaciones con IA: `os.environ["GROQ_API_KEY"] = "..."`
   en una celda antes de correr el resto.

## Despliegue

- **Dónde**: Streamlit Community Cloud.
- **Punto de entrada**: `app.py`, rama `main` de este repositorio.
- **Entorno**: se construye con `requirements.txt` (dependencias) y `runtime.txt`
  (versión de Python).
- **Tema visual**: `.streamlit/config.toml`.
- **Secretos**: `GROQ_API_KEY` se configura aparte, en **Settings → Secrets** del
  panel de la app (no vive en ningún archivo del repositorio).
- **Actualizaciones**: cada `git push` a `main` redespliega la app automáticamente.
