# newshubmx — Sistema Predictivo de Apoyo a la Toma de Decisiones Editoriales

Dashboard de Streamlit que predice, antes de publicar, si una nota va a tener bajo
desempeño de tráfico, y explica por qué en lenguaje editorial (SHAP), con
recomendaciones generadas por LLM y respaldo automático basado en reglas.

## Demo en vivo

**https://sistema-predictivo-news-2cq3rxqzqtpaxhdm49rf7d.streamlit.app/**

Para probar la vista de "Notas" (importar una nota real desde un feed RSS en vez de
escribirla a mano), puedes usar este feed de ejemplo:

```
http://jarochilandia-com.ntc4-p2stl.ezhostingserver.com/rss/noticias.xml
```

## Estructura del repositorio

```
app.py                     Dashboard de Streamlit (la app en producción)
notebook_analisis.ipynb    Pipeline completo: EDA, procesamiento, modelación, SHAP
modelo_riesgo.joblib       Modelo entrenado (XGBoost) que carga la app
datos/                     Datos pequeños que usa la app (histórico agregado, autores)
logo/, rss/                Assets del dashboard
```

## Datos

Los archivos que usa la app (`datos/autores.json`, `datos/historico_categoria.csv`,
`datos/mejor_hora_categoria.json`) ya están en el repositorio.

Los datos **crudos** (`cms-2025.csv`, `google-analytics-2025.csv`,
`articulos_limpio.csv`) **no están en git** por su tamaño (más de 130 MB en total) —
se suben aparte al Classroom del curso. Para volver a correr el notebook desde cero,
descárgalos de ahí y colócalos dentro de la carpeta `datos/` del proyecto, con esos
mismos nombres.

## Cómo correrlo en local

```bash
pip install -r requirements.txt
GROQ_API_KEY=tu_api_key streamlit run app.py
```

La `GROQ_API_KEY` es opcional: sin ella, la app usa automáticamente su plan de
respaldo basado en reglas para las recomendaciones.

La clave real se comparte en un `.txt` en el Drive del Classroom del curso, junto con
los archivos de datos crudos mencionados arriba (no se sube a git por seguridad).
