# anomaly-detection-dashboard

Dashboard interactivo en Streamlit que detecta transacciones anómalas en un flujo
sintético de pagos, usando **Isolation Forest** (scikit-learn).

## Estructura

```
src/
  data_processing.py   # generación de datos sintéticos + preprocesamiento
  model.py              # entrenamiento y scoring con Isolation Forest
  evaluate.py            # métricas y visualizaciones
  dashboard.py            # app Streamlit
  run_pipeline.py          # ejecuta todo el pipeline end-to-end (sin UI)
data/
  synthetic_transactions.csv   # dataset sintético generado (seed=42)
results/
  metrics.json
  isolation_forest_model.joblib
  figuras/
    score_distribution.png
    flagged_scatter.png
```

## Datos

`generate_synthetic_data` genera 5000 transacciones (`transaction_id`, `amount`, `hour`,
`location_code`, `frequency_last_24h`, `is_anomaly`), con un 2% de anomalías inyectadas
de forma controlada:

- Montos atípicamente altos (USD 1500–8000) en cualquier horario.
- Patrones de hora/frecuencia inusuales: transacciones de madrugada con ráfagas de
  frecuencia (15–40 transacciones en 24h) y monto moderado.

Las transacciones normales siguen `amount ~ lognormal(3.2, 0.6)` (rango típico
USD 10–100) y `hour` concentrada en horario laboral (media 13:30, desviación ~3h).

## Cómo correrlo

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt

# Pipeline completo (genera datos, entrena, evalúa, guarda figuras y modelo)
venv\Scripts\python src/run_pipeline.py

# Dashboard interactivo
venv\Scripts\streamlit run src/dashboard.py
```

## Resultados

Ejecutado sobre 5000 transacciones sintéticas (seed=42, contamination=0.02):

| Métrica   | Valor |
|-----------|-------|
| Precision | 0.98  |
| Recall    | 0.98  |
| F1-score  | 0.98  |

Con Isolation Forest y `contamination=0.02` calibrado exactamente a la tasa real de
anomalías inyectadas, el modelo recupera casi todas las anomalías reales (recall 0.98)
con muy pocos falsos positivos (precision 0.98). Los dos grupos de anomalías —montos
extremos y patrones de hora/frecuencia inusuales— son claramente separables en el
espacio de features escaladas, lo cual se refleja en la distribución de scores
(`results/figuras/score_distribution.png`): las anomalías reales se concentran en la
cola izquierda (scores negativos), con solapamiento mínimo con las transacciones
normales.

El dashboard permite ajustar `contamination` en vivo desde la barra lateral: valores
más altos que la tasa real (0.02) aumentan el recall marginalmente pero degradan la
precision al marcar transacciones normales como anómalas.
