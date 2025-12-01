# Multi-Target Prediction System - Quick Reference

## 🚀 Usage Commands

### Training Models

```bash
# Train PM2.5 models (default, backwards compatible)
python main.py

# Train Ozone models with all horizons
python main.py --target ozone

# Train NO₂ with specific horizons
python main.py --target nitrogen_dioxide --horizons 1,12,24

# Train CO for long-term forecasts only
python main.py --target carbon_monoxide --horizons 72,168

# Train PM10
python main.py --target pm10
```

### Making Predictions

```bash
# Predict PM2.5 (default)
python -m src.predict

# Predict Ozone
python -m src.predict --target ozone

# Predict NO₂ with specific horizons
python -m src.predict --target nitrogen_dioxide --horizons 1,24

# Predict CO
python -m src.predict --target carbon_monoxide
```

### Get Help

```bash
# Training help
python main.py --help

# Prediction help
python -m src.predict --help
```

---

## 📊 Available Targets

| Target | Variable Name | Description |
|--------|---------------|-------------|
| PM2.5 | `pm2_5` | Fine particulate matter (<2.5 μm) |
| PM10 | `pm10` | Coarse particulate matter (<10 μm) |
| Ozone | `ozone` | Ground-level ozone (O₃) |
| NO₂ | `nitrogen_dioxide` | Nitrogen dioxide from combustion |

---

## ⏰ Available Horizons

| Horizon | Time Ahead | Use Case |
|---------|------------|----------|
| 1h | 1 hour | Immediate/nowcasting |
| 12h | 12 hours | Half-day forecast |
| 24h | 24 hours | Daily forecast |
| 72h | 3 days | Short-term planning |
| 168h | 1 week | Long-term planning |

---

## 📁 File Outputs

### Models Directory
```
models/
├── xgboost_{target}_{horizon}h.json
```

**Examples:**
- `xgboost_pm2_5_24h.json` - PM2.5 24-hour forecast
- `xgboost_ozone_1h.json` - Ozone 1-hour forecast  
- `xgboost_nitrogen_dioxide_12h.json` - NO₂ 12-hour forecast

### Processed Data
```
data/processed/
├── train_data_{target}.csv
```

**Examples:**
- `train_data_pm2_5.csv` - PM2.5 training data with features
- `train_data_ozone.csv` - Ozone training data with features

---

## 🔑 Key Features

### For Each Target, the System Creates:

1. **Lag Features**: `{target}_lag_1`, `{target}_lag_24`
2. **Rolling Statistics**: `{target}_rolling_mean_24`, `{target}_rolling_std_24`
3. **Cross-Pollutant Features**: All other pollutants as predictors
4. **Meteorological Features**: Temperature, humidity, wind, pressure
5. **Time Features**: Cyclical hour and month encodings

### Example for Ozone:
- **Lags**: `ozone_lag_1`, `ozone_lag_24`
- **Rolling**: `ozone_rolling_mean_24`, `ozone_rolling_std_24`
- **Cross-pollutants**: `pm2_5`, `pm10`, `nitrogen_dioxide`, `carbon_monoxide`
- **Meteo**: `temperature_2m`, `relative_humidity_2m`, `wind_u`, `wind_v`, etc.
- **Time**: `hour_sin`, `hour_cos`, `month_sin`, `month_cos`

---

## 💡 Common Workflows

### Workflow 1: Compare Pollutants

Train models for all pollutants and compare:

```bash
python main.py --target pm2_5 --horizons 1,24
python main.py --target ozone --horizons 1,24
python main.py --target nitrogen_dioxide --horizons 1,24
python main.py --target pm10 --horizons 1,24
```

Then compare metrics (R², MAE, RMSE) to see which pollutants are easier to predict.

### Workflow 2: Focus on Short-Term Forecasts

```bash
python main.py --target ozone --horizons 1,12,24
python -m src.predict --target ozone --horizons 1,12,24
```

### Workflow 3: Batch Training Script

Create `train_all.sh`:
```bash
#!/bin/bash
for target in pm2_5 ozone nitrogen_dioxide pm10; do
    echo "====== Training $target ======"
    python main.py --target $target
done
```

---

## ⚠️ Troubleshooting

### Error: "Model not found"
**Solution:** Train the model first
```bash
python main.py --target ozone
```

### Error: "Missing features in data"
**Solution:** Ensure data is processed with the correct target
```bash
# Delete old processed data and retrain
rm data/processed/train_data_ozone.csv
python main.py --target ozone
```

### Error: "Target not in available targets"
**Solution:** Use one of the valid targets:
- `pm2_5`, `pm10`, `ozone`, `nitrogen_dioxide`

---

## 📈 Performance Metrics

After training, you'll see:

```
================================================================================
  MÉTRICAS EN TEST SET - OZONE (DATOS NO VISTOS)
================================================================================
Horizonte            | MAE      | RMSE     | R2       | MAPE     | Corr     | Skill   
----------------------------------------------------------------------------------------------------
ozone_1h             | 5.234    | 7.891    | 0.8567   | 12.45%   | 0.9234   | +45.23%
ozone_12h            | 8.123    | 11.234   | 0.7234   | 18.90%   | 0.8567   | +32.45%
ozone_24h            | 10.456   | 14.567   | 0.6123   | 23.67%   | 0.7891   | +25.67%
```

**Metrics Explained:**
- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **R²**: Coefficient of determination (higher is better, max 1.0)
- **MAPE**: Mean Absolute Percentage Error
- **Corr**: Correlation coefficient (higher is better, max 1.0)
- **Skill**: Improvement over persistence baseline

---

## 🎯 Best Practices

1. **Start with short horizons** (1h, 24h) before training long-term forecasts
2. **Compare with baseline** - Check that Skill score is positive
3. **Monitor R² scores** - Should be >0.5 for reliable predictions
4. **Use consistent horizons** - Train and predict with same horizons
5. **Keep data updated** - Retrain when new data becomes available

---

## 🔗 Related Files

- **Implementation Plan**: [implementation_plan.md](file:///C:/Users/mati9/.gemini/antigravity/brain/7f5b4711-f504-4e49-8b2f-998f62ef981e/implementation_plan.md)
- **Full Walkthrough**: [walkthrough.md](file:///C:/Users/mati9/.gemini/antigravity/brain/7f5b4711-f504-4e49-8b2f-998f62ef981e/walkthrough.md)
- **Task Checklist**: [task.md](file:///C:/Users/mati9/.gemini/antigravity/brain/7f5b4711-f504-4e49-8b2f-998f62ef981e/task.md)

---

**Ready to predict any pollutant! 🚀**
