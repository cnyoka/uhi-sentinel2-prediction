# Urban Heat Island Prediction from Sentinel-2 Satellite Imagery

Machine learning pipeline that classifies urban heat island (UHI) intensity — Low, Medium, or High — from Sentinel-2 multispectral imagery and 3D building footprint data. Built during my MBAn at Hult International Business School (Spring 2026).

**Headline result: weighted F1 of 0.92 on the held-out test set, up from a 0.40 single-pixel baseline** — per-class F1 of 0.94 (High), 0.94 (Low), 0.88 (Medium).

## The problem

Urban heat islands make dense city areas up to 10°C hotter than their surroundings, and extreme heat is the leading weather-related cause of death globally. The challenge: train a model on ground-measured UHI data from Santiago, Chile and Rio de Janeiro, Brazil, then predict UHI intensity across 14,105 locations in Freetown, Sierra Leone — an unseen city on a different continent with different climate, vegetation, and urban form. A classic geospatial domain-transfer problem.

## Approach

**Multi-scale feature engineering (396 features).** Instead of sampling a single pixel per location, features were extracted from three concentric windows (50m, 100m, 200m) around each point, with four statistics per window (mean, std, 25th/75th percentile). 21 spectral indices were computed from the 12 Sentinel-2 bands — vegetation (NDVI, EVI, SAVI, MSAVI, GNDVI, NDVIre), built-up intensity (NDBI, UI, IBI, BRBA, BCI, NDISI), soil exposure (BSI, RI), water (NDWI, MNDWI, AWEInsh), and heat stress (NBR, NBR2) — plus 25 cross-domain interaction features combining spectral indices with building morphology (e.g., NDVI × building density for green space inside dense urban fabric).

**Cross-region robustness.** Two fixes proved critical for transferring from South America to West Africa: clamping the EVI denominator (which otherwise produced billion-scale values on some tropical pixels and destroyed scaling) and excluding atmosphere-only bands (B01, B09) that were encoding regional atmospheric differences rather than land surface signal.

**Model.** Soft-voting ensemble of three gradient boosting models — two LightGBM configurations (one Optuna-tuned over 50 Bayesian trials with 5-fold stratified CV) and one sklearn GradientBoostingClassifier. A high minimum-leaf-sample constraint (45) guards against memorizing region-specific patterns. Pseudo-labeling on the target city was attempted and correctly rejected by a prediction-distribution safety guard.

**Interpretation.** SHAP analysis ranked all features and drove the key findings below.

## Key findings

- **Regional context beats local surface type.** 200-metre-scale features consistently dominated SHAP importance over 50-metre features — a location's thermal environment is shaped more by what surrounds it than what's directly beneath it.
- **Bare soil exposure is the top predictor** (Redness Index at 200m), not building density. Exposed, unshaded ground acts as a heat radiator — which makes covering bare soil (ground cover, mulch, reflective coatings) one of the cheapest high-impact interventions available to city planners.
- **Water presence and variability rank second** (MNDWI), confirming the cooling effect of blue corridors and fragmented vs. continuous water access.

The notebook includes a full written report: intervention recommendations for planners (cool roofs, blue corridors, urban geometry, waste-heat reduction), proposed data additions (Landsat/MODIS LST, SRTM terrain, VIIRS nighttime lights), and a bibliography.

## Tech stack

Python · pandas · NumPy · rasterio · GeoPandas · Shapely · LightGBM · scikit-learn · Optuna · SHAP · matplotlib/seaborn

## Data

- **Sentinel-2 L2A imagery** — European Space Agency Copernicus mission (public)
- **3D-GloBFP building footprints** — Esch et al. 2024 (public)
- **Ground-truth UHI traversal measurements** — provided as course challenge materials and **not included in this repository**. Download links in the notebook have been redacted (`REDACTED_COURSE_FILE_ID`); the pipeline structure is fully visible and reproducible with equivalent data.

Feature extraction over the full rasters takes roughly 3–4 hours; the modeling steps (9–16) run in minutes once features are cached.

