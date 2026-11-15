ML-Driven PVT-Aware Leakage Estimation, 16 nm LP

Main deliverables:
1. transistor_datasets_80000.xlsx
   - Eight datasets
   - 80,000 SPICE transistor samples

2. transistor_spice_vs_ml_80000.xlsx
   - Four sheets with 20,000 rows each
   - SPICE and ML currents for Id, Ig, Is and Ib

3. three_bit_cla_200_corners.xlsx
   - 200 PVT corners
   - 128 vectors per corner
   - 25,600 CLA simulations

4. saved_models/
   - Four trained joblib model artifacts

5. train_part_a.py and mlfeat.py
   - Training, feature engineering and prediction logic

The saved models were created using the package versions listed in
requirements-original.txt.

To inspect the models:
python load_models.py
