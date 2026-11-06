# Delhivery Last-Mile Delay Intelligence

## 1. Project Overview

This project analyzes last-mile delivery data to identify shipment delays, predict delivery risks, segment delivery hubs, and recommend operational actions.

The project uses Python, Pandas, NumPy, Scikit-learn, and Machine Learning techniques.

### Project Objectives
- Clean and analyze shipment data.
- Identify factors affecting delivery delays.
- Predict delivery time using regression models.
- Classify shipments based on delay risk.
- Segment delivery hubs into operational groups.
- Generate actionable recommendations for new shipments.

## 2. Project Structure

```text
Delhivery-Delay-Intelligence/
│
├── data/
├── src/
│   ├── shipment_prep.py
│   ├── part_a_clean_eda.py
│   ├── part_b_regression.py
│   ├── part_c_classification.py
│   ├── part_d_segments.py
│   ├── package_model.py
│   └── predict_delay.py
│
├── models/
├── outputs/
├── generate_data.py
├── check_data.py
├── test_clean.py
├── test_features.py
├── requirements.txt
└── README.md
```

## 3. How to Run

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
cd Delhivery-Delay-Intelligence
```

### Step 2: Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the project scripts

```bash
python src/part_a_clean_eda.py
python src/part_b_regression.py
python src/part_c_classification.py
python src/part_d_segments.py
python src/package_model.py
python src/predict_delay.py
```

Run the scripts in this order because later stages depend on earlier outputs.

## 4. Data Findings

- Raw shipment records: 1,224
- Cleaned shipment records: 1,200
- Duplicate records removed: 24
- Delivery delay rate: 23.17%
- Delivery delays increase with shipment distance.
- Tier 3 hubs show a higher delay rate than Tier 1 hubs.
- Economy shipments have a lower observed delay rate than Express shipments in this dataset.

### Important Correlations

- Delivery hours vs promised hours: 0.837
- Delivery hours vs distance: 0.401
- Delay status vs distance: 0.545

## 5. Model Results

### Regression

Selected model: Linear Regression

| Metric | Test Result |
|---|---:|
| MAE | 3.54 |
| MSE | 20.19 |
| RMSE | 4.49 |
| R² | 0.950 |

### Classification

Selected model: Random Forest Classifier

Selected threshold: 0.57

| Metric | Test Result |
|---|---:|
| Accuracy | 0.8833 |
| Precision | 0.7500 |
| Recall | 0.7317 |
| F1 Score | 0.7407 |

The classification model identifies shipments that are likely to be delayed.

## 6. Hub Segmentation

K-Means clustering was used to divide 24 delivery hubs into three operational segments.

| Segment | Hubs | Average Delay Rate |
|---|---:|---:|
| Standard | 7 | 2.57% |
| Watch | 8 | 15.50% |
| Escalate | 9 | 46.00% |

The segmentation helps prioritize operational attention based on hub performance.

## 7. New Shipment Predictions

The prediction pipeline processes 48 new shipments.

### Prediction Summary

| Recommended Action | Shipments |
|---|---:|
| Normal monitoring | 42 |
| Priority intervention | 5 |
| Monitor and alert | 1 |

Output file:

`outputs/new_shipment_actions.csv`

The output contains shipment details, predicted delay probability, predicted delay status, hub segment, and recommended action.

## 8. Operations Recommendation

- Standard hubs should continue normal monitoring.
- Watch hubs should receive closer performance monitoring.
- Escalate hubs should receive priority operational intervention.
- High-risk shipments should be identified early.
- Delivery teams should investigate long-distance routes and high-stop-count shipments.

## 9. Production Plan

For production deployment:

- Integrate the prediction pipeline with shipment tracking systems.
- Schedule regular model monitoring.
- Monitor prediction accuracy and data drift.
- Retrain the model when performance declines.
- Provide operational dashboards for hub managers.
- Monitor false positives and false negatives.

## 10. Limitations

- The project uses a generated dataset for analysis and modelling.
- Model performance on real-world shipment data may differ.
- Hub segmentation is based on historical shipment performance.
- Prediction quality depends on the quality of incoming shipment data.
- Further validation is required before real-world deployment.

## 11. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- VS Code
- Git and GitHub

---

**Project:** Delhivery Last-Mile Delay Intelligence  
**Purpose:** Shipment delay prediction, hub segmentation, and operational decision support.