# 🚛 CHASE HORSE | Freight Intelligence Command Center

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?style=for-the-badge&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-blueviolet?style=for-the-badge&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A comprehensive freight rate analytics platform with interactive dashboards, ML-powered predictions, and India-wide route intelligence.**

</div>

---

## 📸 Dashboard Preview

<div align="center">

| Executive Overview | India Route Map | ML Predictions |
|:---:|:---:|:---:|
| KPI Cards, Rate Distribution, Monthly Trends | Interactive Choropleth Map | Random Forest Price Model |

</div>

---

## 🎯 Key Features

### 📈 Executive Overview
- **6 Real-time KPI Cards**: Total shipments, average rates, distance ranges, state coverage
- **Bid Rate Distribution**: Histogram showing winning bid patterns
- **Monthly Rate Trends**: Time series with trend visualization
- **Top Origin States**: Horizontal bar chart of highest-volume states

### 🗺️ Geographic Intelligence
- **Interactive India Choropleth Map**: State-wise freight volume visualization
- **Top Route Analysis**: Busiest and highest-value freight corridors
- **State-wise Breakdown**: Volume and average rate comparisons

### 📊 Segment Analysis
- **Truck Capacity vs Rate**: Median rates across 10+ truck capacities
- **Distance Bins with 95% CI**: Statistical rate analysis by distance segments
- **State Volume & Rate**: Dual-panel comparison charts

### 🔮 Time Series Forecasting
- **Seasonal Decomposition**: Trend, seasonal, and residual components
- **12-Week Rate Forecast**: Polynomial trend + seasonal pattern prediction
- **Confidence Visualization**: Clear forecast with historical context

### 🤖 Machine Learning Model
- **Random Forest Regressor**: 100-tree ensemble for price prediction
- **Actual vs Predicted**: Scatter plot with perfect prediction baseline
- **Feature Importance**: Which factors drive freight rates
- **Performance Metrics**: MAE and R² scores

### 📋 Data Explorer
- **Raw Data Table**: Sortable, paginated data preview
- **Data Quality Dashboard**: Missing value percentages by column
- **Completeness Pie Chart**: Overall data integrity visualization

### 💡 Business Insights
- **Correlation Matrix**: Multi-variable relationship heatmap
- **Pricing Driver Analysis**: Distance impact quantification
- **Seasonality Patterns**: Q4 peak identification
- **Fleet Optimization**: Capacity utilization recommendations

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/chase-horse.git
cd chase-horse

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the dashboard
streamlit run app.py