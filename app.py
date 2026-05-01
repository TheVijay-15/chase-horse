"""
CHASE HORSE | Freight Intelligence Command Center
Streamlit Dashboard for Freight Rate Analysis - Final Version
"""
import streamlit as st
import base64
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="CHASE HORSE | Freight Vision Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# DATA FUNCTIONS
# =============================================

@st.cache_data
def load_and_clean(file):
    """Load and clean freight data"""
    df = pd.read_excel(file, parse_dates=["Timestamp"])
    df = df.copy()
    
    df = df.dropna(subset=['Last_L1_Rate'])
    df['Km'] = df['Km'].fillna(df['Km'].median())
    df['L1_Last_Bid'] = df['L1_Last_Bid'].fillna(df['Last_L1_Rate'])
    
    q1 = df['Last_L1_Rate'].quantile(0.05)
    q3 = df['Last_L1_Rate'].quantile(0.95)
    df = df[(df['Last_L1_Rate'] >= q1) & (df['Last_L1_Rate'] <= q3)]
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    df['Year'] = df['Timestamp'].dt.year
    return df


def get_kpi(df):
    """Calculate KPIs"""
    return {
        'records': len(df),
        'avg_rate': df['Last_L1_Rate'].mean(),
        'median_km': df['Km'].median(),
        'min_km': df['Km'].min(),
        'max_km': df['Km'].max(),
        'states': df['Origin state'].nunique(),
        'pincodes': df['Delivery Pincode'].nunique(),
        'capacities': df['Trucks_Capacity'].nunique(),
        'years': f"{df['Year'].min()}–{df['Year'].max()}"
    }

# =============================================
# CHART FUNCTIONS
# =============================================

def dark_layout(fig, title="", height=400):
    """Apply dark theme"""
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        font=dict(color='#e6edf3', size=12),
        height=height,
        title=dict(text=title, font=dict(size=16, color='#58a6ff')) if title else None,
        margin=dict(t=40 if title else 20, r=20, b=40, l=60),
        xaxis=dict(gridcolor='#21262d', zerolinecolor='#21262d'),
        yaxis=dict(gridcolor='#21262d', zerolinecolor='#21262d'),
        showlegend=True,
        legend=dict(font=dict(color='#8b949e'))
    )
    return fig


def chart_histogram(df):
    """Rate distribution with frequency line"""
    counts, bin_edges = np.histogram(df['Last_L1_Rate'], bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['Last_L1_Rate'], nbinsx=50,
        marker=dict(color='#58a6ff', line=dict(color='#0d1117', width=1)),
        opacity=0.8, name='L1 Rate'
    ))
    
    fig.add_trace(go.Scatter(
        x=bin_centers, y=counts, mode='lines',
        line=dict(color='#f78166', width=2),
        name='Frequency Line'
    ))
    return dark_layout(fig, "Bid Rate Distribution", 380)


def chart_monthly(df):
    """Monthly trend with adjusted Y-axis"""
    m = df.set_index('Timestamp')['Last_L1_Rate'].resample('ME').mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=m.index, y=m.values, mode='lines+markers',
        line=dict(color='#3fb950', width=2.5),
        fill='tozeroy', fillcolor='rgba(63,185,80,0.1)',
        marker=dict(size=4), name='Avg Rate'
    ))
    fig.update_layout(yaxis=dict(range=[0, 40000]))
    return dark_layout(fig, "Monthly Average Rate", 380)


def chart_states(df):
    """State volume chart"""
    s = df.groupby("Origin state")['Last_L1_Rate'].count().sort_values(ascending=True).tail(12)
    fig = go.Figure(go.Bar(
        y=s.index, x=s.values, orientation='h',
        marker=dict(color='#58a6ff'),
        text=s.values, textposition='outside',
        textfont=dict(color='#e6edf3', size=10),
        name='Trips'
    ))
    fig.add_trace(go.Scatter(
        y=s.index, x=s.values, mode='lines+markers',
        line=dict(color='#f78166', width=2),
        marker=dict(size=6), name='Trend'
    ))
    return dark_layout(fig, "Top Origin States by Volume", 420)


def chart_capacity(df):
    """Truck capacity vs rate"""
    c = df.groupby('Trucks_Capacity')['Last_L1_Rate'].median().sort_values(ascending=False).head(10)
    colors = ['#58a6ff','#3fb950','#f78166','#d2a8ff','#ffa657',
              '#79c0ff','#56d364','#ff7b72','#bc8cff','#e3b341']
    fig = go.Figure(go.Bar(
        x=c.index, y=c.values, marker=dict(color=colors[:len(c)]),
        text=[f"₹{v:,.0f}" for v in c.values], textposition='outside',
        textfont=dict(color='#e6edf3', size=10),
        name='Median Rate'
    ))
    fig.add_trace(go.Scatter(
        x=c.index, y=c.values, mode='lines+markers',
        line=dict(color='#e6edf3', width=1, dash='dot'),
        marker=dict(size=4), name='Frequency'
    ))
    return dark_layout(fig, "Median Rate by Truck Capacity", 380)


def chart_distance(df):
    """Rate by Distance Bins"""
    df['KmBin'] = pd.cut(df['Km'], bins=10)
    g = df.groupby('KmBin', observed=True)['Last_L1_Rate'].agg(['mean','sem'])
    
    fig = go.Figure(go.Bar(
        x=[str(b) for b in g.index], y=g['mean'],
        marker=dict(color='#58a6ff'),
        error_y=dict(type='data', array=g['sem']*1.96, visible=True, color='#8b949e'),
        name='Avg Rate'
    ))
    fig.add_trace(go.Scatter(
        x=[str(b) for b in g.index], y=g['mean'], mode='lines+markers',
        line=dict(color='#3fb950', width=2),
        marker=dict(size=6), name='Trend Line'
    ))
    fig.update_xaxes(tickangle=45)
    return dark_layout(fig, "Rate by Distance Bins", 380)


def chart_india_map(df):
    """India choropleth map with all states"""
    state_data = df.groupby('Origin state')['Last_L1_Rate'].count().reset_index()
    state_data.columns = ['State', 'Volume']
    
    name_map = {
        'RAJASTHAN':'Rajasthan','GUJARAT':'Gujarat','MAHARASHTRA':'Maharashtra',
        'UTTAR PRADESH':'Uttar Pradesh','WEST BENGAL':'West Bengal',
        'TAMIL NADU':'Tamil Nadu','KARNATAKA':'Karnataka','HARYANA':'Haryana',
        'PUNJAB':'Punjab','MADHYA PRADESH':'Madhya Pradesh','BIHAR':'Bihar',
        'ANDHRA PRADESH':'Andhra Pradesh','ODISHA':'Odisha','KERALA':'Kerala',
        'ASSAM':'Assam','JHARKHAND':'Jharkhand','CHHATTISGARH':'Chhattisgarh',
        'DELHI':'Delhi','TELANGANA':'Telangana','UTTARAKHAND':'Uttarakhand',
        'HIMACHAL PRADESH':'Himachal Pradesh','JAMMU AND KASHMIR':'Jammu & Kashmir',
    }
    state_data['StateName'] = state_data['State'].map(name_map)
    
    all_states = [
        "Arunachal Pradesh", "Assam", "Chandigarh", "Karnataka", "Manipur", 
        "Meghalaya", "Mizoram", "Nagaland", "Punjab", "Rajasthan", "Sikkim", 
        "Tripura", "Uttarakhand", "Telangana", "Bihar", "Kerala", 
        "Madhya Pradesh", "Andaman & Nicobar", "Gujarat", "Lakshadweep", 
        "Odisha", "Dadra and Nagar Haveli and Daman and Diu", "Ladakh", 
        "Jammu & Kashmir", "Chhattisgarh", "Delhi", "Goa", "Haryana", 
        "Himachal Pradesh", "Jharkhand", "Tamil Nadu", "Uttar Pradesh", 
        "West Bengal", "Andhra Pradesh", "Puducherry", "Maharashtra"
    ]
    
    full_state_df = pd.DataFrame({'StateName': all_states})
    state_data = pd.merge(full_state_df, state_data, on='StateName', how='left').fillna(0)
    
    fig = go.Figure(go.Choropleth(
        geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
        featureidkey='properties.ST_NM',
        locations=state_data['StateName'], z=state_data['Volume'],
        colorscale='Viridis', colorbar_title='Trips',
        marker_line_color='#0d1117', marker_line_width=0.8,
        hovertemplate='<b>%{location}</b><br>Trips: %{z}<extra></extra>'
    ))
    
    fig.update_geos(
        fitbounds="locations", 
        visible=False,
        bgcolor='#0d1117',
        showcountries=False,
        showcoastlines=True,
        coastlinecolor="#21262d"
    )
    
    fig.update_layout(
        template='plotly_dark', height=600,
        paper_bgcolor='#0d1117', margin=dict(l=0,r=0,t=40,b=0),
        title=dict(
            text="Comprehensive Freight Volume Map of India", 
            font=dict(color='#58a6ff', size=20),
            x=0.5, xanchor='center'
        )
    )
    return fig


def chart_decomposition(df):
    """Seasonal decomposition"""
    ts = df.set_index('Timestamp')['Last_L1_Rate'].resample('W').median().dropna()
    if len(ts) < 104:
        fig = go.Figure()
        fig.add_annotation(text="Need 2+ years for decomposition", showarrow=False,
                          font=dict(size=16, color='#8b949e'))
        return dark_layout(fig, "Seasonal Decomposition", 400)
    
    decomp = seasonal_decompose(ts, model='additive', period=52)
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=['Observed','Trend','Seasonal','Residual'])
    
    items = [
        (decomp.observed, '#58a6ff'), (decomp.trend, '#3fb950'),
        (decomp.seasonal, '#f78166'), (decomp.resid, '#d2a8ff')
    ]
    for i, (data, color) in enumerate(items, 1):
        d = data.dropna()
        fig.add_trace(go.Scatter(x=d.index, y=d.values, line=dict(color=color, width=1.5), showlegend=False), row=i, col=1)
    
    fig.update_layout(
        template='plotly_dark', height=600,
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        font=dict(color='#e6edf3'),
        title=dict(text="Seasonal Decomposition (Weekly)", font=dict(size=16, color='#58a6ff'))
    )
    return fig


def chart_forecast(df):
    """Enhanced 12-week intelligent forecast with confidence bands and better visuals"""
    # Resample to weekly median
    ts = df.set_index('Timestamp')['Last_L1_Rate'].resample('W').median().dropna()
    
    if len(ts) < 26:  # Need at least 26 weeks for meaningful forecast
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ Insufficient data for seasonal forecast (need 26+ weeks)",
            showarrow=False, font=dict(size=14, color='#f78166'),
            x=0.5, y=0.5, xanchor='center'
        )
        return dark_layout(fig, "12-Week Intelligence Forecast", 450)
    
    # Use last 104 weeks (2 years) if available, or all data
    ts_limited = ts if len(ts) <= 104 else ts.iloc[-104:]
    values = ts_limited.values
    dates = ts_limited.index
    
    # Apply Holt-Winters Exponential Smoothing with seasonality
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    try:
        # Get seasonality period (52 weeks for yearly cycle, adjust if needed)
        period = min(52, len(values) // 2)
        
        model = ExponentialSmoothing(
            values, 
            seasonal_periods=period,
            trend='add',
            seasonal='add',
            initialization_method='estimated'
        )
        fitted_model = model.fit()
        
        # Forecast 12 weeks
        forecast_steps = 12
        forecast = fitted_model.forecast(forecast_steps)
        
        # Get prediction intervals (via simulation)
        residuals = fitted_model.resid
        std_residual = np.std(residuals) if len(residuals) > 0 else np.std(values) * 0.1
        
        # Generate future dates
        last_date = dates[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=forecast_steps, freq='W')
        
        # Confidence intervals (95%)
        z_score = 1.96
        upper_bound = forecast + z_score * std_residual
        lower_bound = forecast - z_score * std_residual
        lower_bound = np.maximum(lower_bound, 0)  # Rates can't be negative
        
    except Exception as e:
        # Fallback to simpler method if Holt-Winters fails
        from sklearn.linear_model import LinearRegression
        
        X = np.arange(len(values)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(X, values)
        trend = lr.predict(X)
        seasonal = values - trend
        
        # Calculate forecast
        future_X = np.arange(len(values), len(values) + forecast_steps).reshape(-1, 1)
        fc_trend = lr.predict(future_X)
        
        # Repeat seasonal pattern
        seasonal_pattern = seasonal[-period:] if len(seasonal) >= period else seasonal
        fc_seasonal = np.tile(seasonal_pattern, forecast_steps // len(seasonal_pattern) + 1)[:forecast_steps]
        forecast = fc_trend + fc_seasonal
        
        std_residual = np.std(seasonal)
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=forecast_steps, freq='W')
        upper_bound = forecast + z_score * std_residual
        lower_bound = np.maximum(forecast - z_score * std_residual, 0)
    
    # Create enhanced visualization
    fig = go.Figure()
    
    # 1. Historical Data with area fill
    fig.add_trace(go.Scatter(
        x=dates[-52:], 
        y=values[-52:],
        mode='lines',
        name='Historical (Last 52 weeks)',
        line=dict(color='#58a6ff', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(88,166,255,0.08)',
        hovertemplate='Week: %{x|%b %d, %Y}<br>Rate: ₹%{y:,.0f}<extra></extra>'
    ))
    
    # 2. Forecast confidence band (95% interval)
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_dates, future_dates[::-1]]),
        y=np.concatenate([upper_bound, lower_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(240,136,62,0.25)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval',
        hoverinfo='skip',
        showlegend=True
    ))
    
    # 3. Inner confidence band (50% interval)
    inner_upper = forecast + 0.67 * std_residual
    inner_lower = np.maximum(forecast - 0.67 * std_residual, 0)
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_dates, future_dates[::-1]]),
        y=np.concatenate([inner_upper, inner_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(240,136,62,0.5)',
        line=dict(color='rgba(255,255,255,0)'),
        name='50% Confidence Interval',
        hoverinfo='skip',
        showlegend=True
    ))
    
    # 4. Forecast line (main prediction)
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast,
        mode='lines+markers',
        name='📈 Smart Forecast',
        line=dict(color='#f0883e', width=3.5, shape='spline'),
        marker=dict(
            size=10, 
            color='#f0883e', 
            symbol='diamond',
            line=dict(color='#0d1117', width=2)
        ),
        hovertemplate='<b>📊 Forecast</b><br>Date: %{x|%b %d, %Y}<br>Predicted: ₹%{y:,.0f}<extra></extra>'
    ))
    
    # 5. Add moving average line (optional - shows trend direction)
    ma = pd.Series(values).rolling(window=4, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=dates[-min(52, len(ma)):],
        y=ma[-min(52, len(ma)):],
        mode='lines',
        name='4-Week MA (Trend)',
        line=dict(color='rgba(63,185,80,0.6)', width=1.5, dash='dot'),
        hovertemplate='Trend: ₹%{y:,.0f}<extra></extra>'
    ))
    
    # Add vertical line separating historical and forecast
    fig.add_vline(
        x=dates[-1], 
        line_width=2, 
        line_dash="dash", 
        line_color="#f78166",
        opacity=0.7
    )
    
    # Add annotation for forecast start
    fig.add_annotation(
        x=dates[-1], 
        y=0.98, 
        yref="paper",
        text="🔮 FORECAST BEGINS",
        showarrow=False,
        font=dict(size=10, color="#f78166", weight="bold"),
        xanchor='right',
        xshift=-10
    )
    
    # Add current rate indicator
    current_rate = values[-1]
    fig.add_hline(
        y=current_rate,
        line_dash="dot",
        line_color="#8b949e",
        opacity=0.5,
        annotation_text=f"Current: ₹{current_rate:,.0f}",
        annotation_position="bottom right",
        annotation_font_size=10
    )
    
    # Calculate forecast summary stats
    forecast_change = ((forecast[-1] - current_rate) / current_rate) * 100
    forecast_trend = "📈 Rising" if forecast_change > 3 else "📉 Falling" if forecast_change < -3 else "➡️ Stable"
    trend_color = "#3fb950" if forecast_change > 3 else "#f78166" if forecast_change < -3 else "#d2a8ff"
    
    # Add forecast summary as annotation
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"<b>12-Week Outlook</b><br>{forecast_trend} ({forecast_change:+.1f}%)<br>Forecast High: ₹{forecast.max():,.0f}<br>Forecast Low: ₹{forecast.min():,.0f}",
        showarrow=False,
        font=dict(size=11, color="#e6edf3"),
        align="left",
        bgcolor="rgba(13,17,23,0.8)",
        borderpad=8,
        borderwidth=1,
        bordercolor="#58a6ff"
    )
    
    # Update layout with better formatting
    fig.update_layout(
        yaxis=dict(
            title="<b>Rate (₹)</b>",
            gridcolor='#21262d',
            zerolinecolor='#21262d',
            tickformat=',.0f',
            range=[0,50000]
        ),
        xaxis=dict(
            title="<b>Timeline</b>",
            gridcolor='#21262d',
            tickangle=-45
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(13,17,23,0.7)",
            bordercolor="#21262d",
            borderwidth=1
        )
    )
    
    return dark_layout(fig, "12-Week Intelligence Forecast", 500)

def chart_ml(df):
    """ML model"""
    try:
        df2 = df.dropna(subset=['Km','Last_L1_Rate','Origin state','Trucks_Capacity','L1_Last_Bid']).copy()
        le1 = LabelEncoder()
        le2 = LabelEncoder()
        df2['se'] = le1.fit_transform(df2['Origin state'].astype(str))
        df2['ce'] = le2.fit_transform(df2['Trucks_Capacity'].astype(str))
        
        X = df2[['Km','se','ce','L1_Last_Bid']]
        y = df2['Last_L1_Rate']
        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(Xt, yt)
        preds = model.predict(Xv)
        
        r2 = max(0, stats.pearsonr(yv, preds)[0]**2)
        mae = np.mean(np.abs(yv-preds))
        
        fi = pd.DataFrame({
            'Feature': ['Distance','State','Capacity','L1 Bid'],
            'Imp': model.feature_importances_
        }).sort_values('Imp')
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=yv, y=preds, mode='markers', marker=dict(color='#58a6ff', size=5, opacity=0.5), name='Predictions'))
        mx = max(yv.max(), preds.max())
        fig1.add_trace(go.Scatter(x=[0,mx], y=[0,mx], mode='lines', line=dict(color='#3fb950', dash='dash'), name='Perfect Fit'))
        
        fig2 = go.Figure(go.Bar(y=fi['Feature'], x=fi['Imp'], orientation='h', marker=dict(color=['#f78166','#ffa657','#58a6ff','#3fb950']), name='Importance'))
        
        return dark_layout(fig1, f"Actual vs Predicted (R²={r2:.2f})", 380), dark_layout(fig2, "Feature Importance", 380), mae, r2
    except:
        f = go.Figure()
        f.add_annotation(text="Need more data for ML", showarrow=False, font=dict(color='#8b949e'))
        return dark_layout(f, "", 300), dark_layout(f, "", 300), 0, 0


def chart_correlation(df):
    """Correlation heatmap"""
    cols = ['Km','Last_L1_Rate','L1_Last_Bid','L2_Last_Bid','L3_Last_Bid']
    available_cols = [c for c in cols if c in df.columns]
    corr = df[available_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0,'#1a2332'],[0.5,'#58a6ff'],[1,'#f78166']],
        text=np.round(corr.values,2), texttemplate='%{text}',
        textfont=dict(color='white', size=12)
    ))
    return dark_layout(fig, "Correlation Matrix", 400)


# =============================================
# SIDEBAR
# =============================================
# Load and encode the image
logo_path = r"C:\Users\vijay\OneDrive\Desktop\chase_horse.png"
with open(logo_path, "rb") as img_file:
    b64_logo = base64.b64encode(img_file.read()).decode()

with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:15px 0;'>
        <img src="data:image/png;base64,{b64_logo}" 
             style='width:100%; max-width:220px; height:auto; display:block; margin:0 auto 10px;'>
        <h2 style='color:#58a6ff;margin:5px 0;'>CHASE HORSE</h2>
        <p style='color:#8b949e;font-size:0.8em;'>Freight Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    # ---- DATA LOADING ----
    uploaded = st.file_uploader("📂 Upload Excel File (optional)", type=['xlsx','xls'])
    
    if uploaded is not None:
        df_all = load_and_clean(uploaded)
        st.success(f"{len(df_all):,} records loaded from upload")
    else:
        # Auto-load from repo
        try:
            df_all = load_and_clean("data/Freight_Reconcile_Final.xlsx")
            st.success(f"{len(df_all):,} records auto-loaded")
        except:
            st.error("No data file found. Please upload an Excel file.")
            df_all = None
    
    if df_all is not None:
        st.divider()
        st.markdown("### Filters")
        
        years = sorted(df_all['Year'].unique())
        sel_years = st.multiselect("Year", years, default=years)
        
        states_list = sorted([s for s in df_all['Origin state'].unique() if isinstance(s, str)])
        sel_states = st.multiselect("State", states_list, default=states_list[:3])
        
        df = df_all.copy()
        if sel_years:
            df = df[df['Year'].isin(sel_years)]
        if sel_states:
            df = df[df['Origin state'].isin(sel_states)]
    else:
        df = None


# =============================================
# MAIN CONTENT
# =============================================
# Load and encode the logo
logo_path = r"C:\Users\vijay\OneDrive\Desktop\chase_horse.png"
with open(logo_path, "rb") as img_file:
    b64_logo = base64.b64encode(img_file.read()).decode()

st.markdown(f"""
<div style='display:flex;justify-content:space-between;align-items:center;padding:15px 0;'>
    <div style='display:flex;align-items:center;gap:12px;'>
        <img src="data:image/png;base64,{b64_logo}" 
             style='width:60px;height:60px;object-fit:contain;vertical-align:middle;'>
        <h1 style='color:#e6edf3;margin:0;font-size:1.8em;'>Freight Vision Dashboard</h1>
    </div>
    <span style='color:#3fb950;font-size:0.9em;'>● Live Dashboard</span>
</div>
""", unsafe_allow_html=True)

if df is not None:
    kpi = get_kpi(df)
    
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1:
        st.metric("📊 Total Records", f"{kpi['records']:,}")
    with c2:
        st.metric("💰 Avg Rate", f"₹{kpi['avg_rate']:,.0f}")
    with c3:
        st.metric("🚛 Distance", f"{kpi['min_km']:.0f}–{kpi['max_km']:.0f} km")
    with c4:
        st.metric("📍 States", kpi['states'])
    with c5:
        st.metric("🎯 Pincodes", f"{kpi['pincodes']:,}")
    with c6:
        st.metric("📅 Period", kpi['years'])
    
    tabs = st.tabs([
        "📈 Overview", "🗺️ India Map", "📊 Segments", 
        "🔮 Forecast", "🤖 ML Model", "📋 Data", "💡 Insights"
    ])
    
    # OVERVIEW
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_histogram(df), use_container_width=True)
            st.info("This histogram visualizes the distribution of freight rates across all transactions. The frequency line highlights the density of specific bid price points.")
        with c2:
            st.plotly_chart(chart_monthly(df), use_container_width=True)
            st.info("The monthly average rate chart tracks pricing fluctuations over time. The Y-axis is scaled to 10k to provide a consistent perspective on cost ceilings.")
        
        st.plotly_chart(chart_states(df), use_container_width=True)
        st.info("This bar chart ranks origin states by their total trip volume. The integrated frequency line highlights the volume trend across the top logistics hubs.")
    
    # INDIA MAP
    with tabs[1]:
        st.plotly_chart(chart_india_map(df), use_container_width=True)
        st.success("""
        **🗺️ Geospatial Insights**
        The map displays freight volume across all Indian states, providing a comprehensive view of regional activity. 
        Darker areas indicate higher logistics density, helping pinpoint where your operations are most concentrated.
        """)
        
        st.markdown("### 🔥 Top Routes")
        df['Route'] = df['Origin state'] + ' → ' + df['Delivery location'].str.split(',').str[0]
        top_r = df.groupby('Route')['Last_L1_Rate'].agg(['count','mean']).sort_values('count', ascending=False).head(8)
        for r, row in top_r.iterrows():
            st.markdown(f"- **{r}**: {int(row['count']):,} trips • ₹{row['mean']:,.0f}")
    
    # SEGMENTS
    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_capacity(df), use_container_width=True)
            st.info("This chart compares median freight rates against different truck capacities. It allows for cost-efficiency analysis across various fleet types.")
        with c2:
            st.plotly_chart(chart_distance(df), use_container_width=True)
            st.info("Rates are grouped into distance bins to show how travel length impacts cost. The trend line clarifies the correlation between mileage and pricing.")
    
    # FORECAST
    with tabs[3]:
        st.plotly_chart(chart_decomposition(df), use_container_width=True)
        st.info("Seasonal decomposition breaks down rates into trend, seasonal, and residual components. This provides a clear view of underlying market cycles.")
        
        st.plotly_chart(chart_forecast(df), use_container_width=True)
        st.info("The Intelligence Forecast predicts future rates with 95% confidence intervals. Historical data is focused on the last 52 weeks to maintain clarity.")
    
    # ML MODEL
    with tabs[4]:
        fig1, fig2, mae, r2 = chart_ml(df)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig1, use_container_width=True)
            st.info("This scatter plot compares the model's predicted rates against actual values. A tighter alignment with the dashed line indicates higher accuracy.")
        with c2:
            st.plotly_chart(fig2, use_container_width=True)
            st.info("Feature importance identifies which variables most influence the freight rate. This helps focus on the key drivers of logistics costs.")
        
        if mae > 0:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("MAE", f"₹{mae:,.0f}")
            c2.metric("R²", f"{r2:.2f}")
            c3.metric("Trees", "100")
            c4.metric("Features", "4")
    
    # DATA
    with tabs[5]:
        st.dataframe(df.head(30).style.format({'Last_L1_Rate':'₹{:,.0f}','Km':'{:.0f}'}), use_container_width=True, height=500)
        st.info("This table displays the raw transaction data for detailed review. It includes essential fields like rates, distances, and timestamps.")
        
        st.markdown("### Data Quality")
        miss = pd.DataFrame({'Column': df.columns, 'Missing%': (df.isnull().sum()/len(df)*100).values}).sort_values('Missing%', ascending=False)
        st.dataframe(miss, use_container_width=True)
        st.info("The data quality table monitors missing values across all columns. Maintaining high data integrity is crucial for reliable dashboard insights.")
    
    # INSIGHTS
    with tabs[6]:
        c1, c2 = st.columns([2,1])
        with c1:
            st.plotly_chart(chart_correlation(df), use_container_width=True)
            st.info("The correlation matrix shows the relationship strength between numerical variables. It reveals how factors like distance and bids move together.")
        with c2:
            try:
                r, _ = stats.pearsonr(df['Km'], df['Last_L1_Rate'])
                slope, _, _, _, _ = stats.linregress(df['Km'], df['Last_L1_Rate'])
                
                st.info(f"""
                **📌 Distance Impact**\n
                Correlation: r={r:.3f}\n
                Per 100km = ₹{slope*100:,.0f}
                """)
                st.success(f"""
                **🚛 Fleet Stats**\n
                Avg Distance: {df['Km'].mean():.0f} km\n
                Median Rate: ₹{df['Last_L1_Rate'].median():,.0f}
                """)
                st.warning(f"""
                **📊 Data Coverage**\n
                {df['Year'].min()} – {df['Year'].max()}\n
                {kpi['states']} states active
                """)
            except:
                st.error("Insufficient data for correlation analysis.")
else:
    st.markdown("""
    <div style='text-align:center;padding:100px 20px;'>
        <div style='font-size:80px;'>🚛</div>
        <h2 style='color:#e6edf3;'>CHASE HORSE | Freight Intelligence</h2>
        <p style='color:#8b949e;font-size:18px;'>Upload your Excel file using the sidebar to begin</p>
    </div>
    """, unsafe_allow_html=True)
