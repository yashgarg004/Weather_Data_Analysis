# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import streamlit as st

# Streamlit page setup
st.set_page_config(page_title="Weather Data Dashboard", layout="wide")
st.title("🌦️ Weather Data Dashboard")


# %%
pd.read_csv('data/weather_dataset.csv')

# %%
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# %%
try:
    df = pd.read_csv('data/weather_dataset.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: File not found. Please ensure the dataset is in the data/ folder.")

# %%
df.head()

# %%
print("Dataset Info:")
df.info()

# %%
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().mean() * 100).round(2)

missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage (%)': missing_percentage
})

print("Missing Values Summary:")
missing_data[missing_data['Missing Values'] > 0]

# %%
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Values Heatmap', pad=20)
plt.show()


# %%
# Handle missing values
def handle_missing_values(data):
    """Handle missing values in the dataset"""
    # For temperature columns, fill with monthly mean by city
    temp_cols = ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean']
    for col in temp_cols:
        data[col] = data.groupby(['city_name', data['datetime'].dt.month])[col].transform(
            lambda x: x.fillna(x.mean()))
    
    # For precipitation, fill with 0 (assuming no rain)
    data['precipitation_sum'] = data['precipitation_sum'].fillna(0)
    
    # For wind speed, fill with overall mean by city
    data['wind_speed_10m_max'] = data.groupby('city_name')['wind_speed_10m_max'].transform(
        lambda x: x.fillna(x.mean()))
    
    return data


# %%
df['datetime'] = pd.to_datetime(df['datetime'])

# Apply missing value handling
df_clean = handle_missing_values(df.copy())

# Verify no missing values remain
print("Missing values after cleaning:")
print(df_clean.isnull().sum())

# %%
# Check and remove duplicates
print(f"Number of duplicates before cleaning: {df_clean.duplicated().sum()}")
df_clean = df_clean.drop_duplicates()
print(f"Number of duplicates after cleaning: {df_clean.duplicated().sum()}")

# %%
# Check data types
print("\nData types:")
print(df_clean.dtypes)

# Fix any incorrect data types (example - if city_name was numeric)
df_clean['city_name'] = df_clean['city_name'].astype('category')

# %%
selected_features = [
    'city_name',
    'datetime',
    'temperature_2m_max',
    'temperature_2m_min',
    'temperature_2m_mean',
    'precipitation_sum',
    'wind_speed_10m_max'
]

df_final = df_clean[selected_features].copy()
def engineer_features(data):
    """Create new features from existing data"""
    # Extract time-based features
    data['month'] = data['datetime'].dt.month
    data['year'] = data['datetime'].dt.year
    data['day_of_year'] = data['datetime'].dt.dayofyear


# %%
def engineer_features(data):
    """Create new features from existing data"""
    # Extract time-based features
    data['month'] = data['datetime'].dt.month
    data['year'] = data['datetime'].dt.year
    data['day_of_year'] = data['datetime'].dt.dayofyear
    data['season'] = data['month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Autumn')
    data['temperature_range'] = data['temperature_2m_max'] - data['temperature_2m_min']
    data['had_precipitation'] = data['precipitation_sum'] > 0
    
    return data

df_final = engineer_features(df_final)

df_final.head()
  





# %%
print(f"Number of duplicates: {df_final.duplicated().sum()}")

# %%
numeric_cols = ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean', 
                'precipitation_sum', 'wind_speed_10m_max']

print("Value ranges for numeric columns:")
for col in numeric_cols:
    print(f"{col}: Min={df_final[col].min()}, Max={df_final[col].max()}")

# %%
# Check city name consistency
print("\nUnique city names:")
print(df_final['city_name'].unique())

# %%
# %% [markdown]
## 5. Summary Statistics and Insights (Condensed)

# %%
# Set smaller default font size
plt.rcParams.update({'font.size': 8})

# Basic statistics
print("Overall Statistics:")
st.write(df_final[numeric_cols].describe().round(2))

# %%
# Grouped statistics - by city (compact display)
print("\nStatistics by City:")
city_stats = df_final.groupby('city_name')[numeric_cols].agg(['mean', 'median', 'std'])
st.write(city_stats.style.format("{:.2f}").set_table_styles(
    [{'selector': 'th', 'props': [('font-size', '8pt')]}]
))

# %%
# Grouped statistics - by season (compact display)
print("\nStatistics by Season:")
st.write(df_final.groupby('season')[numeric_cols].agg(['mean', 'median', 'std']).round(2))

# %%
# Compact correlation matrix
plt.figure(figsize=(6, 5))
sns.heatmap(df_final[numeric_cols].corr(), annot=True, cmap='coolwarm', 
            center=0, annot_kws={'size': 7}, fmt='.2f')
plt.title('Variable Correlations', fontsize=9, pad=10)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.show()

# %%
## 6. Patterns & Trends (Compact with Better Legend)

# %%
# Temperature trends with better legend placement
plt.figure(figsize=(9, 4))
for city in df_final['city_name'].unique():
    city_data = df_final[df_final['city_name'] == city]
    plt.plot(city_data['datetime'], city_data['temperature_2m_mean'], 
             linewidth=1, label=city)

plt.title('Temperature Trends by City', fontsize=9, pad=10)
plt.xlabel('Date', fontsize=7)
plt.ylabel('Temp (°C)', fontsize=7)
plt.legend(prop={'size': 6}, framealpha=0.5, 
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# Monthly Temperature Plot (compact with better legend)
plt.figure(figsize=(10, 5))
ax = sns.lineplot(data=df_final, x='month', y='temperature_2m_mean', hue='city_name',
                 ci=None, estimator='mean', marker='o', markersize=5, linewidth=1)
plt.title('Average Monthly Temperature by City', fontsize=10, pad=10)
plt.xlabel('Month', fontsize=8)
plt.ylabel('Mean Temperature (°C)', fontsize=8)
plt.xticks(range(1, 13), ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'],
          fontsize=7)
plt.yticks(fontsize=7)
plt.grid(True, alpha=0.3)

# Move legend outside and make it compact
plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left', 
          fontsize=7, title_fontsize=8, framealpha=0.5)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12, 5))  # Wider figure for better visibility

# Create the bar plot with consistent spacing
ax = sns.barplot(data=df_final, x='city_name', y='precipitation_sum',
                estimator=np.sum, ci=None, 
                palette="Blues_d",  # Darker palette for better visibility
                saturation=0.8,    # More vibrant colors
                width=0.7)         # Optimal bar width

# Improve visibility of all elements
plt.title('Total Precipitation by City', fontsize=12, pad=15, weight='bold')
plt.xlabel('City', fontsize=10, labelpad=10)
plt.ylabel('Total Precipitation (mm)', fontsize=10, labelpad=10)

# Format ticks and labels
plt.xticks(fontsize=9, rotation=45, ha='right')  # Better angled labels
plt.yticks(fontsize=9)
plt.grid(True, axis='y', alpha=0.2, linestyle='--')

# Ensure equal spacing between all bars
positions = range(len(df_final['city_name'].unique()))
ax.set_xticks([p for p in positions])
ax.set_xticklabels(df_final['city_name'].unique())

# Add value labels on top of each bar
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}", 
               (p.get_x() + p.get_width() / 2., p.get_height()),
               ha='center', va='center', 
               xytext=(0, 5), textcoords='offset points',
               fontsize=8)

# Adjust layout
ax.margins(x=0.03)  # Tight but consistent margins
plt.tight_layout()
plt.show()

# %%
# Anomaly Detection (compact output)
print("\nTemperature Anomalies (3σ from mean):")
for city in df_final['city_name'].unique():
    city_data = df_final[df_final['city_name'] == city]
    mean_temp = city_data['temperature_2m_mean'].mean()
    std_temp = city_data['temperature_2m_mean'].std()
    
    anomalies = city_data[
        (city_data['temperature_2m_mean'] > mean_temp + 3*std_temp) | 
        (city_data['temperature_2m_mean'] < mean_temp - 3*std_temp)
    ]
    
    if not anomalies.empty:
        print(f"\n{city}:")
        st.write(anomalies[['datetime', 'temperature_2m_mean']].style.format({
            'temperature_2m_mean': '{:.1f}°C',
            'datetime': lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else ''
        }).set_table_styles([{
            'selector': 'th',
            'props': [('font-size', '8pt')]
        }]))

# %%
# %%
# Enhanced Boxplot with better spacing and visibility
plt.figure(figsize=(14, 7))  # Wider figure for better spacing

# Create the boxplot with adjusted parameters
box = sns.boxplot(data=df_final, x='city_name', y='temperature_2m_mean',
                 palette="coolwarm",  # Color scheme
                 width=0.6,          # Adjust box width
                 linewidth=1.5,      # Thicker box lines
                 fliersize=4)        # Outlier marker size

# Enhance title and labels
plt.title('Temperature Distribution by City (with Outliers)', 
          fontsize=14, pad=20, weight='bold')
plt.xlabel('City', fontsize=12, labelpad=10)
plt.ylabel('Mean Temperature (°C)', fontsize=12, labelpad=10)

# Improve x-axis city labels
box.set_xticklabels(box.get_xticklabels(), 
                   rotation=45, 
                   ha='right', 
                   fontsize=10)

# Add horizontal grid lines
plt.grid(True, axis='y', alpha=0.3, linestyle='--')

# Adjust city spacing
plt.xlim(-0.8, len(df_final['city_name'].unique())-0.2)  # Wider spacing
box.margins(x=0.05)  # Add small margin on sides

# Add annotation for outliers
plt.annotate('Circles show temperature outliers',
            xy=(0.02, 0.95), xycoords='axes fraction',
            fontsize=10, color='gray')

plt.tight_layout()
plt.show()


# %%
# Detect outliers using IQR method
def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers
    temp_outliers = detect_outliers_iqr(df_final, 'temperature_2m_mean')
    print(f"Number of temperature outliers: {len(temp_outliers)}")

# %%
# Handle outliers - cap them at reasonable values
def cap_outliers(data, column):
    """Cap outliers at 1.5*IQR bounds"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    data[column] = data[column].clip(lower_bound, upper_bound)
    return data

df_final = cap_outliers(df_final, 'temperature_2m_mean')

# Verify outliers were handled
temp_outliers = detect_outliers_iqr(df_final, 'temperature_2m_mean')
print(f"Number of temperature outliers after capping: {len(temp_outliers)}")


# %%
# Log transformation for precipitation (many zeros)
df_final['precipitation_log'] = np.log1p(df_final['precipitation_sum'])

plt.figure(figsize=(12, 6))
sns.histplot(df_final['precipitation_log'], bins=30, kde=True)
plt.title('Log-Transformed Precipitation Distribution', pad=20)
plt.xlabel('Log(Precipitation + 1)')
plt.ylabel('Frequency')
plt.show()

# %%
## 8. Data Visualization

# %%
# Temperature distribution by season and city
plt.figure(figsize=(14, 8))
sns.boxplot(data=df_final, x='season', y='temperature_2m_mean', hue='city_name')
plt.title('Temperature Distribution by Season and City', pad=20)
plt.xlabel('Season')
plt.ylabel('Mean Temperature (°C)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%
# Wind speed vs. temperature
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_final, x='temperature_2m_mean', y='wind_speed_10m_max', 
                hue='city_name', alpha=0.6)
plt.title('Wind Speed vs. Temperature by City', pad=20)
plt.xlabel('Mean Temperature (°C)')
plt.ylabel('Max Wind Speed (km/h)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%
# Precipitation patterns by month
plt.figure(figsize=(14, 8))
sns.barplot(data=df_final, x='month', y='precipitation_sum', hue='city_name', 
            estimator=np.mean, ci=None)
plt.title('Average Monthly Precipitation by City', pad=20)
plt.xlabel('Month')
plt.ylabel('Average Precipitation (mm)')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# %%
# Temperature range analysis
plt.figure(figsize=(14, 8))
sns.lineplot(data=df_final, x='month', y='temperature_range', hue='city_name', 
             ci=None, estimator='mean', marker='o')
plt.title('Average Monthly Temperature Range by City', pad=20)
plt.xlabel('Month')
plt.ylabel('Temperature Range (°C)')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# %%
print("Key Findings:")
print("1. Temperature patterns vary significantly by city and season.")
print("2. The hottest months are typically June-August (Summer).")
print("3. Precipitation patterns differ across cities, with some showing distinct wet/dry seasons.")
print("4. Temperature ranges (difference between max and min) are highest in Spring and Autumn.")
print("5. Wind speeds tend to be higher during colder months in most cities.")

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## 📈 Temperature Trend Over Time
# This line chart shows how the temperature changes over time. It helps identify seasonal trends or anomalies.

# %%
# Import the required libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Create a sample dataframe with temperature data
# In a real scenario, you would load your data from a file or database
dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
temperatures = np.random.normal(20, 5, size=30)  # Random temperatures around 20°C
df = pd.DataFrame({'date': dates, 'temperature': temperatures})

# Now use the plotting functions
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='date', y='temperature')
plt.title('Daily Temperature Trends')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../images/temperature_trends.png')
plt.show()

# %% [markdown]
# ## 🌧️ Monthly Precipitation Analysis
# This bar chart displays average precipitation per month, providing insights into seasonal rainfall patterns.

# %%
date_range = pd.date_range(start='2021-01-01', end='2021-12-31', freq='D')
df = pd.DataFrame({
    'date': date_range,
    'precipitation': abs(np.random.randn(len(date_range)) * 10)
})

# Extract month from date
df['month'] = df['date'].dt.month

# Group by month to get average precipitation
monthly_precip = df.groupby('month')['precipitation'].mean().reset_index()

# %%
plt.figure(figsize=(10, 5))
sns.barplot(data=monthly_precip, x='month', y='precipitation', palette='Blues_d')
plt.title('Monthly Average Precipitation')
plt.xlabel('Month')
plt.ylabel('Precipitation (mm)')
plt.tight_layout()
plt.savefig('precipitation_analysis.png')
plt.show()

# %% [markdown]
# ## 🔥 Correlation Heatmap
# This heatmap shows the correlation between temperature, humidity, and other numeric variables.

# %%
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('../images/correlation_heatmap.png')
plt.show()

# %% [markdown]
# ## 🌐 Interactive Temperature Trend (Plotly)
# Hover over the line chart to explore values interactively.

# %%
import numpy as np
import pandas as pd
import plotly.express as px
import os

# Create a DataFrame if it doesn't exist
# This assumes you want to create sample data for demonstration
months = range(1, 13)
dates = pd.date_range(start='2023-01-01', periods=len(months), freq='M')
df = pd.DataFrame({
    'month': months,
    'date': dates
})

# Simulate seasonal temperature (cooler at start & end, hotter mid-year)
df['temperature'] = 20 + 10 * np.sin((df['month'] - 1) * (2 * np.pi / 12)) + np.random.normal(0, 2, size=len(df))

# Ensure 'date' is datetime (already done in our creation, but keeping for clarity)
df['date'] = pd.to_datetime(df['date'])

# Drop NaNs
df = df.dropna(subset=['temperature'])

# Save directory
os.makedirs('../images', exist_ok=True)

# Plot
fig = px.line(df, x='date', y='temperature', title='Interactive Temperature Over Time')
fig.write_html('../images/interactive_plot.html')
fig.show()

# %% [markdown]
# ## 🌡️ Humidity vs Temperature (Plotly Scatter)
# Explore how temperature varies with humidity.

# %%
import numpy as np

# If 'temperature' column doesn't exist, simulate it
if 'temperature' not in df.columns:
    df['temperature'] = 20 + 10 * np.sin((df['month'] - 1) * (2 * np.pi / 12)) + np.random.normal(0, 2, size=len(df))

# Simulate humidity inversely related to temperature, add some noise
np.random.seed(42)
df['humidity'] = 100 - df['temperature'] + np.random.normal(0, 5, size=len(df))

# Clip to 0–100 range
df['humidity'] = df['humidity'].clip(0, 100)


# %%
import plotly.express as px
import os

# Ensure folder exists
os.makedirs('../images', exist_ok=True)

# Create interactive scatter plot
fig = px.scatter(
    df,
    x='humidity',
    y='temperature',
    color='month',  # Optional: color by month to show seasonal patterns
    title='Humidity vs Temperature (Interactive)',
    labels={'humidity': 'Humidity (%)', 'temperature': 'Temperature (°C)'},
    width=900,
    height=500,
    opacity=0.7
)

# Customize layout
fig.update_traces(marker=dict(size=6))
fig.update_layout(margin=dict(l=40, r=40, t=60, b=60))

# Save and display
fig.write_html('../images/humidity_vs_temperature.html')
fig.show()


# %%
import numpy as np

# Simulate realistic precipitation (monsoon pattern)
df['precipitation'] = np.abs(100 * np.sin((df['month'] - 6) * (np.pi / 12))) + np.random.normal(0, 10, size=len(df))
df['precipitation'] = df['precipitation'].clip(0)  # Ensure no negatives


# %%
print("Cleaned columns:", df.columns.tolist())


# %%
import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv(
    'data/weather_dataset.csv',
    parse_dates=['datetime']
)

# Rename columns for consistency
df.rename(columns={
    'datetime': 'date',
    'city_name': 'city',
    'temperature_2m_mean': 'temperature',
    'precipitation_sum': 'precipitation'
}, inplace=True)

# Create month column
df['month'] = df['date'].dt.month

# App Title
st.title("🌦️ Weather Data Dashboard")

# Sidebar for user input
city = st.selectbox("Select City", df['city'].unique())

# Filter data for selected city
df_city = df[df['city'] == city].copy()

# Line chart for temperature trend
st.subheader(f"🌡️ Temperature Trend in {city}")
fig = px.line(df_city, x='date', y='temperature', title=f'Temperature Over Time in {city}')
st.plotly_chart(fig)

# Bar chart for average monthly precipitation
st.subheader(f"☔ Monthly Precipitation in {city}")
monthly_precip = df_city.groupby('month')['precipitation'].mean().reset_index()
fig2 = px.bar(
    monthly_precip,
    x='month',
    y='precipitation',
    title=f'Average Monthly Precipitation in {city}',
    labels={'month': 'Month', 'precipitation': 'Precipitation (mm)'}
)
st.plotly_chart(fig2)


# %%
import pandas as pd

# Load your data into df_final
df_final = pd.read_csv('data/weather_dataset.csv')  # Replace with your actual data source

# Make sure numeric_cols is defined
numeric_cols = df_final.select_dtypes(include=['number']).columns  # This gets all numeric columns

# Now display the descriptive statistics
st.write(df_final[numeric_cols].describe().round(2))


# %%
import pandas as pd

# Example of creating a dataframe (replace with your actual data source)
df_final = pd.read_csv('data/weather_dataset.csv')  # or any other method to create your dataframe

# Then use numeric_cols (make sure this is defined too)
numeric_cols = df_final.select_dtypes(include=['number']).columns

# Now the original line will work
st.write(df_final[numeric_cols].describe().round(2))

# %%
from IPython.display import display


# %%
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# File uploader
uploaded_file = st.file_uploader("Upload Weather Dataset (.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=True)
    st.success("✅ Data loaded successfully!")

    # Show basic info
    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    # Missing values heatmap
    st.subheader("🔍 Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax)
    st.pyplot(fig)

    # Missing values summary
    st.subheader("📊 Missing Value Summary")
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().mean() * 100).round(2)
    missing_data = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage (%)': missing_percentage
    })
    st.dataframe(missing_data[missing_data['Missing Values'] > 0])
else:
    st.warning("📂 Please upload a CSV file to continue.")


# %%
