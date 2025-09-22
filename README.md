# Weather Data Analysis Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌤️ Project Overview

This project provides comprehensive analysis of multi-city weather data through interactive visualizations and statistical insights. Built with Python and Streamlit, it offers a user-friendly dashboard for exploring weather patterns, trends, and correlations across different cities and time periods.

The project combines data cleaning, exploratory data analysis, and interactive visualization to deliver actionable insights about weather conditions, seasonal patterns, and climate trends.

## ✨ Features

- **Interactive Dashboard**: Real-time weather data exploration through Streamlit web interface
- **Multi-City Analysis**: Compare weather patterns across different cities
- **Comprehensive Visualizations**: 
  - Time series plots for temperature, humidity, and precipitation
  - Correlation heatmaps
  - Seasonal trend analysis
  - Statistical distribution plots
- **Data Filtering**: Filter data by date range, city, and weather parameters
- **Statistical Insights**: Summary statistics and trend analysis
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Export Capabilities**: Download filtered data and visualizations

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yashgarg004/Weather_Data_Analysis
   cd Weather_data_analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv weather_env
   
   # On Windows
   weather_env\Scripts\activate
   
   # On macOS/Linux
   source weather_env/bin/activate
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application**
   ```bash
   streamlit run WeatherAnalysis_Extended.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

### Dependencies

The project requires the following Python packages:

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.15.0
scikit-learn>=1.3.0
```

## 📁 Project Structure

```
Weather_data_analysis/
├── images/                     # All visualization outputs
│   ├── temperature_trends.png
│   ├── precipitation_analysis.png
│   ├── correlation_heatmap.png
│   └── interactive_plots.png
├── Weather.ipynb              # Basic weather analysis notebook
├── WeatherAnalysis_Extended.ipynb  # Extended analysis notebook
├── WeatherAnalysis_Extended.py     # Streamlit dashboard application
├── requirements.txt
├── LICENSE
└── README.md


## 🚀 Usage

### Running the Dashboard

1. **Start the application**:
   ```bash
   streamlit run WeatherAnalysis_Extended.py
   ```

2. **Navigate the interface**:
   - Use the sidebar to select cities and date ranges
   - Choose different visualization types from the main panel
   - Apply filters to focus on specific weather parameters
   - Download results using the export buttons


## 📊 Exploratory Data Insights

### 🔹 Line Plot – Temperature Trend Over Time
This plot shows how the temperature changes over time for the selected city. A steady upward trend is observed in summer (May to July) and a dip in winter (December to January), matching seasonal weather patterns.

### 🔹 Box Plot – Monthly Temperature Distribution
This boxplot reveals the spread of temperature each month. July displays a consistent high temperature with minimal variance, while March exhibits more variation.

### 🔹 Scatter Plot – Temperature vs Precipitation
A negative relationship is seen: higher temperatures often align with lower precipitation. This suggests drier periods during hotter months.

### 🔹 Bar Chart – Average Temperature by City
City X records the highest average temperatures, hinting at geographic or climatic influences such as altitude or urban heating.

### 🔹 Heatmap – Correlation Matrix
The correlation matrix shows:
- A strong negative correlation between temperature and humidity.
- Positive correlation between wind speed and gusts.
This helps understand which factors move together and informs feature selection for modeling.

### 🔹 Interactive Plots (Plotly)
Interactive visualizations allow users to:
- Hover to see specific data points.
- Zoom into particular time ranges.
- Dynamically explore seasonal/weather patterns.

---

## ✅ Summary of Key Findings
- **Temperature trends** show clear seasonality.
- **Precipitation** is more common during colder months.
- **Geographic differences** in weather patterns are evident between cities.
- **Strong correlations** exist between temperature, wind, and humidity.

### Outputs of Dashboards and plot and graphs 

## 📊 Dashboard Screenshot

![Dashboard Preview](images/Dashboard.png)

## 📊 Interactive Graphs and plots 

![Average Temperature Over Time](images/Average%20Temp%20Overtime%20Graph.png)
![Monthly Temperature Range by City](images/Avg%20Monthly%20Temp%20Range%20By%20City.png)
![Correlation Heatmap](images/Correlation%20Heatmap.png)
![Wind Speed Distribution](images/Distribution%20Max%20Wind%20speed.png)
![Humidity vs Temperature](images/Humidity%20VS%20Temperature%20(interactive).png)
![Temperature Over Time](images/Temperature%20Overtime%20(inter%20active).png)
![Temperature Trend](images/Temperature%20Trend%20Over%20Time.png)
![Wind Speed vs Temperature](images/Wind%20Speed%20Vs%20Temp%20by%20City.png)
![Precipitation Analysis](images/precipitation_analysis.png)
![Weather Variables Correlation](images/Correlation%20between%20weather%20variables.png)
![Monthly Temperature Distribution](images/Monthly%20Temperature%20distribution.png)
![Temperature Trend Analysis](images/Temperature%20trend.png)
![Temperature vs Precipitation](images/Temperature%20vs%20Precipitation.png)

### Key Features Guide

- **Data Overview**: View summary statistics and data quality metrics
- **Time Series Analysis**: Explore weather trends over time
- **Comparative Analysis**: Compare weather patterns between cities
- **Correlation Analysis**: Discover relationships between weather variables
- **Seasonal Patterns**: Analyze seasonal weather variations

### Jupyter Notebooks

For detailed analysis and methodology:
WeatherAnalysis_Extended.ipynb

## 📊 Data Sources

The project uses weather data from multiple sources:
- Historical weather records from major cities
- Meteorological station data
- Climate databases

*Note: Ensure you have the proper permissions and licenses for any external data sources.*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Yash Garg** - [GitHub Profile](https://github.com/yashgarg004)

## 🙏 Acknowledgments

- Weather data providers and meteorological organizations
- Open-source Python community
- Streamlit team for the amazing framework

---

**⭐ If you find this project helpful, please consider giving it a star!**
