import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


df = pd.read_csv(r"C:\Users\DELL\OneDrive\Documents\weather_data.csv")
df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

df.fillna(df.mean(numeric_only=True), inplace=True)


scaler = MinMaxScaler()
df[['Temperature', 'Humidity', 'Rainfall']] = scaler.fit_transform(df[['Temperature', 'Humidity', 'Rainfall']])


print("\nDescriptive Statistics:")
print(df[['Temperature', 'Humidity', 'Rainfall']].describe())


corr = df[['Temperature', 'Humidity', 'Rainfall']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.clf()


temp_trend = df.groupby('Year')['Temperature'].mean().reset_index()
plt.plot(temp_trend['Year'], temp_trend['Temperature'], marker='o')
plt.title("Average Temperature Trend Over Years")
plt.xlabel("Year")
plt.ylabel("Normalized Temperature")
plt.grid(True)
plt.savefig("temperature_trend.png")
plt.clf()


rainfall_yearly = df.groupby('Year')['Rainfall'].sum().reset_index()
plt.bar(rainfall_yearly['Year'],
rainfall_yearly['Rainfall'], color='skyblue')
plt.title("Total Rainfall by Year")
plt.xlabel("Year")
plt.ylabel("Normalized Rainfall")
plt.savefig("rainfall_bar_chart.png")
plt.clf()


plt.scatter(df['Temperature'], df['Humidity'], alpha=0.6)
plt.title("Temperature vs Humidity")
plt.xlabel("Normalized Temperature")
plt.ylabel("Normalized Humidity")
plt.savefig("temp_vs_humidity.png")
plt.clf()


X = temp_trend['Year'].values.reshape(-1, 1)
y = temp_trend['Temperature'].values
model = LinearRegression()
model.fit(X, y)


future_years = np.arange(2025, 2031).reshape(-1, 1)
future_preds = model.predict(future_years)


plt.plot(temp_trend['Year'], temp_trend['Temperature'], label="Historical Data", marker='o')
plt.plot(future_years, future_preds, color='orange', linestyle='--', label="Predicted Trend")
plt.title("Temperature Forecast using Linear Regression")
plt.xlabel("Year")
plt.ylabel("Normalized Temperature")
plt.legend()
plt.grid(True)
plt.savefig("temperature_forecast.png")
plt.clf()


y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
print(f"\nModel Evaluation:\nMean Squared Error: {mse:.4f}\nRoot Mean Squared Error: {rmse:.4f}")
