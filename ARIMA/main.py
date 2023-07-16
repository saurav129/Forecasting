import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Step 1: Load the data
data = pd.read_csv('sales_data.csv')
data = data.query("Product == 'Airpods'")

# Step 2: Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data.rename(columns={'Date': 'ds', 'Quantity': 'y'}, inplace=True)

# Step 3: Create and fit the Prophet model
model = Prophet(seasonality_mode='multiplicative')
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # seasonality parameters.
model.fit(data)

# Step 4: Forecast future sales
future = model.make_future_dataframe(periods=24, freq='M')  # Forecasting next 2 years.
forecast = model.predict(future)

# Step 5: Visualize the forecast as a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['ds'], data['y'], label='Actual Sales', color='blue', marker='o', s=50)
plt.scatter(forecast['ds'], forecast['yhat'], label='Forecasted Sales', color='red', marker='x', s=50)
plt.xlabel('Date')
plt.ylabel('Sales Quantity')
plt.title('Sales Forecast for the Next 2 Years')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Customizing date tick labels
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: pd.to_datetime(x).strftime('%Y-%m')))
plt.xticks(rotation=45)

plt.show()
