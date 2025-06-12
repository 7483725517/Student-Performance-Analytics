import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("flight_delay_data.csv", parse_dates=['Scheduled_Departure', 'Actual_Departure'])
print(df)

df = df[(df['Departure_Delay_Minutes'] > -10000) & (df['Departure_Delay_Minutes'] < 10000)]

df['Day_of_Week'] = df['Scheduled_Departure'].dt.dayofweek
df['Hour_of_Day'] = df['Scheduled_Departure'].dt.hour


df_model = pd.get_dummies(df, columns=['Airline', 'Origin_Airport', 'Destination_Airport', 'Weather_Condition', 'Delay_Reason'], drop_first=True)

X = df_model.drop(columns=['Flight_ID', 'Scheduled_Departure', 'Actual_Departure', 'Departure_Delay_Minutes'])
y = df_model['Departure_Delay_Minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

print("Shape of dataset:", df.shape)
print("\nColumn Info:\n", df.info())
print("\nMissing Values:\n", df.isnull().sum())

print("\nSummary Statistics:\n", df.describe())

df['Day_of_Week'] = df['Scheduled_Departure'].dt.day_name()
df['Hour_of_Day'] = df['Scheduled_Departure'].dt.hour

df = df[(df['Departure_Delay_Minutes'] > -10000) & (df['Departure_Delay_Minutes'] < 10000)]

plt.figure(figsize=(8,4))
sns.histplot(df['Departure_Delay_Minutes'], bins=50, kde=True)
plt.title("Distribution of Departure Delays")
plt.xlabel("Delay (Minutes)")
plt.show()

plt.figure(figsize=(10,5))
sns.barplot(x='Airline', y='Departure_Delay_Minutes', data=df, estimator='mean')
plt.title("Average Delay by Airline")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,5))
sns.barplot(x='Origin_Airport', y='Departure_Delay_Minutes', data=df, estimator='mean')
plt.title("Average Delay by Origin Airport")
plt.show()

plt.figure(figsize=(10,5))
sns.barplot(x='Weather_Condition', y='Departure_Delay_Minutes', data=df, estimator='mean')
plt.title("Delay by Weather Condition")
plt.show()

plt.figure(figsize=(12,5))
sns.boxplot(x='Hour_of_Day', y='Departure_Delay_Minutes', data=df)
plt.title("Delay vs Hour of Day")
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='Day_of_Week', y='Departure_Delay_Minutes', data=df)
plt.title("Delay vs Day of Week")
plt.xticks(rotation=45)
plt.show()

numeric_df = df.select_dtypes(include='number')
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()