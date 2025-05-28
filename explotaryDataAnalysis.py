import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from a CSV file
data = pd.read_csv('avian_data.csv')
print(data.head())

# Display information about the DataFrame
print(data.info())
print(data.describe(include='all'))

# Create a new 'Date' column by combining 'Year' and 'Month'
data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))

# Plot the total number of cases over time
plt.figure(figsize=(12, 6))
data.groupby('Date')['Cases'].sum().plot()
plt.title('Total Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.grid(True)
plt.show()

# Plot total cases by region using a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=data.groupby('Region')['Cases'].sum().reset_index(), x='Region', y='Cases')
plt.title('Total Cases by Region')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot the frequency of different diagnoses using a count plot
plt.figure(figsize=(10, 5))
sns.countplot(data=data, y='Diagnosis', order=data['Diagnosis'].value_counts().index)
plt.title('Diagnosis Frequency')
plt.tight_layout()
plt.show()


