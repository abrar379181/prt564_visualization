import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('aviandata_final.csv')
print(data.head())
print(data.describe())
print(data.dtypes)

#Univariate (One variable at a time)
# Frequency on total cases (Histogram)

sns.set_style('darkgrid')
plt.figure(figsize=(10,6))
sns.histplot(data['Cases'],bins=10, kde=False)
plt.title('Frequency of Cases')
plt.xlabel('Cases')
plt.ylabel('Frequency')
plt.show()

# Clinical Signs count (Counter plot)

plt.figure(figsize=(14,8))
sns.countplot(data=data, x='Clinical Sign')
plt.title('Clinical Signs Distribution')
plt.xlabel('Clinical Signs')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


#Bivariate
# Scatter Plot (Temperature vs Cases)
plt.figure(figsize=(14,8))
sns.scatterplot(data=data, x='Temparature', y='Cases')
sns.regplot(data=data, x='Temparature', y='Cases',scatter_kws={'color':'red'})
plt.title('Temperature Distribution')
plt.xlabel('Temparature')
plt.ylabel('Cases')
plt.xticks(rotation=90)
plt.show()


#Box Plot  (Clinical signs vs Cases)
plt.figure(figsize=(14,8))
sns.boxplot(data=data, x='Clinical Sign', y='Cases')
plt.title('Clinical Signs using Boxplot')
plt.xlabel('Clinical Signs')
plt.ylabel('Cases')
plt.xticks(rotation=90)
plt.show()

#Heatmap
numerical_data = data.select_dtypes(include=['float64','int64'])
correlation_matrix = numerical_data.corr()
plt.figure(figsize=(14,8))
sns.heatmap(data=correlation_matrix, annot=True,cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


