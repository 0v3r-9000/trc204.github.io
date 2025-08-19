import pandas as pd

df = pd.read_csv('Mall_Customers.csv')
df.head()
from sklearn.cluster import KMeans

km = KMeans(n_clusters=5)
km.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
import matplotlib.pyplot as plt
import seaborn as sns

# Predict the cluster for each data point in the original DataFrame
df['cluster'] = km.predict(df[['Annual Income (k$)','Spending Score (1-100)']])

# Visualize the clusters using a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='cluster', palette='viridis', s=100)
plt.title('KMeans Clustering of Customers by Age and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

plt.legend(title='Cluster')
plt.grid(True)
plt.show()
km.predict([[140, 20]])
