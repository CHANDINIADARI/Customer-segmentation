from flask import Flask, render_template
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__, static_folder='static')

@app.route('/customer')
def satya():
    return render_template('customer.html')

if __name__ == "_main_":
    # Load the customer data
    df = pd.read_csv('Mall_Customers.csv')

    # Perform some exploratory data analysis
    plt.scatter(df['Age'], df['Spending Score (1-100)'])
    plt.xlabel('Age')
    plt.ylabel('Spending Score')
    plt.savefig('static/age_spending_score.png')  # Save the plot as an image
    plt.close()

    # Create and fit the K-means model
    km = KMeans(n_clusters=5)
    predicted = km.fit_predict(df[['Annual Income (k$)', 'Spending Score (1-100)']])

    # Visualize the clustering results
    df['Cluster'] = predicted

    plt.scatter(df[df['Cluster'] == 0]['Annual Income (k$)'], df[df['Cluster'] == 0]['Spending Score (1-100)'], color='green')
    plt.scatter(df[df['Cluster'] == 1]['Annual Income (k$)'], df[df['Cluster'] == 1]['Spending Score (1-100)'], color='red')
    plt.scatter(df[df['Cluster'] == 2]['Annual Income (k$)'], df[df['Cluster'] == 2]['Spending Score (1-100)'], color='black')
    plt.scatter(df[df['Cluster'] == 3]['Annual Income (k$)'], df[df['Cluster'] == 3]['Spending Score (1-100)'], color='cyan')
    plt.scatter(df[df['Cluster'] == 4]['Annual Income (k$)'], df[df['Cluster'] == 4]['Spending Score (1-100)'], color='blue')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.savefig('static/clustering_result.png')  # Save the plot as an image
    plt.close()

    app.run(debug=True)
    