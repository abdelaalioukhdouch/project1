# -*- coding: utf-8 -*-
from flask import Flask, render_template, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def home():
    # Load the dataset
    df = pd.read_excel("database/Full Dataset.xlsx")

    # Preprocessing steps...
    # ...

    # Bar plot function
    def bar_h(df, x):
        counts = df[x].value_counts()
        plt.figure()
        plt.barh(counts.index, counts.values)
        plt.xlabel('Number of clothes')
        plt.ylabel(x)
        plt.title(f'Number of clothes by {x}')
        return plt

    # Plotting
    plt1 = bar_h(df, 'Type')
    plt2 = bar_h(df, 'Washing_instruction')
    plt3 = bar_h(df, 'Drying_instruction')
    plt1.savefig('static/bar_type.png')
    plt2.savefig('static/bar_washing.png')
    plt3.savefig('static/bar_drying.png')

    # Creating the pie chart
    plt.figure()
    df['EI'].value_counts().plot(kind='pie', autopct='%.2f%%')
    plt.title('Distribution of EI (Environmental Impact)')
    plt.savefig('static/pie.png')

    # Creating the correlation heatmap
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='rocket_r', linewidths=1, linecolor='white')
    plt.title('Correlation Matrix')
    plt.savefig('static/correlation.png')

    # Render the HTML template
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_file(f'static/{path}')

if __name__ == '__main__':
    app.run()
