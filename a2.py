import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df= pd.read_csv('Iris Dataset.csv')
print(df.head(5))

labels=['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
for label in labels:
    print('Distribution of :',label)
    sns.boxplot(df[label])
    plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

labels=['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
for label in labels:
    print('Skewness of :',label)
    print(df[label].skew())