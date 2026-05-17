import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df=pd.read_csv('Titanic Dataset.csv')
print(df.head(5))

sns.countplot(x='Gender', hue='Survived', data=df)
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.show()

sns.distplot(df['Age'],kde=False, bins=40)
plt.show()

sns.countplot(x='Gender',data=df)
plt.show()

sns.countplot(x='Survived', hue='SibSp', data=df,palette='mako')
plt.show()

sns.distplot(df['Fare'])
plt.show()

sns.countplot(x='Pclass', hue='Age', data=df,palette='spring')
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()