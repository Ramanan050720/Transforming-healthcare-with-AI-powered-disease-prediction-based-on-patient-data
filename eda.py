
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = pd.read_csv("dataset.csv", encoding="latin-1")
df = df.drop_duplicates()
df.columns = df.columns.str.lower()
df = df.dropna()
df.reset_index(drop=True, inplace=True)

# 1. Distribution of Diseases
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='prognosis', order=df['prognosis'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Distribution of Diseases')
plt.show()

# 2. Symptom Correlation Heatmap
plt.figure(figsize=(20,16))
sns.heatmap(df.drop('prognosis', axis=1).corr(), cmap='coolwarm')
plt.title('Symptom Correlation Heatmap')
plt.show()

# 3. Top Symptoms for First Disease
first_disease = df['prognosis'].unique()[0]
first_disease_data = df[df['prognosis'] == first_disease]
first_disease_symptoms = first_disease_data.drop('prognosis', axis=1).mean()
plt.figure(figsize=(12,8))
first_disease_symptoms.sort_values(ascending=False).head(15).plot(kind='bar')
plt.title(f"Top Symptoms for {first_disease}")
plt.ylabel('Presence Rate')
plt.show()

# 4. Most Common Symptoms Overall
symptom_counts = df.drop('prognosis', axis=1).sum()
symptom_counts.sort_values(ascending=False).head(20).plot(kind='bar', figsize=(10, 6), color='skyblue')
plt.title('Most Common Symptoms')
plt.xlabel('Symptoms')
plt.ylabel('Count')
plt.show()
