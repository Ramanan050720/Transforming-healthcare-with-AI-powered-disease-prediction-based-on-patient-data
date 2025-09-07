import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Data
df = pd.read_csv("dataset.csv", encoding="latin-1")
df = df.drop_duplicates()
df.columns = df.columns.str.lower()
df = df.dropna()
df.reset_index(drop=True, inplace=True)

# KMeans Clustering
symptom_features = df.drop('prognosis', axis=1)
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
df['disease_group'] = kmeans.fit_predict(symptom_features)

# Group Mapping
group_mapping = (
    df.groupby('disease_group')['prognosis']
    .agg(lambda x: x.value_counts().idxmax())
    .to_dict()
)

# Train Model
X = df.drop(['prognosis', 'disease_group'], axis=1)
y = df['disease_group']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save Model & Metadata
joblib.dump(model, "model.pkl")
joblib.dump(group_mapping, "group_mapping.pkl")
joblib.dump(list(X.columns), "symptom_list.pkl")

print("✅ Model trained and saved as model.pkl")
