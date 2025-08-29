import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import StandardScaler

path = "../data/processed.*.data"
files = glob.glob(path)
print("\nfiles:",files)
df= pd.concat(
    (pd.read_csv(f,  sep=",", na_values=0, encoding="latin1") for f in files),
)

columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg","thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
df = df.iloc[:, :14]
df.columns = columns
df = df.apply(pd.to_numeric, errors="coerce")
x = df.mean()
df = df.applymap(lambda Y: x if isinstance(Y, str) else Y)
df = df.fillna(0)
df = df.fillna(0)


features = df.drop("target", axis=1)
target = df["target"]

scaler = StandardScaler()
standardized_data = scaler.fit_transform(features)
standardized_df = pd.DataFrame(standardized_data, columns=features.columns)