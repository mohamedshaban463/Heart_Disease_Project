from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import StandardScaler


heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

df = pd.DataFrame(X)

#Handling the missing values
df = df.fillna(df.mean())

#standardizing data
scaler = StandardScaler()
standardized_data = scaler.fit_transform(df)
standardized_df = pd.DataFrame(standardized_data, columns=df.columns)

def main():
    print("Heart Disease Dataset:")

if __name__ == "__main__":
    main()
