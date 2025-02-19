import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin

# Chargement des données
def load_data(file_path):
    data = pd.read_csv(file_path)

    # Permet de passer la colonne `Name` en lowercase
    data['Name'] = data['Name'].str.lower()
    data['Marque'] = data['Name'].apply(lambda x: "land-rover" if x.lower().startswith("land rover") else x.split()[0])
    def extract_model(name):
        name_parts = name.split()
        if "land" in name.lower():
            return " ".join(name_parts[2:])  # Récupère tout après les deux premiers mots
        else:
            return " ".join(name_parts[1:])  # Récupère tout après le premier mot

    data['Model'] = data['Name'].apply(extract_model)
    
    X = data.drop("Price", axis=1)
    y = data["Price"]
    return X, y

# Transformation personnalisée pour Owner_Type
class OrdinalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mappings = {
            "Owner_Type": {"First": 3, "Second": 2, "Third": 1, "Fourth & Above": 0},
            "Location": {},
            "Name": {},
            "Model": {}
        }
    def fit(self, X, y=None):
        for col in ["Location", "Marque", "Model"]:
            unique_values = sorted(X[col].unique())
            self.mapping[col] = {val: idx for idx, val in enumerate(unique_values)}
        return self
    
    def transform(self, X):
        X = X.copy()
        X["Owner_Type"] = X["Owner_Type"].map(self.mapping)
        return X

# Création de la pipeline de preprocessing
def create_pipeline():
    numerical_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = ColumnTransformer([
        ("binary_transmission", ce.BinaryEncoder(cols=["Transmission"])),
        ("onehot_fuel", ce.OneHotEncoder(cols=["Fuel_Type"])),
        ("encoder", ce.OrdinalEncoder(cols=["Transmission", "Location", "Fuel_Type", "Marque", "Model"]))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, ["Kilometers_Driven", "Engine", "Power", "Seats"]),
        ("cat", categorical_transformer, ["Transmission", "Fuel_Type"]),
        ("num", numerical_transformer, ["Kilometers_Driven", "Engine", "Power", "Seats"]),
        ("cat", categorical_transformer, ["Transmission", "Location", "Fuel_Type", "Marque", "Model"])
    ])
    
    pipeline = Pipeline([
        ("ordinal", OrdinalTransformer()),
        ("preprocessor", preprocessor)
    ])
    
    return pipeline

# Exécution du preprocessing
def preprocess_data(file_path):
    X, y = load_data(file_path)
    pipeline = create_pipeline()
    X_transformed = pipeline.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Sauvegarde des données
X_train, X_test, y_train, y_test = preprocess_data("datas/df_EDA.csv")

pd.DataFrame(X_train).to_csv("datas/split/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("datas/split/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("datas/split/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("datas/split/y_test.csv", index=False)

print("Les fichiers X_train.csv, X_test.csv, y_train.csv, et y_test.csv ont été sauvegardés.")