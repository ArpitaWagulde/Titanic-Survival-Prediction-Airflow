import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from os import path

def get_base_dir():
    this_file = path.abspath(__file__)
    base_dir = path.dirname(path.dirname(this_file))
    return base_dir

def extract_data():
    df = pd.read_csv(path.join(get_base_dir(), "data", "Titanic-Dataset.csv"))
    return df

def extract_preprocessed_data():
    df = pd.read_csv(path.join(get_base_dir(), "data", "preprocessed_titanic_data.csv"))
    return df

def preprocess_data():
    df = extract_data()
    df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    df = df.astype(float)
        
    print("Data after preprocessing:")
    print(df.head())
    print("\nMissing values check:")
    print(df.isnull().sum())

    df.to_csv(path.join(get_base_dir(), "data", "preprocessed_titanic_data.csv"), index=False)

def split_data():
    df = extract_preprocessed_data()
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    return data

def train_model():
    data = split_data()
    X_train = data['X_train']
    y_train = data['y_train']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    model_path = path.join(get_base_dir(), "model", "titanic_model.pkl")
    pd.to_pickle(model, model_path)

def evaluate_model():
    model = pd.read_pickle(path.join(get_base_dir(), "model", "titanic_model.pkl"))
    data = split_data()
    X_test = data['X_test']
    y_test = data['y_test']
    accuracy = model.score(X_test, y_test)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')