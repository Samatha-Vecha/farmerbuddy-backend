import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


DATASET_PATH = 'C:/Users/samat/OneDrive/Documents/FertilizerPredictionMiniProject/backend/Dataset/Processed_Crop_and_Fertilizer_Dataset.csv'
MODEL_PATH = 'C:/Users/samat/OneDrive/Documents/FertilizerPredictionMiniProject/backend/model/decision_tree_model.pkl'
TRANSFORMER_PATH = 'C:/Users/samat/OneDrive/Documents/FertilizerPredictionMiniProject/backend/model/transformer.pkl' 
TARGET_COLUMN = 'Fertilizer'


def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def plot_correlation_matrix(df):
    plt.figure(figsize=(10, 8))
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numerical_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Matrix")
    plt.show()


def preprocess_data(df, target_column):
    try:
        y = df[target_column].copy()
        X = df.drop(target_column, axis=1).copy()

       
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

        
        ct = ColumnTransformer(
            transformers=[
                ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols),
                ('scaler', StandardScaler(), numerical_cols)
            ],
            remainder='passthrough'
        )

        X_transformed = ct.fit_transform(X)
        print("Data preprocessing completed.")
        return X_transformed, y, ct
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return None, None, None


def train_model_with_tuning(X, y):
    try:
       
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

        
        param_grid = {
            'max_depth': [5, 10, None],      
            'min_samples_split': [2, 5],
            'criterion': ['gini', 'entropy'] 
        }

        
        grid_search = GridSearchCV(
            estimator=DecisionTreeClassifier(),
            param_grid=param_grid,
            cv=3,              
            scoring='accuracy', 
            n_jobs=-1          
        )

        
        grid_search.fit(X_train, y_train)

        
        print("Best hyperparameters:", grid_search.best_params_)

       
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

       
        print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print("Classification Report:\n", classification_report(y_test, y_pred))

        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        return best_model
    except Exception as e:
        print(f"Error during model training with tuning: {e}")
        return None


def save_model_and_transformer(model, transformer, model_path, transformer_path):
    try:
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f"Model saved at {model_path}")

        with open(transformer_path, 'wb') as transformer_file:
            pickle.dump(transformer, transformer_file)
        print(f"Transformer saved at {transformer_path}")
    except Exception as e:
        print(f"Error saving model or transformer: {e}")


if __name__ == "__main__":
    df = load_dataset(DATASET_PATH)
    if df is not None:
        plot_correlation_matrix(df)
        X, y, ct = preprocess_data(df, TARGET_COLUMN)
        if X is not None and y is not None:
            model = train_model_with_tuning(X, y)
            if model:
                save_model_and_transformer(model, ct, MODEL_PATH, TRANSFORMER_PATH)
