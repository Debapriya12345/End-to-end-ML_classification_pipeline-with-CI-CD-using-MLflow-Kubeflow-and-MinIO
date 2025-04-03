# 📌 Import the necessary libraries first
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score,recall_score,f1_score
import pickle

# 📌 Read the data from local Folder 

df = pd.read_csv("E:\\GDS_MLOPS\\Project-1\\heart.csv")
print(df.head())

# 📌Split the DataFrame into features and target variable
X = df.iloc[:, :-1]  # Features (all columns except the last)
y = df.iloc[:, -1]   # Target variable (the last column)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌Set up MLflow Tracking
mlflow.set_experiment("Heart Attack Predection")

# ⭐ First Create a Random forest Classifier model
def train_model_Randomforest(X_train, y_train, X_test, y_test):

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    with mlflow.start_run():
        # Fit the model to the training data
        rf_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = rf_classifier.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        fprecision = precision_score(y_test, y_pred, average='weighted')
        frecall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 📌Log Parameters & Metrics in MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", fprecision)
        mlflow.log_metric("recall", frecall)
        mlflow.log_metric("f1_score", f1)

        Randomforest_model = rf_classifier

        # 📌Save Model and Artifacts Separately
        model_path = "models/Randomforest_model.pkl"
        pickle.dump(Randomforest_model, open(model_path, 'wb'))
        print("Saved the pickle File of Randomforest model")

        #📌Log model and artifact in specific Mlflow
        mlflow.sklearn.log_model(Randomforest_model, "HeartClassification_model")
        mlflow.log_artifact(model_path, artifact_path="artifacts")

        # ✅ Register Model in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/HeartClassification_model"
        mlflow.register_model(model_uri, "HeartClassificationModel")

        print(f"Accuracy: {acc}, Precision: {fprecision}, Recall: {frecall}, F1 Score: {f1}")

# ⭐ Create a Gradient Boosting Classifier model
def train_model_GradientBoosting(X_train, y_train, X_test, y_test):

    gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

    with mlflow.start_run():

        # Fit the model to the training data
        gb_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = gb_classifier.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 📌Log Parameters & Metrics in MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        GradientBoosting_model = gb_classifier

        # 📌Save Model and Artifacts Separately
        model_path = "models/GradientBoosting_model.pkl"
        pickle.dump(GradientBoosting_model, open(model_path, 'wb'))
        print("Saved the pickle File of GradientBoosting model")

        #📌Log model and artifact in specific Mlflow
        mlflow.sklearn.log_model(GradientBoosting_model, "HeartClassification_model")
        mlflow.log_artifact(model_path, artifact_path="artifacts")

        # ✅ Register Model in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/HeartClassification_model"
        mlflow.register_model(model_uri, "HeartClassificationModel")

        print(f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# ✅ Run Training
train_model_Randomforest(X_train, y_train, X_test, y_test)

train_model_GradientBoosting(X_train, y_train, X_test, y_test)

print("End Of Code")