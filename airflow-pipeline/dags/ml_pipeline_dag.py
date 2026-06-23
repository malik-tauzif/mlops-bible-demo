from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime

# Task 1: Pull the latest data version from S3 (Chapter 2 concept)
def pull_data():
    print("📥 Pulling latest data version from DVC remote (S3)...")
    print("✅ Data pull simulated successfully.")

# Task 2: Train the model (Chapter 1 concept)
def train_model():
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("🏋️  Training model...")
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Model trained. Accuracy: {acc:.4f}")

# Task 3: Push results back (Chapter 2 concept)
def push_data():
    print("📤 Pushing updated data/model artifacts to S3...")
    print("✅ Push simulated successfully.")


with DAG(
    dag_id="ml_training_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    pull_data_task = PythonOperator(
        task_id="pull_data",
        python_callable=pull_data,
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    push_data_task = PythonOperator(
        task_id="push_data",
        python_callable=push_data,
    )

    # THIS is the actual "race official's instructions" — defining ORDER
    pull_data_task >> train_model_task >> push_data_task
