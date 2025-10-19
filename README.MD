# Titanic Survival Prediction with Airflow

This project implements a machine learning pipeline to predict passenger survival on the Titanic using Apache Airflow to orchestrate the workflow.

## Workflow

The project's workflow is defined in an Airflow DAG. This DAG orchestrates the following steps:

1.  **Data Preprocessing:** Cleans and prepares the data for model training. This includes handling missing values, encoding categorical features, and scaling numerical features.
2.  **Model Training:** Trains a machine learning model to predict survival.
3.  **Model Evaluation:** Evaluates the model's performance on a test set. A branching operator has also been setup to evaluate if the end was success or failure based on the accuracy.

-----

## Getting Started

### 1. Environment Setup
You can set up a virtual environment and install the necessary dependencies using the following commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Initialize Airflow
Once you have everything set up, you can run Airflow in standalone mode:

Point the AIRFLOW_HOME variable to config folder of this cloned repository. The variable set is only for this instance of terminal and will not affect others.

```bash
export AIRFLOW_HOME=.
airflow standalone
```

### 3. Access the Airflow UI:
A login password for the admin user will be shown in the terminal or in
config/simple_auth_manager_passwords.json.generated

Open your browser and go to:

http://localhost:8080

Login using the admin credentials

### 4. Run the DAG:

1. In the Airflow UI, search for titanic_pipeline DAG
2. Click trigger DAG to start execution.

-----

## Results

The model's performance is evaluated using accuracy. The results are logged and can be viewed in the Airflow UI. This allows for easy tracking of model performance over time. 