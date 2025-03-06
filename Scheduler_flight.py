from celery import shared_task, chain
import subprocess
import logging
from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables
load_dotenv()

# Database Connection
DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise ValueError("Database URL not found in environment variables.")

engine = create_engine(DB_URL)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@shared_task
def fetch_inference_data():
    """Fetch real-time inference data"""
    try:
        inference_data = "Datas/FLight_inference_data.csv"  # Ensure the file path exists
        logging.info(f"Fetched inference data from {inference_data}")
        return inference_data
    except Exception as e:
        logging.error(f"Error fetching inference data: {e}")
        return None


@shared_task
def trigger_inference_pipeline(inference_data_path):
    """Run the inference pipeline on fetched data"""
    if not inference_data_path:
        logging.error("No inference data provided.")
        return None

    try:
        logging.info(f"Running inference on data: {inference_data_path}")
        subprocess.run(["python", "Inference_Flight.py", inference_data_path], check=True)
        inference_results = "Datas/FLight_inference_results.csv"
        logging.info(f"Inference completed, results saved at {inference_results}")
        return inference_results
    except subprocess.CalledProcessError as e:
        logging.error(f"Inference script failed: {e}")
        return None


@shared_task
def store_inference_results(inference_results_path):
    """Store inference results in the database"""
    if not inference_results_path:
        logging.error("No inference results provided.")
        return None

    try:
        df = pd.read_csv(inference_results_path)  # Ensure file exists
        with engine.begin() as conn:
            df.to_sql("inference_results", con=conn, if_exists="append", index=False, method="multi")

        logging.info("Inference results stored in DB successfully.")
        return inference_results_path  # Pass the results for training
    except SQLAlchemyError as e:
        logging.error(f"Database error: {e}")
        return None
    except FileNotFoundError:
        logging.error(f"File not found: {inference_results_path}")
        return None
    except Exception as e:
        logging.error(f"Error storing results: {e}")
        return None


@shared_task
def retrain_model(training_data_path):
    """Retrain the model using new inference data"""
    if not training_data_path:
        logging.error("No training data provided.")
        return None

    try:
        logging.info(f"Retraining model using data from: {training_data_path}")
        subprocess.run(["python", "flight_model.py", training_data_path], check=True)
        logging.info("Model retrained successfully.")
        return "Model retrained successfully"
    except subprocess.CalledProcessError as e:
        logging.error(f"Model retraining failed: {e}")
        return None


# Define Celery task pipeline
pipeline = chain(
    fetch_inference_data.s(),
    trigger_inference_pipeline.s(),
    store_inference_results.s(),
    retrain_model.s()
)

# Trigger the pipeline asynchronously
pipeline.apply_async()
