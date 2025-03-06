from celery import Celery
from celery.schedules import crontab

app = Celery(
    "scheduler_pipeline",
    broker="redis://localhost:6379/0",  
    backend="redis://localhost:6379/0"  
)

app.conf.beat_schedule = {
    "fetch_training_data_daily": {
        "task": "tasks.fetch_training_data",
        "schedule": crontab(hour=0, minute=0),  
    },
    "train_model_daily": {
        "task": "tasks.trigger_training_pipeline",
        "schedule": crontab(hour=1, minute=0),  
    },
    "fetch_inference_data_every_30min": {
        "task": "tasks.fetch_inference_data",
        "schedule": crontab(minute="*/30"),  
    },
    "run_inference_every_30min": {
        "task": "tasks.trigger_inference_pipeline",
        "schedule": crontab(minute="*/30"),  
    },
}  
app.conf.timezone = "UTC"
