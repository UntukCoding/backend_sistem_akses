import os
from apscheduler.schedulers.background import BackgroundScheduler
from django_apscheduler.jobstores import DjangoJobStore
from django.core.management import call_command

def start_scheduler():
    # Cegah job dijalankan dua kali saat 'runserver' reload
    if os.environ.get('RUN_MAIN') != 'true':
        return

    scheduler = BackgroundScheduler()
    scheduler.add_jobstore(DjangoJobStore(), "default")

    from .cron import update_room_status  # sesuaikan dengan path fungsi kamu

    scheduler.add_job(
        update_room_status,
        'interval',
        minutes=10,
        id='update_room_status_job',
        replace_existing=True,
    )

    scheduler.start()
    print("✅ APScheduler started.")
