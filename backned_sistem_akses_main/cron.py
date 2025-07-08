import logging
from datetime import timedelta
from django.utils import timezone
from .models import Monitoringruangan, Accesslogruangan

# Setup logger
logger = logging.getLogger(__name__)

def update_room_status():
    now = timezone.now()
    batas_waktu = now - timedelta(minutes=5)

    for monitor in Monitoringruangan.objects.all():
        logs = Accesslogruangan.objects.filter(
            room_id=monitor.room_id,
            access_time__gte=batas_waktu
        )
        if logs.exists():
            monitor.room_status = 'Ada Pengunjung'
            logger.info(f"[{now}] Ruang {monitor.room_id} status: Ada Pengunjung")
        else:
            monitor.room_status = 'Tidak Ada Pengunjung'
            logger.info(f"[{now}] Ruang {monitor.room_id} status: Tidak Ada Pengunjung")
        monitor.save()
