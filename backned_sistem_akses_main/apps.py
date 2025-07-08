from django.apps import AppConfig


class BacknedSistemAksesMainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'backned_sistem_akses_main'
    def ready(self):
        from . import apschedule
        apschedule.start_scheduler()
