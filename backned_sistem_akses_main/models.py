from django.db import models
import os,secrets,string
# Create your models here.
def generate_uid():
    return ''.join(secrets.choice(string.digits)for _ in range(10))
def generate_uid2():
    first_digit=secrets.choice('123456789')
    random_digit=''.join(secrets.choice(string.digits) for i in range(5))
    return first_digit+random_digit
def upload_image_training(instance,filename):
    return os.path.join('imagetraining',str(instance.id_user.username),filename)
def upload_image_access_user(instance,filename):
    print(instance)
    return os.path.join('tracking',str(instance.room_id.room_name),filename)

class Datauser(models.Model):
    id_user=models.CharField(max_length=10,primary_key=True,default=generate_uid2,editable=False)
    username=models.CharField(max_length=255)
    nrp_user=models.BigIntegerField(null=True)
    nip_user=models.BigIntegerField(null=True)
    status=models.CharField(max_length=255)
    create_at=models.DateTimeField(auto_now_add=True)


class Userimagetraining(models.Model):
    id_user=models.ForeignKey(Datauser,on_delete=models.CASCADE,to_field='id_user',related_name='datauser')
    image_user=models.ImageField(upload_to=upload_image_training,default='',blank=True,null=True)
    created_at=models.DateTimeField(auto_now_add=True)
    update_at=models.DateTimeField(null=True,blank=True)

class Ruangan(models.Model):
    room_id=models.CharField(default=generate_uid,primary_key=True,editable=False,max_length=10)
    room_name=models.CharField(max_length=255,unique=True)
    room_description=models.TextField(default='')
    room_location=models.CharField(max_length=255)



class Accesslogruangan(models.Model):
    log_id=models.CharField(default=generate_uid,primary_key=True,editable=False,max_length=10)
    id_user=models.ForeignKey(Datauser,on_delete=models.CASCADE,to_field='id_user',related_name='log_access_user',null=True)
    room_id=models.ForeignKey(Ruangan,on_delete=models.CASCADE,to_field='room_id',related_name='room_id_access',null=True)
    image=models.ImageField(upload_to=upload_image_access_user,default='',blank=True,null=True)
    access_time=models.DateTimeField(auto_now_add=True)
    status=models.CharField(max_length=255)


class Devicesruangan(models.Model):
    id_device=models.CharField(default=generate_uid,primary_key=True,editable=False,max_length=10)
    room_id=models.ForeignKey(Ruangan,on_delete=models.CASCADE,to_field='room_id',related_name='room_id_devices',unique=True)
    status=models.CharField(max_length=255)
    create_at=models.DateTimeField(auto_now_add=True)
    updated_at=models.DateTimeField(auto_now=True,null=True)

class Logdeviceruangan(models.Model):
    id_log_device=models.CharField(default=generate_uid,primary_key=True,editable=False,max_length=10)
    deviceruangan_id=models.ForeignKey(Devicesruangan,on_delete=models.CASCADE,to_field='id_device',related_name='deviceruangan_id2')
    status=models.CharField(max_length=255)
    update_at=models.DateTimeField(null=True)

class Monitoringruangan(models.Model):
    monitoring_id=models.CharField(default=generate_uid,primary_key=True,editable=False,max_length=10)
    access_log=models.ManyToManyField(Accesslogruangan,related_name='log_access_monitor')
    device_status=models.ForeignKey(Devicesruangan,on_delete=models.CASCADE,to_field='id_device',related_name='device_status_monitor',null=True)
    room_id=models.ForeignKey(Ruangan,on_delete=models.CASCADE,unique=True,to_field='room_id',related_name='room_id_monitor')
    create_at=models.DateTimeField(auto_now_add=True)
    update_at=models.DateTimeField(auto_now=True)
    room_status=models.CharField(max_length=255)