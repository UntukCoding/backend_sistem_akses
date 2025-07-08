from rest_framework import serializers
from .models import Accesslogruangan,Devicesruangan,Datauser,Monitoringruangan,Ruangan,Userimagetraining,Logdeviceruangan
from . import models
import os

class Imageusertrainingserializer(serializers.ModelSerializer):
    class Meta:
        model=Userimagetraining
        fields='__all__'

class Datauserserializer(serializers.ModelSerializer):
    imageuser=Imageusertrainingserializer(read_only=True,many=True)
    image_list=serializers.ListField(
        child=serializers.ImageField(max_length=100000,allow_empty_file=False,use_url=False)
        ,write_only=True
    )
    class Meta:
        model=Datauser
        fields=[
            'id_user',
            'username',
            'nrp_user',
            'nip_user',
            'status',
            'create_at',
            'imageuser',
            'image_list'
        ]
    def create(self, validated_data):
        image_list=validated_data.pop('image_list')
        datauser=Datauser.objects.create(**validated_data)
        for image in image_list:
            image_name=os.path.basename(image.name)
            save_path=os.path.join('media','imagetraining',datauser.username,image_name)
            if not os.path.exists(save_path):
                Userimagetraining.objects.create(id_user=datauser,image_user=image)
        return datauser
class Datauserserialforimagetraining(serializers.ModelSerializer):
    datauser=Imageusertrainingserializer(read_only=True,many=True)
    class Meta:
        model=Datauser
        fields=[
            'id_user',
            'username',
            'nrp_user',
            'nip_user',
            'status',
            'create_at',
            'datauser',
        ]

class Datauserserialonly(serializers.ModelSerializer):
    class Meta:
        model=Datauser
        fields=[
            'id_user',
            'username',
            'nrp_user',
            'nip_user',
            'status',
            'create_at',
        ]

class Ruanganserializer(serializers.ModelSerializer):
    class Meta:
        model=Ruangan
        fields=[
            'room_id',
            'room_name',
            'room_description',
            'room_location'
        ]

class Logaccesserializer(serializers.ModelSerializer):
    # id_user=Datauserserialonly(read_only=True)
    # room_id=Ruanganserializer(read_only=True)
    class Meta:
        model=Accesslogruangan
        fields=[
            'log_id',
            'id_user',
            'room_id',
            'image',
            'status',
            'access_time'
        ]
class Logaccesserializer2(serializers.ModelSerializer):
    id_user=Datauserserialonly(read_only=True)
    room_id=Ruanganserializer(read_only=True)
    class Meta:
        model=Accesslogruangan
        fields=[
            'log_id',
            'id_user',
            'room_id',
            'image',
            'status',
            'access_time'
        ]

class Deviceruanganserializer(serializers.ModelSerializer):
    room_id=Ruanganserializer(read_only=True)
    class Meta:
        model=Devicesruangan
        fields=[
            'id_device',
            'status',
            'create_at',
            'updated_at',
            'room_id'
        ]
class Deviceruanganserializer2(serializers.ModelSerializer):
    # room_id=Ruanganserializer(read_only=True)
    class Meta:
        model=Devicesruangan
        fields=[
            'id_device',
            'status',
            'create_at',
            'updated_at',
            'room_id'
        ]


class Monitoringruanganserializer(serializers.ModelSerializer):
    access_log=serializers.PrimaryKeyRelatedField(
        many=True,
        read_only=True
    )
    class Meta:
        model=Monitoringruangan
        fields=[
            'monitoring_id',
            'access_log',
            'device_status',
            'room_id',
            'create_at',
            'update_at',
            'room_status'
        ]

class Monitoringruanganserializer2(serializers.ModelSerializer):
    access_log=Logaccesserializer(read_only=True,many=True)
    room_id=Ruanganserializer(read_only=True)
    device_status=Deviceruanganserializer(read_only=True)
    class Meta:
        model=Monitoringruangan
        fields=[
            'monitoring_id',
            'access_log',
            'device_status',
            'room_id',
            'create_at',
            'update_at',
            'room_status'
        ]


class Monitoringruanganserializer3(serializers.ModelSerializer):
    access_log=serializers.SerializerMethodField()
    def get_access_log(self, obj):
        return Logaccesserializer2(
            obj.access_log.order_by('access_time'), many=True,read_only=True
        ).data
    room_id=Ruanganserializer(read_only=True)
    device_status=Deviceruanganserializer(read_only=True)
    class Meta:
        model=Monitoringruangan
        fields=[
            'monitoring_id',
            'access_log',
            'device_status',
            'room_id',
            'create_at',
            'update_at',
            'room_status'
        ]

class Logdeviceruanganserialzier(serializers.ModelSerializer):
    class Meta:
        model=Logdeviceruangan
        fields='__all__'