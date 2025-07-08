from django.urls import path
from . import views
urlpatterns=[

    path('testingip/',views.Testingip.as_view(),name='testingip'),
    ###for db user
    path('createimage/',views.Createuserandimagetraining.as_view(),name='createimage'),
    path('getimageall/',views.Getuserandimagetrainingall.as_view(),name='getimageall'),
    path('getimageone/',views.Getuserandimagetrainingone.as_view(),name='getimageone'),
    path('updateuserandimagepath/',views.Updateuser.as_view(),name='updateuserandimagepath'),
    path('deleteuserandimage/',views.Deleteuser.as_view(),name='deleteuser'),

    ##for db userimagetraining
    path('addimagenew/',views.Addimagetraining.as_view(),name='addimage'),
    path('getimagetrainingall/',views.Getimagetrainingall.as_view(),name='getimagetrainingall'),
    path('getimageforoneuser/',views.Getimagefromoneuser.as_view(),name='getimagetraingoneuser'),
    path('updateimagetraining/',views.Updateimagetraining.as_view(),name='updateimagetraining'),
    path('deleteimagetraining/',views.Deleteimagetraining.as_view(),name='deleteimagetraining'),

    ###for db ruangan
    path('createdataruangan/',views.Createdataruangan.as_view(),name='createdataruangan'),
    path('getdataruanganone/',views.Getdataruanganone.as_view(),name='getdataruanganone'),
    path('getdataruanganall/',views.Getdataruanganall.as_view(),name='getdataruanganall'),
    path('updatedataruangan/',views.Updatedataruangan.as_view(),name='updatedataruangan'),
    path('deletedataruangan/',views.Deletedataruangan.as_view(),name='deletedataruangan'),


    ### for db logaccess
    path('createlogaccess/',views.Createlogaccess.as_view(),name='createlogacces'),
    path('getacceslogall/',views.Getaccesslogruanganall.as_view(),name='getaccesslogall'),
    path('getacceslogone/',views.Getaccesslogruanganone.as_view(),name='getaccesslogone'),
    path('deletelogacces/',views.Deleteaccesslogruangan.as_view(),name='deleteacceslog'),

    ###for db device ruangan
    path('createdeviceruangan/',views.Creatadeviceruangan.as_view(),name='createdeviceruangan'),
    path('getdatadeviceruanganall/',views.Getdeviceruanganall.as_view(),name='getdatadeviceruanganall'),
    path('getdatadeviceruanganone/',views.Getdeviceruanganone.as_view(),name='getdatadeviceruanganone'),
    path('updatedeviceruangan/',views.Updatedeviceruangan.as_view(),name='updatedeviceruangan'),
    path('deletedeviceruangan/',views.Deletedatadeviceruangan.as_view(),name='deletedeviceruangan'),

    ### for db logdeviceruangan
    path('createhistorylogdeviceruangan/',views.Createlogdeviceruangan.as_view(),name='createhistorydeviceruangan'),
    path('gethistorylogdeviceruangan/',views.Getlogdeviceruangan.as_view(),name='gethistorydeviceruangan'),
    path('deletehistorylogdeviceruangan/',views.Deletelogdeviceruangan.as_view(),name='deletehistorydeviceruangan'),

    ###for db monitoring ruangan
    path('createdatamonitoringruangan/',views.Createdatamonitoringruangan.as_view(),name='createdatamonitor'),
    path('getmonitoruanganall/',views.Getmonitorruanganall.as_view(),name='getdatamonitor'),
    path('updatedatamonitor/',views.Updatemonitorruangan.as_view(),name='updatedatamonitor'),
    path('deletedatamonitor/',views.Deletedatamonitor.as_view(),name='deletedatamonitor'),

]