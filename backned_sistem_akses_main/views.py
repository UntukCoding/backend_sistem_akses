from django.shortcuts import render
import os.path,shutil,string,secrets,tempfile

from django.core.files.base import ContentFile
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser,FormParser,JSONParser
from rest_framework.views import APIView
from rest_framework import status
import cv2,os,numpy as np
from PIL import Image
from . import serializer,models

# Create your views here.

def get_trained_images(log_file, user_folder):
    """Membaca daftar gambar yang sudah dilatih dari file log."""
    trained_images = set()
    if os.path.exists(log_file):
        with open(log_file, "r") as file:
            trained_images = set(file.read().splitlines())
    return trained_images

def update_trained_images(log_file, new_images):
    """Menambahkan gambar baru yang telah dilatih ke file log."""
    with open(log_file, "a") as file:
        for image in new_images:
            file.write(f"{image}\n")
def update_trained_images2(log_file, new_images):
    """
    Menulis ulang log file dengan gambar baru yang telah dilatih.
    """
    with open(log_file, "w") as file:  # Gunakan mode "w" untuk menimpa log lama
        for image in new_images:
            file.write(f"{image}\n")
def non_max_suppression_fast(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)  # bisa juga gunakan area

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:-1]]

        idxs = np.delete(idxs, np.concatenate(([len(idxs)-1], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def train_or_update_user_data(training_dir, model_save_path, target_user, target_label):
    print(target_user)
    print(target_label)
    face_recognizer = cv2.face.LBPHFaceRecognizer.create()
    haar_name='haarcascade_frontalface_default.xml'
    haar_loc=os.path.join('hasiltraining',haar_name)
    face_cascade = cv2.CascadeClassifier(haar_loc)

    log_file = f"{target_user}_trained_images.log"
    trained_images = get_trained_images(log_file, target_user)
    faces = []
    labels = []

    user_path = os.path.join(training_dir, target_user)
    new_images = []

    # Pastikan folder user ada
    if os.path.isdir(user_path):
        for file in os.listdir(user_path):
            if file.endswith(("jpg", "jpeg", "png")):
                image_path = os.path.join(user_path, file)
                if image_path in trained_images:
                    continue  # Skip image yang sudah dilatih sebelumnya

                image = Image.open(image_path).convert("L")
                image_np = np.array(image, "uint8")
                detected_faces = face_cascade.detectMultiScale(image_np, 1.2, 5)
                for (x, y, w, h) in detected_faces:
                    faces.append(image_np[y:y+h, x:x+w])
                    labels.append(target_label)
                    print(labels)
                    print(f"Detected faces in {file}: {detected_faces}")
                    new_images.append(image_path)

        print(f"Data wajah ditemukan untuk user {target_user}: {len(faces)} gambar baru.")
    else:
        print(f"Folder user {target_user} tidak ditemukan.")

    if faces:
        # Update model dengan data baru
        if os.path.exists(model_save_path):
            face_recognizer.read(model_save_path)

        face_recognizer.update(faces, np.array(labels))
        face_recognizer.save(model_save_path)
        print(np.array(labels))
        print(f"Data baru untuk user {target_user} ditambahkan ke model dan disimpan di {model_save_path}.")

        # Update daftar gambar yang telah dilatih
        update_trained_images(log_file, new_images)
    else:
        print(f"Tidak ada wajah baru untuk user {target_user}.")

def train_replace_user_data(training_dir, model_save_path, target_user, target_label):
    """
    Mengganti semua data wajah user yang ada di log dan menggantinya dengan gambar baru dari folder user.
    """
    face_recognizer = cv2.face.LBPHFaceRecognizer.create()
    haar_name='haarcascade_frontalface_default.xml'
    haar_loc=os.path.join('hasiltraining',haar_name)
    face_cascade = cv2.CascadeClassifier(haar_loc)

    log_file = f"{target_user}_trained_images.log"
    faces = []
    labels = []

    user_path = os.path.join(training_dir, target_user)
    new_images = []

    # Pastikan folder user ada
    if os.path.isdir(user_path):
        # Hapus semua entri lama di log
        if os.path.exists(log_file):
            os.remove(log_file)
            print(f"Log file lama untuk {target_user} dihapus.")

        # Proses semua gambar di folder user
        for file in os.listdir(user_path):
            if file.endswith(("jpg", "jpeg", "png")):
                image_path = os.path.join(user_path, file)
                image = Image.open(image_path).convert("L")
                image_np = np.array(image, "uint8")
                detected_faces = face_cascade.detectMultiScale(image_np, 1.2, 5)
                for (x, y, w, h) in detected_faces:
                    faces.append(image_np[y:y+h, x:x+w])
                    labels.append(target_label)
                    new_images.append(image_path)

        print(f"Data wajah ditemukan untuk user {target_user}: {len(faces)} gambar.")
    else:
        print(f"Folder user {target_user} tidak ditemukan.")
        return  # Tidak ada folder, tidak ada pelatihan

    if faces:
        # Latih ulang model dari awal atau update model
        if os.path.exists(model_save_path):
            face_recognizer.read(model_save_path)
        else:
            print("Model baru akan dibuat.")

        face_recognizer.train(faces, np.array(labels))
        face_recognizer.save(model_save_path)
        print(f"Model untuk user {target_user} disimpan di {model_save_path}.")

        # Update log file dengan gambar baru
        update_trained_images2(log_file, new_images)
    else:
        print(f"Tidak ada data wajah yang valid untuk user {target_user}.")

def clear_log_for_user(log_file):
    """Hapus semua referensi log untuk user tertentu."""
    if os.path.exists(log_file):
        os.remove(log_file)

def recognize_from_image(image, model_path, label_to_user):
    """
    Melakukan proses pengenalan wajah pada gambar yang diberikan.
    """
    # Load the trained model
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    recognizer.read(model_path)

    # Load the face detection model
    haar_name='haarcascade_frontalface_default.xml'
    haar_loc=os.path.join('hasiltraining',haar_name)
    face_cascade = cv2.CascadeClassifier(haar_loc)

    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah pada gambar
    raw_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
    faces=non_max_suppression_fast(raw_faces,overlapThresh=0.3)
    results = []
    for (x, y, w, h) in faces:
        face_image = gray[y:y+h, x:x+w]
        try:
            label, confidence = recognizer.predict(face_image)
            print(label)
            print(confidence)
        except Exception as e:
            # Jika error saat prediksi
            print(f"Error predicting face: {e}")
            continue
        if confidence >= 50:
            username = 'Unknown'
            id_user=None
            status = "Unauthorized"
        else:
            username = label_to_user.get(label, "Unknown")
            id_user=label
            status = "Authorized"
        print(id_user)
        print(username)
        color= (0, 255, 0) if confidence < 50 else (0, 0, 255)
        cv2.putText(image, username, (x+100,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        results.append({
            "id_user": id_user,
            "username":username,
            "status": status,
            "confidence":"  {0}%".format(round(100 - confidence)),
            "face_image": image
        })
    print('result from def recognize image',results)
    return results

def get_true_label_from_path(image_path):
    # Assuming the label is the first part of the filename, split by "_"
    label_str = os.path.basename(image_path).split("_")[0]
    return str(label_str)

def evaluate_model(image_path, modelsave_path, know_label):
    true_labels = []
    predicted_labels = []
    results = []

    TP, TN, FP, FN = 0, 0, 0, 0
    # Ambil true_label dari nama file
    true_label = os.path.splitext(os.path.basename(image_path))[0].split('_')[0]
    true_labels.append(true_label)
    face_recognizer = cv2.face.LBPHFaceRecognizer.create()
    face_recognizer.read(modelsave_path)
    haar_name='haarcascade_frontalface_default.xml'
    haar_loc=os.path.join('hasiltraining',haar_name)
    face_cascade = cv2.CascadeClassifier(haar_loc)
    # Baca gambar dan lakukan deteksi wajah
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:  # Jika wajah terdeteksi
        (x, y, w, h) = faces[0]
        face = image[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face)
        print(confidence)
        if confidence < 45:
            predicted_label = know_label.get(label, "Unknown")
            label_text ="  {0}%".format(round(100 - confidence))
            print(predicted_label)
        else:
            predicted_label = "Unknown"
            label_text ="  {0}%".format(round(100 - confidence))
    else:  # Jika tidak ada wajah yang terdeteksi
        predicted_label = "Unknown"
        label_text = None

    predicted_labels.append(predicted_label)
    if true_label == predicted_label:
        category = "TP" if true_label != "Unknown" else "TN"
        if category == "TP":
            TP += 1
        else:
            TN += 1
    else:
        category = "FP" if predicted_label != "Unknown" else "FN"
        if category == "FP":
            FP += 1
        else:
            FN += 1

    results.append({
        "image_name": os.path.basename(image_path),
        "true_label": true_label,
        "predicted_label": predicted_label,
        "confidence_score": label_text if len(faces) > 0 else None,
        "category": category
    })
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    return {
        "results": results,
        "accuracy": accuracy
    }


class Testingip(APIView):
    def get(self,request):
        word_request=request.query_params.get('word',None)
        return Response(data={
            'status':"ok",
            'pesan':'sudah benar',
            "data":word_request
        },status=status.HTTP_200_OK)
class Createuserandimagetraining(APIView):
    parser_classes=[MultiPartParser,FormParser]
    def post(self,request):
        username=request.data.get('username')
        items=models.Datauser.objects.filter(username=username)
        print(items)
        if items.exists():
            return Response('Username sudah ada',status=status.HTTP_400_BAD_REQUEST)
        else:
            print(request.data)
            serial=serializer.Datauserserializer(data=request.data)
            if serial.is_valid():
                serial.save()
                training_dir=os.path.join('media','imagetraining')
                model_name='lbph_model.xml'
                model_save_path=os.path.join('hasiltraining',model_name)
                train_or_update_user_data(
                    training_dir=training_dir,
                    model_save_path=model_save_path,
                    target_user=serial.data.get('username'),
                    target_label=int(serial.data.get('id_user'))
                )
                return Response(serial.data,status=status.HTTP_200_OK)
            else:
                return Response(serial.errors,status=status.HTTP_400_BAD_REQUEST)


class Getuserandimagetrainingall(APIView):
    def get(self,request):
        user=models.Datauser.objects.all()
        items=serializer.Datauserserialforimagetraining(user,many=True)
        return Response(items.data,status=status.HTTP_200_OK)

class Getuserandimagetrainingone(APIView):
    def get(self,request):
        username=request.query_params.get('username',None)
        items=models.Datauser.objects.filter(username=username).first()
        serial=serializer.Datauserserialforimagetraining(items)
        if items is None:
            return Response(serial.errors,status=status.HTTP_400_BAD_REQUEST)
        else:
            print(int(serial.data.get('id_user')))
            return Response(serial.data,status=status.HTTP_200_OK)


class Updateuser(APIView):
    def patch(self,request):
        username=request.query_params.get('username',None)
        items=models.Datauser.objects.get(username=username)
        serial=serializer.Datauserserialforimagetraining(items,data=request.data,partial=True)
        if serial.is_valid():
            serial.save()
            return Response(serial.data,status=status.HTTP_200_OK)
        else:
            return Response(serial.errors,status=status.HTTP_400_BAD_REQUEST)


class Deleteuser(APIView):
    def delete(self,request):
        username = request.query_params.get('username', None)
        try:
            items = models.Datauser.objects.get(username=username)
            log_file = f"{items.username}_trained_images.log"
        except models.Datauser.DoesNotExist:
            return Response("Username not found", status=status.HTTP_400_BAD_REQUEST)
        imagetrain = models.Userimagetraining.objects.filter(id_user=items.id_user)
        if not imagetrain.exists():
            return Response("Images not found", status=status.HTTP_400_BAD_REQUEST)
        for images in imagetrain:
            image_path = images.image_user.path
            if os.path.exists(image_path):
                os.remove(image_path)
            images.delete()
            clear_log_for_user(log_file)
        # Delete the directory if it exists
        image_directory = os.path.join('media', 'imagetraining', items.username)
        if os.path.exists(image_directory):
            try:
                shutil.rmtree(image_directory)
            except PermissionError:
                return Response("Permission denied while deleting image directory", status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        items.delete()
        return Response("User data and image have been deleted", status=status.HTTP_200_OK)


class Addimagetraining(APIView):
    parser_classes=[MultiPartParser,FormParser]
    def post(self,request):
        username=request.data.get('username',None)
        items=models.Datauser.objects.get(username=username)
        print(items.username)
        images=request.FILES.getlist('image_list')
        if not items:
            return Response({
                "error":"username has invalid"
            },status=status.HTTP_400_BAD_REQUEST)
        if not images:
            return Response({"error": "No images found."}, status=status.HTTP_400_BAD_REQUEST)
        saved_images=[]
        for image in images:
            serial=serializer.Imageusertrainingserializer(data={
                'id_user':items.id_user ,
                'image_user':image
            })
            if serial.is_valid():
                serial.save()
                training_dir=os.path.join('media','imagetraining')
                model_name='lbph_model.xml'
                model_save_path=os.path.join('hasiltraining',model_name)
                train_or_update_user_data(
                    training_dir=training_dir,
                    model_save_path=model_save_path,
                    target_user=items.username,
                    target_label=int(items.id_user)
                )
                saved_images.append(serial.data)
            else:
                return Response(serial.errors,status=status.HTTP_400_BAD_REQUEST)
        return Response(saved_images,status=status.HTTP_200_OK)


class Getimagetrainingall(APIView):
    def get(self,request):
        imageall=models.Userimagetraining.objects.all()
        items=serializer.Imageusertrainingserializer(imageall,many=True)
        return Response(items.data,status=status.HTTP_200_OK)

class Getimagefromoneuser(APIView):
    def get(self,request):
        username=self.request.query_params.get("username",None)
        items_user=models.Datauser.objects.get(username=username)
        items_image=models.Userimagetraining.objects.filter(id_user=items_user.id_user)
        serial=serializer.Imageusertrainingserializer(items_image,many=True)
        if not items_user:
            return Response({
                "error":"username has invalid"
            },status=status.HTTP_400_BAD_REQUEST)
        if not items_image:
            return Response({"error": "No images found."}, status=status.HTTP_400_BAD_REQUEST)
        return Response(serial.data,status=status.HTTP_200_OK)

class Updateimagetraining(APIView):
    def patch(self,request):
        data=request.query_params.get('username')
        images=request.FILES.getlist('images')
        items_user=models.Datauser.objects.get(username=data)
        items_image=models.Userimagetraining.objects.filter(id_user=items_user.id_user)
        saved_images=[]
        if not items_user:
            return Response({
                "error":"username has invalid"
            },status=status.HTTP_400_BAD_REQUEST)
        if not items_image:
            return Response({"error": "No images found."}, status=status.HTTP_400_BAD_REQUEST)
        for image in items_image:
            if os.path.isfile(image.image_user.path):
                os.remove(image.image_user.path)
            image.delete()
        for imagedata in images:
            serial=serializer.Imageusertrainingserializer(data={
                'id_user':items_user.id_user,
                'image_user':imagedata
            },partial=True)
            if serial.is_valid():
                serial.save()
                training_dir=os.path.join('media','imagetraining')
                model_name='lbph_model.xml'
                model_save_path=os.path.join('hasiltraining',model_name)
                train_replace_user_data(
                    training_dir=training_dir,
                    model_save_path=model_save_path,
                    target_user=items_user.username,
                    target_label=int(items_user.id_user)
                )
                saved_images.append(serial.data)
            else:
                return Response(serial.errors,status=status.HTTP_400_BAD_REQUEST)
            return Response(saved_images,status=status.HTTP_200_OK)

class Deleteimagetraining(APIView):
    def delete(self,request):
        data=request.query_params.get('username')
        items_user=models.Datauser.objects.get(username=data)
        log_file = f"{items_user.username}_trained_images.log"
        items_image=models.Userimagetraining.objects.filter(id_user=items_user.id_user)
        if not items_user:
            return Response({
                "error":"username has invalid"
            },status=status.HTTP_400_BAD_REQUEST)
        if not items_image:
            return Response({"error": "No images found."}, status=status.HTTP_400_BAD_REQUEST)
        for image in items_image:
            if os.path.isfile(image.image_user.path):
                os.remove(image.image_user.path)
            image.delete()
            clear_log_for_user(log_file)
        image_directory = os.path.join('media', 'imagetraining', items_user.username)
        if os.path.exists(image_directory):
            try:
                shutil.rmtree(image_directory)
            except PermissionError:
                return Response("Permission denied while deleting image directory", status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response({"data":"image user has been deleted"},status=status.HTTP_200_OK)


class Createdataruangan(APIView):
    def post(self,request):
        room_name=request.data.get('room_name')
        items=models.Ruangan.objects.filter(room_name=room_name)
        if items.exists():
            return Response({
                "error":'data lab sudah ada'
            },status=status.HTTP_400_BAD_REQUEST)
        else:
            serial=serializer.Ruanganserializer(data=request.data)
            if serial.is_valid():
                serial.save()
                return Response(serial.data,status=status.HTTP_200_OK)
            else:
                return Response(serial.errors,status=status.HTTP_400_BAD_REQUEST)
class Getdataruanganone(APIView):
    def get(self,request):
        room_name=request.query_params.get('room_name')
        items=models.Ruangan.objects.get(room_name=room_name)
        serial=serializer.Ruanganserializer(items)
        if not items:
            return Response({
                "error":"data ruangan lab tidak ada"
            },status.HTTP_200_OK)
        else:
            return Response(serial.data,status.HTTP_200_OK)
class Getdataruanganall(APIView):
    def get(self,request):
        items=models.Ruangan.objects.all()
        print(items)
        serial=serializer.Ruanganserializer(items,many=True)
        if not items:
            return Response({
                "error":"data ruangan not found"
            },status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serial.data,status.HTTP_200_OK)

class Updatedataruangan(APIView):
    def patch(self,request):
        room_name=request.query_params.get('room_name')
        items=models.Ruangan.objects.get(room_name=room_name)
        serial=serializer.Ruanganserializer(items,data=request.data,partial=True)
        if serial.is_valid():
            serial.save()
            return Response(serial.data,status.HTTP_200_OK)
        else:
            return Response(serial.errors,status.HTTP_400_BAD_REQUEST)

class Deletedataruangan(APIView):
    def delete(self,request):
        room_name=request.query_params.get('room_name')
        items=models.Ruangan.objects.get(room_name=room_name)
        if not items:
            return Response({
                "error":'data ruangan tidak ditemukan'
            },status.HTTP_400_BAD_REQUEST)
        else:
            items.delete()
            return Response({
                "data":"data ruangan berhasil dihapus"
            },status.HTTP_200_OK)


class Createlogaccess(APIView):
    def post(self,request):
        # try:
        # Ambil file gambar dari request
        image_file = request.FILES.get("images")
        print('request from raspi',image_file)
        if not image_file:
            return Response({"error": "No image file provided"}, status=status.HTTP_400_BAD_REQUEST)

        room_name = request.data.get("room_name")
        room = models.Ruangan.objects.filter(room_name=room_name).get()
        print(room.room_name)
        if not room:
            return Response({"error": "Room name not found"}, status=status.HTTP_400_BAD_REQUEST)

        # Load gambar ke OpenCV
        image_bytes = image_file.read()
        np_image = np.frombuffer(image_bytes,np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Load data pengguna dari database
        label_to_user = {int(user.id_user): user.username for user in models.Datauser.objects.all()}
        # print(label_to_user)

        # Jalankan face recognition
        model_name = 'lbph_model.xml'
        model_save_path = os.path.join('hasiltraining', model_name)
        results = recognize_from_image(image, model_save_path, label_to_user)
        print(results)
        # Simpan hasil ke database
        image_result = []
        for result in results:
            if not result["id_user"]:
                _, image_buffer = cv2.imencode(".jpeg", result['face_image'])
                file_name = f"Unknown_{room.room_id}.jpeg"
                log_serializer = serializer.Logaccesserializer(data={
                    'id_user': None,
                    'room_id': room.room_id,
                    'image': ContentFile(image_buffer.tobytes(), name=file_name),
                    'status': result['status']
                })
                if log_serializer.is_valid():
                    log_serializer.save()
                    image_result.append(log_serializer.data)
                else:
                    print(f"Error saving log for unknown user: {log_serializer.errors}")
                    return Response(log_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
                return Response(data={
                    'results':image_result,
                    'confidence':result['confidence']
                }, status=status.HTTP_200_OK)
            _, image_buffer = cv2.imencode(".jpeg", result['face_image'])
            file_name = f"{result['username']}_{result['status']}_{room.room_id}.jpeg"
            log_serializer = serializer.Logaccesserializer(data={
                'id_user': result['id_user'],
                'room_id': room.room_id,
                'image': ContentFile(image_buffer.tobytes(), name=file_name),
                'status': result['status']
            })
            if log_serializer.is_valid():
                log_serializer.save()
                image_result.append(log_serializer.data)
            else:
                print(f"Error saving log for known user: {log_serializer.errors}")
                return Response(log_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            return Response(data={
                'results':image_result,
                'confidence':result['confidence']
            }, status=status.HTTP_200_OK)

class Getaccesslogruanganall(APIView):
    def get(self,request):
        items=models.Accesslogruangan.objects.all().order_by('-access_time')[:10]
        serial=serializer.Logaccesserializer2(items,many=True)
        if not items:
            return Response({
                'error':"data ruangan sedang kosong atau error"
            })
        else:
            return Response(serial.data,status.HTTP_200_OK)

class Getaccesslogruanganone(APIView):
    def get(self,request):
        room_request=self.request.query_params.get('room_request')
        room=models.Ruangan.objects.get(room_name=room_request)
        access_log=models.Accesslogruangan.objects.filter(room_id=room.room_id).order_by('access_time')[:10]
        serial=serializer.Logaccesserializer2(access_log,many=True)
        if not access_log or not room:
            return Response({
                'error':'data ruangan atau access log tidak ditemukan'
            })
        else:
            return Response(serial.data,status.HTTP_200_OK)

class Deleteaccesslogruangan(APIView):
    def delete(self,request):
        room_request=self.request.query_params.get('room_request')
        print(room_request)
        item_room=models.Ruangan.objects.get(room_name=room_request)
        item_log=models.Accesslogruangan.objects.filter(room_id=item_room.room_id)
        if not item_room:
            return Response({
                "error":"room request has invalid"
            },status=status.HTTP_400_BAD_REQUEST)
        elif not item_log:
            return Response({"error": "data log tidak tersedia."}, status=status.HTTP_400_BAD_REQUEST)
        else:
            for items in item_log:
                if os.path.isfile(items.image.path):
                    os.remove(items.image.path)
                items.delete()
            image_directory = os.path.join('media', 'tracking', item_room.room_name)
            if os.path.exists(image_directory):
                try:
                    shutil.rmtree(image_directory)
                except PermissionError:
                    return Response("Permission denied while deleting image directory", status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                return Response({"data":"image user has been deleted"},status=status.HTTP_200_OK)


class Creatadeviceruangan(APIView):
    def post(self,request):
        room_name=request.data.get('room_name')
        item_room=models.Ruangan.objects.filter(room_name=room_name).first()
        if not item_room:
            return Response({
                "error":'room request has invalid'
            },status.HTTP_400_BAD_REQUEST)
        room_id=models.Devicesruangan.objects.filter(room_id=item_room.room_id)
        if room_id.exists():
            return Response({
                "error":'data device ruangan sudah ada'
            },status.HTTP_400_BAD_REQUEST)
        serial=serializer.Deviceruanganserializer2(data={
            'room_id':item_room.room_id,
            'status':"online"
        })
        if serial.is_valid():
            serial.save()
            return Response(serial.data,status.HTTP_200_OK)
        else:
            return Response(serial.errors,status.HTTP_400_BAD_REQUEST)

class Getdeviceruanganall(APIView):
    def get(self,request):
        item_room=models.Devicesruangan.objects.all()
        serial=serializer.Deviceruanganserializer(item_room,many=True)
        if not item_room:
            return Response({
                "error":'data device ruangan not found'
            },status.HTTP_400_BAD_REQUEST)
        return Response(serial.data,status.HTTP_200_OK)

class Getdeviceruanganone(APIView):
    def get(self,request):
        room_name=request.query_params.get('room_name')
        item_room=models.Ruangan.objects.filter(room_name=room_name).first()
        item_device=models.Devicesruangan.objects.filter(room_id=item_room.room_id)
        if not item_room:
            return Response({
                "error":'room request has invalid'
            },status.HTTP_400_BAD_REQUEST)
        if not item_device:
            return Response({
                'error':"data device ruangan tidak ditemukan"
            })
        serial=serializer.Deviceruanganserializer(item_device,many=True)
        return Response(serial.data,status.HTTP_200_OK)

class Updatedeviceruangan(APIView):
    def patch(self,request):
        room_name=request.query_params.get('room_name')
        item_room=models.Ruangan.objects.filter(room_name=room_name).first()
        item_device=models.Devicesruangan.objects.filter(room_id=item_room.room_id).first()
        if not item_room:
            return Response({
                "error":'room request has invalid'
            },status.HTTP_400_BAD_REQUEST)
        if not item_device:
            return Response({
                'error':"data device ruangan tidak ditemukan"
            },status.HTTP_400_BAD_REQUEST)
        serial=serializer.Deviceruanganserializer(item_device,data=request.data,partial=True)
        if serial.is_valid():
            serial.save()
            # hisory_serial=serializer.Logdeviceruanganserialzier(data={
            #     'deviceruangan_id':item_device.id_device,
            #     'status':request.data.get('status')
            # })
            # if hisory_serial.is_valid():
            #     hisory_serial.save()
            return Response(serial.data,status.HTTP_200_OK)
        else:
            return Response(serial.errors,status.HTTP_400_BAD_REQUEST)

class Deletedatadeviceruangan(APIView):
    def delete(self,request):
        room_name=request.query_params.get('room_name')
        item_room=models.Ruangan.objects.filter(room_name=room_name).first()
        item_device=models.Devicesruangan.objects.get(room_id=item_room.room_id)
        if not item_room:
            return Response({
                "error":'room request has invalid'
            },status.HTTP_400_BAD_REQUEST)
        elif not item_device:
            return Response({
                'error':"data device ruangan tidak ditemukan"
            },status.HTTP_400_BAD_REQUEST)
        else:
            item_device.delete()
            return Response({
                'data':"data device ruangan berhasil dihapus"
            },status.HTTP_200_OK)

class Createlogdeviceruangan(APIView):
    def post(self,request):
        room_name=request.data.get('room_name')
        item_room=models.Ruangan.objects.filter(room_name=room_name).first()
        if not item_room:
            return Response({
                "error":'room request has invalid'
            },status.HTTP_400_BAD_REQUEST)
        room_id=models.Devicesruangan.objects.filter(room_id=item_room.room_id).first()
        if not room_id:
            return Response({
                "error":'data device ruangan sudah ada'
            },status.HTTP_400_BAD_REQUEST)
        serial=serializer.Logdeviceruanganserialzier(data={
            'deviceruangan_id':room_id.id_device,
            'status':request.data.get('status'),
            'update_at':request.data.get('update_at')
        })
        if serial.is_valid():
            serial.save()
            return Response(serial.data,status.HTTP_200_OK)
        else:
            return Response(serial.errors,status.HTTP_400_BAD_REQUEST)
class Getlogdeviceruangan(APIView):
    def get(self,request):
        room_name=request.query_params.get('device_log_id')
        item_logdevice=models.Logdeviceruangan.objects.filter(deviceruangan_id=room_name)
        if not item_logdevice:
            return Response({
                'error':"data device ruangan tidak ditemukan"
            })
        serial=serializer.Logdeviceruanganserialzier(item_logdevice,many=True)
        return Response(serial.data,status.HTTP_200_OK)
class Deletelogdeviceruangan(APIView):
    def delete(self,request):
        room_name=request.query_params.get('room_name')
        item_room=models.Ruangan.objects.filter(room_name=room_name).first()
        item_device=models.Devicesruangan.objects.get(room_id=item_room.room_id)
        item_log=models.Logdeviceruangan.objects.filter(deviceruangan_id=item_device.id_device)
        if not item_room:
            return Response({
                "error":'room request has invalid'
            },status.HTTP_400_BAD_REQUEST)
        elif not item_device:
            return Response({
                'error':"data device ruangan tidak ditemukan"
            },status.HTTP_400_BAD_REQUEST)
        elif not item_log:
            return Response({
                'error':"data log device ruangan tidak ditemukan"
            },status.HTTP_400_BAD_REQUEST)
        else:
            item_log.delete()
            return Response({
                'message':"data log device berhasil dihapus"
            }, status.HTTP_200_OK)
class Createdatamonitoringruangan(APIView):
    def post(self,request):
        room_name=request.data.get('room_name')
        room_id=models.Ruangan.objects.get(room_name=room_name)
        if not room_id:
            return Response({
                'error':"room yang anda minta tidak ditemukan"
            },status.HTTP_400_BAD_REQUEST)
        access_id=models.Accesslogruangan.objects.filter(room_id=room_id).all().order_by('access_time')
        access_log_list = models.Accesslogruangan.objects.filter(room_id=room_id).order_by('access_time').values_list('log_id', 'access_time')
        print(list(access_log_list))
        device_id=models.Devicesruangan.objects.filter(room_id=room_id).first()
        if not access_id :
            return Response({
                'error':'data log untuk ruangan sesuai request anda tidak ditemukan'
            },status.HTTP_400_BAD_REQUEST)
        if not device_id:
            return Response({
                'error':"data device ruangan sesuai request anda tidak ditemukan"
            },status.HTTP_400_BAD_REQUEST)
        # serial_log=serializer.Logaccesserializer(access_id,many=True)
        serial_monitor=serializer.Monitoringruanganserializer(data={
            # 'access_log':serial_log.data,
            'device_status':device_id.id_device,
            'room_id':room_id.room_id,
            'room_status':'Sedang ada pengunjung'
        })
        if serial_monitor.is_valid():
            monitor=serial_monitor.save()
            monitor.access_log.add(*access_id)
            monitor.save()
            return Response(data={
                "message":"data monitor ruangan berhasil dibuat",
                "data":serial_monitor.data
            },status=status.HTTP_200_OK)
        else:
            return Response(serial_monitor.errors,status.HTTP_400_BAD_REQUEST)

class Getmonitorruanganall(APIView):
    def get(self,request):
        items=models.Monitoringruangan.objects.all()
        serial=serializer.Monitoringruanganserializer3(items,many=True)
        if not items:
            return Response({
                'error':"data ruangan sedang kosong atau error"
            },status.HTTP_400_BAD_REQUEST)
        else:
            return Response(data={
                "message":"Data ruangan Berhasil didapatkan",
                "data":serial.data
            },status=status.HTTP_200_OK)

class Updatemonitorruangan(APIView):
    def patch(self,request):
        room_name=request.query_params.get('room_name')
        room_id=models.Ruangan.objects.get(room_name=room_name)
        if not room_id:
            return Response({'error': 'data ruangan tidak ditemukan'}, status=status.HTTP_400_BAD_REQUEST)
        monitoring = models.Monitoringruangan.objects.get(room_id=room_id)
        if not monitoring:
            return Response({'error': 'Data Monitoringruangan tidak ditemukan untuk room_name tersebut'}, status=status.HTTP_400_BAD_REQUEST)
        access_id=models.Accesslogruangan.objects.filter(room_id=room_id)
        exis_log=monitoring.access_log.all()
        new_logs = access_id.exclude(log_id__in=exis_log.values_list('log_id', flat=True))
        if new_logs.exists():
            monitoring.access_log.add(*new_logs)
        monitoring.save()
        serial=serializer.Monitoringruanganserializer2(monitoring,partial=True)
        return Response(data={
            "message":"Update data monitor ruangan berhasil",
            "data":serial.data
        },status=status.HTTP_200_OK)

class Deletedatamonitor(APIView):
    def delete(self,request):
        room_name = request.query_params.get('room_name')
        room_id=models.Ruangan.objects.get(room_name=room_name)
        if not room_id:
            return Response({'error': 'data ruangan tidak ditemukan'}, status=status.HTTP_400_BAD_REQUEST)
        monitoring=models.Monitoringruangan.objects.get(room_id=room_id.room_id)
        if not monitoring:
            return Response({'error': 'Data Monitoringruangan tidak ditemukan untuk room_name tersebut'}, status=status.HTTP_400_BAD_REQUEST)
        monitoring.access_log.clear()
        monitoring.delete()
        return Response({'message': 'Data Monitoringruangan dan relasi access_log berhasil dihapus'}, status=status.HTTP_200_OK)


# class Createvaluation(APIView):
#     parser_classes = [MultiPartParser,FormParser]
#
#     def post(self, request):
#         images = request.FILES.getlist('images')  # Mengambil daftar file yang diunggah
#         if not images:
#             return Response({"error": "No images were uploaded."}, status=400)
#
#         results = []
#         response_data={}
#         for image in images:
#             temp_image_path=os.path.join('tmp',image.name)
#             with open(temp_image_path, 'wb') as temp_file:
#                 temp_file.write(image.read())
#             try:
#                 model_name='lbph_model.xml'
#                 model_save_path=os.path.join('hasiltraining',model_name)
#
#                 label_to_user = {int(user.id_user): user.username for user in models.Datauser.objects.all()}
#                 image_results = evaluate_model(temp_image_path, model_save_path,label_to_user)
#
#                 # Simpan hasil evaluasi ke database
#                 for result in image_results['results']:
#                     serialized_result = serializer.Evaluationserial(data={
#                         'image_name':result['image_name'],
#                         'true_label':result['true_label'],
#                         'predicted_label':result['predicted_label'],
#                         'confidence_score':result['confidence_score'],
#                         'category':result['category']
#                     })
#                     if serialized_result.is_valid():
#                         serialized_result.save()
#                         results.append(serialized_result.data)
#                     else:
#                         print(f"Error saving log for unknown user: {serialized_result.errors}")
#                         return Response(serialized_result.errors, status=status.HTTP_400_BAD_REQUEST)
#                 response_data={
#                     'results':results,
#                     'accuracy':image_results['accuracy']
#                 }
#             finally:
#                 # Hapus file sementara
#                 print('file pengujian telah disimpan')
#
#         return Response(response_data, status=status.HTTP_200_OK)
