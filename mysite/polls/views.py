from django.shortcuts import render, redirect
import os
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from.models import Inputemo as Input_table
from .forms import EmoForm
from .sustain import train_generator


var = False

def handler(request):
    if var:
        obj = Input_table.objects.values_list('file', flat=True).order_by('-emo_id')[:1]
        path_to_file = obj[0]
        data = np.array(transform_data('polls/my_data/' + str(path_to_file), 224))
        res = predict(data)
        print(res)
    else:
        res = [None]

    return render(request, "index.html", {'response': res[0]})


def uploading(request):
    if request.method == 'POST':
        form = EmoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            global var
            var = True
            return redirect('homepage')
    else:
        form = EmoForm()
    return render(request, 'upload.html', {'form': form})


def transform_data(path, RESIZE = 224):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    _, ftype = os.path.splitext(path)
    if ftype == ".png":
        img = read(path)
        img = cv2.resize(img, (RESIZE, RESIZE))
        IMG.append(np.array(img))

    return IMG


def predict(data):
    model = load_model(r'polls/breast-cancer-model.h5')
    pred = model.predict(data)
    preds = np.argmax(pred, axis=1)
    return preds