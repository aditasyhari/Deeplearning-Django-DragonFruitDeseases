from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
# from starlette.applications import Starlette
# from starlette.middleware.cors import CORSMiddleware
# from starlette.responses import HTMLResponse, JSONResponse

# Create your views here.
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import efficientnet.tfkeras
from tensorflow.keras.models import load_model


# app = Starlette()
# app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])


# parameters
input_size = (250, 250)
channel = (3,)
input_shape = input_size + channel
labels = ['Antraks', 'Bercak Merah', 'Busuk Batang', 'Busuk Hitam', 'Kudis', 'Mosaik']

# prepocessing function
def preprocess(img, input_size):
    nimg = img.convert('RGB').resize(input_size, resample = 0)
    img_arr = (np.array(nimg)) / 255
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis = 0)

# Load Model
MODEL_PATH = './models/pitaya/model10.h5'
model = load_model(MODEL_PATH, compile = False)




def index(request):
    context = {'a':1}
    return render(request,'index.html',context)

def predictImage(request):
    print(request)
    print(request.POST.dict())

    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    testimage = '.'+filePathName

    # predict the image
    img = Image.open(testimage)
    X = preprocess(img, input_size)
    X = reshape([X])
    y = model.predict(X)

    predictedLabel = labels[np.argmax(y)]
    predictedAcc = np.max(y)*100
    # print(labels[np.argmax(y)], np.max(y))

    context = {'filePathName':filePathName,'predictedLabel':predictedLabel,'predictedAcc':predictedAcc}
    return render(request,'index.html',context)

# @app.route('/mobile', methods=['POST'])
async def mobile(request) :
    fileObj = request.FILES['file']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    testimage = '.'+filePathName

    # predict the image
    img = Image.open(testimage)
    X = preprocess(img, input_size)
    X = reshape([X])
    y = model.predict(X)

    predictedLabel = labels[np.argmax(y)]
    # print(labels[np.argmax(y)], np.max(y))

    return JsonResponse({'result': str(predictedLabel)})