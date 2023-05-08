import numpy as np
import tensorflow as tf
import urllib.request as urlrequest
import imageio
from io import BytesIO
from django.http import HttpResponse
from django.shortcuts import render
from skimage.transform import resize
from urllib.request import urlopen

# Importing model form
from .forms import PredictForm


def home(request):
    html = "<html><body>Tensorflow version is %s.</body></html>" % tf.__version__
    return HttpResponse(html)


def predict_form(request):
    form = PredictForm(request.POST or None)
    result = None
    if form.is_valid():
        url = form.cleaned_data['url']
        with urlopen(url) as file:
            image = imageio.imread(BytesIO(file.read()), pilmode='L')

        image = resize(image, (28, 28))
        result = predict_image(image)
        form.save()

    return render(request, 'tensorapp/predict.html', {'form':form, 'prediction':result})


def predict_image(img):
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    with open("tensorapp/tensormodel/fashion_model.json", 'r') as f:
        model_json = f.read()

    model = tf.keras.models.model_from_json(model_json)
    model.load_weights("tensorapp/tensormodel/fashion_model.h5")

    image_reshaped = tf.reshape(img, [1, 28*28])
    prediction = model.predict(image_reshaped)
    return classes[np.argmax(prediction[0])]
