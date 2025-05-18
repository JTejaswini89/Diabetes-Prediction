from django.shortcuts import render
import os
import pickle

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH=os.path.join(BASE_DIR,'ml_model','diabetes.pkl')
with open(MODEL_PATH,'rb') as file:
        model=pickle.load(file)

# Create your views here.
def diabetes_fn(request):
    prediction = None
    if request.method == 'POST':
        pregnancies = float(request.POST.get('pregnancies'))
        glucose = float(request.POST.get('glucose'))
        bloodpressure = float(request.POST.get('bloodpressure'))
        skinthickness = float(request.POST.get('skinthickness'))
        insulin = float(request.POST.get('insulin'))
        bmi = float(request.POST.get('bmi'))
        diabetespedigreefunction = float(request.POST.get('diabetespedigreefunction'))
        age = float(request.POST.get('age'))
        
        features = [
            pregnancies, glucose, bloodpressure, skinthickness,
            insulin, bmi, diabetespedigreefunction, age
        ]
        
        prediction = model.predict([features])  
        
    return render(request, 'diabetes.html', {'prediction': prediction[0] if prediction is not None else None})

