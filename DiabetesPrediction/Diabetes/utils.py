import os
import pickle

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH=os.path.join(BASE_DIR,'ml_model','diabetes.pkl')
with open(MODEL_PATH,'rb') as file:
        model=pickle.load(file)
def predict_results(features):
        prediction=utils.predict_result(features)
        return prediction[0]
