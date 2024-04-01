from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json
import pickle


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    
    Typical_Chest_Pain: int
    Region_RWMA: int
    Current_Smoker: int
    Age: int
    Length: int
    EF_TTE: int
    ESR: int
    BMI: int
    K: float

coronary_artery_model = pickle.load(open('cat_model.pickle','rb'))


@app.post('/coronary_artery_prediction')
def brain_stroke_pred(input_parameters : model_input):

    input_data = input_parameters.model_dump_json()
    input_dictionary = json.loads(input_data)
    
    Typical_Chest_Pain = input_dictionary['Typical_Chest_Pain']
    Region_RWMA = input_dictionary['Region_RWMA']
    Current_Smoker = input_dictionary['Current_Smoker']
    Age = input_dictionary['Age']
    Length = input_dictionary['Length']
    EF_TTE = input_dictionary['EF_TTE']
    ESR = input_dictionary['ESR']
    BMI = input_dictionary['BMI']
    K = input_dictionary['K']


    input_list = [Typical_Chest_Pain, Region_RWMA, Current_Smoker, Age, Length, EF_TTE, ESR, BMI, K]

    try:
        prediction = coronary_artery_model.predict([input_list])
        prediction_result = 'CAD Positive' if prediction[0] == 0 else 'Normal'
        print(f"\nThe predicted outcome is: {prediction_result}")
    except Exception as e:
        print(f"Error during prediction: {e}")

    return prediction_result

