from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Literal
import joblib

app = FastAPI(title='Income Prediction API', version='1.0')

# CORS: lets index.html call the API when it is opened from a file or another address
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # fine for local development; restrict before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)

#universal model loading
MODEL_PATH = Path(__file__).parent / 'model.pkl'
INDEX_PATH = Path(__file__).parent / 'index.html'
model = joblib.load(MODEL_PATH)

#Literal restricts each categorical field to the exact values the model was trained on. User can only input values which are in this literals for this    

Workclass = Literal[
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
    "State-gov", "Without-pay", "Never-worked", "Unknown",
]
MaritalStatus = Literal[
    "Married-civ-spouse", "Divorced", "Never-married", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse",
]
Occupation = Literal[
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces", "Unknown",
]
Relationship = Literal[
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried",
]
Race = Literal["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
Sex = Literal["Male", "Female"]
NativeCountry = Literal[
    "United-States", "Cambodia", "Canada", "China", "Columbia", "Cuba",
    "Dominican-Republic", "Ecuador", "El-Salvador", "England", "France", "Germany",
    "Greece", "Guatemala", "Haiti", "Holand-Netherlands", "Honduras", "Hong",
    "Hungary", "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos",
    "Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines",
    "Poland", "Portugal", "Puerto-Rico", "Scotland", "South", "Taiwan", "Thailand",
    "Trinadad&Tobago", "Vietnam", "Yugoslavia", "Unknown",
]

class IncomeInputs(BaseModel):

    model_config = ConfigDict(populate_by_name=True)

    #'Unknown' defaults ONLY where 'Unknown' existed in training data
    #and some columns contained '?' as null values this would generate an error

    #categorical columns
    workclass: Workclass = "Unknown"
    marital_status: MaritalStatus = Field(alias="marital-status")
    occupation: Occupation = "Unknown"
    relationship: Relationship
    race: Race
    sex: Sex
    native_country: NativeCountry = Field(default="Unknown", alias="native-country")
    

    #numerical columns 
    age: int = Field(ge=17, le=100)
    education_num: int = Field(alias="education-num", ge=1, le=16)
    capital_gain: float = Field(default=0, alias="capital-gain", ge=0)
    capital_loss: float = Field(default=0, alias="capital-loss", ge=0)
    hours_per_week: float = Field(alias="hours-per-week", gt=0, le=99)

@app.get("/")
def home():
    return FileResponse('index.html')

@app.post("/predict")
def predict(data : IncomeInputs):
 
    df = pd.DataFrame([data.model_dump(by_alias=True)])
    pred = int(model.predict(df)[0])
    prob_high = float(model.predict_proba(df)[0][1])

    return {
        "prediction" : '>50K' if pred == 1 else '<=50K',
        'probability_over_50K' : round(prob_high, 4)
    }