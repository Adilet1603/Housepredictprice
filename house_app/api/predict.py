from fastapi import APIRouter
from pydantic import BaseModel
import joblib

predict_router = APIRouter(prefix='/predict', tags=['predict Price'])

model = joblib.load('mmodel.pkl')
scaler = joblib.load('mscaler.pkl')


nei = ['Blueste', 'BrDale', '_rkSide', 'ClearCr', 'CollgCr',
       'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV',
       'Mitchel', 'NAmes', 'NPkVill', 'NWAmes', 'NoRidge',
       'NridgHt', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW',
       'Somerst', 'StoneBr', 'Timber', 'Veenker']


class HousePredictSchema(BaseModel):
    GrLivArea: int
    YearBuilt: int
    GarageCars: int
    TotalBsmtSF: int
    FullBath: int
    OverallQual: int
    Neighborhood: str

@predict_router.post('/predict')
async def predict(house: HousePredictSchema):
    house_dict = house.dict()


    new_neighborhood = house_dict.pop('Neighborhood')


    neighborhood_0_1 = [1 if new_neighborhood == i else 0 for i in nei]


    features = list(house_dict.values()) + neighborhood_0_1

    scaled_data = scaler.transform([features])
    pred = model.predict(scaled_data)[0]
    return {"Price": round(pred)}