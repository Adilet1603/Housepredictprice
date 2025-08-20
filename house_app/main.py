import uvicorn
from fastapi import FastAPI
from house_app.api import predict

house_app = FastAPI()
house_app.include_router(predict.predict_router)


if __name__== '__main__':
    uvicorn.run(house_app, host='127.0.0.1', port=8000)