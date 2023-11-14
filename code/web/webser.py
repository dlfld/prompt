from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class PosItem(BaseModel):
    model_id: str
    model_detail_id: str
    text: str

# 调用指定的模型进行词性标注
@app.post("/pos")
def postdate(posItem: PosItem):  
    return postag(posItem)



class PosTrain(BaseModel):
    id: str
    data_id: str
    data_name: str
    model_id:str
    plm_id:str
    model_name:str
    train_size_id:str

# 调用指定的模型进行词性标注
@app.post("/train")
def postdate(posTrain: PosTrain):  
    return pos_train(posTrain)
