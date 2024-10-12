from pydantic import BaseModel
from typing import List

class DataModel(BaseModel):
    Textos_espanol: List[str]  # Recibe una lista de textos

    def columns(self):
        return ["Textos_espanol"]
