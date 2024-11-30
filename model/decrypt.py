from pydantic import BaseModel

class DecryptModel(BaseModel):
    encrypt_text: str