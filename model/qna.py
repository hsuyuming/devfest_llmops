from pydantic import BaseModel, Field

class QnA(BaseModel):
    question: str = Field(default="What were alphabet revenues in 2022?")