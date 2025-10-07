# src/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class PredictIn(BaseModel):
    text: str = Field(..., description="Input text to classify")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional decision threshold")

class PredictOut(BaseModel):
    pred: str = Field(..., description="Predicted label: 'spam' or 'ham'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Spam probability in [0,1]")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Threshold used for the decision")

class BatchIn(BaseModel):
    texts: List[str] = Field(..., description="List of texts to classify")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional decision threshold")

class BatchOut(BaseModel):
    pred: List[str] = Field(..., description="Predicted labels for each input")
    confidence: List[float] = Field(..., description="Spam probabilities in [0,1]")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Threshold used for the decision")
