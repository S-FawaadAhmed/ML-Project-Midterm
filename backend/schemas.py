from pydantic import BaseModel, Field
from typing import Optional, Dict


class OrderPredictionInput(BaseModel):
    """Every field is optional. Each model checks its own required features."""
    Quantity: Optional[int] = Field(None, ge=1, le=100, description="Order quantity")
    Unit_Price_INR: Optional[float] = Field(None, gt=0, description="Unit price in INR")
    Discount: Optional[float] = Field(None, ge=0.0, le=1.0, description="Discount rate between 0.0 and 1.0")
    Category: Optional[str] = Field(None, example="Technology")
    Subcategory: Optional[str] = Field(None, example="Phones")
    Region: Optional[str] = Field(None, example="North")
    Segment: Optional[str] = Field(None, example="Consumer")
    Channel: Optional[str] = Field(None, example="Online")
    Payment_Method: Optional[str] = Field(None, example="Credit Card")
    Order_Month: Optional[int] = Field(None, ge=1, le=12, description="Month of order (1-12)")
    Order_DayOfWeek: Optional[int] = Field(None, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")


class PredictionResponse(BaseModel):
    model_key: str
    model_name: str
    target: str
    task: str
    predicted_profit_inr: Optional[float] = None
    predicted_class: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    status: str = "success"


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
