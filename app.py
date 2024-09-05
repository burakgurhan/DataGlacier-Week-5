import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional


# Load the pre-trained model
model = joblib.load("random_forest_model.joblib")

# Define a Pydantic model for the input data
class InputData(BaseModel):
    age: int
    bmi: int
    children: Optional[int] = None

# Create a FastAPI app
app = FastAPI()

# Define a route to handle predictions
@app.post("/predict")
def predict(input_data: InputData):
    # Convert input data to a Pandas DataFrame
    data = pd.DataFrame(input_data.dict(), index=[0])

    # Make a prediction using the loaded model
    prediction = model.predict(data)[0]

    return {"prediction": prediction}

if __name__ == "__main__":
    app.run(debug=True)