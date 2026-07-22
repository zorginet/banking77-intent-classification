"""A service for deploying an intent classification model using FastAPI."""

import json
import os
import warnings

import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Hiding UserWarning messages from third-party libraries to keep logs clean
warnings.filterwarnings("ignore", category=UserWarning)

# Initializing the FastAPI app with metadata
app = FastAPI(
    title="Intent Classification on Banking77",
    description="This API classifies customer intents based on the Banking77 dataset.",
)


# MLP architecture for classifying embeddings from SentenceTransformer
class IntentMLP(nn.Module):
    def __init__(self, input_dim):
        super(IntentMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(512, 77)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# Defining the absolute path to the folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Local paths to the saved models and class dictionary
MODEL_ST_PATH = os.path.join(BASE_DIR, "models/sentence_transformer_local")
MODEL_MLP_PATH = os.path.join(BASE_DIR, "models/intent_mlp_st_weights.pt")
INTENTS_MAP_PATH = os.path.join(BASE_DIR, "models/intents_map.json")

# Loading the mapping of identifiers to textual intent names
with open(INTENTS_MAP_PATH, "r", encoding="utf-8") as f:
    intents_map = json.load(f)

# Initializing models and loading trained weights on CPU for inference
st_model = SentenceTransformer(MODEL_ST_PATH)
mlp_model = IntentMLP(input_dim=st_model.get_embedding_dimension())
mlp_model.load_state_dict(torch.load(MODEL_MLP_PATH, map_location=torch.device("cpu")))
mlp_model.eval()


# Input data schema for the /predict endpoint
class PredictRequest(BaseModel):
    text: str = Field(...)

    model_config = {
        "json_schema_extra": {
            "example": {"text": "I want to check my account balance."}
        }
    }


# Output data schema for the server response
class PredictResponse(BaseModel):
    intent: str
    confidence: float = Field(..., ge=0, le=100)
    model_config = {
        "json_schema_extra": {
            "example": {"intent": "check_balance", "confidence": 95.0}
        }
    }


# Root endpoint for quick service health check
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Banking77 Intent Classification API!",
        "description": "This API classifies customer intents based on the Banking77 dataset.",
    }


# Endpoint for classifying customer text
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    text_input = request.text

    # Generating text embedding and preparing batch dimension
    embedding = st_model.encode(text_input, convert_to_tensor=True)
    embedding = embedding.unsqueeze(0)

    # Forward pass through the MLP and calculation of final probabilities
    with torch.no_grad():
        logits = mlp_model(embedding)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        intent = intents_map.get(str(predicted_class.item()), "unknown_intent")
    return PredictResponse(
        intent=intent, confidence=round(float(confidence.item()) * 100, 2)
    )
