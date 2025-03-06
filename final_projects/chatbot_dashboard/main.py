#uvicorn main:app --reload

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class TextInput(BaseModel):
    text: str

class QAInput(BaseModel):
    context: str
    question: str

# Function for Text Summarization (Dummy Response)
@app.post("/summarize")
async def summarize(input_data: TextInput):
    try:
        summary = f"This is a dummy summary of the text: {input_data.text[:50]}..."
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Function for Next Word Prediction (Dummy Response)
@app.post("/predict_next_word")
async def predict_next_word(input_data: TextInput):
    try:
        prediction = input_data.text + " [Predicted Next Word]"
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Function for Sentiment Analysis (Dummy Response)
@app.post("/sentiment")
async def sentiment(input_data: TextInput):
    try:
        sentiment = "POSITIVE"
        confidence = 0.99  # Assuming high confidence
        return {"sentiment": sentiment, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Function for Chatbot (Dummy Response)
@app.post("/chat")
async def chat(input_data: TextInput):
    try:
        # Simulating a dummy chatbot response
        response = f"Bot: Hello! You said: {input_data.text}"
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Function for Question Answering (Dummy Response)
@app.post("/qa")
async def qa(input_data: QAInput):
    try:
        answer = "This is a dummy answer based on the context provided."
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Function for Image Generation (Dummy Response)
@app.post("/generate_image")
async def generate_image(input_data: TextInput):
    try:
        image_url = "https://example.com/dummy_image.png"
        return {"image_url": image_url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
