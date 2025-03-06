from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import Optional

app = FastAPI()

# Initialize the Hugging Face pipelines for different use cases
summarizer = pipeline("summarization")
next_word_predictor = pipeline("text-generation", model="gpt2")
sentiment_analyzer = pipeline("sentiment-analysis")
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")
qa_model = pipeline("question-answering")

# Define the input models using Pydantic
class TextInput(BaseModel):
    text: str

class QAInput(BaseModel):
    context: str
    question: str

# Function for Text Summarization
@app.post("/summarize")
async def summarize(input_data: TextInput):
    try:
        summary = summarizer(input_data.text)
        return {"summary": summary[0]['summary_text']}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Function for Next Word Prediction
@app.post("/predict_next_word")
async def predict_next_word(input_data: TextInput):
    try:
        prediction = next_word_predictor(input_data.text, max_length=50, num_return_sequences=1)
        return {"prediction": prediction[0]['generated_text']}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Function for Sentiment Analysis
@app.post("/sentiment")
async def sentiment(input_data: TextInput):
    try:
        sentiment = sentiment_analyzer(input_data.text)
        return {"sentiment": sentiment[0]['label'], "confidence": sentiment[0]['score']}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Function for Chatbot
@app.post("/chat")
async def chat(input_data: TextInput):
    try:
        response = chatbot(input_data.text)
        return {"response": response[0]['generated_text']}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Function for Question Answering
@app.post("/qa")
async def qa(input_data: QAInput):
    try:
        answer = qa_model(question=input_data.question, context=input_data.context)
        return {"answer": answer['answer']}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate_image")
async def generate_image(input_data: TextInput):
    #as now this is a place holder
    raise HTTPException(status_code=400, detail="Image generation feature not implemented.")
