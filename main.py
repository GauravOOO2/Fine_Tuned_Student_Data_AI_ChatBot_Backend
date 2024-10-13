import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",  # Allow your frontend's URL for local development
    "https://fine-tuned-student-data-ai-chatbot.onrender.com",  # Allow your deployed frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Configure Gemini API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise RuntimeError("GEMINI_API_KEY is not set in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

# MongoDB setup
MONGODB_URI = os.getenv("MONGODB_URI")
if MONGODB_URI is None:
    raise RuntimeError("MONGODB_URI is not set in environment variables.")

client = AsyncIOMotorClient(MONGODB_URI)
db = client['chatbotDB']  # Database name
chat_collection = db['chatHistory']  # Collection name

# Pydantic model for request body
class PromptRequest(BaseModel):
    prompt: str

# Function to save chat messages to MongoDB
async def save_chat_to_db(sender: str, text: str):
    try:
        chat_message = {
            "sender": sender,
            "text": text,
            "timestamp": datetime.utcnow()
        }
        print(f"Saving chat message from {sender} to MongoDB...")
        await chat_collection.insert_one(chat_message)
        print(f"Chat message from {sender} saved successfully.")
    except Exception as e:
        print(f"Error saving chat message: {str(e)}")
        raise e

@app.post("/generate-content/")
async def generate_content(request: PromptRequest):
    try:
        print("Received prompt:", request.prompt)  # Log the prompt received
        
        fine_tuned_model_id = "tunedModels/school-model-7461"
        model = genai.GenerativeModel(model_name=fine_tuned_model_id)
        
        print("Generating content...")  # Log before the generation call
        response = model.generate_content(request.prompt)
        print("Content generated:", response.text)  # Log after generation
        
        chat_message = {
            "messages": [
                {"sender": "user", "text": request.prompt},
                {"sender": "bot", "text": response.text}
            ],
            "timestamp": datetime.utcnow()
        }
        
        print("Saving to MongoDB...")  # Log before DB operation
        await chat_collection.insert_one(chat_message)
        print("Saved successfully!")  # Log after successful save
        
        return {"generated_text": response.text}
    
    except Exception as e:
        print("Error occurred:", str(e))  # Log the error
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint to get chat history from MongoDB
@app.get("/chat-history/")
async def get_chat_history():
    try:
        print("Fetching chat history from MongoDB...")  # Log fetching history
        chat_history = await chat_collection.find().to_list(100)
        print(f"Fetched {len(chat_history)} chat messages.")  # Log count
        
        chat_history = [
            {
                "_id": str(msg["_id"]),  # Convert ObjectId to string
                "messages": msg.get("messages", []),  # Use .get() to avoid KeyError
                "timestamp": msg.get("timestamp")  # Access the timestamp
            }
            for msg in chat_history
        ]
        
        return chat_history
    except Exception as e:
        print("Error fetching chat history:", str(e))  # Log the error
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {str(e)}")

# Root endpoint to verify server is running
@app.get("/")
def read_root():
    return {"message": "Gemini API FastAPI server is running"}

