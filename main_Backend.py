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
    "http://localhost:3000",  # Allow your frontend's URL
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
    chat_message = {
        "sender": sender,
        "text": text,
        "timestamp": datetime.utcnow()
    }
    await chat_collection.insert_one(chat_message)

@app.post("/generate-content/")
async def generate_content(request: PromptRequest):
    try:
        # Use your fine-tuned model instead of the base model
        fine_tuned_model_id = "tunedModels/school-model-7461"
        
        # Initialize the fine-tuned model
        model = genai.GenerativeModel(model_name=fine_tuned_model_id)
        
        # Generate content using the prompt provided in the request
        response = model.generate_content(request.prompt)
        
        # Save user prompt and AI-generated response in MongoDB as a single document
        chat_message = {
            "messages": [
                {"sender": "user", "text": request.prompt},
                {"sender": "bot", "text": response.text}
            ],
            "timestamp": datetime.utcnow()
        }
        await chat_collection.insert_one(chat_message)
        
        # Return the AI-generated content as JSON
        return {"generated_text": response.text}
    
    except Exception as e:
        # Return an HTTP 500 error if something goes wrong
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint to get chat history from MongoDB
@app.get("/chat-history/")
async def get_chat_history():
    try:
        # Fetch last 100 chat messages from MongoDB
        chat_history = await chat_collection.find().to_list(100)
        
        # Convert BSON documents to dictionaries
        chat_history = [
            {
                "_id": str(msg["_id"]),  # Convert ObjectId to string
                "messages": msg.get("messages", []),  # Use .get() to avoid KeyError
                "timestamp": msg.get("timestamp")  # Access the timestamp
            }
            for msg in chat_history
        ]
        
        print("Fetched chat history:", chat_history)  # Debugging
        return chat_history
    except Exception as e:
        print("Error fetching chat history:", str(e))  # Log the error for debugging
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {str(e)}")

# Root endpoint to verify server is running
@app.get("/")
def read_root():
    return {"message": "Gemini API FastAPI server is running"}
