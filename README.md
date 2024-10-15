
# **AI Chatbot for Student Data Management**

## Overview

This project is an AI-driven chatbot fine-tuned with the Gemini 1.5 Flash LLM model. It interacts with a dataset of fictional student information, answering questions about student profiles while handling both in-dataset and out-of-dataset queries. The chatbot is built with a **FastAPI** backend and a **React** frontend, ensuring seamless communication and efficient response handling. It also features chat history storage using **MongoDB**, with session persistence via local storage, and provides users with a dedicated history page for reviewing past interactions.

## Features

- **AI-Driven Responses**: Fine-tuned with student data using Gemini 1.5 Flash LLM to provide precise answers to queries.
- **FastAPI Backend**: Efficient API handling and scalable architecture for seamless server-client communication.
- **React Frontend**: Interactive user interface with real-time communication and response rendering.
- **MongoDB Integration**: Stores chat history dynamically, allowing real-time retrieval and review of previous conversations.
- **Local Storage**: Persists chat history across sessions for better user experience and continuity.
- **Dedicated History Page**: Displays a complete log of past conversations for easy reference.
- **In-Dataset and Out-of-Dataset Handling**: Accurately manages both known and unknown queries with appropriate responses.

## Tech Stack

- **Backend**: FastAPI, Python
- **Frontend**: React, JavaScript
- **Database**: MongoDB (for chat history)
- **Model**: Gemini 1.5 Flash LLM (fine-tuned with student data)
- **Deployment**: Google Cloud SDK, Google AI Studio
- **Storage**: Local Storage (for session persistence)

## Setup and Installation

### Prerequisites

Ensure you have the following installed:

- **Python 3.7+**
- **Node.js and npm** (for React frontend)
- **MongoDB** (for storing chat history)
- **Google Cloud SDK** (for model fine-tuning and deployment)
- **FFmpeg** (for Flask backend audio handling)
- **Gemini API Key** and **Model ID**

### Backend Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/GauravOOO2/Fine_Tuned_Student_Data_AI_ChatBot_Backend.git
   cd Fine_Tuned_Student_Data_AI_ChatBot_Backend
   ```

   ```bash
   git clone https://github.com/GauravOOO2/Fine_Tuned_Student_Data_AI_ChatBot_FrontEnd.git
   cd Fine_Tuned_Student_Data_AI_ChatBot_FrontEnd
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**

   Create a `.env` file in the backend folder and add your credentials:

   ```bash
   GEMINI_API_KEY=<your-gemini-api-key>
   MONGODB_URI=<your-mongodb-uri>
   ```

4. **Run the FastAPI server:**

   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup

1. **Navigate to the frontend folder:**

   ```bash
   cd ../frontend
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Start the React development server:**

   ```bash
   npm start
   ```

### MongoDB Setup

1. **Ensure MongoDB is running locally or provide a connection URI.** The chat history is stored dynamically in MongoDB.
2. **Set the MongoDB URI in the backend's `.env` file.**

### Model Fine-Tuning

1. **Fine-tune the Gemini 1.5 Flash LLM** using **Google Cloud SDK** and **AI Studio**. The dataset used contains fictional student information in JSON format.

2. **Model ID and API Key** should be included in the `.env` file.

## Usage

Once both the backend and frontend servers are running, you can access the chatbot by navigating to:

```
http://localhost:3000
```

1. **Chat with the Bot**: Ask questions about students, and the chatbot will respond using the fine-tuned Gemini model.
2. **Chat History**: Check the "History" tab to view past conversations.
3. **Session Persistence**: The chatbot will remember past conversations using local storage.

## Example Queries

- "What is the age of John Doe?"
- "Who is Jane Smith's oncologist?"
- "Which school does Alex attend?"
- "What is the percentage of Lily Thompson?"

For any out-of-dataset queries, the chatbot will gracefully handle them with appropriate fallback responses.

## Project Structure

```
.
├── backend
│   ├── app
│   │   ├── main.py         # FastAPI backend implementation
│   │   ├── database.py     # MongoDB connection and schema
│   │   └── models.py       # Defines schema for chat history
├── frontend
│   ├── public              # Public files (index.html, etc.)
│   ├── src
│   │   ├── components      # React components
│   │   ├── pages           # Pages for chat and history
│   │   ├── App.js          # Main React component
│   │   └── index.js        # Entry point for React app
└── README.md
```

## Future Enhancements

- **Support for Multiple Datasets**: Extend the chatbot’s ability to handle multiple datasets or domains.
- **Enhanced Natural Language Understanding**: Improve chatbot’s understanding and flexibility in answering more complex queries.
- **User Authentication**: Add user login and personalized chat history for individual users.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code follows the existing code style and includes relevant tests.


## Contact

If you have any questions or suggestions, feel free to reach out:

- **Gaurav Varma** – [LinkedIn](https://www.linkedin.com/in/gaurav-varm-oo21/)  
- **GitHub** – [YourGitHubProfile](https://github.com/GauravOOO2)

