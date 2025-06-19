import logging
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from astrapy import DataAPIClient
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG Implementation API with Fallback Mechanism")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE")
ASTRA_DB_COLLECTION = os.getenv("ASTRA_DB_COLLECTION")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Astra DB client
astra_client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
db = astra_client.get_database(ASTRA_DB_API_ENDPOINT)

# Initialize Google Generative AI client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Logging setup
logging.basicConfig(level=logging.INFO)

@app.post("/generate")
async def generate(request: Request):
    try:
        # Parse JSON data
        data = await request.json()
        logging.info("Received request data: %s", data)

        messages = data.get('messages', [])
        latest_message = messages[-1].get('content') if messages else None

        if not latest_message:
            logging.warning("No message content found in request.")
            raise HTTPException(status_code=400, detail="No message provided.")

        logging.info("Processing latest message: %s", latest_message)

        # Extract additional disease details from the request
        predicted_class = data.get("predicted_class", "").strip().lower()
        severity = float(data.get("severity", 0))
        severity_class = data.get("severity_class", "")

        # Step 0: Handle healthy plant case
        if predicted_class == "healthy" and severity == 0:
            logging.info("Detected healthy plant with 0 severity, generating general care advice.")

            general_care_prompt = """
            The tea plant appears to be healthy with no signs of disease.

            As an AI assistant, provide general advice on how to maintain the health of tea plants. 
            Include tips on watering, pruning, fertilization, sunlight exposure, and disease prevention best practices.
            Provide a response within 400 words and give response like a plant healthcare professional.
            Answer should be easy to understand by a non-technical user like farmers and plants caretakers.
            Response should be academically correct and to the point do not greet or interact with the user.
            """.strip()

            try:
                chat_response = gemini_client.chats.create(
                    model="gemini-1.5-flash",
                    history=[types.Content(role="user", parts=[types.Part(text=general_care_prompt)])],
                )
                response_text = chat_response.send_message(general_care_prompt).text
                logging.info("Gemini model response for healthy plant: %s", response_text)

            except Exception as e:
                logging.error("Error interacting with Gemini chat API for healthy plant: %s", e, exc_info=True)
                raise HTTPException(status_code=500, detail="Error generating general care advice.")

            return JSONResponse(content={"response": response_text})

        # Step 1: Query database for disease context
        try:
            collection = db.get_collection(ASTRA_DB_COLLECTION)
            logging.info("Connected to Astra collection: %s", ASTRA_DB_COLLECTION)

            results = collection.find(
                filter={"disease_name": predicted_class},
                limit=1
            )

            documents = list(results)

            if documents:
                disease_context = documents[0].get("text", "")
                logging.info("Found disease context in database.")
            else:
                disease_context = None
                logging.warning("No matching disease found in database.")

        except Exception as e:
            logging.error("Error querying Astra DB: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Error querying Astra DB.")

        # Step 2: Fallback mechanism
        if not disease_context:
            logging.info("Falling back to Gemini for response.")

            fallback_prompt = f"""
            You are an AI assistant knowledgeable about tea plant diseases. The user has provided details about a disease:

            Disease Name: {predicted_class}
            Severity: {severity}%
            Severity Classification: {severity_class}

            Based on this information, generate a detailed and human-like response about the disease, including possible prevention methods.
            Provide a response within 500 words and give response like a plant healthcare professional.
            Answer should be easy to understand by a non-technical user like farmers and plants caretakers.
            Response should be academically correct and to the point do not greet or interact with the user.
            """.strip()

            try:
                chat_response = gemini_client.chats.create(
                    model="gemini-1.5-flash",
                    history=[types.Content(role="user", parts=[types.Part(text=fallback_prompt)])],
                )
                response_text = chat_response.send_message(fallback_prompt).text
                logging.info("Gemini model response: %s", response_text)

            except Exception as e:
                logging.error("Error interacting with Gemini chat API: %s", e, exc_info=True)
                raise HTTPException(status_code=500, detail="Error interacting with Gemini chat.")

            return JSONResponse(content={"response": response_text})

        # Step 3: Construct response with context from database
        full_context = f"Disease Details:\n{disease_context}\n\nSeverity: {severity}%\nSeverity Classification: {severity_class}".strip()

        response_prompt = f"""
        You are an AI assistant. Use the context below to answer the user's question accurately.
        Provide a response within 500 words and give response like a plant healthcare professional.
        Answer should be easy to understand by a non-technical user like farmers and plants caretakers.
        Response should be academically correct and to the point do not greet or interact with the user.

        ------------------------
        Context:
        {full_context}
        ------------------------

        Question: {latest_message}
        """.strip()

        try:
            chat_response = gemini_client.chats.create(
                model="gemini-1.5-flash",
                history=[types.Content(role="user", parts=[types.Part(text=response_prompt)])],
            )
            response_text = chat_response.send_message(response_prompt).text
            logging.info("Gemini model response: %s", response_text)

        except Exception as e:
            logging.error("Error interacting with Gemini chat API: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Error interacting with Gemini chat.")

        return JSONResponse(content={"response": response_text})

    except Exception as err:
        logging.error("Unhandled error: %s", err, exc_info=True)
        raise HTTPException(status_code=500, detail=str(err))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8725)
