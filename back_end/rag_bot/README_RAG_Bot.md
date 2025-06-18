
# 🤖 RAG Bot: Retrieval-Augmented Generation with Fallback

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-✔️-green)
![GeminiAPI](https://img.shields.io/badge/Google_Gemini-✔️-orange)
![AstraDB](https://img.shields.io/badge/Astra_DB-✔️-purple)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

> **A FastAPI-based chatbot leveraging Retrieval-Augmented Generation (RAG) with fallback mechanisms for enriched and reliable responses.**

---

## 🖋️ About

The **RAG Bot** is an advanced conversational AI application designed to:
- Retrieve relevant information about tea plant diseases from **Astra DB**.
- Use **Google's Generative AI (Gemini)** for enhanced response generation.
- Provide fallback responses when database context is missing.
- Offer real-time interaction through a high-performance API.

---

## 🚀 Features

- **Retrieval-Augmented Generation (RAG)**: Combines database retrieval with generative AI to produce context-rich responses.
- **Fallback Mechanism**: Ensures meaningful outputs even when the database lacks relevant information.
- **Disease Contextualization**: Uses severity metrics and disease classifications to tailor advice.
- **API Integration**: Provides endpoints for seamless client interaction.
- **Scalable and Secure**: Utilizes environment variables and robust error handling.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or above
- Required libraries (see `requirements.txt`)
- Astra DB and Gemini API credentials

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-bot.git
   cd rag-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in a `.env` file:
   ```plaintext
   ASTRA_DB_NAMESPACE=your_db_namespace
   ASTRA_DB_COLLECTION=your_collection_name
   ASTRA_DB_API_ENDPOINT=your_api_endpoint
   ASTRA_DB_APPLICATION_TOKEN=your_application_token
   GEMINI_API_KEY=your_gemini_api_key
   ```

4. Start the server:
   ```bash
   uvicorn rag_bot:app --host 0.0.0.0 --port 56125
   ```

5. Access the API at `http://127.0.0.1:56125`.

---

## 📂 File Structure

```plaintext
📦 RAG_Bot
├── rag_bot.py                # Main API logic for RAG bot
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .env                      # Environment variables (not included in repo for security)
```

---

## 🖼️ API Endpoints

### `/generate`
- **Method**: POST
- **Input**: JSON payload with the following fields:
  - `messages`: List of user messages in the conversation.
  - `predicted_class`: Disease name predicted (optional).
  - `severity`: Severity percentage (optional).
  - `severity_class`: Severity category (optional).
- **Response**: JSON with the generated response.

---

## 🧪 How It Works

1. User sends a message via the `/generate` endpoint.
2. The bot retrieves disease-related context from Astra DB.
3. If context is unavailable, it falls back to Google's Gemini API for a detailed response.
4. Outputs include a human-like, contextually accurate answer.

---

## 🔗 Dependencies

1. **Astra DB**: [Astra DB Documentation](https://www.datastax.com/astra)
2. **Google Gemini API**: [Gemini API Documentation](https://ai.google)

---

## 🎨 Preview

![Preview](https://via.placeholder.com/800x400?text=RAG+Bot+Interaction+Preview)

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and create a pull request with your feature or bug fix.

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
