# AgriGPT – Multimodal AI Farming Assistant

AgriGPT is an end-to-end **multimodal, multi-agent agricultural advisory system** designed to provide reliable, domain-specific, and safety-aware guidance to farmers. The system supports **text queries, image-based crop diagnosis, and multi-turn conversations**, while intelligently routing requests to specialized expert agents such as crop management, pest diagnostics, irrigation, yield analysis, and government subsidies.

Unlike normal chatbots, AgriGPT decomposes user's complex intent and coordinates multiple constrained expert agents, ensuring accurate responses, reduced hallucinations, and explainable decision-making.

---

## Key Features:

- Multi-agent architecture with strict role boundaries
- Multimodal support (text-only, image-only, text + image)
- Vision-based crop pest and disease diagnostics
- LLM-based intent routing with confidence scoring
- Retrieval-Augmented Generation (RAG) for subsidy information
- Session-based conversational memory
- Safety-first prompting and execution constraints
- Production-grade FastAPI backend
- Modern, animated React frontend

---



## System Architecture Overview:



### Backend (FastAPI + Python)

- Intent routing using LLM relevance scoring
- Multi-agent orchestration and execution
- Image-based vision analysis using Groq vision models
- Retrieval-based subsidy guidance using FAISS
- Session memory and query history logging
- Weather data integration
- Robust input validation and retry logic

### Frontend (React + TypeScript)

- Real-time conversational chat interface
- Image upload with preview and validation
- Multimodal chat flow (text + image)
- Weather-aware UI header
- Conversation history viewer

### LLM Models

 - Text Model: llama-3.1-70b-versatile
 - Vision Model: llama-3.2-vision-preview

---

##  Agent-Based Design:

AgriGPT uses a **multi-agent system**, where each agent is a domain specialist with enforced boundaries.

| Agent | Responsibility |
|-------|----------------|
| CropAgent | Crop cultivation practices, fertilizers, soil preparation |
| PestAgent | Pest, disease, and visible symptom diagnosis (image-first) |
| IrrigationAgent | Irrigation frequency and water management |
| YieldAgent | Yield-related issue analysis and limiting factors |
| SubsidyAgent | Government subsidy and scheme information (RAG-based) |
| FormatterAgent | Final synthesis and presentation-only agent |

Each agent **cannot exceed its domain scope**, preventing unsafe or hallucinated advice.

---

##  Intelligent Query Routing:

AgriGPT includes an LLM-powered intent router that:

- Scores all available agents on a **0–100 relevance scale**
- Selects **one dominant (primary) agent**
- Adds **supporting agents only if relevance exceeds thresholds**
- Enforces a maximum of **3 agents per query**
- Automatically includes **PestAgent** when an image is present

This ensures focused, efficient, and interpretable responses.

---

##  Multimodal Query Handling:

AgriGPT supports three interaction modes:

- **Text-only queries** → Routed via intent scoring
- **Image-only queries** → Direct PestAgent diagnosis
- **Text + Image queries** → Combined reasoning across agents


---

##  Retrieval-Augmented Generation (RAG):

Subsidy-related queries use a verified data pipeline:

- FAISS vector store
- Sentence-transformer embeddings
- Curated `subsidies.json` dataset

This ensures:

- No hallucinated government schemes
- Eligibility and benefits grounded in official data
- Transparent, explainable responses

---

##  Conversational Memory:

- Frontend stores a persistent `session_id`
- Backend maintains the **last 10 messages per session**
- Used for contextual routing (pronouns, follow-ups, references)
- Prevents loss of conversational coherence

---

##  Logging and Observability:

Every interaction is logged with:

- Timestamp (UTC)
- Agent name
- Query type (text / image / multimodal)
- Truncated response content
- Optional metadata

Logs automatically rotate after reaching size limits to ensure stability.

---

##  Weather Integration:

The frontend integrates location-based weather insights:

- Temperature
- Humidity
- Wind speed
- General condition (sunny / cloudy / rainy)

Powered by the OpenWeather API and used to enhance farming context.

---

##  Technology Stack:

### Backend

- Python 3.10+
- FastAPI
- Groq LLMs (text + vision)
- LangChain
- FAISS
- Pydantic Settings
- OpenWeather API

### Frontend

- React + TypeScript
- Vite
- Tailwind CSS
- Framer Motion
- Axios
- Zustand 

---

##  Running the Project Locally:

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Environment Configuration

Create a `.env` file in the backend directory:

```env
GROQ_API_KEY=your_groq_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

---

##  API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/ask/text` | Text-only farming queries |
| `/ask/image` | Image-only crop diagnosis |
| `/ask/chat` | Multimodal chat (text + image) |
| `/weather/current` | Location-based weather |
| `/health` | Backend health check |

---

##  License

This project is intended for academic purposes only.

---

##  Authors

**Akshay Keerthi Adhikasavan Suresh**  , **Anish Gawde** 

---  
