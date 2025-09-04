# RAG Chatbot

## Backend
First, cd to folder backend;

create file .env with content:
```
LANGSMITH_API_KEY=YOUR_LANGSMITH_API_KEY
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
MILVUS_URI=YOUR_MILVUS_URI
MILVUS_TOKEN=YOUR_MILVUS_TOKEN
```

to run backend, run:
```
uvicorn app:app --reload --port 8000
```

or run file main.py:
```
python main.py
```

## Frontend
First, cd to folder frontend;

to install npm libraries, run:
```
npm install
```

to run frontend, run:
```
npm run dev
```

to access the chatbot, go to link:
```
localhost:3000
```

## Expriment
To run the experiment, cd to folder backend and run:
```
python experiment.py
```