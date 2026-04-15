# Medical Claim Processing Pipeline

## Overview

This project is a backend system that processes medical claim PDFs and extracts relevant information such as patient details, discharge summaries, and billing data. The system uses a structured workflow to classify document pages and extract key information using language models.


## Technologies Used

* FastAPI for building the API
* LangChain and OpenAI for language model integration
* LangGraph for workflow orchestration
* PyPDF for PDF text extraction
* Pydantic for data validation
* Python dotenv for environment variable management


## Setup Instructions

### 1. Create the repository

```
git clone <repository-url>
cd assignment-health
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file and add:

```
OPENAI_API_KEY=your_api_key_here
```

### 4. Run the application

```
python main.py
```

### 5. Access API documentation

Open:

```
http://127.0.0.1:8000/docs
```

---

## API Endpoint

### POST /api/process

#### Request

* claim_id: 123
* file: PDF file



```
{
  "claim_id": "123",
  "status": "success",
  "data": {
    "id_details": {},
    "discharge_details": {},
    "billing_details": {}
  }
}
```

---

## How It Works

1. The PDF is uploaded through the API
2. Text is extracted from each page
3. Each page is classified into a document category
4. Relevant pages are processed by specific extraction components
5. The extracted data is combined into a final response



