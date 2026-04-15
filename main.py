import io
import os
from typing import TypedDict, Annotated, List, Dict, Any, Literal

from fastapi import FastAPI, UploadFile, File, Form
from pypdf import PdfReader
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# ==============================
# Load ENV
# ==============================
load_dotenv()

# ==============================
# Reducer
# ==============================
def merge_dicts(dict1: dict, dict2: dict) -> dict:
    return {**dict1, **dict2}

# ==============================
# State
# ==============================
class ClaimState(TypedDict):
    claim_id: str
    pdf_bytes: bytes
    pages_text: Dict[int, str]
    categorized_pages: Dict[str, List[str]]
    extracted_data: Annotated[Dict[str, Any], merge_dicts]

# ==============================
# Types
# ==============================
DocType = Literal[
    "claim_forms", "cheque_or_bank_details", "identity_document",
    "itemized_bill", "discharge_summary", "prescription",
    "investigation_report", "cash_receipt", "other"
]

# ==============================
# Models
# ==============================
class PageClassification(BaseModel):
    page_type: DocType

class IdentityExtraction(BaseModel):
    patient_name: str | None = None
    dob: str | None = None
    id_numbers: List[str] = []
    policy_details: str | None = None

class DischargeExtraction(BaseModel):
    diagnosis: str | None = None
    admit_date: str | None = None
    discharge_date: str | None = None
    physician_details: str | None = None

class ItemizedBillItem(BaseModel):
    item_name: str
    cost: float

class BillExtraction(BaseModel):
    items: List[ItemizedBillItem] = []
    total_amount: float | None = None

# ==============================
# LLM Setup
# ==============================
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ==============================
# TEXT EXTRACTION (NO OCR)
# ==============================
def extract_text_simple(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text = {}

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or "Empty Page"
        pages_text[i] = text

    return pages_text

# ==============================
# Nodes
# ==============================
def segregator_node(state: ClaimState):
    pdf_bytes = state["pdf_bytes"]

    pages_text = extract_text_simple(pdf_bytes)

    categorized_pages = {doc: [] for doc in DocType.__args__}

    classification_llm = llm.with_structured_output(PageClassification)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert medical document classifier.

Classify the page into one of:
- claim_forms
- cheque_or_bank_details
- identity_document
- itemized_bill
- discharge_summary
- prescription
- investigation_report
- cash_receipt
- other

Return ONLY the category name."""),
        ("human", "{page_content}")
    ])

    chain = prompt | classification_llm

    for i, text in pages_text.items():
        print(f"\n--- Page {i} ---")
        print(text[:200])  # Debug

        try:
            result = chain.invoke({"page_content": text[:2000]})
            categorized_pages[result.page_type].append(text)
        except Exception:
            categorized_pages["other"].append(text)

    return {
        "pages_text": pages_text,
        "categorized_pages": categorized_pages,
        "extracted_data": {}
    }


def id_agent_node(state: ClaimState):
    pages = state["categorized_pages"].get("identity_document", [])

    if not pages:
        return {"extracted_data": {"id_details": None}}

    content = "\n---\n".join(pages[:3])

    try:
        extractor = llm.with_structured_output(IdentityExtraction)
        result = extractor.invoke(f"""
Extract clearly:
- patient_name
- dob
- id_numbers
- policy_details

Text:
{content}
""")
        return {"extracted_data": {"id_details": result.model_dump()}}
    except Exception:
        return {"extracted_data": {"id_details": None}}


def discharge_agent_node(state: ClaimState):
    pages = state["categorized_pages"].get("discharge_summary", [])

    if not pages:
        return {"extracted_data": {"discharge_details": None}}

    content = "\n---\n".join(pages[:3])

    try:
        extractor = llm.with_structured_output(DischargeExtraction)
        result = extractor.invoke(f"""
Extract clearly:
- diagnosis
- admit_date
- discharge_date
- physician_details

Text:
{content}
""")
        return {"extracted_data": {"discharge_details": result.model_dump()}}
    except Exception:
        return {"extracted_data": {"discharge_details": None}}


def bill_agent_node(state: ClaimState):
    pages = state["categorized_pages"].get("itemized_bill", [])

    if not pages:
        return {"extracted_data": {"billing_details": None}}

    content = "\n---\n".join(pages[:3])

    try:
        extractor = llm.with_structured_output(BillExtraction)
        result = extractor.invoke(f"""
Extract:
- item_name
- cost
- total_amount

Text:
{content}
""")
        return {"extracted_data": {"billing_details": result.model_dump()}}
    except Exception:
        return {"extracted_data": {"billing_details": None}}


def aggregator_node(state: ClaimState):
    return state

# ==============================
# Graph
# ==============================
workflow = StateGraph(ClaimState)

workflow.add_node("segregator", segregator_node)
workflow.add_node("id_agent", id_agent_node)
workflow.add_node("discharge", discharge_agent_node)
workflow.add_node("bill", bill_agent_node)
workflow.add_node("aggregator", aggregator_node)

workflow.add_edge(START, "segregator")

workflow.add_edge("segregator", "id_agent")
workflow.add_edge("segregator", "discharge")
workflow.add_edge("segregator", "bill")

workflow.add_edge("id_agent", "aggregator")
workflow.add_edge("discharge", "aggregator")
workflow.add_edge("bill", "aggregator")

workflow.add_edge("aggregator", END)

graph_app = workflow.compile()

# ==============================
# FastAPI
# ==============================
app = FastAPI()

@app.post("/api/process")
async def process_claim_api(
    claim_id: str = Form(...),
    file: UploadFile = File(...)
):
    pdf_bytes = await file.read()

    initial_state = {
        "claim_id": claim_id,
        "pdf_bytes": pdf_bytes,
        "pages_text": {},
        "categorized_pages": {},
        "extracted_data": {}
    }

    try:
        final_state = graph_app.invoke(initial_state)

        return {
            "claim_id": claim_id,
            "status": "success",
            "data": final_state["extracted_data"]
        }

    except Exception as e:
        return {
            "claim_id": claim_id,
            "status": "error",
            "message": str(e)
        }

# ==============================
# Run
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)