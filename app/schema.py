# from pydantic import BaseModel, Field
# from typing import List

# class QueryRequest(BaseModel):
#     documents: str = Field(..., description="URL to the PDF document")
#     questions: List[str] = Field(..., description="List of questions to answer")

# class QueryResponse(BaseModel):
#     answers: List[str] = Field(..., description="List of answers corresponding to the questions")

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List

class QueryRequest(BaseModel):
    documents: str = Field(
        ..., 
        description="URL to the PDF document",
        min_length=10,
        max_length=2000
    )
    questions: List[str] = Field(
        ..., 
        description="List of questions to answer",
        min_items=1,
        max_items=50
    )
    
    @validator('documents')
    def validate_document_url(cls, v):
        """Validate that the document URL is properly formatted."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Document URL must start with http:// or https://')
        return v
    
    @validator('questions')
    def validate_questions(cls, v):
        """Validate questions list."""
        if not v:
            raise ValueError('At least one question is required')
        
        # Filter out empty or whitespace-only questions
        filtered_questions = [q.strip() for q in v if q and q.strip()]
        
        if not filtered_questions:
            raise ValueError('At least one non-empty question is required')
        
        # Check for reasonable question length
        for i, q in enumerate(filtered_questions):
            if len(q) > 500:
                raise ValueError(f'Question {i+1} is too long (max 500 characters)')
            if len(q) < 5:
                raise ValueError(f'Question {i+1} is too short (min 5 characters)')
        
        return filtered_questions

    class Config:
        schema_extra = {
            "example": {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
                "questions": [
                    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                    "What is the waiting period for pre-existing diseases (PED) to be covered?",
                    "Does this policy cover maternity expenses, and what are the conditions?"
                ]
            }
        }


class QueryResponse(BaseModel):
    answers: List[str] = Field(
        ..., 
        description="List of answers corresponding to the questions"
    )
    
    @validator('answers')
    def validate_answers(cls, v):
        """Ensure answers list is not empty."""
        if not v:
            raise ValueError('Answers list cannot be empty')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "answers": [
                    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
                    "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
                    "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months."
                ]
            }
        }