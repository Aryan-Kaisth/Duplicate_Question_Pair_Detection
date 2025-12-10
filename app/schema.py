from pydantic import BaseModel, Field
from typing import Annotated

class QuestionPairInput(BaseModel):
    question1: Annotated[
        str,
        Field(
            ...,
            description="First question provided by the user",
            min_length=10,
            max_length=1000,
            examples=[
                "How can I learn machine learning?",
                "What is the best way to prepare for data science?"
            ]
        )
    ]

    question2: Annotated[
        str,
        Field(
            ...,
            description="Second question provided by the user",
            min_length=10,
            max_length=1000,
            examples=[
                "What are good resources to learn machine learning?",
                "How should I start learning data science?"
            ]
        )
    ]

class DuplicatePredictionOutput(BaseModel):
    is_duplicate: bool = Field(
        ...,
        description="Whether the two questions are duplicates"
    )

    label: str = Field(
        ...,
        description="Prediction result in readable form",
        examples=[
            "Duplicate question pair",
            "Not a duplicate question pair"
        ]
    )

