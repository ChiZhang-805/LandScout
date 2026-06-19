from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventType(StrEnum):
    LAND_SUPPLY = "land_supply"
    LAND_TRANSACTION = "land_transaction"
    MAJOR_PROJECT = "major_project"
    INFRASTRUCTURE = "infrastructure"
    INDUSTRIAL_PROJECT = "industrial_project"
    PUBLIC_SERVICE = "public_service"
    INVESTMENT_SIGNING = "investment_signing"
    PROJECT_APPROVAL = "project_approval"
    PLANNING_POLICY = "planning_policy"
    RESIDENTIAL_SUPPLY = "residential_supply"
    OTHER = "other"


class Evidence(BaseModel):
    source_id: str
    url: str
    quote: str
    fetched_at: str = ""
    content_hash: str = ""
    confidence: float = Field(default=0.75, ge=0, le=1)

    @field_validator("quote")
    @classmethod
    def quote_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("quote must not be blank")
        return value


class GovernmentEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    source_id: str
    source_url: str
    event_type: EventType
    title: str
    summary: str = ""
    district: str | None = None
    address: str | None = None
    project_name: str | None = None
    event_date: str | None = None
    amount_wanyuan: float | None = None
    area_sqm: float | None = None
    lat: float | None = None
    lon: float | None = None
    geo_confidence: float = Field(default=0.0, ge=0, le=1)
    evidence: list[Evidence] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    raw_doc_hash: str = ""
    needs_review: bool = False
    review_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GovernmentSignalExtraction(BaseModel):
    document_classification: str
    events: list[GovernmentEvent] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)


class StrictExtractionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractedEvidence(StrictExtractionModel):
    quote: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @field_validator("quote")
    @classmethod
    def quote_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("quote must not be blank")
        return value


class ExtractedGovernmentEvent(StrictExtractionModel):
    event_type: EventType
    title: str
    summary: str
    district: str | None
    address: str | None
    project_name: str | None
    event_date: str | None
    amount_wanyuan: float | None
    area_sqm: float | None
    evidence: list[ExtractedEvidence] = Field(min_length=1)
    tags: list[str]


class LLMGovernmentSignalExtraction(StrictExtractionModel):
    document_classification: str
    events: list[ExtractedGovernmentEvent]
    review_notes: list[str]
