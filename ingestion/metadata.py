"""
Metadata definitions and NGO profile schemas for the knowledge base.
Follows the requirements in Section 2 and Section 8 of the project spec.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    """Metadata schema representing an ingested authentic source document."""
    document_id: str = Field(..., description="Unique document ID, e.g. DOC-DARPAN-001")
    document_name: str = Field(..., description="Full title of the document")
    category: str = Field(..., description="One of the 10 official document categories")
    organization: str = Field(..., description="Publishing body or organization")
    year: str = Field(default="2023", description="Publication year or reporting period")
    source_url: str = Field(default="", description="Official web source link")
    file_name: str = Field(..., description="Raw file name")
    page_number: Optional[str] = Field(default="1", description="Page number or section")
    focus_area: str = Field(default="General", description="Primary focus area")
    operating_area: Optional[str] = Field(default="India", description="Operating area")
    target_beneficiaries: Optional[str] = Field(default="", description="Beneficiaries reached")

class DocumentChunk(BaseModel):
    """Schema for a chunked segment of text ready for vector embedding."""
    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any]

class NGOProfile(BaseModel):
    """
    Schema capturing detailed organizational attributes for each covered NGO.
    Directly satisfies Section 2 of the project requirements.
    """
    ngo_name: str
    registration_info: str
    location: str
    main_focus_area: str
    programs: List[str]
    target_beneficiaries: str
    services_provided: str
    project_objectives: str
    activities_carried_out: str
    reported_outcomes: str
    govt_schemes: str
    csr_funding: str
    fcra_info: str
    reporting_period: str
    source_document: str
    source_organization: str
