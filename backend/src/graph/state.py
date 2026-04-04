import operator
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage

# schema for single compliance issue
class ComplianceIssue(TypedDict):
    category: str
    description: str
    severity: str
    timestamp: Optional[str]
    
class VideoAuditState(TypedDict):
    video_url: str
    video_id: str

    local_file_path: Optional[str]
    video_metadata: Optional[Dict[str, Any]]
    transcript: Optional[str]
    OCR_text: List[str]
    
    compliance_results: Annotated[List[ComplianceIssue], operator.add]

    final_status: str
    final_report: str

    errors: Annotated[List[str], operator.add]
    
    
    
    
    