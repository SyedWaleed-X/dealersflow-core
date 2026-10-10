from typing import TypedDict, Optional, List, Dict, Any


class AgentPipelineState(TypedDict):
    """
    Typed shared state passing across the 4 LangGraph nodes.
    Every node reads existing keys and updates specific output channels.
    """

    # Ingestion inputs
    audio_path: Optional[str]
    raw_text: Optional[str]

    # Node 1 Output: ASR Transcription
    transcript: str

    # Node 2 Output: Structured Entity Extraction & Normalization
    extracted_lead: Optional[Dict[str, Any]]
    normalized_lead: Optional[Dict[str, Any]]

    # Node 3 Output: Valuation & Vector Matchmaking
    valuation_result: Optional[Dict[str, Any]]
    matched_leads: List[Dict[str, Any]]

    # Node 4 Output: Outbound Synthesis & WhatsApp Link
    whatsapp_message: str
    whatsapp_deep_link: str

    # Error isolation channel
    error: Optional[str]
    is_success: bool