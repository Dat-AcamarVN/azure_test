from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import uuid
import inspect
import logging

@dataclass
class PatentInfo:
    """Patent information model based on provided fields"""
    patent_id: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    claims: Optional[str] = None
    description: Optional[str] = None
    assignee: Optional[str] = None
    filing_date: Optional[str] = None
    inventor: Optional[str] = None
    language: Optional[str] = None
    patent_office: Optional[str] = None
    priority_date: Optional[str] = None
    publication_date: Optional[str] = None
    country: Optional[str] = None
    status: Optional[str] = None
    application_number: Optional[str] = None
    cpc: Optional[str] = None
    ipc: Optional[str] = None
    combined_vector: Optional[List[float]] = None
    chunk_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    user_id: Optional[str] = None
    partition_key: Optional[str] = None
    id: Optional[str] = None  # UUID for document id

    def __post_init__(self):
        """Initialize timestamps and generate IDs"""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()
        if not self.id:
            self.id = str(uuid.uuid4())  # Always generate UUID for id
        if self.priority_date and len(self.priority_date) >= 7:
            self.partition_key = self.priority_date[:7]
        else:
            self.partition_key = "unknown"

    def update_timestamps(self):
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatentInfo':
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'PatentInfo':
        data = json.loads(json_str)
        return cls.from_dict(data)

@dataclass
class PatentChunk:
    id: str
    patent_id: str
    text: str
    field: str
    partition_key: str
    created_at: str
    updated_at: str
    chunk_index: int
    chunk_size: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SearchInfo:
    search_by: str
    search_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchInfo':
        return cls(
            search_by=str(data.get("search_by", "")),
            search_value=str(data.get("search_value", ""))
        )

@dataclass
class FilterInfoWithPageInput:
    page_number: int
    page_size: int
    sort_by: str
    sort_order: str
    search_infos: List[SearchInfo]
    search_type: str = "text"

    def __post_init__(self):
        if self.page_number < 1:
            self.page_number = 1
        if self.sort_order.lower() not in ["asc", "desc"]:
            self.sort_order = "desc"
        else:
            self.sort_order = self.sort_order.lower()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "page_size": self.page_size,
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
            "search_infos": [search_info.to_dict() for search_info in self.search_infos],
            "search_type": self.search_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FilterInfoWithPageInput':
        search_infos = []
        if "search_infos" in data and isinstance(data["search_infos"], list):
            for search_info_data in data["search_infos"]:
                if isinstance(search_info_data, dict):
                    search_infos.append(SearchInfo.from_dict(search_info_data))

        return cls(
            page_number=int(data.get("page_number", 1)),
            page_size=int(data.get("page_size", 10)),
            sort_by=str(data.get("sort_by", "created_at")),
            sort_order=str(data.get("sort_order", "desc")),
            search_infos=search_infos,
            search_type=str(data.get("search_type", "text"))
        )

    def extract_query(self) -> str:
        query_fields = ["query", "text", "search", "q"]
        for search_info in self.search_infos:
            if search_info.search_by.lower() in query_fields:
                return search_info.search_value
        return "*"

    def get_search_parameters(self) -> Dict[str, Any]:
        params = {}
        for search_info in self.search_infos:
            field = search_info.search_by.lower()
            value = search_info.search_value
            if field == "vector_field" and self.search_type in ["vector", "hybrid", "semantic"]:
                params["vector_field"] = value
            elif field == "similarity_threshold" and self.search_type in ["vector", "hybrid", "semantic"]:
                try:
                    params["similarity_threshold"] = float(value)
                except (ValueError, TypeError):
                    pass
            elif field == "text_weight" and self.search_type == "hybrid":
                try:
                    params["text_weight"] = float(value)
                except (ValueError, TypeError):
                    pass
            elif field == "vector_weight" and self.search_type == "hybrid":
                try:
                    params["vector_weight"] = float(value)
                except (ValueError, TypeError):
                    pass
            elif field == "fuzziness" and self.search_type == "fuzzy":
                try:
                    params["fuzziness"] = int(value)
                except (ValueError, TypeError):
                    pass
        return params
