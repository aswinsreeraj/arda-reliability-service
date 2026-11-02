from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GateOutcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    GATE_OUTCOME_UNSPECIFIED: _ClassVar[GateOutcome]
    GATE_OUTCOME_PASS: _ClassVar[GateOutcome]
    GATE_OUTCOME_REVIEW_REQUIRED: _ClassVar[GateOutcome]
GATE_OUTCOME_UNSPECIFIED: GateOutcome
GATE_OUTCOME_PASS: GateOutcome
GATE_OUTCOME_REVIEW_REQUIRED: GateOutcome

class GateRequest(_message.Message):
    __slots__ = ("tenant_id", "agent_id", "case_description", "proposed_decision", "reliability_threshold")
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    CASE_DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    PROPOSED_DECISION_FIELD_NUMBER: _ClassVar[int]
    RELIABILITY_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    tenant_id: str
    agent_id: str
    case_description: str
    proposed_decision: str
    reliability_threshold: float
    def __init__(self, tenant_id: _Optional[str] = ..., agent_id: _Optional[str] = ..., case_description: _Optional[str] = ..., proposed_decision: _Optional[str] = ..., reliability_threshold: _Optional[float] = ...) -> None: ...

class GateResponse(_message.Message):
    __slots__ = ("confidence", "outcome", "reason_code", "neighbors")
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    REASON_CODE_FIELD_NUMBER: _ClassVar[int]
    NEIGHBORS_FIELD_NUMBER: _ClassVar[int]
    confidence: float
    outcome: GateOutcome
    reason_code: str
    neighbors: _containers.RepeatedCompositeFieldContainer[ReferenceNeighbor]
    def __init__(self, confidence: _Optional[float] = ..., outcome: _Optional[_Union[GateOutcome, str]] = ..., reason_code: _Optional[str] = ..., neighbors: _Optional[_Iterable[_Union[ReferenceNeighbor, _Mapping]]] = ...) -> None: ...

class ReferenceNeighbor(_message.Message):
    __slots__ = ("reference_example_id", "label", "similarity")
    REFERENCE_EXAMPLE_ID_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    reference_example_id: str
    label: str
    similarity: float
    def __init__(self, reference_example_id: _Optional[str] = ..., label: _Optional[str] = ..., similarity: _Optional[float] = ...) -> None: ...
