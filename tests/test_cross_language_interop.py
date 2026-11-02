"""Cross-language signing and canonical serialization interoperability tests.

Executes actual Go runtime programs via the Go compiler to compute canonical JSON
serialization and HMAC-SHA256 signatures, and verifies that the Python
Reliability Service produces byte-for-byte identical canonical payloads and
accepts Go-generated signatures.

Integration Status Note:
The actual upstream Go review producer service ('arda.human-review-service') does
not yet exist in the Go repository ('/home/aswin/second-project/arda').
These tests prove runtime mathematical and byte-level interoperability between
Go's standard library ('crypto/hmac', 'encoding/json') and Python's implementation,
while end-to-end service deployment remains pending.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from arda_reliability_service.domain.exceptions import (
    InvalidSignatureError,
    UnauthorizedProducerError,
)
from arda_reliability_service.infrastructure.messaging.authorizer import (
    TrustedProducerAuthorizer,
    build_canonical_payload,
    compute_hmac_signature,
    format_canonical_timestamp,
)
from arda_reliability_service.infrastructure.messaging.events import (
    HumanResolutionEventDTO,
)

GO_SIGNER_SOURCE = r"""package main

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

type CanonicalPayload struct {
	AgentID                 string `json:"agent_id"`
	CaseID                  string `json:"case_id"`
	EventID                 string `json:"event_id"`
	EventType               string `json:"event_type"`
	EventVersion            string `json:"event_version"`
	HumanDecision           string `json:"human_decision"`
	KeyID                   string `json:"key_id"`
	OriginalCaseDescription string `json:"original_case_description"`
	ProducerID              string `json:"producer_id"`
	ProposedDecision        string `json:"proposed_decision"`
	ReviewerID              string `json:"reviewer_id"`
	TenantID                string `json:"tenant_id"`
	Timestamp               string `json:"timestamp"`
}

type SignRequest struct {
	Secret  string           `json:"secret"`
	Payload CanonicalPayload `json:"payload"`
}

type SignResponse struct {
	Canonical string `json:"canonical"`
	Signature string `json:"signature"`
}

func main() {
	inputBytes, err := io.ReadAll(os.Stdin)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read error: %v\n", err)
		os.Exit(1)
	}

	var req SignRequest
	if err := json.Unmarshal(inputBytes, &req); err != nil {
		fmt.Fprintf(os.Stderr, "unmarshal error: %v\n", err)
		os.Exit(1)
	}

	// Canonical JSON serialization with HTML escaping disabled
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(req.Payload); err != nil {
		fmt.Fprintf(os.Stderr, "encode error: %v\n", err)
		os.Exit(1)
	}
	canonicalBytes := bytes.TrimRight(buf.Bytes(), "\n")

	// HMAC-SHA256 signature
	mac := hmac.New(sha256.New, []byte(req.Secret))
	mac.Write(canonicalBytes)
	sig := hex.EncodeToString(mac.Sum(nil))

	resp := SignResponse{
		Canonical: string(canonicalBytes),
		Signature: sig,
	}

	out, _ := json.Marshal(resp)
	fmt.Println(string(out))
}
"""


def run_go_signer(payload: dict[str, Any], secret: str) -> tuple[str, str]:
    """Execute the Go signing program with given payload and secret."""
    go_bin = shutil.which("go")
    if not go_bin:
        pytest.skip("Go compiler ('go') not found in PATH")

    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = Path(tmpdir) / "main.go"
        src_file.write_text(GO_SIGNER_SOURCE, encoding="utf-8")

        req_data = {
            "secret": secret,
            "payload": {
                "agent_id": payload["agent_id"],
                "case_id": payload["case_id"],
                "event_id": payload["event_id"],
                "event_type": payload.get(
                    "event_type", "arda.reliability.human_resolution"
                ),
                "event_version": payload.get("event_version", "v1"),
                "human_decision": payload["human_decision"],
                "key_id": payload.get("key_id", "v1"),
                "original_case_description": payload["original_case_description"],
                "producer_id": payload.get("producer_id", "arda.human-review-service"),
                "proposed_decision": payload.get("proposed_decision", "APPROVE"),
                "reviewer_id": payload["reviewer_id"],
                "tenant_id": payload["tenant_id"],
                "timestamp": payload["timestamp"],
            },
        }

        proc = subprocess.run(
            [go_bin, "run", str(src_file)],
            input=json.dumps(req_data),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Go signer failed:\n{proc.stderr}")

        res = json.loads(proc.stdout.strip())
        return res["canonical"], res["signature"]


def test_cross_language_canonical_bytes_identical_standard_ascii() -> None:
    """Verify that Python and Go produce byte-for-byte identical canonical payloads for ASCII fields."""
    secret = "secret-interop-test-123"
    tenant_id = str(uuid4())
    agent_id = str(uuid4())
    timestamp = format_canonical_timestamp(datetime.now(UTC))

    event_dict = {
        "agent_id": agent_id,
        "case_id": "CASE-STANDARD-001",
        "event_id": "EVT-ASCII-999",
        "event_type": "arda.reliability.human_resolution",
        "event_version": "v1",
        "human_decision": "APPROVE",
        "key_id": "v1",
        "original_case_description": "Standard invoice payment of $500.00 to verified vendor.",
        "producer_id": "arda.human-review-service",
        "proposed_decision": "APPROVE",
        "reviewer_id": "reviewer_alice",
        "tenant_id": tenant_id,
        "timestamp": timestamp,
        "resolved_at": timestamp,
    }

    go_canonical, go_signature = run_go_signer(event_dict, secret)
    py_canonical_bytes = build_canonical_payload(event_dict)
    py_signature = compute_hmac_signature(secret, py_canonical_bytes)

    # 1. Byte-for-byte equivalence
    assert py_canonical_bytes == go_canonical.encode("utf-8")
    assert py_signature == go_signature

    # 2. Python authorizer accepts Go-generated signature
    event_dict["signature"] = go_signature
    dto = HumanResolutionEventDTO.model_validate(event_dict)
    authorizer = TrustedProducerAuthorizer(
        producer_secrets={"arda.human-review-service": {"v1": secret}}
    )
    authorizer.authorize_event(dto.to_domain())


def test_cross_language_unicode_and_special_character_equivalence() -> None:
    """Verify Python and Go equivalence with Unicode accents, currency symbols, HTML tags, and quotes."""
    secret = "super-secret-unicode-key-456"
    tenant_id = str(uuid4())
    agent_id = str(uuid4())
    timestamp = "2026-11-03T12:00:00Z"

    # Narrative with accents (€, é, ñ), HTML brackets (<script>), ampersands (&), escaped quotes, and newlines
    case_description = (
        "Transfer €50,000 to Société Générale (café & naïve résumé).\n"
        "Security check: <alert type='high'>\"Suspicious activity\" & fraud score > 0.85</alert>."
    )

    event_dict = {
        "agent_id": agent_id,
        "case_id": "CASE-UNICODE-888",
        "event_id": "EVT-UNICODE-001",
        "event_type": "arda.reliability.human_resolution",
        "event_version": "v1",
        "human_decision": "REJECT",
        "key_id": "v2",
        "original_case_description": case_description,
        "producer_id": "arda.human-review-service",
        "proposed_decision": "APPROVE",
        "reviewer_id": "senior_reviewer_bob",
        "tenant_id": tenant_id,
        "timestamp": timestamp,
        "resolved_at": timestamp,
    }

    go_canonical, go_signature = run_go_signer(event_dict, secret)
    py_canonical_bytes = build_canonical_payload(event_dict)
    py_signature = compute_hmac_signature(secret, py_canonical_bytes)

    # Byte-for-byte check
    assert py_canonical_bytes == go_canonical.encode("utf-8")
    assert py_signature == go_signature

    # Verification passes
    event_dict["signature"] = go_signature
    dto = HumanResolutionEventDTO.model_validate(event_dict)
    authorizer = TrustedProducerAuthorizer(
        producer_secrets={"arda.human-review-service": {"v2": secret}}
    )
    # Provide fixed now matching timestamp to avoid expiration
    authorizer.authorize_event(
        dto.to_domain(), now=datetime(2026, 11, 3, 12, 1, 0, tzinfo=UTC)
    )


def test_cross_language_signature_tamper_detection() -> None:
    """Verify that any modification of Go-signed payload is immediately detected and rejected by Python."""
    secret = "tamper-proof-secret-777"
    tenant_id = str(uuid4())
    agent_id = str(uuid4())
    timestamp = format_canonical_timestamp(datetime.now(UTC))

    event_dict = {
        "agent_id": agent_id,
        "case_id": "CASE-TAMPER-001",
        "event_id": "EVT-TAMPER-001",
        "event_type": "arda.reliability.human_resolution",
        "event_version": "v1",
        "human_decision": "APPROVE",
        "key_id": "v1",
        "original_case_description": "Approved budget allocation for Q4 operations.",
        "producer_id": "arda.human-review-service",
        "proposed_decision": "APPROVE",
        "reviewer_id": "charlie_approver",
        "tenant_id": tenant_id,
        "timestamp": timestamp,
        "resolved_at": timestamp,
    }

    _, go_signature = run_go_signer(event_dict, secret)
    event_dict["signature"] = go_signature

    authorizer = TrustedProducerAuthorizer(
        producer_secrets={"arda.human-review-service": {"v1": secret}}
    )

    # 1. Tamper human_decision
    tampered_decision = dict(event_dict)
    tampered_decision["human_decision"] = "REJECT"
    with pytest.raises(InvalidSignatureError):
        authorizer.authorize_event(
            HumanResolutionEventDTO.model_validate(tampered_decision).to_domain()
        )

    # 2. Tamper case_description
    tampered_desc = dict(event_dict)
    tampered_desc["original_case_description"] = "Malicious unapproved allocation."
    with pytest.raises(InvalidSignatureError):
        authorizer.authorize_event(
            HumanResolutionEventDTO.model_validate(tampered_desc).to_domain()
        )

    # 3. Tamper tenant_id
    tampered_tenant = dict(event_dict)
    tampered_tenant["tenant_id"] = str(uuid4())
    with pytest.raises(InvalidSignatureError):
        authorizer.authorize_event(
            HumanResolutionEventDTO.model_validate(tampered_tenant).to_domain()
        )


def test_cross_language_key_rotation_overlap() -> None:
    """Verify that during key rotation, both old and new Go-generated signatures are verified."""
    v1_secret = "old-secret-v1"
    v2_secret = "new-secret-v2"
    tenant_id = str(uuid4())
    agent_id = str(uuid4())
    timestamp = format_canonical_timestamp(datetime.now(UTC))

    authorizer = TrustedProducerAuthorizer(
        producer_secrets={
            "arda.human-review-service": {
                "v1": v1_secret,
                "v2": v2_secret,
            }
        }
    )

    # Go signs event A with v1 key
    event_v1 = {
        "agent_id": agent_id,
        "case_id": "CASE-ROT-001",
        "event_id": "EVT-ROT-V1",
        "event_type": "arda.reliability.human_resolution",
        "event_version": "v1",
        "human_decision": "APPROVE",
        "key_id": "v1",
        "original_case_description": "Legacy key rotation test case A.",
        "producer_id": "arda.human-review-service",
        "proposed_decision": "APPROVE",
        "reviewer_id": "reviewer_alice",
        "tenant_id": tenant_id,
        "timestamp": timestamp,
        "resolved_at": timestamp,
    }
    _, sig_v1 = run_go_signer(event_v1, v1_secret)
    event_v1["signature"] = sig_v1
    authorizer.authorize_event(
        HumanResolutionEventDTO.model_validate(event_v1).to_domain()
    )

    # Go signs event B with v2 key
    event_v2 = {
        "agent_id": agent_id,
        "case_id": "CASE-ROT-002",
        "event_id": "EVT-ROT-V2",
        "event_type": "arda.reliability.human_resolution",
        "event_version": "v1",
        "human_decision": "REJECT",
        "key_id": "v2",
        "original_case_description": "New key rotation test case B.",
        "producer_id": "arda.human-review-service",
        "proposed_decision": "APPROVE",
        "reviewer_id": "reviewer_bob",
        "tenant_id": tenant_id,
        "timestamp": timestamp,
        "resolved_at": timestamp,
    }
    _, sig_v2 = run_go_signer(event_v2, v2_secret)
    event_v2["signature"] = sig_v2
    authorizer.authorize_event(
        HumanResolutionEventDTO.model_validate(event_v2).to_domain()
    )

    # Event claiming key_id v3 is rejected as unknown key
    event_v3 = dict(event_v2)
    event_v3["key_id"] = "v3"
    with pytest.raises(UnauthorizedProducerError, match="Unknown key_id 'v3'"):
        authorizer.authorize_event(
            HumanResolutionEventDTO.model_validate(event_v3).to_domain()
        )
