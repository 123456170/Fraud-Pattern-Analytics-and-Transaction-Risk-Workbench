from __future__ import annotations

import io
import json
import math
import os
import hashlib
import time
import traceback
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


APP_NAME = "Fraud Pattern Analytics and Transaction Risk Workbench"
APP_VERSION = "0.1.1-mvp"
RULESET_VERSION = "fraud-rules-0.1.1"
SYNTHETIC_NOTICE = (
    "All records, labels, balances, devices, IPs, counterparties, and outcomes in this demo are synthetic. "
    "This is decision-support software for human investigators. It does not make autonomous financial decisions "
    "and has not been validated with real customer data."
)

VIDEO_DIR = "demo_output"
VIDEO_FILENAME = "fraud_workbench_demo.mp4"

COUNTRY_CURRENCY = {
    "US": "USD",
    "GB": "GBP",
    "DE": "EUR",
    "FR": "EUR",
    "SG": "SGD",
    "AE": "AED",
    "BR": "BRL",
    "IN": "INR",
    "NG": "NGN",
    "MX": "MXN",
}

CHANNELS = [
    "online_banking",
    "mobile_app",
    "wire",
    "atm",
    "card_pos",
    "branch",
]

SEGMENTS = [
    "Retail",
    "SME",
    "Corporate",
    "Treasury",
    "Lending_Platform",
    "Payment_Operator",
]

FRAUD_SCENARIOS = [
    "account_takeover",
    "structuring",
    "velocity",
    "new_payee",
    "balance_drain",
]

ORG_NAMES = [
    "Northwind Trading",
    "Helios Logistics",
    "Aurora Payments",
    "Blue Harbor Capital",
    "Cedar Treasury Services",
    "Delta Lending Labs",
    "Everline Retail",
    "Fjord Manufacturing",
    "Granite Settlements",
    "Horizon Freight",
    "Ironbridge Finance",
    "Juniper Commerce",
]

REQUIRED_TX_COLUMNS = [
    "txn_id",
    "account_id",
    "timestamp_utc",
    "amount",
    "currency",
    "direction",
    "counterparty_id",
    "counterparty_country",
    "channel",
    "device_id",
    "ip_hash",
]

CLEAN_TX_COLUMNS = [
    "txn_id",
    "account_id",
    "timestamp_utc",
    "amount",
    "currency",
    "direction",
    "counterparty_id",
    "counterparty_country",
    "channel",
    "device_id",
    "ip_hash",
    "note",
    "source_system",
    "is_fraud_synthetic",
    "label_source",
    "scenario",
    "consent_id",
    "source_type",
]

ERROR_COLUMNS = [
    "row_number",
    "field",
    "issue",
    "provided_value",
    "suggested_fix",
    "severity",
]

FEATURE_COLUMNS = [
    "txn_id",
    "account_id",
    "timestamp_utc",
    "amount",
    "currency",
    "direction",
    "counterparty_id",
    "counterparty_country",
    "channel",
    "device_id",
    "ip_hash",
    "risk_score",
    "confidence",
    "reason_codes",
    "explanation",
    "data_quality",
    "is_fraud_synthetic",
    "label_source",
    "scenario",
    "account_country",
    "account_segment",
    "velocity_1h_count",
    "velocity_1h_amount",
    "amount_z",
    "balance_drain_ratio",
    "new_counterparty",
    "device_changed",
    "shared_network",
    "night_activity",
    "ruleset_version",
    "source_type",
    "source_system",
]

ALERT_COLUMNS = [
    "alert_id",
    "txn_id",
    "account_id",
    "txn_timestamp",
    "amount",
    "currency",
    "created_at",
    "updated_at",
    "model_score",
    "risk_score",
    "status",
    "assigned_to",
    "reviewer",
    "decision",
    "override_reason",
    "feedback",
    "reason_codes",
    "explanation",
    "data_quality",
    "confidence",
    "ruleset_version",
    "source_type",
]

AUDIT_COLUMNS = [
    "event_id",
    "timestamp_utc",
    "actor_role",
    "actor_name",
    "action",
    "entity_type",
    "entity_id",
    "details",
]

IMPORT_COLUMNS = [
    "import_id",
    "filename",
    "checksum",
    "imported_at",
    "rows_total",
    "rows_valid",
    "rows_committed",
    "rows_skipped",
    "status",
    "notes",
]

ROLE_PERMISSIONS = {
    "Administrator": {
        "view",
        "export",
        "reset_demo",
        "regenerate_video",
        "change_rules",
        "approve",
        "override",
        "archive",
        "assign",
        "triage",
        "add_feedback",
        "import",
        "run_simulation",
        "restore_snapshot",
    },
    "Analyst/Operator": {
        "view",
        "export",
        "triage",
        "assign",
        "add_feedback",
        "request_review",
        "import",
        "run_simulation",
    },
    "Reviewer": {
        "view",
        "export",
        "approve",
        "override",
        "close",
        "archive",
        "add_feedback",
        "audit",
    },
    "Read-only Stakeholder": {
        "view",
        "export",
    },
}

DEFAULT_RULES: Dict[str, Any] = {
    "alert_threshold": 55,
    "amount_z_trigger": 2.5,
    "velocity_window_hours": 1,
    "velocity_count_trigger": 5,
    "travel_hours_max": 6,
    "new_payee_amount": 5000,
    "structuring_low": 9000,
    "structuring_high": 9999,
    "structuring_min_count": 2,
    "balance_drain_trigger": 0.60,
    "night_start": 0,
    "night_end": 5,
    "weights": {
        "AMOUNT_OUTLIER": 18,
        "VELOCITY_SPIKE": 22,
        "IMPOSSIBLE_TRAVEL": 20,
        "NEW_COUNTERPARTY_HIGH_VALUE": 15,
        "STRUCTURING_NEAR_THRESHOLD": 16,
        "BALANCE_DRAIN": 14,
        "DEVICE_OR_IP_CHANGE": 10,
        "SHARED_DEVICE_NETWORK": 12,
        "NIGHT_ACTIVITY": 5,
        "DATA_QUALITY_LOW": 3,
    },
}

PRODUCT_BRIEF = """
### Product brief

**Problem:** Investigator teams receive fragmented payment and account activity signals. Important patterns are buried
in raw transactions, balances, device/IP changes, counterparty novelty, and manual notes. Current workflows often depend
on spreadsheet triage, inconsistent reason codes, and limited auditability.

**Narrow MVP workflow:** Ingest or simulate payment/account activity, compute transparent deterministic risk scores and
reason codes, prioritize a review queue, allow analyst/reviewer actions, preserve audit history, and export a case pack.

**Primary user in MVP:** A single operator/analyst on a local Windows machine reviewing synthetic or permissioned sample
data.

**Measurable MVP improvements to test later with real users:**
- Reduce time to locate high-priority items by presenting ranked alerts with reason codes.
- Improve consistency of triage language through controlled reason codes.
- Preserve evidence through audit events and exportable case packs.
- Detect data-quality problems before they silently affect review.

**Important boundary:** This release is decision support. It must not be represented as regulated financial advice,
autonomous fraud detection, or validated customer-impacting software until tested with real, permissioned data and
qualified domain reviewers.
"""

BACKLOG_ROWS = [
    {"tier": "MVP", "priority": 1, "capability": "Synthetic demo data loader and live simulation"},
    {"tier": "MVP", "priority": 2, "capability": "Deterministic risk scoring with reason codes"},
    {"tier": "MVP", "priority": 3, "capability": "CSV/XLSX import preview, validation, and error report"},
    {"tier": "MVP", "priority": 4, "capability": "Alert queue with lifecycle statuses and audit trail"},
    {"tier": "MVP", "priority": 5, "capability": "Case pack export and snapshot export"},
    {"tier": "MVP", "priority": 6, "capability": "Generated local demo video and interactive dashboard"},
    {"tier": "Post-MVP", "priority": 7, "capability": "SQLite persistence and multi-user role server"},
    {"tier": "Post-MVP", "priority": 8, "capability": "Time-based model evaluation harness with held-out data"},
    {"tier": "Post-MVP", "priority": 9, "capability": "Graph/shared-device/network features with provenance"},
    {"tier": "Post-MVP", "priority": 10, "capability": "Configurable rule catalog and version rollback"},
    {"tier": "Enterprise", "priority": 11, "capability": "SSO/directory integration and granular RBAC"},
    {"tier": "Enterprise", "priority": 12, "capability": "PostgreSQL, backup/restore automation, and HA deployment"},
    {"tier": "Enterprise", "priority": 13, "capability": "SIEM/GRC export, retention policies, and legal hold"},
    {"tier": "Enterprise", "priority": 14, "capability": "Model governance registry and approved-model rollback"},
]

ACCEPTANCE_CHECKLIST_ROWS = [
    {"id": "AC-1", "criterion": "App starts on Windows from README commands and loads synthetic demo data automatically.", "observable_pass": "Dashboard shows non-zero transactions and alerts without manual setup.", "fail_condition": "Blank dashboard, import error, or empty dataset."},
    {"id": "AC-2", "criterion": "App generates and embeds a local demo video file.", "observable_pass": "demo_output/fraud_workbench_demo.mp4 exists or GIF fallback is displayed.", "fail_condition": "No video/media output and no explanatory fallback."},
    {"id": "AC-3", "criterion": "Valid sample CSV can be previewed and committed.", "observable_pass": "Import summary shows committed rows and new alerts/features rebuild.", "fail_condition": "Commit corrupts existing records or silently overwrites source IDs."},
    {"id": "AC-4", "criterion": "Malformed rows produce actionable row-level errors.", "observable_pass": "Error report includes row number, field, issue, and suggested fix.", "fail_condition": "Generic failure or no downloadable error report."},
    {"id": "AC-5", "criterion": "Core scoring runs without UI.", "observable_pass": "Smoke tests can score a hand-checkable suspicious transaction higher than a normal transaction.", "fail_condition": "Scoring depends on Streamlit widgets or fails on typed inputs."},
    {"id": "AC-6", "criterion": "Each alert has transparent reason codes and explanation JSON.", "observable_pass": "Alert detail shows contributions, weights, data-quality warnings, and ruleset version.", "fail_condition": "Score appears without input values or rationale."},
    {"id": "AC-7", "criterion": "Reviewer actions are audited.", "observable_pass": "Assign, review, clear, escalate, override, feedback, and archive create audit events.", "fail_condition": "Material changes are not traceable to actor, role, timestamp, and entity."},
    {"id": "AC-8", "criterion": "Export produces usable case pack and snapshot.", "observable_pass": "Downloads include CSV/JSON/ZIP artifacts with stable filenames.", "fail_condition": "Export omits evidence, reason codes, or audit history."},
    {"id": "AC-9", "criterion": "Smoke tests pass from clean launch.", "observable_pass": "Governance tab shows all smoke tests passed.", "fail_condition": "Any core test fails or is skipped without explanation."},
]

RISK_REGISTER_ROWS = [
    {"risk": "Domain validity gap", "category": "Domain", "likelihood": "High in demo", "impact": "High", "mitigation": "Use deterministic rules, human review, synthetic labeling, and no real-world performance claims until evaluated with permissioned data."},
    {"risk": "Data quality defects", "category": "Data", "likelihood": "Medium", "impact": "High", "mitigation": "Preview, row-level validation, referential integrity checks, error download, and non-overwriting import."},
    {"risk": "Sensitive data exposure", "category": "Security", "likelihood": "Medium", "impact": "High", "mitigation": "Synthetic demo only, no paid APIs, no secrets in code, export controls, audit logging, and production HTTPS/auth requirements."},
    {"risk": "Investigator over-reliance", "category": "Adoption", "likelihood": "Medium", "impact": "High", "mitigation": "Decision-support framing, mandatory reason codes, human approval boundaries, and override audit."},
    {"risk": "Operational failure during import/simulation", "category": "Operations", "likelihood": "Medium", "impact": "Medium", "mitigation": "Safe staging, skip-existing re-import, snapshot export/restore, smoke tests, and structured error messages."},
]

DATA_DICTIONARY_ROWS = [
    {"field": "txn_id", "type": "string", "unit": "identifier", "meaning": "Stable transaction identifier", "allowed_values": "unique non-empty", "source": "source system / import / simulator", "sensitivity": "internal", "validation_rule": "required, unique, no silent overwrite"},
    {"field": "account_id", "type": "string", "unit": "identifier", "meaning": "Account reference", "allowed_values": "must exist in accounts catalog", "source": "core banking / payments reference", "sensitivity": "confidential in production", "validation_rule": "required, referential integrity"},
    {"field": "timestamp_utc", "type": "datetime", "unit": "UTC timestamp", "meaning": "Event occurrence time", "allowed_values": "ISO-8601, not far future/past", "source": "source system", "sensitivity": "internal", "validation_rule": "parseable, timezone-aware, range-checked"},
    {"field": "amount", "type": "float", "unit": "currency units", "meaning": "Transaction amount", "allowed_values": "> 0 and <= 10,000,000 in MVP", "source": "payment ledger", "sensitivity": "confidential in production", "validation_rule": "numeric, positive, bounded"},
    {"field": "currency", "type": "string", "unit": "ISO-4217", "meaning": "Transaction currency", "allowed_values": "3-letter alphabetic", "source": "account/payment metadata", "sensitivity": "internal", "validation_rule": "required, format checked"},
    {"field": "direction", "type": "string", "unit": "enum", "meaning": "Debit or credit", "allowed_values": "debit, credit", "source": "ledger", "sensitivity": "internal", "validation_rule": "required, enum"},
    {"field": "counterparty_id", "type": "string", "unit": "identifier", "meaning": "Payee/payer reference", "allowed_values": "non-empty", "source": "payment instruction", "sensitivity": "confidential in production", "validation_rule": "required"},
    {"field": "counterparty_country", "type": "string", "unit": "ISO-3166 alpha-2", "meaning": "Counterparty geography", "allowed_values": "2-letter code preferred", "source": "payment metadata", "sensitivity": "internal", "validation_rule": "format warning if invalid"},
    {"field": "channel", "type": "string", "unit": "enum", "meaning": "Initiation channel", "allowed_values": "online_banking, mobile_app, wire, atm, card_pos, branch", "source": "channel telemetry", "sensitivity": "internal", "validation_rule": "known channel preferred"},
    {"field": "device_id", "type": "string", "unit": "identifier", "meaning": "Device fingerprint/hash", "allowed_values": "non-empty", "source": "digital channel", "sensitivity": "confidential in production", "validation_rule": "required"},
    {"field": "ip_hash", "type": "string", "unit": "hash", "meaning": "Hashed IP reference", "allowed_values": "non-empty", "source": "digital channel", "sensitivity": "confidential in production", "validation_rule": "required"},
    {"field": "risk_score", "type": "float", "unit": "0-100", "meaning": "Deterministic priority score", "allowed_values": "0-100", "source": "domain engine", "sensitivity": "internal", "validation_rule": "bounded, explainable"},
    {"field": "reason_codes", "type": "string", "unit": "controlled list", "meaning": "Human-readable drivers", "allowed_values": "catalog codes", "source": "domain engine", "sensitivity": "internal", "validation_rule": "traceable to contributions"},
    {"field": "explanation", "type": "json", "unit": "structured evidence", "meaning": "Inputs, signals, contributions, warnings", "allowed_values": "valid JSON", "source": "domain engine", "sensitivity": "internal", "validation_rule": "must parse and include ruleset version"},
    {"field": "status", "type": "string", "unit": "workflow enum", "meaning": "Alert lifecycle state", "allowed_values": "new, under_review, cleared, escalated, archived", "source": "review workflow", "sensitivity": "internal", "validation_rule": "role-gated transitions"},
]

WORKFLOW_TEXT = """
### Central workflow

1. Data entry: synthetic simulator, CSV/XLSX import, or future source-system adapter creates transaction records.
2. Validation: schema, type, range, duplicate, referential, and timezone checks produce preview and row-level errors.
3. Scoring: deterministic domain engine computes features, reason codes, confidence, and data-quality warnings.
4. Prioritization: alerts above threshold enter the review queue, ranked by score and freshness.
5. Review: analyst assigns, investigates, adds feedback, and requests reviewer action.
6. Approval/decision: reviewer clears, escalates, overrides with reason, or archives.
7. Audit/export: every material change is audited; case packs and snapshots are exportable.
8. Exception path: missing/contradictory/late/out-of-range data is quarantined in the error report and never silently committed.
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_utc_timestamp(value: Any) -> pd.Timestamp:
    """
    Safely convert datetime, Timestamp, string, or None to a timezone-aware UTC pandas Timestamp.

    This avoids the pandas error:
    "Cannot pass a datetime or Timestamp with tzinfo with the tz parameter."
    """
    if value is None:
        return utc_now()

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return utc_now()
        ts = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Unable to parse timestamp: {value!r}")
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return utc_now()
        if value.tzinfo is None:
            return value.tz_localize("UTC")
        return value.tz_convert("UTC")

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return pd.Timestamp(value).tz_localize("UTC")
        return pd.Timestamp(value).tz_convert("UTC")

    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Unable to parse timestamp: {value!r}")
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() in {"true", "1", "yes", "y", "fraud", "suspicious"}


def json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, set):
        return sorted(list(obj))
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


def empty_df(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def ensure_columns(df: Optional[pd.DataFrame], columns: List[str]) -> pd.DataFrame:
    if df is None:
        return empty_df(columns)
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = None
    return out[columns]


def concat_df(old: Optional[pd.DataFrame], new: Optional[pd.DataFrame]) -> pd.DataFrame:
    if old is None or len(old) == 0:
        return new.copy() if new is not None else pd.DataFrame()
    if new is None or len(new) == 0:
        return old.copy()
    return pd.concat([old, new], ignore_index=True)


def can(role: str, action: str) -> bool:
    return action in ROLE_PERMISSIONS.get(role, set())


def add_audit(
    db: Dict[str, Any],
    actor_role: str,
    actor_name: str,
    action: str,
    entity_type: str,
    entity_id: str,
    details: str = "",
) -> None:
    event = {
        "event_id": stable_id("EVT", utc_now().isoformat(), actor_role, actor_name, action, entity_id, details),
        "timestamp_utc": utc_now().isoformat(),
        "actor_role": actor_role,
        "actor_name": actor_name,
        "action": action,
        "entity_type": entity_type,
        "entity_id": str(entity_id),
        "details": details,
    }
    db["audit"] = ensure_columns(concat_df(db.get("audit"), pd.DataFrame([event])), AUDIT_COLUMNS)


def make_txn_row(
    acc: pd.Series,
    ts: Any,
    amount: float,
    direction: str,
    counterparty_id: str,
    counterparty_country: str,
    channel: str,
    device_id: str,
    ip_hash: str,
    is_fraud: bool,
    label_source: str,
    scenario: str,
    note: str = "",
    source_system: str = "payments_sim",
    txn_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "txn_id": txn_id,
        "account_id": str(acc["account_id"]),
        "timestamp_utc": normalize_utc_timestamp(ts),
        "amount": round(float(amount), 2),
        "currency": str(acc["currency"]),
        "direction": str(direction).lower(),
        "counterparty_id": str(counterparty_id),
        "counterparty_country": str(counterparty_country).upper(),
        "channel": str(channel).lower(),
        "device_id": str(device_id),
        "ip_hash": str(ip_hash),
        "note": str(note),
        "source_system": str(source_system),
        "is_fraud_synthetic": bool(is_fraud),
        "label_source": str(label_source),
        "scenario": str(scenario),
        "consent_id": f"CONSENT-SYN-{acc['account_id']}",
        "source_type": "synthetic",
    }


def generate_demo_dataset(seed: int = 42) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    countries = list(COUNTRY_CURRENCY.keys())
    n_accounts = 36
    days = 45
    start = utc_now() - timedelta(days=days)

    account_rows: List[Dict[str, Any]] = []
    for i in range(n_accounts):
        country = countries[int(rng.integers(0, len(countries)))]
        segment = SEGMENTS[int(rng.integers(0, len(SEGMENTS)))]
        if segment == "Retail":
            typical = float(np.exp(rng.normal(math.log(180.0), 0.65)))
            avg_balance = typical * float(rng.uniform(4.0, 18.0))
            org = f"Synthetic Individual {i + 1:03d}"
        else:
            typical = float(np.exp(rng.normal(math.log(7500.0), 0.75)))
            avg_balance = typical * float(rng.uniform(3.0, 25.0))
            org = f"{ORG_NAMES[i % len(ORG_NAMES)]} {i + 1:03d}"

        account_rows.append(
            {
                "account_id": f"ACC-{i + 1:04d}",
                "customer_id": f"CUST-{i + 1:04d}",
                "organization": org,
                "segment": segment,
                "country": country,
                "currency": COUNTRY_CURRENCY[country],
                "opened_at": (start - timedelta(days=int(rng.integers(30, 900)))).isoformat(),
                "avg_daily_balance": round(avg_balance, 2),
                "typical_txn_amount": round(typical, 2),
                "home_device_id": f"DEV-HOME-{i + 1:04d}",
                "home_ip_hash": hashlib.sha1(f"home-ip-{i}".encode("utf-8")).hexdigest()[:12],
                "risk_tier": rng.choice(["low", "medium", "high"]),
                "consent_status": "synthetic_demo_consent",
            }
        )

    accounts = pd.DataFrame(account_rows)

    balance_rows: List[Dict[str, Any]] = []
    for _, acc in accounts.iterrows():
        bal = float(acc["avg_daily_balance"])
        for d in range(days + 1):
            day_ts = normalize_utc_timestamp(start + timedelta(days=d)).normalize()
            bal = max(1.0, bal * float(rng.normal(1.0, 0.035)))
            balance_rows.append(
                {
                    "account_id": acc["account_id"],
                    "date": day_ts,
                    "balance": round(bal, 2),
                }
            )
    balances = pd.DataFrame(balance_rows)

    payee_pool = [f"CP-{j:04d}" for j in range(1, 181)]
    payee_country = {p: countries[int(rng.integers(0, len(countries)))] for p in payee_pool}

    tx_rows: List[Dict[str, Any]] = []

    for _, acc in accounts.iterrows():
        regular = [payee_pool[int(rng.integers(0, len(payee_pool)))] for _ in range(int(rng.integers(3, 10)))]
        n_norm = int(rng.integers(25, 75))

        for _ in range(n_norm):
            ts = start + timedelta(
                days=float(rng.uniform(0, days)),
                hours=float(rng.uniform(0, 24)),
                minutes=float(rng.uniform(0, 60)),
            )
            amount = float(np.exp(rng.normal(math.log(max(1.0, float(acc["typical_txn_amount"]))), 0.55)))
            amount = round(min(max(amount, 1.0), 250000.0), 2)

            cp = regular[int(rng.integers(0, len(regular)))] if rng.random() < 0.82 else payee_pool[int(rng.integers(0, len(payee_pool)))]
            cp_country = payee_country[cp] if rng.random() < 0.90 else str(acc["country"])
            device = str(acc["home_device_id"]) if rng.random() < 0.96 else f"DEV-TEMP-{int(rng.integers(10000, 99999))}"
            ip = str(acc["home_ip_hash"]) if rng.random() < 0.96 else hashlib.sha1(f"temp-ip-{rng.random()}".encode("utf-8")).hexdigest()[:12]
            direction = "debit" if rng.random() < 0.86 else "credit"
            channel = CHANNELS[int(rng.integers(0, len(CHANNELS)))]

            tx_rows.append(
                make_txn_row(
                    acc=acc,
                    ts=ts,
                    amount=amount,
                    direction=direction,
                    counterparty_id=cp,
                    counterparty_country=cp_country,
                    channel=channel,
                    device_id=device,
                    ip_hash=ip,
                    is_fraud=False,
                    label_source="synthetic_normal",
                    scenario="normal",
                    note="Synthetic normal activity",
                    source_system="core_banking_sim",
                )
            )

    fraud_accounts = accounts.sample(min(12, len(accounts)), random_state=int(rng.integers(0, 10000)))["account_id"].tolist()

    for acct_id in fraud_accounts:
        acc = accounts[accounts["account_id"] == acct_id].iloc[0]
        scenario = FRAUD_SCENARIOS[int(rng.integers(0, len(FRAUD_SCENARIOS)))]
        day_offset = int(rng.integers(0, days))
        base_ts = start + timedelta(
            days=day_offset,
            hours=int(rng.integers(0, 24)),
            minutes=int(rng.integers(0, 60)),
        )

        if scenario == "account_takeover":
            foreign = [c for c in countries if c != acc["country"]]
            cp_country = foreign[int(rng.integers(0, len(foreign)))] if foreign else "SG"
            for k in range(int(rng.integers(2, 5))):
                ts = base_ts + timedelta(minutes=5 * k + int(rng.integers(0, 10)))
                amount = round(float(acc["typical_txn_amount"]) * float(rng.uniform(3.0, 9.0)), 2)
                tx_rows.append(
                    make_txn_row(
                        acc=acc,
                        ts=ts,
                        amount=amount,
                        direction="debit",
                        counterparty_id=f"CP-ATO-{acct_id[-4:]}-{k:02d}",
                        counterparty_country=cp_country,
                        channel="wire",
                        device_id=f"DEV-ATO-{int(rng.integers(10000, 99999))}",
                        ip_hash=hashlib.sha1(f"ato-ip-{rng.random()}".encode("utf-8")).hexdigest()[:12],
                        is_fraud=True,
                        label_source=f"synthetic_{scenario}",
                        scenario=scenario,
                        note="Synthetic account-takeover pattern",
                        source_system="payments_sim",
                    )
                )

        elif scenario == "structuring":
            for k in range(int(rng.integers(5, 9))):
                ts = base_ts + timedelta(minutes=15 * k + int(rng.integers(0, 10)))
                amount = round(float(rng.uniform(9000.0, 9950.0)), 2)
                tx_rows.append(
                    make_txn_row(
                        acc=acc,
                        ts=ts,
                        amount=amount,
                        direction="debit",
                        counterparty_id=f"CP-STR-{acct_id[-4:]}-{k:02d}",
                        counterparty_country=str(acc["country"]),
                        channel="online_banking",
                        device_id=str(acc["home_device_id"]),
                        ip_hash=str(acc["home_ip_hash"]),
                        is_fraud=True,
                        label_source=f"synthetic_{scenario}",
                        scenario=scenario,
                        note="Synthetic near-threshold structuring pattern",
                        source_system="payments_sim",
                    )
                )

        elif scenario == "velocity":
            for k in range(int(rng.integers(10, 16))):
                ts = base_ts + timedelta(seconds=30 * k + int(rng.integers(0, 15)))
                amount = round(float(rng.uniform(80.0, 650.0)), 2)
                tx_rows.append(
                    make_txn_row(
                        acc=acc,
                        ts=ts,
                        amount=amount,
                        direction="debit",
                        counterparty_id=f"CP-VEL-{acct_id[-4:]}-{k:02d}",
                        counterparty_country=str(acc["country"]),
                        channel="card_pos",
                        device_id=str(acc["home_device_id"]),
                        ip_hash=str(acc["home_ip_hash"]),
                        is_fraud=True,
                        label_source=f"synthetic_{scenario}",
                        scenario=scenario,
                        note="Synthetic rapid velocity pattern",
                        source_system="payments_sim",
                    )
                )

        elif scenario == "new_payee":
            for k in range(int(rng.integers(1, 3))):
                ts = base_ts + timedelta(hours=k)
                amount = round(float(acc["typical_txn_amount"]) * float(rng.uniform(4.0, 12.0)), 2)
                cp_country = countries[int(rng.integers(0, len(countries)))]
                tx_rows.append(
                    make_txn_row(
                        acc=acc,
                        ts=ts,
                        amount=amount,
                        direction="debit",
                        counterparty_id=f"CP-NEW-{int(rng.integers(10000, 99999))}",
                        counterparty_country=cp_country,
                        channel="wire",
                        device_id=str(acc["home_device_id"]),
                        ip_hash=str(acc["home_ip_hash"]),
                        is_fraud=True,
                        label_source=f"synthetic_{scenario}",
                        scenario=scenario,
                        note="Synthetic new high-value payee pattern",
                        source_system="payments_sim",
                    )
                )

        else:  # balance_drain
            acct_balances = balances[balances["account_id"] == acct_id].sort_values("date")
            bal = float(acct_balances.iloc[-1]["balance"]) if not acct_balances.empty else float(acc["avg_daily_balance"])
            amount = round(bal * float(rng.uniform(0.65, 0.95)), 2)
            tx_rows.append(
                make_txn_row(
                    acc=acc,
                    ts=base_ts,
                    amount=amount,
                    direction="debit",
                    counterparty_id=f"CP-DRAIN-{acct_id[-4:]}",
                    counterparty_country=str(acc["country"]),
                    channel="wire",
                    device_id=str(acc["home_device_id"]),
                    ip_hash=str(acc["home_ip_hash"]),
                    is_fraud=True,
                    label_source=f"synthetic_{scenario}",
                    scenario=scenario,
                    note="Synthetic balance-drain pattern",
                    source_system="treasury_sim",
                )
            )

    transactions = pd.DataFrame(tx_rows)
    transactions["timestamp_utc"] = pd.to_datetime(transactions["timestamp_utc"], utc=True, errors="coerce")
    transactions = transactions.sort_values("timestamp_utc").reset_index(drop=True)
    transactions["txn_id"] = [f"TXN-{i + 1:06d}" for i in range(len(transactions))]
    transactions = ensure_columns(transactions, CLEAN_TX_COLUMNS)

    db = {
        "accounts": accounts,
        "transactions": transactions,
        "balances": balances,
        "features": empty_df(FEATURE_COLUMNS),
        "alerts": empty_df(ALERT_COLUMNS),
        "audit": empty_df(AUDIT_COLUMNS),
        "imports": empty_df(IMPORT_COLUMNS),
        "risk_engine": None,
        "demo_video_path": None,
    }
    return db


class FraudRiskEngine:
    def __init__(self, accounts_df: pd.DataFrame, balances_df: pd.DataFrame, rules: Dict[str, Any]):
        self.rules = rules
        self.accounts = accounts_df.copy()
        self.acc_map = self.accounts.set_index("account_id").to_dict("index") if not self.accounts.empty else {}

        self.bal_map: Dict[str, Dict[str, float]] = defaultdict(dict)
        if balances_df is not None and not balances_df.empty:
            bal = balances_df.copy()
            bal["date"] = pd.to_datetime(bal["date"], utc=True, errors="coerce")
            for acct_id, grp in bal.groupby("account_id"):
                self.bal_map[str(acct_id)] = {
                    str(row["date"].date()): to_float(row["balance"], 0.0)
                    for _, row in grp.iterrows()
                    if pd.notna(row["date"])
                }

        self.hist_tx: Dict[str, List[Tuple[pd.Timestamp, float, str, str, str, str]]] = defaultdict(list)
        self.cp_seen: Dict[str, set] = defaultdict(set)
        self.device_accounts: Dict[str, set] = defaultdict(set)
        self.ip_accounts: Dict[str, set] = defaultdict(set)
        self.day_struct: Dict[Tuple[str, str], int] = defaultdict(int)

    def score_row(self, row: pd.Series | Dict[str, Any]) -> Dict[str, Any]:
        get = row.get if isinstance(row, dict) else lambda k, d=None: row.get(k, d)

        acct_id = str(get("account_id", "") or "")
        acc = self.acc_map.get(acct_id, {})
        txn_id = str(get("txn_id", "") or stable_id("TXN", acct_id, str(get("timestamp_utc", "")), str(get("amount", ""))))
        ts = normalize_utc_timestamp(get("timestamp_utc", utc_now()))

        amount = to_float(get("amount", 0.0), 0.0)
        currency = str(get("currency", acc.get("currency", "USD"))).upper()
        direction = str(get("direction", "debit")).lower()
        cp = str(get("counterparty_id", "") or "")
        cp_country = str(get("counterparty_country", acc.get("country", "")) or "").upper()
        channel = str(get("channel", "") or "").lower()
        device = str(get("device_id", "") or "")
        ip = str(get("ip_hash", "") or "")
        note = str(get("note", "") or "")
        source_system = str(get("source_system", "unknown") or "unknown")
        source_type = str(get("source_type", "unknown") or "unknown")
        is_fraud = to_bool(get("is_fraud_synthetic", False))
        label_source = str(get("label_source", "unknown") or "unknown")
        scenario = str(get("scenario", "unknown") or "unknown")

        account_country = str(acc.get("country", "") or "")
        account_segment = str(acc.get("segment", "unknown") or "unknown")
        home_device = str(acc.get("home_device_id", "") or "")
        home_ip = str(acc.get("home_ip_hash", "") or "")
        typical_amount = to_float(acc.get("typical_txn_amount", amount + 1.0), amount + 1.0)
        avg_balance = to_float(acc.get("avg_daily_balance", 1.0), 1.0)

        warnings: List[str] = []
        missing: List[str] = []
        for field, value in [
            ("counterparty_id", cp),
            ("counterparty_country", cp_country),
            ("channel", channel),
            ("device_id", device),
            ("ip_hash", ip),
        ]:
            if not value:
                missing.append(field)
        if missing:
            warnings.append("Missing fields: " + ", ".join(missing))
        if amount <= 0:
            warnings.append("Non-positive amount")
        if not acc:
            warnings.append("Unknown account reference")

        history = self.hist_tx.get(acct_id, [])
        window_hours = to_float(self.rules.get("velocity_window_hours", 1), 1.0)
        recent = []
        for item in history:
            delta_hours = (ts - item[0]).total_seconds() / 3600.0
            if 0 <= delta_hours <= window_hours:
                recent.append(item)

        vel_count = len(recent) + 1
        vel_amount = amount + sum(float(item[1]) for item in recent)

        prev = history[-1] if history else None
        travel_signal = 0.0
        travel_detail = "No prior transaction"
        hours_since_prev = None
        if prev is not None:
            prev_ts, prev_amt, prev_country, prev_dev, prev_ip, prev_cp = prev
            hours_since_prev = (ts - prev_ts).total_seconds() / 3600.0
            travel_max = to_float(self.rules.get("travel_hours_max", 6), 6.0)
            if prev_country and cp_country and prev_country != cp_country and 0 <= hours_since_prev <= travel_max:
                travel_signal = 1.0
            if hours_since_prev is not None:
                travel_detail = (
                    f"hours_since_prev={hours_since_prev:.2f}, prev_country={prev_country}, txn_country={cp_country}"
                )

        new_cp = bool(cp and cp not in self.cp_seen.get(acct_id, set()))

        struct_low = to_float(self.rules.get("structuring_low", 9000), 9000.0)
        struct_high = to_float(self.rules.get("structuring_high", 9999), 9999.0)
        near_threshold = struct_low <= amount <= struct_high
        day_key = (acct_id, str(ts.date()))
        struct_count = self.day_struct.get(day_key, 0) + (1 if near_threshold else 0)
        struct_signal = 1.0 if near_threshold and struct_count >= int(self.rules.get("structuring_min_count", 2)) else 0.0

        bal = self.bal_map.get(acct_id, {}).get(str(ts.date()), avg_balance)
        balance_missing = str(ts.date()) not in self.bal_map.get(acct_id, {})
        ratio = amount / bal if bal > 0 else 0.0
        drain_trigger = to_float(self.rules.get("balance_drain_trigger", 0.6), 0.6)
        drain_signal = 0.0
        if ratio > drain_trigger:
            drain_signal = min(1.0, max(0.0, (ratio - drain_trigger) / max(1e-6, 1.0 - drain_trigger)))

        device_changed = bool(device and home_device and device != home_device)
        ip_changed = bool(ip and home_ip and ip != home_ip)
        device_signal = 0.0
        if device_changed and ip_changed:
            device_signal = 0.8
        elif device_changed or ip_changed:
            device_signal = 0.4

        shared_device = bool(device and len(self.device_accounts.get(device, set()) - {acct_id}) > 0)
        shared_ip = bool(ip and len(self.ip_accounts.get(ip, set()) - {acct_id}) > 0)
        network_signal = 1.0 if shared_device or shared_ip else 0.0

        hour = int(ts.hour)
        night_start = int(self.rules.get("night_start", 0))
        night_end = int(self.rules.get("night_end", 5))
        night_signal = 1.0 if night_start <= hour <= night_end else 0.0

        z = 0.0
        med = amount
        mad = 1.0
        if len(history) >= 8:
            vals = np.array([float(item[1]) for item in history], dtype=float)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med))) + 1e-6
            z = (amount - med) / (1.4826 * mad)

        amt_trigger = to_float(self.rules.get("amount_z_trigger", 2.5), 2.5)
        amount_signal = 0.0
        if z > amt_trigger:
            amount_signal = min(1.0, max(0.0, (z - amt_trigger) / 3.0))

        vel_trigger = to_float(self.rules.get("velocity_count_trigger", 5), 5.0)
        vel_count_signal = min(1.0, max(0.0, (vel_count - vel_trigger) / 6.0))
        vel_amount_signal = 0.0
        if typical_amount > 0:
            vel_amount_signal = min(1.0, max(0.0, (vel_amount - 3.0 * typical_amount) / (10.0 * typical_amount)))
        velocity_signal = max(vel_count_signal, vel_amount_signal)

        new_payee_amt = to_float(self.rules.get("new_payee_amount", 5000), 5000.0)
        new_payee_signal = 0.0
        if new_cp and amount > new_payee_amt:
            new_payee_signal = min(1.0, amount / max(1.0, 2.0 * new_payee_amt))

        signals = {
            "AMOUNT_OUTLIER": amount_signal,
            "VELOCITY_SPIKE": velocity_signal,
            "IMPOSSIBLE_TRAVEL": travel_signal,
            "NEW_COUNTERPARTY_HIGH_VALUE": new_payee_signal,
            "STRUCTURING_NEAR_THRESHOLD": struct_signal,
            "BALANCE_DRAIN": drain_signal,
            "DEVICE_OR_IP_CHANGE": device_signal,
            "SHARED_DEVICE_NETWORK": network_signal,
            "NIGHT_ACTIVITY": night_signal,
            "DATA_QUALITY_LOW": min(1.0, len(missing) / 4.0) if missing else 0.0,
        }

        details = {
            "AMOUNT_OUTLIER": f"amount={amount:.2f}, prior_median={med:.2f}, z={z:.2f}",
            "VELOCITY_SPIKE": f"count_1h={vel_count}, amount_1h={vel_amount:.2f}",
            "IMPOSSIBLE_TRAVEL": travel_detail,
            "NEW_COUNTERPARTY_HIGH_VALUE": f"counterparty={cp}, new={new_cp}, threshold={new_payee_amt:.2f}",
            "STRUCTURING_NEAR_THRESHOLD": f"band={struct_low:.0f}-{struct_high:.0f}, same_day_count={struct_count}",
            "BALANCE_DRAIN": f"balance={bal:.2f}, ratio={ratio:.3f}",
            "DEVICE_OR_IP_CHANGE": f"device_changed={device_changed}, ip_changed={ip_changed}",
            "SHARED_DEVICE_NETWORK": f"shared_device={shared_device}, shared_ip={shared_ip}",
            "NIGHT_ACTIVITY": f"utc_hour={hour}",
            "DATA_QUALITY_LOW": f"missing_fields={missing}",
        }

        weights = self.rules.get("weights", {})
        contributions: List[Dict[str, Any]] = []
        for code, value in signals.items():
            value = float(min(1.0, max(0.0, value)))
            if value > 0.01:
                weight = to_float(weights.get(code, 0.0), 0.0)
                points = weight * value
                if points > 0.01:
                    contributions.append(
                        {
                            "code": code,
                            "signal": round(value, 3),
                            "weight": round(weight, 2),
                            "points": round(points, 2),
                            "detail": details.get(code, ""),
                        }
                    )

        score = float(sum(c["points"] for c in contributions))
        if amount <= 0:
            score = 0.0
        score = round(min(100.0, max(0.0, score)), 1)

        confidence = 1.0
        confidence -= 0.10 * len(missing)
        if not acc:
            confidence -= 0.25
        if balance_missing:
            confidence -= 0.05
        confidence = round(max(0.35, min(1.0, confidence)), 2)

        contributions_sorted = sorted(contributions, key=lambda x: x["points"], reverse=True)
        reason_codes = ", ".join(c["code"] for c in contributions_sorted)

        explanation = {
            "ruleset_version": RULESET_VERSION,
            "formula": "risk_score = min(100, sum(weight_i * signal_i)); signals are clipped to [0,1]",
            "transaction": {
                "txn_id": txn_id,
                "account_id": acct_id,
                "timestamp_utc": ts.isoformat(),
                "amount": amount,
                "currency": currency,
                "direction": direction,
                "counterparty_id": cp,
                "counterparty_country": cp_country,
                "channel": channel,
                "device_id": device,
                "ip_hash": ip,
                "note": note,
                "source_system": source_system,
                "source_type": source_type,
            },
            "account_context": {
                "account_country": account_country,
                "account_segment": account_segment,
                "home_device_id": home_device,
                "home_ip_hash": home_ip,
                "typical_txn_amount": typical_amount,
                "reference_balance": bal,
            },
            "features": {
                "amount_z": round(z, 3),
                "velocity_1h_count": vel_count,
                "velocity_1h_amount": round(vel_amount, 2),
                "balance_drain_ratio": round(ratio, 4),
                "new_counterparty": new_cp,
                "device_changed": device_changed,
                "ip_changed": ip_changed,
                "shared_device": shared_device,
                "shared_ip": shared_ip,
                "night_activity": bool(night_signal > 0),
                "same_day_near_threshold_count": struct_count,
            },
            "signals": {k: round(float(v), 3) for k, v in signals.items()},
            "contributions": contributions_sorted,
            "warnings": warnings,
            "confidence": confidence,
            "data_quality": "ok" if not warnings else "; ".join(warnings),
            "synthetic_notice": SYNTHETIC_NOTICE,
        }

        feature = {
            "txn_id": txn_id,
            "account_id": acct_id,
            "timestamp_utc": ts,
            "amount": round(amount, 2),
            "currency": currency,
            "direction": direction,
            "counterparty_id": cp,
            "counterparty_country": cp_country,
            "channel": channel,
            "device_id": device,
            "ip_hash": ip,
            "risk_score": score,
            "confidence": confidence,
            "reason_codes": reason_codes,
            "explanation": json.dumps(explanation, default=json_default),
            "data_quality": "ok" if not warnings else "; ".join(warnings),
            "is_fraud_synthetic": is_fraud,
            "label_source": label_source,
            "scenario": scenario,
            "account_country": account_country,
            "account_segment": account_segment,
            "velocity_1h_count": vel_count,
            "velocity_1h_amount": round(vel_amount, 2),
            "amount_z": round(z, 3),
            "balance_drain_ratio": round(ratio, 4),
            "new_counterparty": new_cp,
            "device_changed": device_changed,
            "shared_network": bool(shared_device or shared_ip),
            "night_activity": bool(night_signal > 0),
            "ruleset_version": RULESET_VERSION,
            "source_type": source_type,
            "source_system": source_system,
        }

        # Update incremental state after scoring.
        self.hist_tx[acct_id].append((ts, amount, cp_country, device, ip, cp))
        if cp:
            self.cp_seen[acct_id].add(cp)
        if device:
            self.device_accounts[device].add(acct_id)
        if ip:
            self.ip_accounts[ip].add(acct_id)
        if near_threshold:
            self.day_struct[day_key] = struct_count

        return feature


def build_features_and_engine(
    transactions_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    balances_df: pd.DataFrame,
    rules: Dict[str, Any],
) -> Tuple[pd.DataFrame, FraudRiskEngine]:
    tx = transactions_df.copy()
    if not tx.empty:
        tx["timestamp_utc"] = pd.to_datetime(tx["timestamp_utc"], utc=True, errors="coerce")
        tx = tx.sort_values("timestamp_utc").reset_index(drop=True)
    engine = FraudRiskEngine(accounts_df, balances_df, rules)
    rows: List[Dict[str, Any]] = []
    if not tx.empty:
        for _, row in tx.iterrows():
            rows.append(engine.score_row(row))
    features = ensure_columns(pd.DataFrame(rows), FEATURE_COLUMNS)
    if not features.empty:
        features["timestamp_utc"] = pd.to_datetime(features["timestamp_utc"], utc=True, errors="coerce")
        features["risk_score"] = pd.to_numeric(features["risk_score"], errors="coerce").fillna(0.0)
        features["confidence"] = pd.to_numeric(features["confidence"], errors="coerce").fillna(1.0)
        features["is_fraud_synthetic"] = features["is_fraud_synthetic"].apply(to_bool)
    return features, engine


def feature_to_alert(f: pd.Series | Dict[str, Any], threshold: float) -> Dict[str, Any]:
    get = f.get if isinstance(f, dict) else lambda k, d=None: f.get(k, d)
    now_iso = utc_now().isoformat()
    txn_ts = get("timestamp_utc", now_iso)
    try:
        txn_ts_iso = normalize_utc_timestamp(txn_ts).isoformat()
    except Exception:
        txn_ts_iso = str(txn_ts)

    score = to_float(get("risk_score", 0.0), 0.0)
    return {
        "alert_id": stable_id("ALR", get("txn_id", ""), get("ruleset_version", RULESET_VERSION)),
        "txn_id": str(get("txn_id", "")),
        "account_id": str(get("account_id", "")),
        "txn_timestamp": txn_ts_iso,
        "amount": to_float(get("amount", 0.0), 0.0),
        "currency": str(get("currency", "")),
        "created_at": txn_ts_iso,
        "updated_at": now_iso,
        "model_score": score,
        "risk_score": score,
        "status": "new",
        "assigned_to": "",
        "reviewer": "",
        "decision": "",
        "override_reason": "",
        "feedback": "",
        "reason_codes": str(get("reason_codes", "")),
        "explanation": str(get("explanation", "{}")),
        "data_quality": str(get("data_quality", "ok")),
        "confidence": to_float(get("confidence", 1.0), 1.0),
        "ruleset_version": str(get("ruleset_version", RULESET_VERSION)),
        "source_type": str(get("source_type", "unknown")),
    }


def create_alerts_from_features(features_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if features_df is None or features_df.empty:
        return empty_df(ALERT_COLUMNS)
    fs = features_df.copy()
    fs["risk_score"] = pd.to_numeric(fs["risk_score"], errors="coerce").fillna(0.0)
    subset = fs[fs["risk_score"] >= float(threshold)]
    rows = [feature_to_alert(row, threshold) for _, row in subset.iterrows()]
    return ensure_columns(pd.DataFrame(rows), ALERT_COLUMNS)


def merge_alerts(existing: Optional[pd.DataFrame], new: Optional[pd.DataFrame]) -> pd.DataFrame:
    existing = ensure_columns(existing, ALERT_COLUMNS)
    new = ensure_columns(new, ALERT_COLUMNS)
    if not new.empty:
        new = new.drop_duplicates(subset=["alert_id"], keep="first")
    if existing.empty:
        return new
    if new.empty:
        return existing
    existing_ids = set(existing["alert_id"].astype(str))
    add = new[~new["alert_id"].astype(str).isin(existing_ids)]
    return ensure_columns(concat_df(existing, add), ALERT_COLUMNS)


def refresh_db(db: Dict[str, Any], rules: Dict[str, Any]) -> None:
    features, engine = build_features_and_engine(
        db.get("transactions", empty_df(CLEAN_TX_COLUMNS)),
        db.get("accounts", pd.DataFrame()),
        db.get("balances", pd.DataFrame()),
        rules,
    )
    db["features"] = features
    db["risk_engine"] = engine
    new_alerts = create_alerts_from_features(features, rules.get("alert_threshold", 55))
    db["alerts"] = merge_alerts(db.get("alerts"), new_alerts)


def simulate_one_event(
    db: Dict[str, Any],
    rules: Dict[str, Any],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    rng = rng or np.random.default_rng()
    accounts = db["accounts"]
    if accounts.empty:
        raise RuntimeError("Cannot simulate because accounts table is empty.")

    suspicious = bool(rng.random() < 0.38)
    recent_ids = db["transactions"].tail(30)["account_id"].dropna().astype(str).unique() if not db["transactions"].empty else []

    if suspicious and len(recent_ids) > 0 and rng.random() < 0.65:
        acct_id = str(recent_ids[int(rng.integers(0, len(recent_ids)))])
        matches = accounts[accounts["account_id"] == acct_id]
        acc = matches.iloc[0] if not matches.empty else accounts.iloc[int(rng.integers(0, len(accounts)))]
    else:
        acc = accounts.iloc[int(rng.integers(0, len(accounts)))]

    now = utc_now()
    countries = list(COUNTRY_CURRENCY.keys())

    if suspicious:
        scenario = FRAUD_SCENARIOS[int(rng.integers(0, len(FRAUD_SCENARIOS)))]
        if scenario == "account_takeover":
            foreign = [c for c in countries if c != acc["country"]]
            cp_country = foreign[int(rng.integers(0, len(foreign)))] if foreign else "SG"
            amount = round(float(acc["typical_txn_amount"]) * float(rng.uniform(3.5, 8.5)), 2)
            row = make_txn_row(
                acc=acc,
                ts=now,
                amount=amount,
                direction="debit",
                counterparty_id=f"CP-LIVE-ATO-{int(rng.integers(1000, 9999))}",
                counterparty_country=cp_country,
                channel="wire",
                device_id=f"DEV-LIVE-{int(rng.integers(10000, 99999))}",
                ip_hash=hashlib.sha1(f"live-ato-{rng.random()}".encode("utf-8")).hexdigest()[:12],
                is_fraud=True,
                label_source=f"synthetic_live_{scenario}",
                scenario=scenario,
                note="Live synthetic account-takeover event",
                source_system="live_simulator",
            )
        elif scenario == "structuring":
            amount = round(float(rng.uniform(9000, 9950)), 2)
            row = make_txn_row(
                acc=acc,
                ts=now,
                amount=amount,
                direction="debit",
                counterparty_id=f"CP-LIVE-STR-{int(rng.integers(1000, 9999))}",
                counterparty_country=str(acc["country"]),
                channel="online_banking",
                device_id=str(acc["home_device_id"]),
                ip_hash=str(acc["home_ip_hash"]),
                is_fraud=True,
                label_source=f"synthetic_live_{scenario}",
                scenario=scenario,
                note="Live synthetic near-threshold event",
                source_system="live_simulator",
            )
        elif scenario == "velocity":
            amount = round(float(rng.uniform(120, 900)), 2)
            row = make_txn_row(
                acc=acc,
                ts=now,
                amount=amount,
                direction="debit",
                counterparty_id=f"CP-LIVE-VEL-{int(rng.integers(1000, 9999))}",
                counterparty_country=str(acc["country"]),
                channel="card_pos",
                device_id=str(acc["home_device_id"]),
                ip_hash=str(acc["home_ip_hash"]),
                is_fraud=True,
                label_source=f"synthetic_live_{scenario}",
                scenario=scenario,
                note="Live synthetic velocity event",
                source_system="live_simulator",
            )
        elif scenario == "new_payee":
            amount = round(float(acc["typical_txn_amount"]) * float(rng.uniform(4.0, 10.0)), 2)
            row = make_txn_row(
                acc=acc,
                ts=now,
                amount=amount,
                direction="debit",
                counterparty_id=f"CP-LIVE-NEW-{int(rng.integers(10000, 99999))}",
                counterparty_country=countries[int(rng.integers(0, len(countries)))],
                channel="wire",
                device_id=str(acc["home_device_id"]),
                ip_hash=str(acc["home_ip_hash"]),
                is_fraud=True,
                label_source=f"synthetic_live_{scenario}",
                scenario=scenario,
                note="Live synthetic new-payee event",
                source_system="live_simulator",
            )
        else:
            engine = db.get("risk_engine")
            today = str(now.date())
            bal = 0.0
            if engine is not None:
                bal = engine.bal_map.get(str(acc["account_id"]), {}).get(today, float(acc["avg_daily_balance"]))
            else:
                bal = float(acc["avg_daily_balance"])
            amount = round(max(10.0, bal * float(rng.uniform(0.65, 0.95))), 2)
            row = make_txn_row(
                acc=acc,
                ts=now,
                amount=amount,
                direction="debit",
                counterparty_id=f"CP-LIVE-DRAIN-{int(rng.integers(1000, 9999))}",
                counterparty_country=str(acc["country"]),
                channel="wire",
                device_id=str(acc["home_device_id"]),
                ip_hash=str(acc["home_ip_hash"]),
                is_fraud=True,
                label_source="synthetic_live_balance_drain",
                scenario="balance_drain",
                note="Live synthetic balance-drain event",
                source_system="live_simulator",
            )
    else:
        amount = float(np.exp(rng.normal(math.log(max(1.0, float(acc["typical_txn_amount"]))), 0.45)))
        amount = round(min(max(amount, 1.0), 150000.0), 2)
        row = make_txn_row(
            acc=acc,
            ts=now,
            amount=amount,
            direction="debit" if rng.random() < 0.85 else "credit",
            counterparty_id=f"CP-LIVE-{int(rng.integers(1, 500)):04d}",
            counterparty_country=str(acc["country"]) if rng.random() < 0.9 else countries[int(rng.integers(0, len(countries)))],
            channel=CHANNELS[int(rng.integers(0, len(CHANNELS)))],
            device_id=str(acc["home_device_id"]) if rng.random() < 0.94 else f"DEV-LIVE-{int(rng.integers(10000, 99999))}",
            ip_hash=str(acc["home_ip_hash"]) if rng.random() < 0.94 else hashlib.sha1(f"live-ip-{rng.random()}".encode("utf-8")).hexdigest()[:12],
            is_fraud=False,
            label_source="synthetic_live_normal",
            scenario="normal",
            note="Live synthetic normal event",
            source_system="live_simulator",
        )

    row["txn_id"] = f"TXN-LIVE-{len(db['transactions']) + 1:06d}"
    row["source_type"] = "simulated_live"

    if db.get("risk_engine") is None:
        refresh_db(db, rules)

    feature = db["risk_engine"].score_row(pd.Series(row))

    db["transactions"] = ensure_columns(concat_df(db["transactions"], pd.DataFrame([row])), CLEAN_TX_COLUMNS)
    feature_df = ensure_columns(pd.DataFrame([feature]), FEATURE_COLUMNS)
    db["features"] = ensure_columns(concat_df(db["features"], feature_df), FEATURE_COLUMNS)

    alert = None
    if to_float(feature["risk_score"], 0.0) >= to_float(rules.get("alert_threshold", 55), 55.0):
        alert = feature_to_alert(feature, rules.get("alert_threshold", 55))
        db["alerts"] = ensure_columns(concat_df(db["alerts"], pd.DataFrame([alert])), ALERT_COLUMNS)
        add_audit(db, "System", "risk_engine", "alert_created", "alert", alert["alert_id"], f"score={alert['risk_score']}")

    return row, feature, alert


def render_video_frame(db: Dict[str, Any], step: int, total_steps: int, rules: Dict[str, Any]) -> np.ndarray:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=80)

    tx = db.get("transactions", pd.DataFrame()).copy()
    if not tx.empty:
        tx["timestamp_utc"] = pd.to_datetime(tx["timestamp_utc"], utc=True, errors="coerce")
        tx = tx.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
        max_ts = tx["timestamp_utc"].max()
        recent = tx[tx["timestamp_utc"] >= max_ts - timedelta(days=14)]
        if not recent.empty:
            daily = recent.set_index("timestamp_utc").resample("D").size()
            axes[0, 0].bar(range(len(daily)), daily.values.astype(float), color="#4C78A8")
            step_size = max(1, len(daily) // 6)
            axes[0, 0].set_xticks(range(0, len(daily), step_size))
            axes[0, 0].set_xticklabels([d.strftime("%m-%d") for d in daily.index[::step_size]], rotation=0, fontsize=8)
        else:
            axes[0, 0].text(0.5, 0.5, "No recent transactions", ha="center", va="center")
    else:
        axes[0, 0].text(0.5, 0.5, "No transactions", ha="center", va="center")
    axes[0, 0].set_title("Daily transaction volume, last 14 days", fontsize=10)
    axes[0, 0].set_ylabel("Count")

    features = db.get("features", pd.DataFrame()).copy()
    if not features.empty:
        scores = pd.to_numeric(features["risk_score"], errors="coerce").dropna()
        axes[0, 1].hist(scores, bins=20, color="#F58518", alpha=0.75)
        axes[0, 1].axvline(float(rules.get("alert_threshold", 55)), color="red", linestyle="--", linewidth=1.5)
    else:
        axes[0, 1].text(0.5, 0.5, "No scores", ha="center", va="center")
    axes[0, 1].set_title("Risk score distribution", fontsize=10)
    axes[0, 1].set_xlabel("Score")
    axes[0, 1].set_ylabel("Count")

    reason_list: List[str] = []
    if not features.empty and "reason_codes" in features.columns:
        for rc in features["reason_codes"].dropna():
            reason_list.extend(str(rc).split(", "))
    if reason_list:
        vc = pd.Series(reason_list).value_counts().head(6)
        axes[1, 0].barh(vc.index[::-1], vc.values[::-1], color="#54A24B")
        axes[1, 0].tick_params(axis="y", labelsize=8)
    else:
        axes[1, 0].text(0.5, 0.5, "No reason codes yet", ha="center", va="center")
    axes[1, 0].set_title("Top reason codes", fontsize=10)
    axes[1, 0].set_xlabel("Frequency")

    alerts = db.get("alerts", pd.DataFrame())
    if alerts is not None and not alerts.empty:
        status = alerts["status"].value_counts()
        axes[1, 1].bar(status.index.astype(str), status.values, color="#E45756")
        axes[1, 1].tick_params(axis="x", rotation=20, labelsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "No alerts yet", ha="center", va="center")
    axes[1, 1].set_title("Alert queue status", fontsize=10)
    axes[1, 1].set_ylabel("Count")

    fig.suptitle(
        f"{APP_NAME} — synthetic live demo step {step + 1}/{total_steps}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return np.ascontiguousarray(img)


def write_video_frames(frames: List[np.ndarray], out_path: str, fps: int = 12) -> Optional[str]:
    if imageio is None or not frames:
        return None
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    try:
        with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=6) as writer:
            for frame in frames:
                writer.append_data(frame)
        return out_path
    except Exception:
        try:
            gif_path = os.path.splitext(out_path)[0] + ".gif"
            imageio.mimsave(gif_path, frames, format="GIF", duration=1.0 / float(fps), loop=0)
            return gif_path
        except Exception:
            return None


def generate_demo_video(
    db: Dict[str, Any],
    rules: Dict[str, Any],
    steps: int = 36,
    fps: int = 12,
    progress_cb: Optional[Any] = None,
) -> Optional[str]:
    if imageio is None:
        return None

    frames: List[np.ndarray] = []
    rng = np.random.default_rng(2026)
    for i in range(steps):
        try:
            simulate_one_event(db, rules, rng)
            frames.append(render_video_frame(db, i, steps, rules))
        except Exception:
            break
        if progress_cb:
            progress_cb((i + 1) / float(steps), f"Rendering demo frame {i + 1}/{steps}")

    if not frames:
        return None

    out_path = os.path.join(os.getcwd(), VIDEO_DIR, VIDEO_FILENAME)
    return write_video_frames(frames, out_path, fps=fps)


def initialize_demo(progress_cb: Optional[Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rules = json.loads(json.dumps(DEFAULT_RULES))

    if progress_cb:
        progress_cb(0.05, "Generating synthetic accounts, balances, and transactions")
    db = generate_demo_dataset(seed=42)

    if progress_cb:
        progress_cb(0.20, "Scoring historical synthetic transactions with deterministic rules")
    refresh_db(db, rules)

    add_audit(db, "System", "bootstrap", "dataset_created", "database", "demo", f"transactions={len(db['transactions'])}")

    video_path = None
    if progress_cb:
        progress_cb(0.35, "Recording local synthetic live demo video")
    try:
        video_path = generate_demo_video(
            db,
            rules,
            steps=36,
            fps=12,
            progress_cb=lambda p, msg: progress_cb(0.35 + 0.65 * float(p), msg) if progress_cb else None,
        )
    except Exception:
        video_path = None

    db["demo_video_path"] = video_path
    if video_path:
        add_audit(db, "System", "bootstrap", "demo_video_generated", "database", "demo", str(video_path))

    if progress_cb:
        progress_cb(1.0, "Demo ready")
    return db, rules


def sample_import_csv(db: Dict[str, Any]) -> bytes:
    accounts = db.get("accounts", pd.DataFrame()).head(3)
    rows = []
    cols = REQUIRED_TX_COLUMNS + ["note", "source_system", "is_fraud_synthetic", "label_source", "scenario", "consent_id"]

    if accounts.empty:
        return pd.DataFrame(columns=cols).to_csv(index=False).encode("utf-8")

    for i, (_, acc) in enumerate(accounts.iterrows(), start=1):
        rows.append(
            {
                "txn_id": f"TXN-IMPORT-EXAMPLE-{i:04d}",
                "account_id": acc["account_id"],
                "timestamp_utc": (utc_now() - timedelta(hours=i)).isoformat(),
                "amount": round(1250.50 + i * 10.25, 2),
                "currency": acc["currency"],
                "direction": "debit",
                "counterparty_id": f"CP-EXAMPLE-{i:04d}",
                "counterparty_country": acc["country"],
                "channel": "online_banking",
                "device_id": acc["home_device_id"],
                "ip_hash": acc["home_ip_hash"],
                "note": "Valid synthetic import example",
                "source_system": "example_file",
                "is_fraud_synthetic": False,
                "label_source": "example_label",
                "scenario": "normal",
                "consent_id": f"CONSENT-EXAMPLE-{i:04d}",
            }
        )
    return pd.DataFrame(rows, columns=cols).to_csv(index=False).encode("utf-8")


def malformed_sample_csv() -> bytes:
    rows = [
        {
            "txn_id": "TXN-BAD-0001",
            "account_id": "ACC-0001",
            "timestamp_utc": "2026-01-01T10:00:00Z",
            "amount": -25.0,
            "currency": "USD",
            "direction": "debit",
            "counterparty_id": "CP-BAD-0001",
            "counterparty_country": "US",
            "channel": "online_banking",
            "device_id": "DEV-BAD",
            "ip_hash": "IP-BAD",
            "note": "Negative amount",
        },
        {
            "txn_id": "TXN-BAD-0002",
            "account_id": "ACC-MISSING",
            "timestamp_utc": "not-a-date",
            "amount": 100.0,
            "currency": "USD",
            "direction": "transfer",
            "counterparty_id": "",
            "counterparty_country": "US",
            "channel": "wire",
            "device_id": "DEV-BAD",
            "ip_hash": "IP-BAD",
            "note": "Bad date, unknown account, invalid direction, missing counterparty",
        },
        {
            "txn_id": "TXN-BAD-0001",
            "account_id": "ACC-0001",
            "timestamp_utc": "2026-01-01T10:05:00Z",
            "amount": 9999999999.0,
            "currency": "USA",
            "direction": "credit",
            "counterparty_id": "CP-BAD-0003",
            "counterparty_country": "US",
            "channel": "branch",
            "device_id": "DEV-BAD",
            "ip_hash": "IP-BAD",
            "note": "Duplicate txn_id, out-of-range amount, bad currency",
        },
    ]
    cols = REQUIRED_TX_COLUMNS + ["note"]
    return pd.DataFrame(rows, columns=cols).to_csv(index=False).encode("utf-8")


def read_uploaded_file(file: Any) -> pd.DataFrame:
    name = str(getattr(file, "name", "")).lower()
    data = file.getvalue()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    if name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(data), engine="openpyxl")
    raise ValueError("Only CSV and XLSX files are supported in this MVP.")


def validate_transactions_upload(
    df: pd.DataFrame,
    existing_txn_ids: set,
    known_account_ids: set,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    cleaned: List[Dict[str, Any]] = []

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing_cols = [c for c in REQUIRED_TX_COLUMNS if c not in df.columns]
    if missing_cols:
        for col in missing_cols:
            errors.append(
                {
                    "row_number": 1,
                    "field": col,
                    "issue": "Required column is missing",
                    "provided_value": "",
                    "suggested_fix": f"Add column '{col}' to the file header.",
                    "severity": "error",
                }
            )
        return (
            empty_df(CLEAN_TX_COLUMNS),
            ensure_columns(pd.DataFrame(errors), ERROR_COLUMNS),
            {"total_rows": len(df), "valid_rows": 0, "error_count": len(errors), "warning_count": 0},
        )

    seen_txn: set = set()
    now = utc_now()
    error_count = 0
    warning_count = 0

    for pos, (_, row) in enumerate(df.iterrows(), start=2):
        row_has_error = False

        def add_error(field: str, issue: str, value: Any, fix: str, severity: str = "error") -> None:
            nonlocal error_count, warning_count, row_has_error
            errors.append(
                {
                    "row_number": pos,
                    "field": field,
                    "issue": issue,
                    "provided_value": "" if value is None else str(value),
                    "suggested_fix": fix,
                    "severity": severity,
                }
            )
            if severity == "error":
                error_count += 1
                row_has_error = True
            else:
                warning_count += 1

        for col in REQUIRED_TX_COLUMNS:
            val = row.get(col)
            if val is None or (isinstance(val, float) and math.isnan(val)) or str(val).strip() == "":
                add_error(col, "Required field is missing", val, f"Provide a non-empty value for '{col}'.")

        txn_id = str(row.get("txn_id", "")).strip()
        if txn_id:
            if txn_id in seen_txn:
                add_error("txn_id", "Duplicate transaction ID inside the same file", txn_id, "Make txn_id unique per row.")
            elif txn_id in existing_txn_ids:
                add_error("txn_id", "Transaction ID already exists; safe re-import will skip this row", txn_id, "Use a new txn_id or accept skip-on-commit behavior.", severity="warning")
            seen_txn.add(txn_id)

        ts = pd.to_datetime(row.get("timestamp_utc"), utc=True, errors="coerce")
        if pd.isna(ts):
            add_error("timestamp_utc", "Unparseable timestamp", row.get("timestamp_utc"), "Use ISO-8601 UTC, e.g. 2026-01-31T14:30:00Z.")
        else:
            if ts > now + timedelta(days=2):
                add_error("timestamp_utc", "Timestamp is too far in the future", ts.isoformat(), "Correct the event time or time zone.")
            if ts < now - timedelta(days=3650):
                add_error("timestamp_utc", "Timestamp is outside supported historical range", ts.isoformat(), "Use a date within the last 10 years for MVP validation.")

        amount = to_float(row.get("amount"), math.nan)
        if math.isnan(amount):
            add_error("amount", "Amount is not numeric", row.get("amount"), "Use a positive decimal number.")
        elif amount <= 0:
            add_error("amount", "Amount must be greater than zero", amount, "Use a positive transaction amount.")
        elif amount > 10_000_000:
            add_error("amount", "Amount exceeds MVP validation range", amount, "Confirm the value or split the record according to source-system rules.")

        currency = str(row.get("currency", "")).strip().upper()
        if not (len(currency) == 3 and currency.isalpha()):
            add_error("currency", "Currency must be a 3-letter code", currency, "Use ISO-4217 style codes such as USD, EUR, GBP.")

        direction = str(row.get("direction", "")).strip().lower()
        if direction not in {"debit", "credit"}:
            add_error("direction", "Direction must be debit or credit", direction, "Use 'debit' or 'credit'.")

        account_id = str(row.get("account_id", "")).strip()
        if account_id not in known_account_ids:
            add_error("account_id", "Unknown account_id violates referential integrity", account_id, "Add the account to the reference catalog or correct the account ID.")

        channel = str(row.get("channel", "")).strip().lower()
        if channel and channel not in CHANNELS:
            add_error("channel", "Channel is not in the reference catalog", channel, "Use one of: " + ", ".join(CHANNELS), severity="warning")

        cp_country = str(row.get("counterparty_country", "")).strip().upper()
        if cp_country and not (len(cp_country) == 2 and cp_country.isalpha()):
            add_error("counterparty_country", "Country should be a 2-letter code", cp_country, "Use ISO-3166 alpha-2 when available.", severity="warning")

        if not row_has_error:
            cleaned.append(
                {
                    "txn_id": txn_id,
                    "account_id": account_id,
                    "timestamp_utc": ts,
                    "amount": round(amount, 2),
                    "currency": currency,
                    "direction": direction,
                    "counterparty_id": str(row.get("counterparty_id", "")).strip(),
                    "counterparty_country": cp_country,
                    "channel": channel,
                    "device_id": str(row.get("device_id", "")).strip(),
                    "ip_hash": str(row.get("ip_hash", "")).strip(),
                    "note": str(row.get("note", "") or ""),
                    "source_system": str(row.get("source_system", "uploaded_file") or "uploaded_file"),
                    "is_fraud_synthetic": to_bool(row.get("is_fraud_synthetic", False)),
                    "label_source": str(row.get("label_source", "imported_label") or "imported_label"),
                    "scenario": str(row.get("scenario", "imported") or "imported"),
                    "consent_id": str(row.get("consent_id", "CONSENT-IMPORTED") or "CONSENT-IMPORTED"),
                    "source_type": "imported_file",
                }
            )

    valid_df = ensure_columns(pd.DataFrame(cleaned), CLEAN_TX_COLUMNS)
    errors_df = ensure_columns(pd.DataFrame(errors), ERROR_COLUMNS)
    stats = {
        "total_rows": len(df),
        "valid_rows": len(valid_df),
        "error_count": error_count,
        "warning_count": warning_count,
    }
    return valid_df, errors_df, stats


def commit_staged_import(db: Dict[str, Any], rules: Dict[str, Any], staged: Dict[str, Any], actor_role: str, actor_name: str) -> Dict[str, Any]:
    existing_ids = set(db["transactions"]["txn_id"].astype(str)) if not db["transactions"].empty else set()
    staged_df = staged["df"].copy()
    new_df = staged_df[~staged_df["txn_id"].astype(str).isin(existing_ids)]
    skipped = len(staged_df) - len(new_df)

    import_id = stable_id("IMP", staged.get("filename", "upload"), utc_now().isoformat())
    committed = 0

    old_tx = db["transactions"]
    old_features = db["features"]
    old_alerts = db["alerts"]
    old_engine = db.get("risk_engine")
    old_imports = db.get("imports")

    try:
        if not new_df.empty:
            new_df["source_type"] = "imported_file"
            db["transactions"] = ensure_columns(concat_df(db["transactions"], new_df), CLEAN_TX_COLUMNS)
            refresh_db(db, rules)
            committed = len(new_df)
            add_audit(db, actor_role, actor_name, "import_committed", "database", import_id, f"committed={committed}, skipped={skipped}")

        record = {
            "import_id": import_id,
            "filename": staged.get("filename", "upload"),
            "checksum": staged.get("checksum", ""),
            "imported_at": utc_now().isoformat(),
            "rows_total": staged.get("stats", {}).get("total_rows", len(staged_df)),
            "rows_valid": staged.get("stats", {}).get("valid_rows", len(staged_df)),
            "rows_committed": committed,
            "rows_skipped": skipped,
            "status": "committed" if committed > 0 else "skipped_existing",
            "notes": f"errors={staged.get('stats', {}).get('error_count', 0)}, warnings={staged.get('stats', {}).get('warning_count', 0)}",
        }
        db["imports"] = ensure_columns(concat_df(db.get("imports"), pd.DataFrame([record])), IMPORT_COLUMNS)
        return record
    except Exception:
        db["transactions"] = old_tx
        db["features"] = old_features
        db["alerts"] = old_alerts
        db["risk_engine"] = old_engine
        db["imports"] = old_imports
        raise


def set_alert_value(db: Dict[str, Any], alert_id: str, field: str, value: Any) -> None:
    if db["alerts"].empty:
        return
    mask = db["alerts"]["alert_id"].astype(str) == str(alert_id)
    if not mask.any():
        return
    db["alerts"].loc[mask, field] = value
    db["alerts"].loc[mask, "updated_at"] = utc_now().isoformat()


def alert_action(
    db: Dict[str, Any],
    alert_id: str,
    actor_role: str,
    actor_name: str,
    action: str,
    **kwargs: Any,
) -> None:
    if action == "assign":
        set_alert_value(db, alert_id, "assigned_to", actor_name)
        set_alert_value(db, alert_id, "status", "under_review")
    elif action == "start_review":
        set_alert_value(db, alert_id, "status", "under_review")
        set_alert_value(db, alert_id, "reviewer", actor_name)
    elif action == "clear":
        set_alert_value(db, alert_id, "status", "cleared")
        set_alert_value(db, alert_id, "decision", "cleared")
        set_alert_value(db, alert_id, "reviewer", actor_name)
    elif action == "escalate":
        set_alert_value(db, alert_id, "status", "escalated")
        set_alert_value(db, alert_id, "decision", "escalated")
        set_alert_value(db, alert_id, "reviewer", actor_name)
    elif action == "archive":
        set_alert_value(db, alert_id, "status", "archived")
        set_alert_value(db, alert_id, "decision", "archived")
    elif action == "override":
        set_alert_value(db, alert_id, "risk_score", float(kwargs.get("score", 0.0)))
        set_alert_value(db, alert_id, "override_reason", str(kwargs.get("reason", "")))
        set_alert_value(db, alert_id, "status", "under_review")
        set_alert_value(db, alert_id, "reviewer", actor_name)
    elif action == "feedback":
        set_alert_value(db, alert_id, "feedback", str(kwargs.get("text", "")))

    add_audit(db, actor_role, actor_name, f"alert_{action}", "alert", alert_id, json.dumps(kwargs, default=json_default))


def parse_explanation(alert_row: pd.Series) -> Dict[str, Any]:
    try:
        raw = alert_row.get("explanation", "{}")
        if isinstance(raw, dict):
            return raw
        return json.loads(str(raw))
    except Exception:
        return {"parse_error": "Explanation JSON could not be parsed."}


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    alert_rate = (tp + fp) / len(y_true) if len(y_true) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": far,
        "alert_rate": alert_rate,
    }


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    ranks = pd.Series(all_scores).rank(method="average").to_numpy()
    rank_sum_pos = float(np.sum(ranks[: len(pos)]))
    u = rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0
    denominator = len(pos) * len(neg)
    return float(u / denominator) if denominator > 0 else float("nan")


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    p = successes / trials
    denom = 1.0 + (z * z) / trials
    center = (p + (z * z) / (2.0 * trials)) / denom
    half = (z * math.sqrt((p * (1.0 - p)) / trials + (z * z) / (4.0 * trials * trials))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return float(lo), float(hi)


def run_evaluation(db: Dict[str, Any], threshold: float) -> Optional[Dict[str, Any]]:
    features = db.get("features", pd.DataFrame()).copy()
    if features.empty:
        return None

    features["timestamp_utc"] = pd.to_datetime(features["timestamp_utc"], utc=True, errors="coerce")
    features["risk_score"] = pd.to_numeric(features["risk_score"], errors="coerce").fillna(0.0)
    features["amount"] = pd.to_numeric(features["amount"], errors="coerce").fillna(0.0)
    features["y"] = features["is_fraud_synthetic"].apply(to_bool).astype(int)

    max_ts = features["timestamp_utc"].max()
    if pd.isna(max_ts):
        test = features.copy()
        cutoff = pd.NaT
    else:
        cutoff = max_ts - timedelta(days=7)
        test = features[features["timestamp_utc"] >= cutoff].copy()

    if test.empty:
        test = features.copy()

    if pd.isna(cutoff):
        train = features.iloc[:-1].copy() if len(features) > 1 else pd.DataFrame()
    else:
        train = features[features["timestamp_utc"] < cutoff].copy()

    y_test = test["y"].to_numpy()
    model_pred = (test["risk_score"] >= float(threshold)).astype(int).to_numpy()
    model_m = binary_metrics(y_test, model_pred)
    model_auc = auroc(y_test, test["risk_score"].to_numpy())

    if not train.empty and train["amount"].notna().any():
        p95 = float(train["amount"].quantile(0.95))
    else:
        p95 = float(features["amount"].quantile(0.95)) if features["amount"].notna().any() else 0.0

    baseline_pred = (test["amount"] > p95).astype(int).to_numpy()
    baseline_m = binary_metrics(y_test, baseline_pred)
    baseline_auc = auroc(y_test, test["amount"].to_numpy())

    prevalence = float(np.mean(y_test)) if len(y_test) > 0 else 0.0

    def fmt_interval(successes: int, trials: int) -> str:
        lo, hi = wilson_interval(successes, trials)
        if math.isnan(lo) or math.isnan(hi):
            return "n/a"
        return f"{lo:.3f}-{hi:.3f}"

    metrics_rows = [
        {
            "model": "Deterministic rule engine",
            "threshold": float(threshold),
            "tp": model_m["tp"],
            "fp": model_m["fp"],
            "fn": model_m["fn"],
            "tn": model_m["tn"],
            "precision": round(model_m["precision"], 4),
            "precision_ci": fmt_interval(model_m["tp"], model_m["tp"] + model_m["fp"]),
            "recall": round(model_m["recall"], 4),
            "recall_ci": fmt_interval(model_m["tp"], model_m["tp"] + model_m["fn"]),
            "f1": round(model_m["f1"], 4),
            "false_alarm_rate": round(model_m["false_alarm_rate"], 4),
            "alert_rate": round(model_m["alert_rate"], 4),
            "auroc": round(model_auc, 4) if not math.isnan(model_auc) else None,
        },
        {
            "model": "Simple amount baseline (>95th percentile)",
            "threshold": round(p95, 2),
            "tp": baseline_m["tp"],
            "fp": baseline_m["fp"],
            "fn": baseline_m["fn"],
            "tn": baseline_m["tn"],
            "precision": round(baseline_m["precision"], 4),
            "precision_ci": fmt_interval(baseline_m["tp"], baseline_m["tp"] + baseline_m["fp"]),
            "recall": round(baseline_m["recall"], 4),
            "recall_ci": fmt_interval(baseline_m["tp"], baseline_m["tp"] + baseline_m["fn"]),
            "f1": round(baseline_m["f1"], 4),
            "false_alarm_rate": round(baseline_m["false_alarm_rate"], 4),
            "alert_rate": round(baseline_m["alert_rate"], 4),
            "auroc": round(baseline_auc, 4) if not math.isnan(baseline_auc) else None,
        },
        {
            "model": "Random/prevalence reference",
            "threshold": None,
            "tp": None,
            "fp": None,
            "fn": None,
            "tn": None,
            "precision": round(prevalence, 4),
            "precision_ci": "n/a",
            "recall": round(prevalence, 4),
            "recall_ci": "n/a",
            "f1": round(prevalence, 4),
            "false_alarm_rate": round(prevalence, 4),
            "alert_rate": 1.0,
            "auroc": 0.5,
        },
    ]

    subgroup_rows: List[Dict[str, Any]] = []
    for col in ["account_segment", "channel", "counterparty_country"]:
        if col not in test.columns:
            continue
        for value, grp in test.groupby(col, dropna=False):
            if len(grp) < 5:
                continue
            y = grp["y"].to_numpy()
            pred = (grp["risk_score"] >= float(threshold)).astype(int).to_numpy()
            m = binary_metrics(y, pred)
            subgroup_rows.append(
                {
                    "group_by": col,
                    "group": str(value),
                    "n": int(len(grp)),
                    "positives": int(np.sum(y)),
                    "alerts": int(np.sum(pred)),
                    "tp": m["tp"],
                    "fp": m["fp"],
                    "fn": m["fn"],
                    "precision": round(m["precision"], 4),
                    "recall": round(m["recall"], 4),
                    "f1": round(m["f1"], 4),
                }
            )

    return {
        "summary": {
            "synthetic_only": True,
            "ruleset_version": RULESET_VERSION,
            "threshold": float(threshold),
            "test_rows": int(len(test)),
            "test_positives": int(np.sum(y_test)),
            "prevalence": round(prevalence, 4),
            "time_cutoff_utc": cutoff.isoformat() if pd.notna(cutoff) else None,
            "baseline_amount_p95_train": round(p95, 2),
            "limitation": "Metrics are computed on synthetic labeled data. They validate software plumbing and rule behavior, not real-world fraud performance.",
        },
        "metrics_rows": metrics_rows,
        "subgroup_rows": subgroup_rows,
        "test_df": test,
    }


def export_snapshot_zip(db: Dict[str, Any], rules: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ["accounts", "transactions", "balances", "alerts", "audit", "imports"]:
            df = db.get(name, pd.DataFrame())
            if df is None:
                df = pd.DataFrame()
            zf.writestr(f"{name}.csv", df.to_csv(index=False))
        meta = {
            "app_name": APP_NAME,
            "app_version": APP_VERSION,
            "ruleset_version": RULESET_VERSION,
            "rules": rules,
            "exported_at_utc": utc_now().isoformat(),
            "synthetic_notice": SYNTHETIC_NOTICE,
        }
        zf.writestr("meta.json", json.dumps(meta, indent=2, default=json_default))
    return buf.getvalue()


def load_snapshot_from_zip(zip_bytes: bytes, rules: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rules = rules or json.loads(json.dumps(DEFAULT_RULES))
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())

        def read_csv(name: str) -> pd.DataFrame:
            if name not in names:
                return pd.DataFrame()
            return pd.read_csv(zf.open(name))

        accounts = read_csv("accounts.csv")
        transactions = read_csv("transactions.csv")
        balances = read_csv("balances.csv")
        alerts = read_csv("alerts.csv")
        audit = read_csv("audit.csv")
        imports = read_csv("imports.csv")

    if accounts.empty or transactions.empty:
        raise ValueError("Snapshot must contain accounts.csv and transactions.csv.")

    transactions["timestamp_utc"] = pd.to_datetime(transactions["timestamp_utc"], utc=True, errors="coerce")
    balances["date"] = pd.to_datetime(balances["date"], utc=True, errors="coerce")

    db = {
        "accounts": accounts,
        "transactions": ensure_columns(transactions, CLEAN_TX_COLUMNS),
        "balances": balances,
        "alerts": ensure_columns(alerts, ALERT_COLUMNS),
        "audit": ensure_columns(audit, AUDIT_COLUMNS),
        "imports": ensure_columns(imports, IMPORT_COLUMNS),
        "features": empty_df(FEATURE_COLUMNS),
        "risk_engine": None,
        "demo_video_path": None,
    }
    refresh_db(db, rules)
    add_audit(db, "System", "restore", "snapshot_restored", "database", "restored", f"transactions={len(db['transactions'])}")
    return db, rules


def case_pack_markdown(alert_row: pd.Series, db: Dict[str, Any]) -> str:
    expl = parse_explanation(alert_row)
    txn_match = db["transactions"][db["transactions"]["txn_id"].astype(str) == str(alert_row["txn_id"])]
    txn = txn_match.iloc[0] if not txn_match.empty else pd.Series(dtype=object)
    audit_rows = db["audit"][db["audit"]["entity_id"].astype(str) == str(alert_row["alert_id"])] if not db["audit"].empty else pd.DataFrame()

    lines = [
        f"# Case pack: {alert_row['alert_id']}",
        "",
        f"- Generated at: {utc_now().isoformat()}",
        f"- Ruleset version: {alert_row.get('ruleset_version', RULESET_VERSION)}",
        f"- Status: {alert_row.get('status', '')}",
        f"- Risk score: {alert_row.get('risk_score', '')}",
        f"- Confidence: {alert_row.get('confidence', '')}",
        f"- Reason codes: {alert_row.get('reason_codes', '')}",
        f"- Data quality: {alert_row.get('data_quality', '')}",
        "",
        "## Transaction evidence",
        "",
        f"- Transaction ID: {alert_row.get('txn_id', '')}",
        f"- Account ID: {alert_row.get('account_id', '')}",
        f"- Timestamp UTC: {alert_row.get('txn_timestamp', '')}",
        f"- Amount: {alert_row.get('amount', '')} {alert_row.get('currency', '')}",
        f"- Direction: {txn.get('direction', '')}",
        f"- Counterparty: {txn.get('counterparty_id', '')} ({txn.get('counterparty_country', '')})",
        f"- Channel: {txn.get('channel', '')}",
        f"- Device: {txn.get('device_id', '')}",
        f"- IP hash: {txn.get('ip_hash', '')}",
        f"- Source system: {txn.get('source_system', '')}",
        f"- Source type: {alert_row.get('source_type', '')}",
        "",
        "## Explanation summary",
        "",
        "```json",
        json.dumps(expl, indent=2, default=json_default),
        "```",
        "",
        "## Audit history",
        "",
    ]

    if audit_rows.empty:
        lines.append("No audit events recorded for this alert.")
    else:
        for _, a in audit_rows.iterrows():
            lines.append(f"- {a['timestamp_utc']} | {a['actor_role']} / {a['actor_name']} | {a['action']} | {a['details']}")

    lines.extend(
        [
            "",
            "## Governance boundary",
            "",
            "- This case pack is decision support for authorized human review.",
            "- It does not constitute regulated financial advice or an autonomous decision.",
            "- Synthetic demo data must not be used to claim real-world customer impact.",
        ]
    )
    return "\n".join(lines)


def run_smoke_tests() -> pd.DataFrame:
    results: List[Dict[str, Any]] = []

    def record(name: str, fn: Any) -> None:
        try:
            msg = fn()
            results.append({"test": name, "passed": True, "message": msg or "OK"})
        except AssertionError as exc:
            results.append({"test": name, "passed": False, "message": f"Assertion failed: {exc}"})
        except Exception as exc:
            results.append({"test": name, "passed": False, "message": f"{type(exc).__name__}: {exc}"})

    def test_current_dataset() -> str:
        db = st.session_state.get("db")
        if not db:
            raise AssertionError("Session database is missing.")
        assert not db["transactions"].empty, "Transactions table is empty."
        assert not db["features"].empty, "Features table is empty."
        assert bool(db["features"]["risk_score"].between(0, 100).all()), "Risk scores are outside 0-100."
        return f"transactions={len(db['transactions'])}, features={len(db['features'])}"

    def test_hand_checkable_scoring() -> str:
        acc_df = pd.DataFrame(
            [
                {
                    "account_id": "ACC-TEST",
                    "currency": "USD",
                    "country": "US",
                    "home_device_id": "DEV-HOME",
                    "home_ip_hash": "IP-HOME",
                    "typical_txn_amount": 100.0,
                    "avg_daily_balance": 1000.0,
                    "segment": "Retail",
                }
            ]
        )
        bal_df = pd.DataFrame(
            [
                {
                    "account_id": "ACC-TEST",
                    "date": normalize_utc_timestamp(utc_now()).normalize(),
                    "balance": 1000.0,
                }
            ]
        )
        engine = FraudRiskEngine(acc_df, bal_df, DEFAULT_RULES)
        acc = acc_df.iloc[0]

        normal = make_txn_row(
            acc=acc,
            ts=utc_now() - timedelta(days=1),
            amount=100.0,
            direction="debit",
            counterparty_id="CP-TEST-1",
            counterparty_country="US",
            channel="online_banking",
            device_id="DEV-HOME",
            ip_hash="IP-HOME",
            is_fraud=False,
            label_source="test",
            scenario="normal",
            txn_id="TXN-TEST-NORMAL",
        )
        suspicious = make_txn_row(
            acc=acc,
            ts=utc_now(),
            amount=9500.0,
            direction="debit",
            counterparty_id="CP-TEST-NEW",
            counterparty_country="SG",
            channel="wire",
            device_id="DEV-NEW",
            ip_hash="IP-NEW",
            is_fraud=True,
            label_source="test",
            scenario="account_takeover",
            txn_id="TXN-TEST-SUSPICIOUS",
        )

        f_normal = engine.score_row(pd.Series(normal))
        f_susp = engine.score_row(pd.Series(suspicious))
        assert f_susp["risk_score"] > f_normal["risk_score"], (
            f"Suspicious score {f_susp['risk_score']} was not greater than normal score {f_normal['risk_score']}."
        )
        return f"normal={f_normal['risk_score']}, suspicious={f_susp['risk_score']}"

    def test_import_validation_rejects_bad_row() -> str:
        bad = pd.DataFrame(
            [
                {
                    "txn_id": "TXN-SMOKE-BAD",
                    "account_id": "ACC-TEST",
                    "timestamp_utc": "2026-01-01T00:00:00Z",
                    "amount": -10,
                    "currency": "USD",
                    "direction": "debit",
                    "counterparty_id": "CP-1",
                    "counterparty_country": "US",
                    "channel": "online_banking",
                    "device_id": "DEV-1",
                    "ip_hash": "IP-1",
                }
            ]
        )
        valid, errors, stats = validate_transactions_upload(bad, set(), {"ACC-TEST"})
        assert not errors.empty, "Validation did not produce errors for a negative amount."
        assert valid.empty, "Invalid row was incorrectly marked valid."
        return f"errors={len(errors)}"

    def test_alert_threshold_behavior() -> str:
        db = st.session_state.get("db")
        if not db or db["features"].empty:
            raise AssertionError("No features available for alert threshold test.")
        high = create_alerts_from_features(db["features"], 101)
        low = create_alerts_from_features(db["features"], 0)
        assert high.empty, "Threshold above 100 should produce no alerts."
        assert len(low) == len(db["features"]), "Threshold 0 should produce one alert per scored transaction."
        return f"alerts_at_zero={len(low)}"

    def test_snapshot_export() -> str:
        db = st.session_state.get("db")
        rules = st.session_state.get("rules", DEFAULT_RULES)
        if not db:
            raise AssertionError("Session database is missing.")
        blob = export_snapshot_zip(db, rules)
        assert len(blob) > 100, "Snapshot export is suspiciously small."
        return f"bytes={len(blob)}"

    record("Current synthetic dataset loads", test_current_dataset)
    record("Hand-checkable suspicious score exceeds normal score", test_hand_checkable_scoring)
    record("Import validation rejects malformed row", test_import_validation_rejects_bad_row)
    record("Alert threshold behavior is observable", test_alert_threshold_behavior)
    record("Snapshot export produces bytes", test_snapshot_export)

    return pd.DataFrame(results)


def render_sidebar(db: Dict[str, Any], rules: Dict[str, Any]) -> None:
    st.sidebar.title("Workbench controls")
    st.sidebar.caption(SYNTHETIC_NOTICE)

    roles = list(ROLE_PERMISSIONS.keys())
    current_role = st.session_state.get("role", "Administrator")
    if current_role not in roles:
        current_role = "Administrator"
    role = st.sidebar.selectbox("Acting role", roles, index=roles.index(current_role))
    st.session_state.role = role

    actor = st.sidebar.text_input("Investigator / operator name", value=st.session_state.get("actor_name", "Demo User"))
    st.session_state.actor_name = actor or "Demo User"

    st.sidebar.divider()
    st.sidebar.subheader("Demo and simulation")

    if can(role, "regenerate_video") and st.sidebar.button("Regenerate demo MP4", key="regen_video"):
        try:
            with st.spinner("Recording a short synthetic live demo video..."):
                path = generate_demo_video(db, rules, steps=30, fps=12)
            db["demo_video_path"] = path
            if path:
                st.sidebar.success(f"Video saved to: {path}")
            else:
                st.sidebar.warning("Video writer unavailable. Dashboard remains interactive.")
        except Exception as exc:
            st.sidebar.error(f"Video regeneration failed: {exc}")

    if can(role, "reset_demo") and st.sidebar.button("Reset synthetic demo", key="reset_demo"):
        try:
            with st.spinner("Rebuilding synthetic dataset and recording demo..."):
                new_db, new_rules = initialize_demo()
            st.session_state.db = new_db
            st.session_state.rules = new_rules
            st.session_state.staged_import = None
            st.session_state.import_preview = None
            st.session_state.import_checksum = None
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Reset failed: {exc}")
            return

    steps = st.sidebar.number_input("Live events to run", min_value=1, max_value=100, value=10, step=1, key="live_steps")
    if can(role, "run_simulation") and st.sidebar.button("Run live events", key="run_live"):
        rng = np.random.default_rng(int(time.time_ns()))
        try:
            with st.spinner(f"Running {int(steps)} live synthetic events..."):
                for _ in range(int(steps)):
                    simulate_one_event(db, rules, rng)
            st.sidebar.success(f"Added {int(steps)} live synthetic events.")
        except Exception as exc:
            st.sidebar.error(f"Simulation failed: {exc}")

    st.sidebar.divider()
    st.sidebar.subheader("Risk threshold")
    current_threshold = int(rules.get("alert_threshold", 55))
    if can(role, "change_rules"):
        new_threshold = st.sidebar.slider("Live alert threshold", 0, 100, current_threshold, key="threshold_slider")
        if new_threshold != current_threshold:
            rules["alert_threshold"] = new_threshold
            new_alerts = create_alerts_from_features(db.get("features", empty_df(FEATURE_COLUMNS)), new_threshold)
            db["alerts"] = merge_alerts(db.get("alerts"), new_alerts)
            add_audit(db, role, st.session_state.actor_name, "threshold_changed", "rules", "alert_threshold", f"{current_threshold}->{new_threshold}")
    else:
        st.sidebar.write(f"Alert threshold: **{current_threshold}**")
        st.sidebar.caption("Only Administrators can change thresholds in this MVP.")

    st.sidebar.divider()
    st.sidebar.subheader("Data import")
    st.sidebar.caption("Upload CSV or XLSX in the Import tab. This sidebar only shows status.")
    if st.session_state.get("staged_import"):
        st.sidebar.info(f"Staged import ready: {st.session_state['staged_import'].get('filename', 'upload')}")
    else:
        st.sidebar.write("No staged import.")


def render_header(db: Dict[str, Any], rules: Dict[str, Any]) -> None:
    st.title(APP_NAME)
    st.caption(f"Version {APP_VERSION} | Ruleset {RULESET_VERSION}")
    st.warning(SYNTHETIC_NOTICE, icon="⚠️")

    alerts = db.get("alerts", empty_df(ALERT_COLUMNS))
    features = db.get("features", empty_df(FEATURE_COLUMNS))
    transactions = db.get("transactions", empty_df(CLEAN_TX_COLUMNS))

    open_statuses = ["new", "under_review", "escalated"]
    open_count = int(len(alerts[alerts["status"].isin(open_statuses)])) if not alerts.empty else 0
    avg_score = float(features["risk_score"].mean()) if not features.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(transactions):,}")
    c2.metric("Open alerts", f"{open_count:,}")
    c3.metric("Average risk score", f"{avg_score:.1f}")
    c4.metric("Alert threshold", str(int(rules.get("alert_threshold", 55))))

    video_path = db.get("demo_video_path")
    if video_path and os.path.exists(video_path):
        st.subheader("Generated working demo video")
        if str(video_path).lower().endswith(".mp4"):
            st.video(video_path, format="video/mp4", loop=True, autoplay=False, muted=True)
        else:
            st.image(video_path, use_container_width=True)
        st.caption(f"Local file: `{video_path}`")
    else:
        st.info("Demo video file was not generated on this run. The interactive dashboard below still works.")


def render_live_tab(db: Dict[str, Any], rules: Dict[str, Any]) -> None:
    st.subheader("Live synthetic activity")

    features = db.get("features", empty_df(FEATURE_COLUMNS)).copy()
    alerts = db.get("alerts", empty_df(ALERT_COLUMNS)).copy()

    if features.empty:
        st.info("No scored records yet.")
        return

    features["timestamp_utc"] = pd.to_datetime(features["timestamp_utc"], utc=True, errors="coerce")
    recent = features.sort_values("timestamp_utc").tail(80)

    fig = px.line(
        recent,
        x="timestamp_utc",
        y="risk_score",
        color="is_fraud_synthetic",
        title="Recent risk scores by transaction",
        labels={"risk_score": "Risk score", "is_fraud_synthetic": "Synthetic label"},
    )
    fig.add_hline(y=float(rules.get("alert_threshold", 55)), line_dash="dash", line_color="red", annotation_text="Alert threshold")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Latest scored events")
        show_cols = ["timestamp_utc", "txn_id", "account_id", "amount", "currency", "risk_score", "reason_codes", "data_quality"]
        st.dataframe(recent[show_cols].sort_values("timestamp_utc", ascending=False).head(20), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Top open alerts")
        if alerts.empty:
            st.write("No alerts.")
        else:
            open_alerts = alerts[alerts["status"].isin(["new", "under_review", "escalated"])]
            top = open_alerts.sort_values("risk_score", ascending=False).head(10)
            st.dataframe(
                top[["alert_id", "account_id", "txn_timestamp", "amount", "risk_score", "status", "reason_codes"]],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("#### Actionable interpretation")
    st.write(
        "Use the score and reason codes to prioritize review. A high score is not proof of fraud; it is a transparent "
        "ranking signal built from deterministic features such as velocity, geography, device change, counterparty novelty, "
        "balance drain, and data quality."
    )


def render_alert_detail(alert_row: pd.Series, db: Dict[str, Any], role: str) -> None:
    alert_id = str(alert_row["alert_id"])
    expl = parse_explanation(alert_row)

    st.divider()
    st.subheader(f"Alert {alert_id}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk score", f"{to_float(alert_row.get('risk_score', 0.0)):.1f}")
    m2.metric("Confidence", f"{to_float(alert_row.get('confidence', 1.0)):.2f}")
    m3.metric("Status", str(alert_row.get("status", "new")))
    m4.metric("Source type", str(alert_row.get("source_type", "unknown")))

    st.markdown("##### Transaction evidence")
    txn_match = db["transactions"][db["transactions"]["txn_id"].astype(str) == str(alert_row["txn_id"])]
    if txn_match.empty:
        st.warning("Original transaction row not found. Evidence may have been restored from a partial snapshot.")
    else:
        txn = txn_match.iloc[0]
        ev = {
            "txn_id": txn.get("txn_id"),
            "account_id": txn.get("account_id"),
            "timestamp_utc": str(txn.get("timestamp_utc")),
            "amount": txn.get("amount"),
            "currency": txn.get("currency"),
            "direction": txn.get("direction"),
            "counterparty_id": txn.get("counterparty_id"),
            "counterparty_country": txn.get("counterparty_country"),
            "channel": txn.get("channel"),
            "device_id": txn.get("device_id"),
            "ip_hash": txn.get("ip_hash"),
            "source_system": txn.get("source_system"),
            "synthetic_label": to_bool(txn.get("is_fraud_synthetic", False)),
            "label_source": txn.get("label_source"),
        }
        st.json(ev)

    st.markdown("##### Reason code contributions")
    contributions = expl.get("contributions", [])
    if contributions:
        cdf = pd.DataFrame(contributions)
        st.dataframe(cdf, use_container_width=True, hide_index=True)
    else:
        st.write("No positive contributions recorded.")

    st.markdown("##### Data quality and warnings")
    st.write(f"- Data quality: **{alert_row.get('data_quality', 'ok')}**")
    warnings = expl.get("warnings", [])
    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.success("No data-quality warnings for this record.")

    st.markdown("##### Lineage")
    st.write(
        f"- Transaction ID: `{alert_row.get('txn_id', '')}`  "
        f"- Ruleset: `{alert_row.get('ruleset_version', RULESET_VERSION)}`  "
        f"- Source type: `{alert_row.get('source_type', '')}`  "
        f"- Source system: `{expl.get('transaction', {}).get('source_system', '')}`"
    )

    st.markdown("##### Full explanation JSON")
    with st.expander("Open explanation JSON"):
        st.json(expl)

    st.markdown("##### Audit history")
    audit = db.get("audit", empty_df(AUDIT_COLUMNS))
    if audit.empty:
        st.write("No audit events.")
    else:
        rel = audit[audit["entity_id"].astype(str) == alert_id].sort_values("timestamp_utc", ascending=False)
        if rel.empty:
            st.write("No audit events for this alert.")
        else:
            st.dataframe(rel[["timestamp_utc", "actor_role", "actor_name", "action", "details"]], use_container_width=True, hide_index=True)

    st.markdown("##### Investigator actions")
    actor = st.session_state.get("actor_name", "Demo User")
    c1, c2, c3, c4 = st.columns(4)

    if can(role, "assign") and c1.button("Assign to me", key=f"assign_{alert_id}"):
        alert_action(db, alert_id, role, actor, "assign")
        st.rerun()

    if can(role, "triage") and c2.button("Start review", key=f"review_{alert_id}"):
        alert_action(db, alert_id, role, actor, "start_review")
        st.rerun()

    if can(role, "approve") and c3.button("Clear", key=f"clear_{alert_id}"):
        alert_action(db, alert_id, role, actor, "clear")
        st.rerun()

    if can(role, "approve") and c4.button("Escalate", key=f"escalate_{alert_id}"):
        alert_action(db, alert_id, role, actor, "escalate")
        st.rerun()

    if can(role, "add_feedback"):
        feedback = st.text_area("Investigator feedback", value=str(alert_row.get("feedback", "") or ""), key=f"fb_{alert_id}")
        if st.button("Save feedback", key=f"save_fb_{alert_id}"):
            alert_action(db, alert_id, role, actor, "feedback", text=feedback)
            st.rerun()

    if can(role, "override"):
        with st.expander("Manual override with audit reason"):
            new_score = st.number_input(
                "Override score",
                min_value=0.0,
                max_value=100.0,
                value=to_float(alert_row.get("risk_score", 0.0), 0.0),
                step=1.0,
                key=f"ovr_score_{alert_id}",
            )
            reason = st.text_input("Override reason", key=f"ovr_reason_{alert_id}")
            if st.button("Apply override", key=f"ovr_apply_{alert_id}"):
                if not reason.strip():
                    st.error("Override requires a written reason.")
                else:
                    alert_action(db, alert_id, role, actor, "override", score=new_score, reason=reason)
                    st.rerun()

    if can(role, "archive"):
        if st.button("Archive case", key=f"archive_{alert_id}"):
            alert_action(db, alert_id, role, actor, "archive")
            st.rerun()


def render_alert_queue(db: Dict[str, Any], role: str) -> None:
    st.subheader("Alert queue")

    alerts = db.get("alerts", empty_df(ALERT_COLUMNS)).copy()
    if alerts.empty:
        st.info("No alerts have been generated yet. Run live events or lower the threshold.")
        return

    status_options = ["new", "under_review", "escalated", "cleared", "archived"]
    selected_status = st.multiselect("Status", status_options, default=["new", "under_review", "escalated"])
    min_score = st.slider("Minimum score", 0, 100, 0)
    account_search = st.text_input("Search account ID", "")

    filt = alerts.copy()
    if selected_status:
        filt = filt[filt["status"].isin(selected_status)]
    filt["risk_score"] = pd.to_numeric(filt["risk_score"], errors="coerce").fillna(0.0)
    filt = filt[filt["risk_score"] >= min_score]
    if account_search:
        filt = filt[filt["account_id"].astype(str).str.contains(account_search, case=False, na=False)]

    filt = filt.sort_values(["risk_score", "created_at"], ascending=[False, False])

    show_cols = ["alert_id", "account_id", "txn_timestamp", "amount", "currency", "risk_score", "status", "assigned_to", "reviewer", "reason_codes", "confidence", "data_quality"]
    st.dataframe(filt[show_cols], use_container_width=True, hide_index=True)

    if filt.empty:
        st.warning("No alerts match the current filters.")
        return

    alert_ids = filt["alert_id"].astype(str).tolist()
    default_index = 0
    selected_id = st.session_state.get("selected_alert_id")
    if selected_id in alert_ids:
        default_index = alert_ids.index(selected_id)

    if "queue_alert_select" in st.session_state and st.session_state["queue_alert_select"] not in alert_ids:
        st.session_state["queue_alert_select"] = alert_ids[0]

    chosen = st.selectbox("Select alert for detail and action", alert_ids, index=default_index, key="queue_alert_select")
    st.session_state.selected_alert_id = chosen

    row = alerts[alerts["alert_id"].astype(str) == chosen]
    if row.empty:
        st.error("Selected alert no longer exists.")
        return
    render_alert_detail(row.iloc[0], db, role)


def render_import_tab(db: Dict[str, Any], rules: Dict[str, Any], role: str) -> None:
    st.subheader("CSV / XLSX import, validation, and safe re-import")

    dc1, dc2 = st.columns(2)
    dc1.download_button(
        "Download valid sample CSV",
        data=sample_import_csv(db),
        file_name="fraud_import_sample_valid.csv",
        mime="text/csv",
    )
    dc2.download_button(
        "Download malformed sample CSV",
        data=malformed_sample_csv(),
        file_name="fraud_import_sample_malformed.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload transaction file", type=["csv", "xlsx"], key="tx_uploader")

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        checksum = hashlib.sha256(file_bytes).hexdigest()
        if st.session_state.get("import_checksum") != checksum:
            try:
                raw_df = read_uploaded_file(uploaded)
                existing_ids = set(db["transactions"]["txn_id"].astype(str)) if not db["transactions"].empty else set()
                known_ids = set(db["accounts"]["account_id"].astype(str)) if not db["accounts"].empty else set()
                valid_df, errors_df, stats = validate_transactions_upload(raw_df, existing_ids, known_ids)
                st.session_state.import_preview = {
                    "filename": uploaded.name,
                    "checksum": checksum,
                    "df": valid_df,
                    "errors": errors_df,
                    "stats": stats,
                }
                st.session_state.import_checksum = checksum
            except Exception as exc:
                st.session_state.import_preview = None
                st.session_state.import_checksum = checksum
                st.error(f"Could not read file: {exc}")

    preview = st.session_state.get("import_preview")
    if preview:
        st.markdown("#### Import preview")
        stats = preview["stats"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", stats["total_rows"])
        c2.metric("Valid rows", stats["valid_rows"])
        c3.metric("Errors", stats["error_count"])
        c4.metric("Warnings", stats["warning_count"])

        if not preview["df"].empty:
            st.dataframe(preview["df"].head(20), use_container_width=True, hide_index=True)
        else:
            st.warning("No valid rows were produced by validation.")

        if not preview["errors"].empty:
            st.dataframe(preview["errors"], use_container_width=True, hide_index=True)
            st.download_button(
                "Download row-level error report",
                data=preview["errors"].to_csv(index=False).encode("utf-8"),
                file_name="import_error_report.csv",
                mime="text/csv",
            )

        if can(role, "import"):
            if st.button("Stage valid rows for commit"):
                st.session_state.staged_import = preview
                st.success("Rows staged. Committing will skip existing transaction IDs and never overwrite validated source data.")
        else:
            st.info("Your role does not permit importing data.")

    staged = st.session_state.get("staged_import")
    if staged:
        st.markdown("#### Staged import")
        st.write(f"File: `{staged.get('filename', '')}`")
        st.write(f"Valid rows: `{len(staged.get('df', []))}`")
        sc1, sc2 = st.columns(2)
        if can(role, "import") and sc1.button("Commit staged import"):
            try:
                record = commit_staged_import(db, rules, staged, role, st.session_state.get("actor_name", "Demo User"))
                st.session_state.staged_import = None
                st.session_state.import_preview = None
                st.session_state.import_checksum = None
                st.success(f"Committed {record['rows_committed']} rows, skipped {record['rows_skipped']} existing rows.")
            except Exception as exc:
                st.error(f"Import commit failed safely without corrupting existing records: {exc}")
        if sc2.button("Discard staged import"):
            st.session_state.staged_import = None
            st.info("Staged import discarded.")

    imports = db.get("imports", empty_df(IMPORT_COLUMNS))
    if not imports.empty:
        st.markdown("#### Import history")
        st.dataframe(imports.sort_values("imported_at", ascending=False), use_container_width=True, hide_index=True)


def render_analytics_tab(db: Dict[str, Any], rules: Dict[str, Any]) -> None:
    st.subheader("Analytics, baseline evaluation, and limitations")

    features = db.get("features", empty_df(FEATURE_COLUMNS)).copy()
    if features.empty:
        st.info("No scored features available.")
        return

    features["timestamp_utc"] = pd.to_datetime(features["timestamp_utc"], utc=True, errors="coerce")
    features["risk_score"] = pd.to_numeric(features["risk_score"], errors="coerce").fillna(0.0)
    features["is_fraud_synthetic"] = features["is_fraud_synthetic"].apply(to_bool)

    fig1 = px.histogram(
        features,
        x="risk_score",
        color="is_fraud_synthetic",
        nbins=30,
        title="Synthetic label distribution by risk score",
        labels={"risk_score": "Risk score", "is_fraud_synthetic": "Synthetic fraud label"},
    )
    fig1.add_vline(x=float(rules.get("alert_threshold", 55)), line_dash="dash", line_color="red")
    st.plotly_chart(fig1, use_container_width=True)

    reason_list: List[str] = []
    for rc in features["reason_codes"].dropna():
        reason_list.extend(str(rc).split(", "))
    if reason_list:
        vc = pd.Series(reason_list).value_counts().head(12).reset_index()
        vc.columns = ["reason_code", "count"]
        fig2 = px.bar(vc, x="count", y="reason_code", orientation="h", title="Reason code frequency")
        st.plotly_chart(fig2, use_container_width=True)

    daily = features.set_index("timestamp_utc").resample("D").agg(
        transactions=("txn_id", "count"),
        avg_score=("risk_score", "mean"),
        synthetic_positives=("is_fraud_synthetic", "sum"),
    ).reset_index()
    fig3 = sp.make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Scatter(x=daily["timestamp_utc"], y=daily["transactions"], name="Transactions", mode="lines+markers"))
    fig3.add_trace(go.Scatter(x=daily["timestamp_utc"], y=daily["avg_score"], name="Average score", mode="lines+markers"), secondary_y=True)
    fig3.update_layout(title="Daily transaction volume and average risk score")
    fig3.update_yaxes(title_text="Count", secondary_y=False)
    fig3.update_yaxes(title_text="Score", secondary_y=True)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### Reproducible evaluation on held-out time window")
    eval_result = run_evaluation(db, float(rules.get("alert_threshold", 55)))
    if eval_result is None:
        st.info("Evaluation requires scored features.")
        return

    st.json(eval_result["summary"])
    metrics_df = pd.DataFrame(eval_result["metrics_rows"])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    if eval_result["subgroup_rows"]:
        st.markdown("#### Subgroup performance")
        st.dataframe(pd.DataFrame(eval_result["subgroup_rows"]), use_container_width=True, hide_index=True)
    else:
        st.write("No subgroups met the minimum size threshold.")

    with st.expander("Interpretation and limits"):
        st.markdown(
            """
            - These metrics are computed on **synthetic** labels created to exercise the software.
            - They do not establish real-world fraud detection performance.
            - The deterministic engine is intentionally interpretable: each score is a weighted sum of clipped signals.
            - Ranking quality, precision/recall, false-alarm rate, and subgroup tables should be reviewed by qualified
              domain practitioners before any production claim.
            - Machine learning is not required for this MVP. If ML is added later, it must be compared to this deterministic
              baseline and evaluated on permissioned, time-split, independently reviewed data.
            """
        )


def render_reports_tab(db: Dict[str, Any], rules: Dict[str, Any]) -> None:
    st.subheader("Exports, case packs, backup, and restore")

    c1, c2, c3, c4 = st.columns(4)
    alerts = db.get("alerts", empty_df(ALERT_COLUMNS))
    features = db.get("features", empty_df(FEATURE_COLUMNS))
    audit = db.get("audit", empty_df(AUDIT_COLUMNS))

    c1.download_button(
        "Download alerts CSV",
        data=alerts.to_csv(index=False).encode("utf-8") if not alerts.empty else b"",
        file_name="fraud_alerts.csv",
        mime="text/csv",
        disabled=alerts.empty,
    )
    c2.download_button(
        "Download scored features CSV",
        data=features.to_csv(index=False).encode("utf-8") if not features.empty else b"",
        file_name="fraud_features.csv",
        mime="text/csv",
        disabled=features.empty,
    )
    c3.download_button(
        "Download audit CSV",
        data=audit.to_csv(index=False).encode("utf-8") if not audit.empty else b"",
        file_name="fraud_audit.csv",
        mime="text/csv",
        disabled=audit.empty,
    )

    eval_result = run_evaluation(db, float(rules.get("alert_threshold", 55)))
    eval_json = json.dumps(eval_result["summary"], indent=2, default=json_default) if eval_result else "{}"
    c4.download_button(
        "Download evaluation JSON",
        data=eval_json.encode("utf-8"),
        file_name="fraud_evaluation_summary.json",
        mime="application/json",
    )

    st.download_button(
        "Download full snapshot ZIP",
        data=export_snapshot_zip(db, rules),
        file_name="fraud_workbench_snapshot.zip",
        mime="application/zip",
    )

    st.markdown("#### Case pack for a selected alert")
    if alerts.empty:
        st.info("No alerts available for case-pack export.")
    else:
        alert_ids = alerts["alert_id"].astype(str).tolist()
        if "case_pack_alert" in st.session_state and st.session_state["case_pack_alert"] not in alert_ids:
            st.session_state["case_pack_alert"] = alert_ids[0]
        chosen = st.selectbox("Alert ID", alert_ids, key="case_pack_alert")
        row = alerts[alerts["alert_id"].astype(str) == chosen].iloc[0]
        md = case_pack_markdown(row, db)
        st.download_button(
            "Download case pack Markdown",
            data=md.encode("utf-8"),
            file_name=f"case_pack_{chosen}.md",
            mime="text/markdown",
        )
        with st.expander("Preview case pack"):
            st.markdown(md)

    st.markdown("#### Restore snapshot")
    st.caption("Restoring replaces the current in-memory demo database with the snapshot tables and rebuilds features.")
    restore_file = st.file_uploader("Upload snapshot ZIP", type=["zip"], key="restore_zip")
    confirm = st.checkbox("I understand this replaces the current demo data", key="confirm_restore")
    role = st.session_state.get("role", "Administrator")
    if restore_file is not None and confirm:
        if can(role, "restore_snapshot"):
            if st.button("Restore snapshot"):
                try:
                    new_db, new_rules = load_snapshot_from_zip(restore_file.getvalue(), rules)
                    st.session_state.db = new_db
                    st.session_state.rules = new_rules
                    st.session_state.staged_import = None
                    st.session_state.import_preview = None
                    st.session_state.import_checksum = None
                    st.success("Snapshot restored.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Restore failed: {exc}")
        else:
            st.warning("Only Administrators can restore snapshots in this MVP.")


def render_governance_tab(db: Dict[str, Any], rules: Dict[str, Any]) -> None:
    st.subheader("Governance, roles, data dictionary, risks, and tests")

    with st.expander("Product brief and workflow", expanded=False):
        st.markdown(PRODUCT_BRIEF)
        st.markdown(WORKFLOW_TEXT)

    st.markdown("#### Role permission matrix")
    perm_rows = []
    all_actions = sorted({a for perms in ROLE_PERMISSIONS.values() for a in perms})
    for role, perms in ROLE_PERMISSIONS.items():
        row = {"Role": role}
        for action in all_actions:
            row[action] = "Yes" if action in perms else "No"
        perm_rows.append(row)
    st.dataframe(pd.DataFrame(perm_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Prioritized backlog")
    st.dataframe(pd.DataFrame(BACKLOG_ROWS), use_container_width=True, hide_index=True)

    st.markdown("#### Measurable acceptance checklist")
    st.dataframe(pd.DataFrame(ACCEPTANCE_CHECKLIST_ROWS), use_container_width=True, hide_index=True)

    st.markdown("#### Risk register")
    st.dataframe(pd.DataFrame(RISK_REGISTER_ROWS), use_container_width=True, hide_index=True)

    st.markdown("#### Data dictionary")
    st.dataframe(pd.DataFrame(DATA_DICTIONARY_ROWS), use_container_width=True, hide_index=True)

    st.markdown("#### Smoke tests")
    st.caption("These tests verify software behavior on synthetic data. They do not validate real-world domain performance.")
    if st.button("Run smoke tests", key="run_smoke"):
        with st.spinner("Running smoke tests..."):
            st.session_state.smoke_results = run_smoke_tests()
    results = st.session_state.get("smoke_results")
    if results is not None:
        st.dataframe(results, use_container_width=True, hide_index=True)
        passed = int(results["passed"].sum())
        total = int(len(results))
        if passed == total:
            st.success(f"All smoke tests passed: {passed}/{total}")
        else:
            st.error(f"Smoke test failures: {total - passed}/{total}")

    st.markdown("#### Audit log")
    audit = db.get("audit", empty_df(AUDIT_COLUMNS))
    if audit.empty:
        st.write("No audit events.")
    else:
        st.dataframe(audit.sort_values("timestamp_utc", ascending=False).head(200), use_container_width=True, hide_index=True)

    st.markdown("#### Human approval boundaries")
    st.markdown(
        """
        - The system prioritizes and explains; authorized humans decide.
        - Overrides require a written reason and create an audit event.
        - No autonomous lending, payment blocking, or regulated financial advice is performed.
        - Production deployment requires authenticated roles, server-side authorization, HTTPS, secrets management,
          retention policy, backup/restore testing, and domain review by qualified practitioners.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "db" not in st.session_state:
        prog = st.progress(0.0, text="Starting synthetic fraud analytics demo...")
        try:
            def cb(value: float, message: str) -> None:
                prog.progress(min(1.0, max(0.0, float(value))), text=message)

            db, rules = initialize_demo(cb)
            st.session_state.db = db
            st.session_state.rules = rules
            st.session_state.role = "Administrator"
            st.session_state.actor_name = "Demo User"
            st.session_state.selected_alert_id = None
            st.session_state.staged_import = None
            st.session_state.import_preview = None
            st.session_state.import_checksum = None
        except Exception as exc:
            prog.empty()
            st.error(f"Initialization failed: {exc}")
            st.code(traceback.format_exc())
            st.stop()
        prog.empty()

    db = st.session_state.db
    rules = st.session_state.rules
    role = st.session_state.get("role", "Administrator")

    render_sidebar(db, rules)
    render_header(db, rules)

    tabs = st.tabs(
        [
            "Live workbench",
            "Alert queue",
            "Import & data quality",
            "Analytics & evaluation",
            "Reports & export",
            "Governance",
        ]
    )

    with tabs[0]:
        render_live_tab(db, rules)
    with tabs[1]:
        render_alert_queue(db, role)
    with tabs[2]:
        render_import_tab(db, rules, role)
    with tabs[3]:
        render_analytics_tab(db, rules)
    with tabs[4]:
        render_reports_tab(db, rules)
    with tabs[5]:
        render_governance_tab(db, rules)


if __name__ == "__main__":
    main()