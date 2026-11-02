"""Reproducible dataset generator for ARDA Reliability Service synthetic reference examples.

Domain: Financial Transaction Dispute & Refund Adjudication.
Agent: Transaction Dispute Investigator (agent_id: 00000000-0000-0000-0000-000000000002)
Tenant: Acme Fintech / ARDA Synthetic Tenant (tenant_id: 00000000-0000-0000-0000-000000000001)
Total Examples: Exactly 100 (50 APPROVE, 50 REJECT)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TENANT_ID = "00000000-0000-0000-0000-000000000001"
AGENT_ID = "00000000-0000-0000-0000-000000000002"
SOURCE = "synthetic"
STATUS = "active"


def build_raw_template(
    amount: float,
    tenure: int,
    prior_disputes: int,
    delay_days: int,
    merchant: str,
    category: str,
    evidence_summary: str,
) -> str:
    """Standardized raw-value description template for embedding experiments."""
    return (
        f"Transaction amount is ${amount:.2f} at {merchant} ({category}). "
        f"Customer tenure is {tenure} months with {prior_disputes} prior disputes. "
        f"Dispute initiated {delay_days} days following transaction settlement. "
        f"Evidence: {evidence_summary}"
    )


def build_ratio_template(
    amount: float,
    tenure: int,
    prior_disputes: int,
    delay_days: int,
    merchant: str,
    category: str,
    evidence_summary: str,
) -> str:
    """Standardized ratio-verbalized description template for embedding experiments."""
    dispute_ratio = (prior_disputes / max(1, tenure * 3)) * 100.0
    statutory_pct = (delay_days / 60.0) * 100.0
    return (
        f"Claim on {merchant} ({category}) involves a ${amount:.2f} charge. "
        f"Customer history reflects a {dispute_ratio:.1f}% lifetime dispute rate over {tenure} months. "
        f"Dispute filing consumed {statutory_pct:.1f}% of the allowable 60-day submission timeframe. "
        f"Evidence signals: {evidence_summary}"
    )


def compute_ratio_features(
    amount: float,
    tenure: int,
    prior_disputes: int,
    delay_days: int,
) -> dict[str, float]:
    """Calculate normalized ratios for experimental feature comparison."""
    lifetime_tx_estimate = max(1, tenure * 4)
    return {
        "dispute_to_lifetime_tx_ratio": round(prior_disputes / lifetime_tx_estimate, 4),
        "dispute_filing_to_statutory_window_ratio": round(delay_days / 60.0, 4),
        "amount_to_median_transaction_ratio": round(amount / 125.0, 4),
        "tenure_years": round(tenure / 12.0, 2),
    }


def get_raw_seed_data() -> list[dict[str, Any]]:
    """Return raw data definitions for all 100 synthetic reference examples."""
    cases: list[dict[str, Any]] = [
        # =========================================================================
        # 1. STRAIGHTFORWARD CASES (20 examples: 10 APPROVE, 10 REJECT)
        # =========================================================================
        # --- Straightforward APPROVE (10) ---
        {
            "id_num": 1,
            "case_type": "straightforward",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 249.99,
            "merchant": "Apex Electronics Supply",
            "merchant_category": "electronics",
            "tenure": 36,
            "prior_disputes": 0,
            "delay_days": 18,
            "evidence_signals": [
                "carrier_tracking_lost_in_transit",
                "merchant_unresponsive_14d",
            ],
            "case_description": (
                "Customer placed an online order for an audio amplifier totaling $249.99 with Apex Electronics Supply. "
                "The carrier tracking status indicates the package was lost in transit at a regional distribution facility "
                "over three weeks ago. The cardholder contacted merchant support three times requesting a replacement or refund, "
                "but received no reply after fourteen calendar days. The customer has a 36-month account tenure with zero past chargebacks."
            ),
            "evidence_summary": "Carrier confirms shipment lost in transit; merchant non-responsive for over 14 days.",
            "rationale": (
                "Adjudication policy mandates dispute approval when carrier tracking proves the package was lost in transit "
                "and the merchant fails to provide fulfillment, replacement, or communication within the required 14-day window."
            ),
        },
        {
            "id_num": 2,
            "case_type": "straightforward",
            "category": "duplicate_billing",
            "label": "APPROVE",
            "amount": 89.50,
            "merchant": "Nordic Book Haven",
            "merchant_category": "retail",
            "tenure": 24,
            "prior_disputes": 0,
            "delay_days": 4,
            "evidence_signals": [
                "identical_auth_code_timestamp",
                "single_order_record",
            ],
            "case_description": (
                "Cardholder was billed twice for $89.50 within five seconds at Nordic Book Haven's online checkout. "
                "Both settled records reflect the exact same authorization reference code, terminal ID, and shopping basket items. "
                "The merchant's store order database records only a single order number and single shipment dispatch for this customer."
            ),
            "evidence_summary": "Identical authorization code and timestamp within 5 seconds for a single order record.",
            "rationale": (
                "Clear technical duplicate billing confirmed by twin transactions with identical authorization credentials "
                "and only a single corresponding fulfillment record."
            ),
        },
        {
            "id_num": 3,
            "case_type": "straightforward",
            "category": "unauthorized_transaction",
            "label": "APPROVE",
            "amount": 620.00,
            "merchant": "Global Gaming Emporium",
            "merchant_category": "digital_goods",
            "tenure": 48,
            "prior_disputes": 0,
            "delay_days": 2,
            "evidence_signals": ["foreign_ip_anomaly", "concurrent_local_emv_swipe"],
            "case_description": (
                "A digital gaming gift card transaction of $620.00 was authorized at Global Gaming Emporium from an IP address "
                "originating in Eastern Europe with no 3D Secure verification. Simultaneously, the cardholder executed a verified "
                "chip-and-PIN grocery purchase in Denver, Colorado. The customer reported the incident within 48 hours of posting."
            ),
            "evidence_summary": "Simultaneous physical EMV card swipe in Colorado contradicts unauthenticated foreign IP charge.",
            "rationale": (
                "Definitive physical impossibility proven by concurrent verified local EMV transaction, confirming fraudulent "
                "card-not-present credential theft without cardholder participation."
            ),
        },
        {
            "id_num": 4,
            "case_type": "straightforward",
            "category": "credit_not_processed",
            "label": "APPROVE",
            "amount": 175.25,
            "merchant": "Urban Runner Apparel",
            "merchant_category": "apparel",
            "tenure": 18,
            "prior_disputes": 1,
            "delay_days": 22,
            "evidence_signals": [
                "merchant_warehouse_return_receipt",
                "refund_window_expired",
            ],
            "case_description": (
                "Customer returned two pairs of unworn running shoes totaling $175.25 to Urban Runner Apparel using the merchant's "
                "prepaid return shipping label. Carrier tracking documents delivery to the merchant's Ohio returns warehouse on October 5th, "
                "and the merchant sent an automated return acknowledgment email. However, no refund credit was posted within the merchant's "
                "stated 10-business-day refund timeline, and thirty calendar days have elapsed."
            ),
            "evidence_summary": "Confirmed delivery of returned merchandise to warehouse; merchant failed to credit within SLA.",
            "rationale": (
                "Customer fulfilled return obligations with carrier delivery proof and merchant acknowledgment; merchant failed to "
                "issue credit within statutory and published return refund timelines."
            ),
        },
        {
            "id_num": 5,
            "case_type": "straightforward",
            "category": "subscription_cancellation",
            "label": "APPROVE",
            "amount": 49.99,
            "merchant": "CloudVault Backup",
            "merchant_category": "saas",
            "tenure": 30,
            "prior_disputes": 0,
            "delay_days": 8,
            "evidence_signals": [
                "cancellation_confirmation_timestamped",
                "subsequent_unauthorized_charge",
            ],
            "case_description": (
                "Customer was charged a recurring monthly fee of $49.99 by CloudVault Backup on November 1st. "
                "The customer submitted a timestamped cancellation confirmation email from CloudVault dated October 18th stating the "
                "subscription was terminated and no further charges would occur. System logs demonstrate zero data storage or login "
                "activity following the October 18th cancellation."
            ),
            "evidence_summary": "Documented cancellation confirmation email dated 13 days prior to recurring charge; zero post-cancellation usage.",
            "rationale": (
                "Merchant billed account after issuing formal cancellation acknowledgment; customer maintains clear proof of timely "
                "termination and zero post-cancellation service utilization."
            ),
        },
        {
            "id_num": 6,
            "case_type": "straightforward",
            "category": "defective_or_not_as_described",
            "label": "APPROVE",
            "amount": 850.00,
            "merchant": "Vintage Horology Ltd",
            "merchant_category": "luxury_goods",
            "tenure": 12,
            "prior_disputes": 0,
            "delay_days": 15,
            "evidence_signals": ["certified_appraisal_counterfeit", "merchant_refusal"],
            "case_description": (
                "Customer purchased an advertised authentic 1970s Swiss mechanical watch for $850.00 from Vintage Horology Ltd. "
                "Upon delivery, the customer had the timepiece inspected by an authorized Swiss brand service center, which produced "
                "a formal appraisal certifying the movement is a modern counterfeit replica. The merchant rejected the customer's return "
                "request citing their 'all vintage sales final' policy."
            ),
            "evidence_summary": "Authorized service center certified timepiece as counterfeit; merchant refused return.",
            "rationale": (
                "Sale of counterfeit goods violates payment network rules and consumer protection statutes; strict 'final sale' clauses "
                "are void when merchandise is certified counterfeit."
            ),
        },
        {
            "id_num": 7,
            "case_type": "straightforward",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 420.00,
            "merchant": "Skyline Express Airways",
            "merchant_category": "travel",
            "tenure": 60,
            "prior_disputes": 0,
            "delay_days": 10,
            "evidence_signals": [
                "airline_cancellation_technical",
                "refusal_of_mandated_cash_refund",
            ],
            "case_description": (
                "Cardholder booked a $420.00 domestic flight ticket with Skyline Express Airways. The airline canceled the flight "
                "six hours prior to departure due to crew scheduling difficulties and did not rebook the passenger on an alternate flight. "
                "The airline issued an expiring travel voucher rather than the full cash refund requested by the customer in accordance "
                "with Department of Transportation consumer guidelines."
            ),
            "evidence_summary": "Carrier canceled flight due to internal operations and improperly withheld cash refund.",
            "rationale": (
                "Regulatory rules and network dispute guidelines entitle consumers to full cash reversals when transportation carriers "
                "cancel service and fail to accommodate or refund."
            ),
        },
        {
            "id_num": 8,
            "case_type": "straightforward",
            "category": "defective_or_not_as_described",
            "label": "APPROVE",
            "amount": 135.00,
            "merchant": "Precision Optics Direct",
            "merchant_category": "medical_supplies",
            "tenure": 28,
            "prior_disputes": 0,
            "delay_days": 12,
            "evidence_signals": [
                "prescription_parameter_mismatch",
                "return_shipping_carrier_proof",
            ],
            "case_description": (
                "Customer ordered custom prescription reading glasses for $135.00 from Precision Optics Direct, uploading a doctor's "
                "prescription specifying +2.25 diopters. The delivered glasses were manufactured at +1.25 diopters, verified by the "
                "prescribing optometrist's lensometer measurement report. The customer shipped the glasses back to the merchant with "
                "signature tracking, but the merchant did not issue a refund or corrected pair."
            ),
            "evidence_summary": "Optometrist report confirms incorrect lens prescription; merchant received return shipment without refunding.",
            "rationale": (
                "Objective evidence confirms goods delivered did not match agreed medical specifications, and customer documented return "
                "of merchandise to merchant."
            ),
        },
        {
            "id_num": 9,
            "case_type": "straightforward",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 780.00,
            "merchant": "Artisan Oak Woodworks",
            "merchant_category": "furniture",
            "tenure": 40,
            "prior_disputes": 0,
            "delay_days": 45,
            "evidence_signals": ["merchant_bankruptcy_notice", "zero_fulfillment"],
            "case_description": (
                "Customer paid a 50% deposit of $780.00 for custom dining chairs from Artisan Oak Woodworks with delivery scheduled for "
                "within 30 days. Prior to delivery, the merchant ceased business operations, shuttered its workshop, and filed for Chapter 7 "
                "bankruptcy protection. The customer received no merchandise, and bankruptcy court notices verify liquidation of the entity."
            ),
            "evidence_summary": "Merchant ceased operations and filed bankruptcy before fulfilling custom furniture order.",
            "rationale": (
                "Merchant non-fulfillment caused by complete cessation of business and liquidation warrants immediate dispute approval "
                "for advance deposits."
            ),
        },
        {
            "id_num": 10,
            "case_type": "straightforward",
            "category": "unauthorized_transaction",
            "label": "APPROVE",
            "amount": 310.00,
            "merchant": "QuickFuel Automated Dispenser",
            "merchant_category": "fuel",
            "tenure": 50,
            "prior_disputes": 0,
            "delay_days": 3,
            "evidence_signals": [
                "gas_pump_skimmer_compromise",
                "issuer_security_alert",
            ],
            "case_description": (
                "Cardholder disputes four consecutive fuel dispenser charges totaling $310.00 at QuickFuel Automated Dispenser in Dallas, "
                "Texas. Law enforcement and the payment network published a fraud bulletin identifying that physical skimmers were found "
                "on those exact pump terminals during the dispute period. The customer's registered billing address and vehicle registration "
                "are located in Portland, Oregon, with verified local tap-to-pay usage on the same afternoon."
            ),
            "evidence_summary": "Published skimming compromise on pump terminals; cardholder verified present in Oregon.",
            "rationale": (
                "Independent confirmation of terminal skimming compromise combined with conclusive geographical alibi warrants full dispute approval."
            ),
        },
        # --- Straightforward REJECT (10) ---
        {
            "id_num": 11,
            "case_type": "straightforward",
            "category": "goods_not_received",
            "label": "REJECT",
            "amount": 329.00,
            "merchant": "Gadget Hub Online",
            "merchant_category": "electronics",
            "tenure": 14,
            "prior_disputes": 2,
            "delay_days": 16,
            "evidence_signals": [
                "carrier_signed_delivery_gps",
                "household_signature_match",
            ],
            "case_description": (
                "Customer claims a $329.00 noise-cancelling headphone order from Gadget Hub Online was never received. "
                "Merchant provided UPS carrier delivery documentation showing delivery to the customer's verified billing address, "
                "complete with driver GPS coordinates within 2 meters of the front porch and a signature signed with the cardholder's "
                "last name."
            ),
            "evidence_summary": "Carrier confirms signed delivery matching cardholder surname with GPS coordinates at billing address.",
            "rationale": (
                "Signed carrier delivery confirmation with GPS coordinates matching the cardholder's residence constitutes conclusive "
                "proof of fulfillment under network dispute standards."
            ),
        },
        {
            "id_num": 12,
            "case_type": "straightforward",
            "category": "unauthorized_transaction",
            "label": "REJECT",
            "amount": 500.00,
            "merchant": "First National ATM Network",
            "merchant_category": "atm_cash",
            "tenure": 32,
            "prior_disputes": 1,
            "delay_days": 5,
            "evidence_signals": ["emv_chip_and_pin_verified", "zero_pin_tries_failed"],
            "case_description": (
                "Cardholder claims an unauthorized $500.00 cash withdrawal occurred at an ATM terminal in their neighborhood. "
                "Terminal audit logs confirm the transaction was authenticated using the physical EMV chip card and verified offline PIN "
                "on the first entry attempt, with zero prior incorrect attempts. The card remained in cardholder possession before and after."
            ),
            "evidence_summary": "Physical EMV chip transaction authenticated by correct cardholder PIN on first entry.",
            "rationale": (
                "Successful chip-and-PIN authentication with zero PIN entry failures requires possession of both the physical card and confidential "
                "PIN, rebutting unauthorized claim in the absence of documented coercion."
            ),
        },
        {
            "id_num": 13,
            "case_type": "straightforward",
            "category": "goods_not_received",
            "label": "REJECT",
            "amount": 145.00,
            "merchant": "StreamFlix Media",
            "merchant_category": "digital_goods",
            "tenure": 20,
            "prior_disputes": 0,
            "delay_days": 14,
            "evidence_signals": ["digital_key_activated", "ip_matching_cardholder"],
            "case_description": (
                "Customer claims non-receipt of a $145.00 annual digital video streaming redemption voucher purchased through StreamFlix Media. "
                "Merchant audit telemetry indicates the digital redemption key was sent to the customer's account email address and was "
                "redeemed two hours later from the customer's known home IP address and registered smart TV device."
            ),
            "evidence_summary": "Digital license key redeemed from cardholder home IP address on registered device.",
            "rationale": (
                "Documented redemption of digital license key from verified cardholder IP and known device provides clear evidence of fulfillment."
            ),
        },
        {
            "id_num": 14,
            "case_type": "straightforward",
            "category": "unauthorized_transaction",
            "label": "REJECT",
            "amount": 85.00,
            "merchant": "Prime Ride Dispatch",
            "merchant_category": "transportation",
            "tenure": 26,
            "prior_disputes": 1,
            "delay_days": 10,
            "evidence_signals": ["trip_gps_matches_home_work", "cardholder_phone_gps"],
            "case_description": (
                "Customer claims an $85.00 ride-hailing charge from Prime Ride Dispatch was unauthorized. Merchant GPS telemetry confirms "
                "the ride pickup occurred at the cardholder's residential address and dropped off at their confirmed corporate office. "
                "The ride was ordered from the customer's registered smartphone app using Face ID biometrics."
            ),
            "evidence_summary": "Ride initiated on cardholder phone via Face ID; route directly connected home and workplace.",
            "rationale": (
                "Biometric application authorization combined with pickup and dropoff routes matching cardholder verified locations proves "
                "cardholder authorization."
            ),
        },
        {
            "id_num": 15,
            "case_type": "straightforward",
            "category": "defective_or_not_as_described",
            "label": "REJECT",
            "amount": 450.00,
            "merchant": "Alpine Gear Clearance",
            "merchant_category": "sporting_goods",
            "tenure": 15,
            "prior_disputes": 0,
            "delay_days": 25,
            "evidence_signals": [
                "explicit_final_sale_waiver",
                "cardholder_refusal_to_return",
            ],
            "case_description": (
                "Customer purchased discontinued ski boots for $450.00 during an online liquidation sale. Customer filed a dispute claiming "
                "the boots did not fit comfortably. The checkout invoice prominently displays a bold 'Final Sale - No Returns Accepted' "
                "checkbox that the customer checked, and the customer refused the merchant's request to return the boots for examination."
            ),
            "evidence_summary": "Customer checked explicit final sale waiver and refused to return the boots.",
            "rationale": (
                "Subjective sizing remorse on explicitly disclosed, agreed-upon final sale liquidation items where the buyer refuses return "
                "does not constitute a valid chargeback condition."
            ),
        },
        {
            "id_num": 16,
            "case_type": "straightforward",
            "category": "subscription_cancellation",
            "label": "REJECT",
            "amount": 29.99,
            "merchant": "FitPulse Online Gym",
            "merchant_category": "fitness",
            "tenure": 12,
            "prior_disputes": 0,
            "delay_days": 18,
            "evidence_signals": [
                "active_daily_app_sessions",
                "zero_cancellation_request_record",
            ],
            "case_description": (
                "Customer disputes a $29.99 monthly fitness app membership fee claiming they canceled the subscription months ago. "
                "Merchant server logs demonstrate the customer logged into the application 22 times during the disputed billing month, "
                "completed 14 workout sessions, and never submitted a cancellation request via the in-app portal or support email."
            ),
            "evidence_summary": "Extensive active usage logs during disputed month; no record of cancellation submission.",
            "rationale": (
                "Active utilization of digital subscription services throughout the billing period refutes claims of prior cancellation."
            ),
        },
        {
            "id_num": 17,
            "case_type": "straightforward",
            "category": "goods_not_received",
            "label": "REJECT",
            "amount": 95.00,
            "merchant": "Gourmet Bistro & Grill",
            "merchant_category": "restaurant",
            "tenure": 42,
            "prior_disputes": 0,
            "delay_days": 20,
            "evidence_signals": ["signed_pos_itemized_receipt", "chip_card_read"],
            "case_description": (
                "Customer filed a dispute claiming they never authorized a $95.00 dine-in restaurant bill at Gourmet Bistro & Grill. "
                "The merchant provided a scanned copy of the itemized table receipt with the cardholder's legible signature, along with "
                "point-of-sale logs demonstrating successful chip contact and tip approval."
            ),
            "evidence_summary": "Itemized dining receipt with physical cardholder signature and EMV contact approval.",
            "rationale": (
                "Physical chip insertion coupled with signed merchant itemized receipt proves in-person cardholder authorization and service receipt."
            ),
        },
        {
            "id_num": 18,
            "case_type": "straightforward",
            "category": "unauthorized_transaction",
            "label": "REJECT",
            "amount": 210.00,
            "merchant": "PixelCraft Game Studios",
            "merchant_category": "gaming",
            "tenure": 19,
            "prior_disputes": 4,
            "delay_days": 35,
            "evidence_signals": ["family_ip_console_id", "friendly_fraud_pattern"],
            "case_description": (
                "Customer disputes $210.00 in microtransaction gaming purchases on PixelCraft Game Studios. Merchant records show the "
                "purchases were executed on a PlayStation console linked to the cardholder's residential Wi-Fi network and account, "
                "with digital skins delivered to an avatar named after the cardholder's dependent. The customer has filed 4 similar "
                "gaming disputes in the past 12 months."
            ),
            "evidence_summary": "Transactions executed from family residential console; history of 4 repeated friendly fraud claims.",
            "rationale": (
                "Friendly fraud by household members on authorized domestic devices without cardholder oversight does not qualify as "
                "third-party unauthorized fraud under network rules."
            ),
        },
        {
            "id_num": 19,
            "case_type": "straightforward",
            "category": "duplicate_billing",
            "label": "REJECT",
            "amount": 160.00,
            "merchant": "TechPro Accessories",
            "merchant_category": "electronics",
            "tenure": 11,
            "prior_disputes": 0,
            "delay_days": 12,
            "evidence_signals": [
                "two_distinct_orders",
                "two_distinct_shipping_destinations",
            ],
            "case_description": (
                "Customer claims a duplicate charge of $160.00 at TechPro Accessories. Merchant order records show the customer placed "
                "two separate orders thirty minutes apart: the first for a USB-C dock shipped to the customer's residential address, "
                "and the second for a mechanical keyboard shipped to the customer's corporate business address, each with distinct tracking numbers."
            ),
            "evidence_summary": "Two distinct orders for different items shipped to separate destinations.",
            "rationale": (
                "Dispute is invalidated by merchant records establishing two independent orders with different items and separate shipping destinations."
            ),
        },
        {
            "id_num": 20,
            "case_type": "straightforward",
            "category": "credit_not_processed",
            "label": "REJECT",
            "amount": 75.00,
            "merchant": "AeroShuttle Airport Express",
            "merchant_category": "transportation",
            "tenure": 22,
            "prior_disputes": 0,
            "delay_days": 80,
            "evidence_signals": [
                "dispute_filed_past_deadline",
                "exceeded_statutory_window",
            ],
            "case_description": (
                "Customer claims a refund was promised for an airport shuttle fare of $75.00 from AeroShuttle Airport Express that arrived "
                "45 minutes late. The transaction occurred 80 calendar days prior to dispute submission, well beyond the payment network's "
                "strict 60-day maximum filing deadline from statement posting date, with no extenuating circumstances noted."
            ),
            "evidence_summary": "Dispute initiated 80 days post-settlement, exceeding statutory 60-day deadline.",
            "rationale": (
                "Chargeback filing rules strictly enforce statutory deadlines; claims initiated past the allowable timeframe must be "
                "rejected regardless of underlying merit."
            ),
        },
        # =========================================================================
        # 2. BORDERLINE CASES (20 examples: 10 APPROVE, 10 REJECT)
        # =========================================================================
        # --- Borderline APPROVE (10) ---
        {
            "id_num": 21,
            "case_type": "borderline",
            "category": "defective_or_not_as_described",
            "label": "APPROVE",
            "amount": 340.00,
            "merchant": "Lumina Smart Home",
            "merchant_category": "home_automation",
            "tenure": 24,
            "prior_disputes": 0,
            "delay_days": 58,
            "evidence_signals": [
                "filing_at_day_58",
                "documented_4_week_merchant_troubleshooting",
            ],
            "case_description": (
                "Customer filed a dispute for $340.00 regarding a smart security camera hub from Lumina Smart Home on day 58 of the 60-day "
                "dispute window. The hub suffered repeated firmware crashes preventing video recording. Customer logs show four continuous "
                "weeks of constructive correspondence with merchant technical support attempting firmware re-flashing before the merchant "
                "finally refused an exchange."
            ),
            "evidence_summary": "Filed on day 58 of 60; substantial documentation shows active troubleshooting with merchant.",
            "rationale": (
                "Filing on day 58 remains within the statutory 60-day window; extensive documentation shows the delay resulted from "
                "good-faith customer cooperation with merchant troubleshooting."
            ),
        },
        {
            "id_num": 22,
            "case_type": "borderline",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 139.00,
            "merchant": "Urban Denim Supply",
            "merchant_category": "apparel",
            "tenure": 16,
            "prior_disputes": 1,
            "delay_days": 14,
            "evidence_signals": [
                "delivered_to_unattended_lobby",
                "high_theft_apartment_complex",
            ],
            "case_description": (
                "Customer disputes $139.00 for designer jeans from Urban Denim Supply. Carrier tracking notes the parcel was 'Left in "
                "Building Lobby' of a 120-unit apartment building without a signature or recipient photo. The transaction amount is below "
                "the carrier's mandatory signature threshold, but building management confirms frequent lobby package theft, and the "
                "customer submitted a signed theft statement."
            ),
            "evidence_summary": "Parcel dropped in unsecured multi-unit lobby without recipient signature; tenant filed theft affidavit.",
            "rationale": (
                "Delivery to an unsecured public common area without individual recipient verification fails the carrier standard for secure "
                "delivery when corroborated by tenant affidavit."
            ),
        },
        {
            "id_num": 23,
            "case_type": "borderline",
            "category": "subscription_cancellation",
            "label": "APPROVE",
            "amount": 79.00,
            "merchant": "LearnSphere Academy",
            "merchant_category": "education",
            "tenure": 10,
            "prior_disputes": 0,
            "delay_days": 3,
            "evidence_signals": [
                "cancellation_3_hours_post_billing",
                "zero_service_consumption",
            ],
            "case_description": (
                "Customer enrolled in a monthly coding bootcamp platform for $79.00 with LearnSphere Academy. The recurring subscription "
                "billed at 12:01 AM, and the customer executed cancellation at 3:15 AM the same morning upon reviewing their email receipt. "
                "Platform telemetry confirms zero coursework viewed, zero videos played, and zero logins after the cancellation."
            ),
            "evidence_summary": "Canceled 3 hours after renewal charge with zero platform consumption.",
            "rationale": (
                "Immediate cancellation within hours of renewal charge coupled with verified zero consumption warrants equitable approval "
                "under consumer fairness guidelines."
            ),
        },
        {
            "id_num": 24,
            "case_type": "borderline",
            "category": "defective_or_not_as_described",
            "label": "APPROVE",
            "amount": 195.00,
            "merchant": "Terra Pottery Works",
            "merchant_category": "home_decor",
            "tenure": 30,
            "prior_disputes": 0,
            "delay_days": 10,
            "evidence_signals": ["transit_damaged_replacement", "photos_provided"],
            "case_description": (
                "Customer ordered a ceramic vase set for $195.00 from Terra Pottery Works. The original shipment arrived with a cracked bowl, "
                "and the merchant dispatched a replacement. However, the replacement shipment arrived with two shattered vases due to "
                "inadequate bubble wrap. Customer submitted photos of both damaged shipments and unsealed boxes; merchant refused a refund."
            ),
            "evidence_summary": "Both original and replacement shipments arrived severely damaged with photographic evidence.",
            "rationale": (
                "Merchant failed twice to deliver undamaged merchandise; customer provided complete photographic proof satisfying burden of proof."
            ),
        },
        {
            "id_num": 25,
            "case_type": "borderline",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 260.00,
            "merchant": "Apex Tool Warehouse",
            "merchant_category": "hardware",
            "tenure": 22,
            "prior_disputes": 0,
            "delay_days": 20,
            "evidence_signals": [
                "partial_shipment_weight_discrepancy",
                "merchant_claims_full",
            ],
            "case_description": (
                "Customer ordered three heavy power tools totaling $260.00 from Apex Tool Warehouse. Only two tools were inside the delivered box. "
                "The merchant asserts all three items were packed. However, carrier shipping manifest documentation records the parcel "
                "weight as exactly 14.2 lbs at origin scan, exactly matching the catalog weight of the two delivered items rather than "
                "the 22.8 lbs required for all three."
            ),
            "evidence_summary": "Carrier origin weigh-in manifest proves parcel was 8 lbs lighter than required for complete order.",
            "rationale": (
                "Carrier origin weight records provide objective, third-party physical corroboration that the merchant short-shipped the order."
            ),
        },
        {
            "id_num": 26,
            "case_type": "borderline",
            "category": "unauthorized_transaction",
            "label": "APPROVE",
            "amount": 280.00,
            "merchant": "RapidPay Mobile Transfer",
            "merchant_category": "financial_services",
            "tenure": 52,
            "prior_disputes": 0,
            "delay_days": 5,
            "evidence_signals": ["adjacent_city_ip", "cardholder_clean_5_year_tenure"],
            "case_description": (
                "Customer with 52 months of spotless account history disputes a $280.00 peer-to-peer transfer via RapidPay Mobile Transfer. "
                "The transaction originated from an IP address in an adjacent city 45 miles away, but used an unrecognized browser user-agent "
                "string and did not prompt for two-factor authentication due to a gateway risk engine glitch. Customer maintains no knowledge "
                "of the recipient."
            ),
            "evidence_summary": "Unrecognized browser and session bypass on 52-month clean profile; IP in adjacent city.",
            "rationale": (
                "Unrecognized device fingerprint and security bypass combined with a flawless multi-year customer record support the "
                "unauthorized determination."
            ),
        },
        {
            "id_num": 27,
            "case_type": "borderline",
            "category": "credit_not_processed",
            "label": "APPROVE",
            "amount": 185.00,
            "merchant": "Coastal Getaway Motel",
            "merchant_category": "lodging",
            "tenure": 38,
            "prior_disputes": 0,
            "delay_days": 16,
            "evidence_signals": [
                "cancelled_within_24h_grace_period",
                "disputed_check_in_proximity",
            ],
            "case_description": (
                "Cardholder booked a motel room for $185.00 with Coastal Getaway Motel for arrival the next evening, but canceled within "
                "two hours of booking. State consumer statutes mandate a 24-hour statutory grace period for online bookings made through "
                "regulated booking platforms, but the motel enforced their strict 48-hour pre-check-in no-refund rule."
            ),
            "evidence_summary": "Cancellation submitted within 2 hours of booking under statutory cooling-off grace regulations.",
            "rationale": (
                "Statutory booking cooling-off protections supersede merchant property terms when reservations are canceled almost immediately."
            ),
        },
        {
            "id_num": 28,
            "case_type": "borderline",
            "category": "defective_or_not_as_described",
            "label": "APPROVE",
            "amount": 99.00,
            "merchant": "SoundWave DSP Studio",
            "merchant_category": "software",
            "tenure": 14,
            "prior_disputes": 0,
            "delay_days": 7,
            "evidence_signals": [
                "os_compatibility_crash_logs",
                "support_ticket_unresolved",
            ],
            "case_description": (
                "Customer purchased an audio processing software plugin for $99.00 from SoundWave DSP Studio. The merchant website lists "
                "compatibility with macOS 14.x. Upon installation, the plugin reliably crashed the digital audio workstation. "
                "Customer submitted terminal crash logs and bug reports within 48 hours; merchant support acknowledged the bug but refused "
                "a refund pending an unscheduled patch release."
            ),
            "evidence_summary": "Software crashes persistently on listed compatible OS; merchant admitted bug but refused refund.",
            "rationale": (
                "Software that fails to function on merchant-advertised compatible operating systems is materially defective; vague promises "
                "of future patches do not satisfy fulfillment obligations."
            ),
        },
        {
            "id_num": 29,
            "case_type": "borderline",
            "category": "defective_or_not_as_described",
            "label": "APPROVE",
            "amount": 220.00,
            "merchant": "Summit Auto Rental",
            "merchant_category": "rental_cars",
            "tenure": 26,
            "prior_disputes": 0,
            "delay_days": 21,
            "evidence_signals": [
                "damage_fee_assessed_3_weeks_later",
                "no_check_in_walkaround_sheet",
            ],
            "case_description": (
                "Three weeks following the return of a rental vehicle to Summit Auto Rental, the merchant charged the cardholder's card "
                "an additional $220.00 for alleged windshield chip damage. The customer's vehicle return slip marked 'Vehicle Inspected - No "
                "Damage Found' signed by the attendant at return, and the merchant provided no dated photographic evidence taken at return time."
            ),
            "evidence_summary": "Return slip confirms no damage found at check-in; fee assessed 21 days later without evidence.",
            "rationale": (
                "Signed vehicle return inspection confirming zero damage conclusively overrides post-return damage assessments made weeks later."
            ),
        },
        {
            "id_num": 30,
            "case_type": "borderline",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 165.00,
            "merchant": "Festive Harmony Live",
            "merchant_category": "entertainment",
            "tenure": 19,
            "prior_disputes": 0,
            "delay_days": 15,
            "evidence_signals": [
                "lineup_headliners_cancelled",
                "festival_nature_materially_altered",
            ],
            "case_description": (
                "Cardholder disputes a $165.00 general admission ticket for a music festival promoted by Festive Harmony Live. "
                "Both top headlining artists withdrew due to contract disputes, reducing the festival from eight main-stage acts to three "
                "local performers. The promoter refused refund requests, claiming the terms permitted lineup substitutions without restriction."
            ),
            "evidence_summary": "Over 60% of advertised musical lineup canceled, materially altering the event character.",
            "rationale": (
                "Material alteration of event headliners deprives the buyer of the primary value of the purchase; blanket disclaimer "
                "clauses are unenforceable when performance substance is radically degraded."
            ),
        },
        # --- Borderline REJECT (10) ---
        {
            "id_num": 31,
            "case_type": "borderline",
            "category": "goods_not_received",
            "label": "REJECT",
            "amount": 310.00,
            "merchant": "Urban Tech Depot",
            "merchant_category": "electronics",
            "tenure": 21,
            "prior_disputes": 0,
            "delay_days": 62,
            "evidence_signals": [
                "day_62_filing",
                "exceeded_statutory_60d_window_by_2_days",
            ],
            "case_description": (
                "Customer claims a $310.00 tablet computer from Urban Tech Depot was never delivered. Carrier tracking shows 'Delivered - "
                "Left on Porch' with an ambiguous photo. However, the customer submitted the dispute on calendar day 62 following the "
                "statement posting date, two days past the strict 60-day network cut-off. The customer provided no medical or emergency "
                "documentation explaining the delay."
            ),
            "evidence_summary": "Dispute submitted on day 62, missing statutory 60-day deadline by 48 hours without extenuating proof.",
            "rationale": (
                "Network chargeback submission windows are strict jurisdictional time limits; filing on day 62 forfeits dispute rights "
                "regardless of underlying factual merits."
            ),
        },
        {
            "id_num": 32,
            "case_type": "borderline",
            "category": "goods_not_received",
            "label": "REJECT",
            "amount": 115.00,
            "merchant": "Metro Parcel Boutique",
            "merchant_category": "retail",
            "tenure": 17,
            "prior_disputes": 2,
            "delay_days": 12,
            "evidence_signals": [
                "carrier_front_porch_photo_gps",
                "no_police_theft_report",
            ],
            "case_description": (
                "Customer disputes $115.00 for boutique clothing from Metro Parcel Boutique claiming the package was stolen from their porch. "
                "Carrier records show successful delivery with a clear photograph of the parcel sitting on the front steps, with the house number "
                "clearly visible. The customer admits the photo matches their porch but asserts it was stolen, yet failed to file a police report."
            ),
            "evidence_summary": "Carrier photo confirms delivery to cardholder steps; customer claims porch theft with no police report.",
            "rationale": (
                "Documented photographic delivery to verified premises shifts risk of post-delivery loss to recipient; absence of official police "
                "theft documentation requires rejection."
            ),
        },
        {
            "id_num": 33,
            "case_type": "borderline",
            "category": "subscription_cancellation",
            "label": "REJECT",
            "amount": 45.00,
            "merchant": "ProStream Cloud Network",
            "merchant_category": "digital_goods",
            "tenure": 15,
            "prior_disputes": 1,
            "delay_days": 4,
            "evidence_signals": [
                "cancellation_2_days_after_billing",
                "terms_require_48h_notice",
            ],
            "case_description": (
                "Customer disputes a $45.00 monthly subscription fee for ProStream Cloud Network. The renewal billed on the 1st of the month, "
                "and the customer submitted a cancellation on the 3rd. Merchant terms of service explicitly require cancellation at least 48 hours "
                "prior to billing. Account logs show the customer logged in on the 2nd and streamed a full-length movie."
            ),
            "evidence_summary": "Canceled 2 days post-renewal; account accessed and utilized for media streaming after renewal.",
            "rationale": (
                "Cancellation occurred after renewal date and customer actively utilized paid services during the disputed cycle."
            ),
        },
        {
            "id_num": 34,
            "case_type": "borderline",
            "category": "defective_or_not_as_described",
            "label": "REJECT",
            "amount": 280.00,
            "merchant": "AeroGadget Outlet",
            "merchant_category": "electronics",
            "tenure": 11,
            "prior_disputes": 0,
            "delay_days": 20,
            "evidence_signals": ["cosmetic_packaging_dent", "product_fully_functional"],
            "case_description": (
                "Customer claims a $280.00 portable projector from AeroGadget Outlet was 'defective upon receipt'. Diagnostic logs and customer "
                "admissions confirm the projector functions flawlessly with pristine image projection; however, the outer cardboard retail box "
                "had a 1-inch crease on one corner. The merchant offered a $25 courtesy credit which the customer rejected."
            ),
            "evidence_summary": "Minor cosmetic crease on exterior packaging; electronic device itself is 100% functional.",
            "rationale": (
                "Minor cosmetic imperfections in transit packaging that do not impair merchandise functionality or value do not constitute "
                "material defect."
            ),
        },
        {
            "id_num": 35,
            "case_type": "borderline",
            "category": "unauthorized_transaction",
            "label": "REJECT",
            "amount": 72.00,
            "merchant": "QuickCab City Rides",
            "merchant_category": "transportation",
            "tenure": 25,
            "prior_disputes": 0,
            "delay_days": 8,
            "evidence_signals": [
                "cardholder_device_gps_close",
                "disputed_route_familiar",
            ],
            "case_description": (
                "Customer disputes a $72.00 late-night ride from QuickCab City Rides claiming they were asleep at home. Telemetry shows the "
                "ride was ordered from the cardholder's mobile device with the device's GPS located 300 feet from the pickup spot. "
                "The destination was the cardholder's regular social club. Customer claims a friend may have borrowed the phone."
            ),
            "evidence_summary": "Ride ordered from cardholder handset near pickup; route ended at cardholder regular destination.",
            "rationale": (
                "Allowing third parties access to an unlocked, authorized mobile device falls under cardholder negligence rather than "
                "third-party payment fraud."
            ),
        },
        {
            "id_num": 36,
            "case_type": "borderline",
            "category": "defective_or_not_as_described",
            "label": "REJECT",
            "amount": 350.00,
            "merchant": "Veloce Bike Studio",
            "merchant_category": "sporting_goods",
            "tenure": 13,
            "prior_disputes": 0,
            "delay_days": 18,
            "evidence_signals": [
                "restocking_fee_disclosed",
                "customer_disputes_policy",
            ],
            "case_description": (
                "Customer returned an unridden bicycle helmet and shoes totaling $350.00 to Veloce Bike Studio. The merchant issued a "
                "refund of $297.50, deducting a 15% ($52.50) restocking fee. The customer disputes the $52.50 deduction. The merchant's "
                "return policy prominently discloses the 15% restocking fee on the product page, checkout cart, and physical invoice."
            ),
            "evidence_summary": "Customer disputes 15% restocking fee prominently disclosed prior to purchase.",
            "rationale": (
                "Clearly disclosed, reasonable restocking fees agreed to at the time of purchase are contractually valid and enforceable."
            ),
        },
        {
            "id_num": 37,
            "case_type": "borderline",
            "category": "goods_not_received",
            "label": "REJECT",
            "amount": 65.00,
            "merchant": "CyberRead Publishing",
            "merchant_category": "digital_goods",
            "tenure": 8,
            "prior_disputes": 0,
            "delay_days": 14,
            "evidence_signals": [
                "100_percent_content_downloaded",
                "claimed_accidental_order",
            ],
            "case_description": (
                "Customer purchased an electronic reference manual for $65.00 from CyberRead Publishing. The customer filed a dispute "
                "claiming accidental purchase. Server access telemetry shows the customer downloaded the entire 600-page PDF within "
                "fifteen minutes of purchase and accessed the link four times over the following week."
            ),
            "evidence_summary": "Full digital file downloaded and accessed repeatedly before refund request.",
            "rationale": (
                "Full download and repeated access to digital informational goods forfeits claims of accidental purchase under digital content rules."
            ),
        },
        {
            "id_num": 38,
            "case_type": "borderline",
            "category": "defective_or_not_as_described",
            "label": "REJECT",
            "amount": 890.00,
            "merchant": "Gemstone Gallery Fine Jewelry",
            "merchant_category": "luxury_goods",
            "tenure": 34,
            "prior_disputes": 0,
            "delay_days": 28,
            "evidence_signals": [
                "appraisal_valuation_variance_only",
                "specs_identical",
            ],
            "case_description": (
                "Customer disputes an $890.00 diamond pendant from Gemstone Gallery Fine Jewelry claiming it was misrepresented. An independent "
                "appraiser verified the diamond carat weight (0.75 ct), cut grade (Very Good), color (H), and clarity (SI1) match the "
                "merchant certificate exactly. The appraiser simply estimated fair market retail value at $780.00 rather than $890.00."
            ),
            "evidence_summary": "Physical gemstone specifications match certificate; dispute reflects minor subjective valuation difference.",
            "rationale": (
                "Subjective appraisal price differences do not substantiate 'not as described' claims when physical gemological specifications "
                "match the sales invoice."
            ),
        },
        {
            "id_num": 39,
            "case_type": "borderline",
            "category": "subscription_cancellation",
            "label": "REJECT",
            "amount": 80.00,
            "merchant": "IronClad Athletic Club",
            "merchant_category": "fitness",
            "tenure": 28,
            "prior_disputes": 0,
            "delay_days": 15,
            "evidence_signals": [
                "oral_cancellation_unsupported",
                "contract_requires_certified_mail",
            ],
            "case_description": (
                "Customer disputes an $80.00 monthly membership charge from IronClad Athletic Club claiming they told a desk clerk they "
                "wanted to cancel. The membership contract signed by the cardholder explicitly specifies cancellations must be submitted via "
                "the online portal or written form. Merchant records contain no written cancellation, and the customer cannot name the staff member."
            ),
            "evidence_summary": "Alleged verbal cancellation contradicts clear written contract requirements for portal submission.",
            "rationale": (
                "Uncorroborated oral cancellation assertions do not override explicit, agreed contractual cancellation procedures."
            ),
        },
        {
            "id_num": 40,
            "case_type": "borderline",
            "category": "goods_not_received",
            "label": "REJECT",
            "amount": 215.00,
            "merchant": "SwiftAir Regional",
            "merchant_category": "travel",
            "tenure": 44,
            "prior_disputes": 0,
            "delay_days": 12,
            "evidence_signals": [
                "missed_flight_personal_traffic",
                "flight_departed_on_schedule",
            ],
            "case_description": (
                "Customer disputes a $215.00 non-refundable economy airfare on SwiftAir Regional. Flight records prove the aircraft departed "
                "on time with all operational services fulfilled. The customer arrived at the airport boarding gate twenty minutes after "
                "departure due to highway construction traffic. The airline offered rebooking on the next flight for a standard change fee."
            ),
            "evidence_summary": "Flight operated on schedule; passenger arrived late due to personal traffic delays.",
            "rationale": (
                "Carrier fulfilled its operational transport commitment; passenger missed flight due to personal delays on a non-refundable fare."
            ),
        },
        # =========================================================================
        # 3. INCOMPLETE EVIDENCE CASES (20 examples: 10 APPROVE, 10 REJECT)
        # =========================================================================
        # --- Incomplete Evidence APPROVE (10) ---
        {
            "id_num": 41,
            "case_type": "incomplete_evidence",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 180.00,
            "merchant": "ElectroTech Warehouse",
            "merchant_category": "electronics",
            "tenure": 12,
            "prior_disputes": 0,
            "delay_days": 25,
            "evidence_signals": [
                "merchant_failed_to_respond",
                "default_burden_of_proof",
            ],
            "case_description": (
                "Customer claims a $180.00 portable SSD from ElectroTech Warehouse was never delivered. The dispute processor sent formal "
                "notice and an evidence request to the merchant. The merchant failed to submit any tracking number, invoice, or response "
                "within the mandatory 14-day arbitration response deadline."
            ),
            "evidence_summary": "Merchant submitted zero response or fulfillment evidence within required 14-day timeframe.",
            "rationale": (
                "Payment network dispute procedural rules dictate immediate default approval in favor of the cardholder when the merchant "
                "fails to respond within the statutory response window."
            ),
        },
        {
            "id_num": 42,
            "case_type": "incomplete_evidence",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 95.00,
            "merchant": "Bella Vita Imports",
            "merchant_category": "specialty_food",
            "tenure": 20,
            "prior_disputes": 0,
            "delay_days": 19,
            "evidence_signals": [
                "unverifiable_carrier_tracking",
                "carrier_record_missing",
            ],
            "case_description": (
                "Customer disputes $95.00 for artisanal olive oil from Bella Vita Imports claiming non-delivery. In response, the merchant "
                "provided a tracking code string 'BV-984210'. Querying all domestic and international carrier registries returns no valid "
                "carrier matches or dispatch records for this tracking identifier."
            ),
            "evidence_summary": "Merchant tracking number cannot be verified in any carrier database.",
            "rationale": (
                "Unverifiable tracking references that lack carrier confirmation fail to satisfy the merchant's fundamental burden to prove dispatch."
            ),
        },
        {
            "id_num": 43,
            "case_type": "incomplete_evidence",
            "category": "unauthorized_transaction",
            "label": "APPROVE",
            "amount": 410.00,
            "merchant": "CloudHost Global Services",
            "merchant_category": "cloud_hosting",
            "tenure": 33,
            "prior_disputes": 0,
            "delay_days": 8,
            "evidence_signals": [
                "merchant_failed_ip_logs",
                "cardholder_filed_prompt_claim",
            ],
            "case_description": (
                "Customer with zero past fraud history disputes a $410.00 server hosting invoice from CloudHost Global Services. "
                "Customer states they never opened an account with this provider. The merchant was requested to supply account registration "
                "IP logs, email verification headers, or billing authorization records, but submitted only an automated PDF billing statement."
            ),
            "evidence_summary": "Merchant failed to supply account registration IP logs or identity verification records.",
            "rationale": (
                "In card-not-present fraud disputes, presenting a billing invoice without identity or connection evidence fails to refute "
                "the customer's unauthorized claim."
            ),
        },
        {
            "id_num": 44,
            "case_type": "incomplete_evidence",
            "category": "duplicate_billing",
            "label": "APPROVE",
            "amount": 125.00,
            "merchant": "Cascade Mountain Outfitters",
            "merchant_category": "apparel",
            "tenure": 27,
            "prior_disputes": 1,
            "delay_days": 6,
            "evidence_signals": [
                "merchant_silent_on_second_charge",
                "single_receipt_provided",
            ],
            "case_description": (
                "Customer account shows two consecutive charges of $125.00 from Cascade Mountain Outfitters posted within one minute. "
                "In response to the duplicate billing inquiry, the merchant provided an invoice for one order but submitted no documentation "
                "or transaction record explaining the second identical charge."
            ),
            "evidence_summary": "Merchant verified single order but provided zero evidence supporting the second settled charge.",
            "rationale": (
                "Merchant's failure to provide order documentation for the second charge leaves the duplicate billing claim unrefuted."
            ),
        },
        {
            "id_num": 45,
            "case_type": "incomplete_evidence",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 320.00,
            "merchant": "NextGen VR Emporium",
            "merchant_category": "electronics",
            "tenure": 15,
            "prior_disputes": 0,
            "delay_days": 30,
            "evidence_signals": [
                "merchant_website_offline",
                "business_phone_disconnected",
            ],
            "case_description": (
                "Customer paid $320.00 for VR accessories from NextGen VR Emporium. Two weeks after the transaction, the merchant's web store "
                "went offline with DNS errors, customer support phone lines were disconnected, and formal dispute inquiries were returned as "
                "undeliverable. Customer received no tracking information."
            ),
            "evidence_summary": "Merchant website shuttered and communications non-functional; no evidence of shipping.",
            "rationale": (
                "Total unavailability of merchant communication channels and failure to respond establishes abandoned fulfillment."
            ),
        },
        {
            "id_num": 46,
            "case_type": "incomplete_evidence",
            "category": "credit_not_processed",
            "label": "APPROVE",
            "amount": 140.00,
            "merchant": "Nordic Footwear Co",
            "merchant_category": "apparel",
            "tenure": 22,
            "prior_disputes": 0,
            "delay_days": 18,
            "evidence_signals": [
                "courier_acceptance_scan_only",
                "carrier_tracking_stalled",
            ],
            "case_description": (
                "Customer returned boots for $140.00 to Nordic Footwear Co using the merchant's prepaid return label. The carrier receipt "
                "shows an official origin post office drop-off acceptance scan on October 10th. Tracking subsequent progress stalled with no "
                "further updates. The merchant refused to credit the customer because the parcel never reached their intake dock."
            ),
            "evidence_summary": "Customer possesses official carrier drop-off receipt; transit stalling occurred on merchant prepaid label.",
            "rationale": (
                "When returns are shipped using merchant-provided prepaid labels, the merchant bears the transit loss risk once the customer "
                "secures carrier acceptance."
            ),
        },
        {
            "id_num": 47,
            "case_type": "incomplete_evidence",
            "category": "subscription_cancellation",
            "label": "APPROVE",
            "amount": 39.99,
            "merchant": "DevTools Pro Cloud",
            "merchant_category": "saas",
            "tenure": 31,
            "prior_disputes": 0,
            "delay_days": 10,
            "evidence_signals": [
                "merchant_failed_to_provide_tos_acceptance",
                "claimed_cancellation",
            ],
            "case_description": (
                "Customer disputes a $39.99 monthly developer tooling charge from DevTools Pro Cloud, stating they terminated service via "
                "an in-app toggle. The merchant disputed the chargeback but failed to produce subscription agreement acceptance logs, recurring "
                "billing disclosure records, or cancellation audit records."
            ),
            "evidence_summary": "Merchant cannot produce audit logs proving recurring subscription consent or cancellation history.",
            "rationale": (
                "Merchants must maintain demonstrable electronic records of recurring billing consent; absence of audit logs favors the cardholder."
            ),
        },
        {
            "id_num": 48,
            "case_type": "incomplete_evidence",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 550.00,
            "merchant": "Luxe Decor Imports",
            "merchant_category": "furniture",
            "tenure": 17,
            "prior_disputes": 0,
            "delay_days": 22,
            "evidence_signals": [
                "freight_bill_of_lading_unsigned",
                "no_delivery_signature",
            ],
            "case_description": (
                "Customer disputes $550.00 for an imported coffee table from Luxe Decor Imports. In defense, the merchant provided an internal "
                "freight bill of lading; however, the consignee delivery signature section is blank, with no consignee name, delivery date, "
                "or driver initials recorded."
            ),
            "evidence_summary": "Merchant submitted blank, unsigned bill of lading lacking proof of recipient handover.",
            "rationale": (
                "An unsigned freight manifest without delivery confirmation does not satisfy the evidentiary standard for completed delivery."
            ),
        },
        {
            "id_num": 49,
            "case_type": "incomplete_evidence",
            "category": "credit_not_processed",
            "label": "APPROVE",
            "amount": 210.00,
            "merchant": "Gourmet Catering Co",
            "merchant_category": "catering",
            "tenure": 40,
            "prior_disputes": 0,
            "delay_days": 14,
            "evidence_signals": [
                "merchant_provided_no_contract",
                "claimed_cancellation_refund",
            ],
            "case_description": (
                "Customer disputes a $210.00 event deposit from Gourmet Catering Co. Customer provided text messages canceling the service "
                "three weeks prior to the event. The merchant claimed the deposit was non-refundable but failed to provide any signed contract, "
                "written agreement, or published cancellation policy terms."
            ),
            "evidence_summary": "Merchant claims deposit non-refundable but cannot provide signed agreement or written cancellation terms.",
            "rationale": (
                "Non-refundable deposit terms must be agreed to in writing prior to payment; failure to provide contract terms invalidates retention."
            ),
        },
        {
            "id_num": 50,
            "case_type": "incomplete_evidence",
            "category": "defective_or_not_as_described",
            "label": "APPROVE",
            "amount": 160.00,
            "merchant": "GreenThumb Nursery Online",
            "merchant_category": "gardening",
            "tenure": 16,
            "prior_disputes": 0,
            "delay_days": 11,
            "evidence_signals": [
                "merchant_failed_to_respond_to_dead_plants",
                "default_approval",
            ],
            "case_description": (
                "Customer ordered rare live perennial plants for $160.00 from GreenThumb Nursery Online. Plants arrived withered, desiccated, "
                "and dead due to being shipped without moisture wraps. Customer provided photos upon opening. The merchant did not reply to "
                "customer messages or respond to the dispute claim within 14 days."
            ),
            "evidence_summary": "Customer provided photos of dead plants upon arrival; merchant completely failed to respond.",
            "rationale": (
                "Photographic proof of perishable goods arriving dead coupled with total merchant default justifies full approval."
            ),
        },
        # --- Incomplete Evidence REJECT (10) ---
        {
            "id_num": 51,
            "case_type": "incomplete_evidence",
            "category": "defective_or_not_as_described",
            "label": "REJECT",
            "amount": 750.00,
            "merchant": "Titan Gaming Rigs",
            "merchant_category": "computers",
            "tenure": 14,
            "prior_disputes": 1,
            "delay_days": 20,
            "evidence_signals": [
                "customer_refused_photos_or_diagnostics",
                "unsubstantiated_defect",
            ],
            "case_description": (
                "Customer disputes $750.00 for a gaming desktop from Titan Gaming Rigs, asserting the graphics card was defective upon arrival. "
                "The dispute processor requested photos of the unit, error screens, diagnostic reports, or technician repair receipts. "
                "The customer refused all documentation requests, stating 'my word should be enough'."
            ),
            "evidence_summary": "Customer refused to provide any photos, error logs, or inspection proof of alleged hardware defect.",
            "rationale": (
                "Cardholders hold the initial burden of substantiating 'not as described' or defective claims; refusal to provide basic "
                "factual evidence requires claim denial."
            ),
        },
        {
            "id_num": 52,
            "case_type": "incomplete_evidence",
            "category": "unauthorized_transaction",
            "label": "REJECT",
            "amount": 820.00,
            "merchant": "Omni Jewelers Online",
            "merchant_category": "luxury_goods",
            "tenure": 9,
            "prior_disputes": 2,
            "delay_days": 15,
            "evidence_signals": [
                "customer_refused_fraud_affidavit",
                "no_police_report",
            ],
            "case_description": (
                "Customer claims an $820.00 gold bracelet purchase from Omni Jewelers Online was unauthorized. The bank requested a standard "
                "signed fraud affidavit under penalty of perjury and police report incident number. The customer repeatedly refused to sign the "
                "affidavit or report the alleged theft to police."
            ),
            "evidence_summary": "Customer refused to sign fraud affidavit or report alleged grand theft to law enforcement.",
            "rationale": (
                "Refusal to execute a standard fraud affidavit under penalty of perjury for high-value claims invalidates unauthorized claim processing."
            ),
        },
        {
            "id_num": 53,
            "case_type": "incomplete_evidence",
            "category": "goods_not_received",
            "label": "REJECT",
            "amount": 290.00,
            "merchant": "TimberCraft Furniture",
            "merchant_category": "furniture",
            "tenure": 18,
            "prior_disputes": 0,
            "delay_days": 16,
            "evidence_signals": [
                "customer_refuses_address_verification",
                "household_inquiry_declined",
            ],
            "case_description": (
                "Customer disputes $290.00 for an entryway bench from TimberCraft Furniture. Carrier records show delivery to the building "
                "reception. When asked to confirm whether roommates, family members, or building staff received the package, the customer "
                "declined to verify or cooperate with basic inquiry steps."
            ),
            "evidence_summary": "Carrier confirmed reception delivery; cardholder declined basic household inquiry or verification.",
            "rationale": (
                "Failure of the cardholder to conduct or confirm basic household inquiry following documented premises delivery prevents approval."
            ),
        },
        {
            "id_num": 54,
            "case_type": "incomplete_evidence",
            "category": "credit_not_processed",
            "label": "REJECT",
            "amount": 110.00,
            "merchant": "Apex Rental Car Services",
            "merchant_category": "rental_cars",
            "tenure": 24,
            "prior_disputes": 0,
            "delay_days": 25,
            "evidence_signals": [
                "claimed_verbal_refund_promise",
                "zero_written_evidence",
            ],
            "case_description": (
                "Customer disputes a $110.00 rental car cleaning fee claiming an agent verbally promised on the phone to waive it. "
                "The customer could provide no confirmation number, agent name, email correspondence, or written waiver. "
                "Merchant records show photos of excessive interior staining and a signed rental contract authorizing detailed cleaning fees."
            ),
            "evidence_summary": "Customer claims unrecorded verbal promise contradicted by signed cleaning policy and interior photos.",
            "rationale": (
                "Unsubstantiated claims of verbal waivers cannot overcome written contracts and documented condition photographs."
            ),
        },
        {
            "id_num": 55,
            "case_type": "incomplete_evidence",
            "category": "defective_or_not_as_described",
            "label": "REJECT",
            "amount": 380.00,
            "merchant": "SneakerVault Exclusive",
            "merchant_category": "apparel",
            "tenure": 13,
            "prior_disputes": 1,
            "delay_days": 18,
            "evidence_signals": [
                "unsubstantiated_counterfeit_allegation",
                "refused_authentication",
            ],
            "case_description": (
                "Customer claims a $380.00 pair of limited-edition collectible sneakers from SneakerVault Exclusive are counterfeit. "
                "The merchant provided original manufacturer distribution receipts. The customer was asked to provide a letter of opinion "
                "or inspection report from an authenticated sneaker inspection service, but refused to obtain any third-party verification."
            ),
            "evidence_summary": "Customer claims sneakers counterfeit but refuses third-party authentication inspection.",
            "rationale": (
                "Counterfeit allegations require objective substantiation from an authorized brand expert or reputable third-party authenticator."
            ),
        },
        {
            "id_num": 56,
            "case_type": "incomplete_evidence",
            "category": "unauthorized_transaction",
            "label": "REJECT",
            "amount": 450.00,
            "merchant": "Precision Auto Tech",
            "merchant_category": "automotive",
            "tenure": 30,
            "prior_disputes": 0,
            "delay_days": 14,
            "evidence_signals": ["missing_estimate_comparison", "work_order_signed"],
            "case_description": (
                "Customer disputes $450.00 of a $900.00 auto brake repair from Precision Auto Tech, alleging unauthorized additional repairs. "
                "The merchant produced a work order with the customer's signature authorizing 'brake overhaul and caliper servicing'. "
                "The customer could not produce any conflicting written estimate or limitation restricting repair scope."
            ),
            "evidence_summary": "Merchant produced signed work order authorizing repairs; customer produced no conflicting estimate.",
            "rationale": (
                "Signed authorization on a comprehensive work order establishes consent in the absence of documented written limitations."
            ),
        },
        {
            "id_num": 57,
            "case_type": "incomplete_evidence",
            "category": "defective_or_not_as_described",
            "label": "REJECT",
            "amount": 260.00,
            "merchant": "Sunset Resort & Suites",
            "merchant_category": "lodging",
            "tenure": 25,
            "prior_disputes": 0,
            "delay_days": 15,
            "evidence_signals": [
                "uninhabitable_claim_unsupported",
                "no_front_desk_report",
            ],
            "case_description": (
                "Customer completed a two-night stay for $260.00 at Sunset Resort & Suites, then disputed the charge claiming the room was "
                "uninhabitable due to cleanliness issues. Hotel records confirm the guest stayed both full nights, never reported any issue "
                "to the front desk, never requested a room change, and provided zero photos supporting the condition claim."
            ),
            "evidence_summary": "Guest stayed both full nights without notifying staff or providing photographic evidence of defect.",
            "rationale": (
                "Full consumption of hotel lodging without contemporaneous complaint or photographic evidence precludes post-stay chargeback approval."
            ),
        },
        {
            "id_num": 58,
            "case_type": "incomplete_evidence",
            "category": "duplicate_billing",
            "label": "REJECT",
            "amount": 190.00,
            "merchant": "Highland Dental Care",
            "merchant_category": "medical_services",
            "tenure": 35,
            "prior_disputes": 0,
            "delay_days": 10,
            "evidence_signals": [
                "authorization_hold_confused_for_settlement",
                "single_settled_charge",
            ],
            "case_description": (
                "Customer disputes an alleged duplicate charge of $190.00 at Highland Dental Care. An audit of transaction processor logs "
                "reveals that one entry was a temporary pending authorization that automatically dropped off after 72 hours, leaving only "
                "one single settled debit on the cardholder's statement. The customer provided only a screenshot of pending authorizations."
            ),
            "evidence_summary": "Cardholder confused temporary pending pre-authorization hold with duplicate settled charge.",
            "rationale": (
                "Expired temporary pre-authorization holds that never settled do not represent duplicate charges; only settled transactions are eligible."
            ),
        },
        {
            "id_num": 59,
            "case_type": "incomplete_evidence",
            "category": "defective_or_not_as_described",
            "label": "REJECT",
            "amount": 175.00,
            "merchant": "SkillForge Coding Bootcamps",
            "merchant_category": "education",
            "tenure": 11,
            "prior_disputes": 0,
            "delay_days": 20,
            "evidence_signals": [
                "subjective_curriculum_dissatisfaction",
                "no_misrepresentation_proof",
            ],
            "case_description": (
                "Customer disputes $175.00 for an online Python programming course from SkillForge Coding Bootcamps claiming the material "
                "was 'too basic'. The customer did not provide the syllabus or point to any advertised topic that was omitted. "
                "The merchant syllabus accurately described the introductory curriculum, and the customer completed 90% of exercises."
            ),
            "evidence_summary": "Customer completed 90% of introductory curriculum; provided no proof of syllabus misrepresentation.",
            "rationale": (
                "Subjective dissatisfaction with course difficulty after near-total completion does not constitute actionable misrepresentation."
            ),
        },
        {
            "id_num": 60,
            "case_type": "incomplete_evidence",
            "category": "credit_not_processed",
            "label": "REJECT",
            "amount": 85.00,
            "merchant": "Aura Spa & Wellness",
            "merchant_category": "personal_care",
            "tenure": 16,
            "prior_disputes": 0,
            "delay_days": 30,
            "evidence_signals": [
                "claimed_gift_card_non_receipt",
                "zero_purchase_transaction_record",
            ],
            "case_description": (
                "Customer claims they purchased a digital gift card for $85.00 from Aura Spa & Wellness that was never delivered by email. "
                "However, the customer's account records show no matching settlement or authorization code for this merchant across the "
                "relevant timeframe. The customer could not provide a receipt or bank statement reflecting an actual charge."
            ),
            "evidence_summary": "Customer cannot produce billing receipt; no transaction record exists on payment ledger.",
            "rationale": (
                "Dispute cannot proceed without verification that the disputed financial transaction actually posted to the cardholder account."
            ),
        },
        # =========================================================================
        # 4. CONFLICTING SIGNALS CASES (20 examples: 10 APPROVE, 10 REJECT)
        # =========================================================================
        # --- Conflicting Signals APPROVE (10) ---
        {
            "id_num": 61,
            "case_type": "conflicting_signals",
            "category": "unauthorized_transaction",
            "label": "APPROVE",
            "amount": 1200.00,
            "merchant": "Apex Crypto & Gift Cards",
            "merchant_category": "digital_goods",
            "tenure": 96,
            "prior_disputes": 0,
            "delay_days": 2,
            "evidence_signals": [
                "home_subnet_ip_match",
                "corroborated_infostealer_malware_infection",
            ],
            "case_description": (
                "Customer with an 8-year spotless account history disputes a $1,200.00 gift card purchase at Apex Crypto & Gift Cards. "
                "The IP address matched the customer's residential internet subnet, which superficially suggests friendly fraud. "
                "However, the bank's cybersecurity forensics team confirmed the customer's laptop was infected with a known session-hijacking "
                "infostealer trojan that reverse-proxied traffic through the home router at 4 AM to execute fraudulent gift card purchases."
            ),
            "evidence_summary": "Home IP match contradicted by verified cybersecurity forensic proof of infostealer trojan proxy attack.",
            "rationale": (
                "Forensic proof of malware session hijacking overcomes the IP match, confirming genuine unauthorized account compromise."
            ),
        },
        {
            "id_num": 62,
            "case_type": "conflicting_signals",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 380.00,
            "merchant": "LawnMaster Tools Online",
            "merchant_category": "hardware",
            "tenure": 28,
            "prior_disputes": 0,
            "delay_days": 10,
            "evidence_signals": [
                "carrier_scan_delivered_in_mailbox",
                "physical_impossibility_dimensions",
            ],
            "case_description": (
                "Customer disputes $380.00 for a 45-pound electric lawn mower from LawnMaster Tools Online. Carrier tracking indicates "
                "'Delivered - Left In Mailbox'. The customer submitted photographs of their standard 6x6x18-inch roadside metal mailbox. "
                "It is physically impossible for a 45-pound box measuring 36x24x20 inches to be placed inside this mailbox, indicating an "
                "erroneous or falsified carrier scan."
            ),
            "evidence_summary": "Carrier scan asserts 45-lb lawn mower was left in a 6-inch mailbox; physically impossible.",
            "rationale": (
                "Physical impossibility of carrier delivery status conclusively refutes tracking validity, establishing non-receipt."
            ),
        },
        {
            "id_num": 63,
            "case_type": "conflicting_signals",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 420.00,
            "merchant": "Nordic Designer Apparel",
            "merchant_category": "apparel",
            "tenure": 32,
            "prior_disputes": 0,
            "delay_days": 8,
            "evidence_signals": [
                "merchant_delivery_photo_valid",
                "security_cam_theft_within_90s",
            ],
            "case_description": (
                "Customer disputes $420.00 for designer clothing from Nordic Designer Apparel. The carrier provided a photo of the parcel on the "
                "customer's front porch. However, the customer provided timestamped home security camera footage showing an unidentified porch "
                "pirate following the delivery van and stealing the parcel exactly 75 seconds after drop-off, along with a formal police report."
            ),
            "evidence_summary": "Carrier photo on porch corroborated, but continuous video proves immediate theft by vehicle-following porch pirate.",
            "rationale": (
                "Continuous security video and timely police report demonstrate predatory porch theft before customer could secure possession."
            ),
        },
        {
            "id_num": 64,
            "case_type": "conflicting_signals",
            "category": "unauthorized_transaction",
            "label": "APPROVE",
            "amount": 890.00,
            "merchant": "Global Electronics Direct",
            "merchant_category": "electronics",
            "tenure": 40,
            "prior_disputes": 0,
            "delay_days": 3,
            "evidence_signals": [
                "3ds_otp_entered_successfully",
                "carrier_documented_sim_swap_fraud",
            ],
            "case_description": (
                "Cardholder disputes $890.00 for smartphone hardware from Global Electronics Direct. The transaction records show SMS OTP "
                "was entered correctly for 3D Secure authentication. However, mobile carrier logs confirm that the cardholder's mobile number "
                "was hijacked via an unauthorized SIM swap by identity thieves three hours before the transaction, and the phone was inactive locally."
            ),
            "evidence_summary": "SMS OTP matched, but carrier confirms unauthorized SIM swap occurred 3 hours prior.",
            "rationale": (
                "Carrier-verified SIM swap fraud invalidates SMS-based 3DS authentication as proof of cardholder authorization."
            ),
        },
        {
            "id_num": 65,
            "case_type": "conflicting_signals",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 650.00,
            "merchant": "Artisan Stone Creations",
            "merchant_category": "home_decor",
            "tenure": 25,
            "prior_disputes": 0,
            "delay_days": 14,
            "evidence_signals": [
                "signed_delivery_slip",
                "contractor_forgery_affidavit",
            ],
            "case_description": (
                "Customer disputes a $650.00 stone garden statue from Artisan Stone Creations. Merchant provided a delivery receipt with a "
                "signature scrawled in the cardholder's name. The cardholder provided a notarized affidavit and flight records showing they were "
                "traveling overseas, along with written admission from a third-party roofing contractor that their worker signed the slip and "
                "subsequently damaged/misplaced the item."
            ),
            "evidence_summary": "Signed delivery receipt contradicted by verified overseas travel and roofing contractor admission of signature.",
            "rationale": (
                "Definitive proof that signature was executed by an unauthorized third party without cardholder knowledge negates delivery proof."
            ),
        },
        {
            "id_num": 66,
            "case_type": "conflicting_signals",
            "category": "unauthorized_transaction",
            "label": "APPROVE",
            "amount": 540.00,
            "merchant": "Digital Creative Software",
            "merchant_category": "software",
            "tenure": 18,
            "prior_disputes": 0,
            "delay_days": 4,
            "evidence_signals": [
                "authenticated_login_session",
                "credential_stuffing_vpn_proxy",
            ],
            "case_description": (
                "Customer disputes $540.00 for annual enterprise software licenses from Digital Creative Software. Merchant logs show the "
                "purchase occurred from the cardholder's registered user account. However, cybersecurity incident audit logs indicate the account "
                "was accessed via a credential stuffing attack from a known Tor exit node with an automated API tool, and the billing credit card "
                "CVV was entered incorrectly four times before succeeding."
            ),
            "evidence_summary": "Account login matched, but security audit proves automated credential stuffing from Tor exit node.",
            "rationale": (
                "Evidence of automated credential stuffing and repeated CVV guessing confirms account takeover despite login success."
            ),
        },
        {
            "id_num": 67,
            "case_type": "conflicting_signals",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 310.00,
            "merchant": "EuroCraft Handbags",
            "merchant_category": "apparel",
            "tenure": 29,
            "prior_disputes": 0,
            "delay_days": 20,
            "evidence_signals": [
                "customs_seizure_order",
                "merchant_shipping_manifest_valid",
            ],
            "case_description": (
                "Customer disputes $310.00 for leather goods from EuroCraft Handbags. Merchant produced a valid international postal tracking "
                "number showing export from Italy. However, US Customs and Border Protection provided an official Notice of Seizure and Destruction "
                "indicating the merchandise was confiscated and destroyed at the border due to the merchant's failure to provide mandatory CITES "
                "wildlife material documentation."
            ),
            "evidence_summary": "Valid shipping manifest, but parcel was seized and destroyed by customs due to merchant documentation omission.",
            "rationale": (
                "Confiscation resulting from merchant failure to comply with import documentation requirements constitutes failure of delivery."
            ),
        },
        {
            "id_num": 68,
            "case_type": "conflicting_signals",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 475.00,
            "merchant": "Island Paradise Resort",
            "merchant_category": "travel",
            "tenure": 36,
            "prior_disputes": 0,
            "delay_days": 12,
            "evidence_signals": [
                "non_refundable_booking_policy",
                "state_emergency_evacuation",
            ],
            "case_description": (
                "Customer disputes a $475.00 non-refundable beachfront villa booking with Island Paradise Resort. The hotel claims strict "
                "non-refundable policy terms apply. However, the Governor declared a mandatory State of Emergency and forced civilian evacuation "
                "of the island due to a Category 4 hurricane during the booking window, legally prohibiting public access to the property."
            ),
            "evidence_summary": "Non-refundable terms contradicted by mandatory government hurricane evacuation order prohibiting entry.",
            "rationale": (
                "Frustration of contract by government-mandated emergency evacuation supersedes private non-refundable terms under doctrine of impossibility."
            ),
        },
        {
            "id_num": 69,
            "case_type": "conflicting_signals",
            "category": "unauthorized_transaction",
            "label": "APPROVE",
            "amount": 295.00,
            "merchant": "KidPlay Interactive Games",
            "merchant_category": "gaming",
            "tenure": 24,
            "prior_disputes": 0,
            "delay_days": 6,
            "evidence_signals": [
                "registered_family_tablet",
                "os_parental_gate_bypass_vulnerability",
            ],
            "case_description": (
                "Customer disputes $295.00 of in-app purchases on KidPlay Interactive Games made on a child's tablet. Merchant claims purchases "
                "are non-refundable digital goods. However, the manufacturer issued a security bulletin acknowledging a known iOS biometric "
                "authentication bypass vulnerability that allowed in-app purchases without parental password verification, which the customer "
                "had actively configured."
            ),
            "evidence_summary": "Family device usage, but OS manufacturer confirmed security flaw that bypassed active parental password gate.",
            "rationale": (
                "Documented operating system security flaw bypassing active cardholder parental controls relieves cardholder of liability."
            ),
        },
        {
            "id_num": 70,
            "case_type": "conflicting_signals",
            "category": "unauthorized_transaction",
            "label": "APPROVE",
            "amount": 340.00,
            "merchant": "QuickGas Automated Pump",
            "merchant_category": "fuel",
            "tenure": 45,
            "prior_disputes": 0,
            "delay_days": 4,
            "evidence_signals": [
                "magnetic_stripe_swipe_match",
                "corroborated_pos_skimmer_seizure",
            ],
            "case_description": (
                "Cardholder disputes $340.00 in fuel purchases at QuickGas Automated Pump in Florida. Terminal logs show magnetic stripe data "
                "was swiped. Superficially, stripe matches can suggest cardholder presence. However, state regulators announced the discovery "
                "of Bluetooth skimming hardware at that exact station, and cardholder's corporate employer verified physical attendance at a "
                "conference in Chicago at that exact hour."
            ),
            "evidence_summary": "Magstripe read on terminal, but physical skimmer seized by police and cardholder verified in Chicago.",
            "rationale": (
                "Magnetic stripe data clone by seized Bluetooth skimmer combined with ironclad employer presence alibi proves counterfeit card fraud."
            ),
        },
        # --- Conflicting Signals REJECT (10) ---
        {
            "id_num": 71,
            "case_type": "conflicting_signals",
            "category": "unauthorized_transaction",
            "label": "REJECT",
            "amount": 1400.00,
            "merchant": "Haute Couture Paris",
            "merchant_category": "luxury_goods",
            "tenure": 22,
            "prior_disputes": 1,
            "delay_days": 14,
            "evidence_signals": [
                "unrecognized_shipping_address",
                "biometric_faceid_apple_pay_match",
            ],
            "case_description": (
                "Cardholder disputes a $1,400.00 designer handbag purchase from Haute Couture Paris, claiming fraud because shipping went to an "
                "address in another city. However, payment network records prove the purchase was authorized via Apple Pay on the cardholder's "
                "primary registered iPhone using Face ID biometrics. Further investigation reveals the shipping recipient is the cardholder's "
                "verified biological sibling."
            ),
            "evidence_summary": "Unrecognized shipping destination, but transaction authenticated via primary iPhone Face ID and sent to sibling.",
            "rationale": (
                "Biometric device tokenization on primary registered handset coupled with family recipient relationship negates unauthorized claim."
            ),
        },
        {
            "id_num": 72,
            "case_type": "conflicting_signals",
            "category": "goods_not_received",
            "label": "REJECT",
            "amount": 450.00,
            "merchant": "Apex Outerwear Direct",
            "merchant_category": "apparel",
            "tenure": 16,
            "prior_disputes": 1,
            "delay_days": 18,
            "evidence_signals": [
                "carrier_tracking_missing_signature",
                "public_social_media_photo_wearing_item",
            ],
            "case_description": (
                "Customer claims a $450.00 winter parka from Apex Outerwear Direct was never delivered, citing carrier tracking that lacks a "
                "recipient signature. However, the merchant submitted public social media posts from the customer's verified Instagram account "
                "dated three days after the delivery scan, showing the customer wearing the exact custom-embroidered parka with visible order tags."
            ),
            "evidence_summary": "Carrier lacked signature, but public social media post shows cardholder wearing unique custom parka.",
            "rationale": (
                "Public visual proof of possession of the exact custom merchandise conclusively refutes non-delivery claim."
            ),
        },
        {
            "id_num": 73,
            "case_type": "conflicting_signals",
            "category": "subscription_cancellation",
            "label": "REJECT",
            "amount": 65.00,
            "merchant": "CloudData Pro Storage",
            "merchant_category": "saas",
            "tenure": 20,
            "prior_disputes": 0,
            "delay_days": 10,
            "evidence_signals": [
                "forged_cancellation_email",
                "mail_server_dkim_failure",
            ],
            "case_description": (
                "Customer disputes $65.00 recurring billing from CloudData Pro Storage, submitting a forwarded email purporting to confirm "
                "prior cancellation. Email server header forensics reveal the forwarded confirmation was fabricated: DKIM signatures failed, "
                "the originating IP was not the merchant's server, and the merchant's database shows no such cancellation record."
            ),
            "evidence_summary": "Customer submitted cancellation email, but forensic analysis proves email headers were fabricated.",
            "rationale": (
                "Fabrication of evidence by the cardholder invalidates dispute rights and confirms legitimate billing."
            ),
        },
        {
            "id_num": 74,
            "case_type": "conflicting_signals",
            "category": "defective_or_not_as_described",
            "label": "REJECT",
            "amount": 2500.00,
            "merchant": "Vanguard Advisory Partners",
            "merchant_category": "professional_services",
            "tenure": 30,
            "prior_disputes": 0,
            "delay_days": 25,
            "evidence_signals": [
                "alleged_non_delivery_of_services",
                "signed_milestone_acceptance_delivered",
            ],
            "case_description": (
                "Corporate cardholder disputes a $2,500.00 strategic consulting fee from Vanguard Advisory Partners alleging 'services not "
                "rendered'. In defense, the merchant provided a comprehensive 60-page financial audit deliverable, recorded video transcripts "
                "of three strategy presentation sessions, and a formal milestone acceptance agreement signed by the cardholder."
            ),
            "evidence_summary": "Cardholder claimed non-performance, but merchant produced signed milestone sign-off and delivered audit deck.",
            "rationale": (
                "Executed milestone acceptance and documented presentation delivery confirm full contractual fulfillment of professional services."
            ),
        },
        {
            "id_num": 75,
            "case_type": "conflicting_signals",
            "category": "unauthorized_transaction",
            "label": "REJECT",
            "amount": 180.00,
            "merchant": "Cornerstone Market & Wine",
            "merchant_category": "grocery",
            "tenure": 38,
            "prior_disputes": 0,
            "delay_days": 8,
            "evidence_signals": [
                "customer_abroad_alibi",
                "in_person_pos_chip_near_home",
            ],
            "case_description": (
                "Customer claims $180.00 at Cornerstone Market & Wine was unauthorized because they were vacationing in France. "
                "While customer did travel to France, flight manifests prove they returned to the US two days prior to this transaction. "
                "The transaction was a chip contact insertion at a store 1.5 miles from their home, with loyalty card points applied."
            ),
            "evidence_summary": "Customer asserted European travel alibi, but customs records prove return 48 hours prior; local chip insertion.",
            "rationale": (
                "Customs return records disprove the travel alibi; local physical EMV chip insertion and loyalty card use confirm cardholder presence."
            ),
        },
        {
            "id_num": 76,
            "case_type": "conflicting_signals",
            "category": "subscription_cancellation",
            "label": "REJECT",
            "amount": 120.00,
            "merchant": "Equinox Wellness Clubs",
            "merchant_category": "fitness",
            "tenure": 44,
            "prior_disputes": 0,
            "delay_days": 12,
            "evidence_signals": [
                "claimed_relocation_cancellation",
                "biometric_gym_turnstile_checkins",
            ],
            "case_description": (
                "Customer disputes a $120.00 health club charge from Equinox Wellness Clubs, claiming they moved to another city and canceled. "
                "However, biometric turnstile access logs prove the cardholder checked into the primary local club facility nine times during "
                "the disputed billing month, including two days before the dispute was filed."
            ),
            "evidence_summary": "Customer asserted relocation, but biometric turnstile records prove 9 active visits during disputed month.",
            "rationale": (
                "Demonstrated ongoing physical facility utilization directly contradicts assertions of prior cancellation and relocation."
            ),
        },
        {
            "id_num": 77,
            "case_type": "conflicting_signals",
            "category": "defective_or_not_as_described",
            "label": "REJECT",
            "amount": 620.00,
            "merchant": "Visionary Home Appliances",
            "merchant_category": "appliances",
            "tenure": 19,
            "prior_disputes": 0,
            "delay_days": 18,
            "evidence_signals": [
                "claimed_shattered_on_arrival",
                "online_warranty_registration_subsequent",
            ],
            "case_description": (
                "Customer disputes $620.00 for a robotic vacuum cleaner from Visionary Home Appliances claiming it arrived broken. "
                "Merchant records show the customer registered the product warranty online two weeks after delivery, paired the unit "
                "with their home Wi-Fi, and completed 42 automated cleaning cycles according to cloud telemetry."
            ),
            "evidence_summary": "Customer claimed broken on arrival, but telemetry proves warranty registration and 42 completed cleaning runs.",
            "rationale": (
                "Extensive cloud operational telemetry and warranty registration disprove claims of total delivery defect."
            ),
        },
        {
            "id_num": 78,
            "case_type": "conflicting_signals",
            "category": "unauthorized_transaction",
            "label": "REJECT",
            "amount": 260.00,
            "merchant": "TechEdge Hardware Store",
            "merchant_category": "electronics",
            "tenure": 15,
            "prior_disputes": 2,
            "delay_days": 14,
            "evidence_signals": [
                "unrecognized_charge_claim",
                "household_dependent_match",
            ],
            "case_description": (
                "Cardholder claims unfamiliarity with a $260.00 computer monitor purchase at TechEdge Hardware Store. "
                "Merchant delivery telemetry confirms shipping was addressed to the cardholder's residential address with recipient name "
                "matching the cardholder's adult college student child who shares the residence and uses the shared family home Wi-Fi network."
            ),
            "evidence_summary": "Order delivered to cardholder address and accepted by dependent residing in home.",
            "rationale": (
                "Cardholder permissive access and domestic household receipt without reported card theft is treated as friendly fraud under network rules."
            ),
        },
        {
            "id_num": 79,
            "case_type": "conflicting_signals",
            "category": "duplicate_billing",
            "label": "REJECT",
            "amount": 210.00,
            "merchant": "Prime Steakhouse & Bar",
            "merchant_category": "restaurant",
            "tenure": 27,
            "prior_disputes": 0,
            "delay_days": 9,
            "evidence_signals": [
                "claimed_overcharge_tip_fraud",
                "signed_itemized_receipt_with_tip",
            ],
            "case_description": (
                "Customer claims a $210.00 restaurant bill at Prime Steakhouse & Bar represents a duplicate charge or unauthorized gratuity "
                "over an agreed $170.00 dinner. The merchant provided the original signed credit card merchant copy clearly displaying "
                "a $170.00 food subtotal, $40.00 handwritten tip, and $210.00 final total written in the cardholder's consistent handwriting."
            ),
            "evidence_summary": "Original signed merchant receipt proves customer authorized $170 meal plus $40 gratuity for $210 total.",
            "rationale": (
                "Merchant production of the signed receipt with clearly legibly handwritten gratuity and matching final total refutes overcharge claim."
            ),
        },
        {
            "id_num": 80,
            "case_type": "conflicting_signals",
            "category": "goods_not_received",
            "label": "REJECT",
            "amount": 950.00,
            "merchant": "Fintech Summit International",
            "merchant_category": "events",
            "tenure": 35,
            "prior_disputes": 0,
            "delay_days": 16,
            "evidence_signals": [
                "claimed_event_cancellation",
                "conference_badge_badging_attendance",
            ],
            "case_description": (
                "Customer disputes $950.00 for a delegate pass to Fintech Summit International, asserting the conference was canceled. "
                "The event organizer submitted official conference proceedings proving the event was held successfully, alongside RFID badge "
                "scanner logs showing the cardholder's badge checked into eight keynote sessions over two days."
            ),
            "evidence_summary": "Cardholder claimed event was canceled, but RFID badge records prove cardholder attended keynotes.",
            "rationale": (
                "Conclusive proof of physical attendance and participation completely disproves false non-provision claims."
            ),
        },
        # =========================================================================
        # 5. CONTRASTIVE PAIRS (20 examples: 10 pairs = 10 APPROVE, 10 REJECT)
        # =========================================================================
        # --- Pair 1: Filing Timeline (Statutory 60-day Window) ---
        {
            "id_num": 81,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_01_filing_timeline",
            "category": "unauthorized_transaction",
            "label": "APPROVE",
            "amount": 350.00,
            "merchant": "Nova Gadgets Online",
            "merchant_category": "electronics",
            "tenure": 24,
            "prior_disputes": 0,
            "delay_days": 59,
            "evidence_signals": [
                "filing_day_59",
                "unauthorized_cnp_claim",
                "clean_record",
            ],
            "case_description": (
                "Cardholder with zero past disputes filed a claim regarding an unauthorized online transaction of $350.00 at Nova Gadgets Online. "
                "The dispute was submitted exactly 59 calendar days after the account statement containing the transaction was generated. "
                "The transaction was a card-not-present charge from an unrecognized domestic IP without 3DS authentication."
            ),
            "evidence_summary": "Dispute filed 59 days post-statement, within statutory 60-day window on unauthenticated CNP charge.",
            "rationale": (
                "Submission on day 59 satisfies strict statutory 60-day filing limits; unauthenticated card-not-present nature warrants approval."
            ),
        },
        {
            "id_num": 82,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_01_filing_timeline",
            "category": "unauthorized_transaction",
            "label": "REJECT",
            "amount": 350.00,
            "merchant": "Nova Gadgets Online",
            "merchant_category": "electronics",
            "tenure": 24,
            "prior_disputes": 0,
            "delay_days": 61,
            "evidence_signals": ["filing_day_61", "exceeded_statutory_window_by_24h"],
            "case_description": (
                "Cardholder with zero past disputes filed a claim regarding an unauthorized online transaction of $350.00 at Nova Gadgets Online. "
                "The dispute was submitted exactly 61 calendar days after the account statement containing the transaction was generated. "
                "The transaction was a card-not-present charge from an unrecognized domestic IP without 3DS authentication."
            ),
            "evidence_summary": "Dispute filed 61 days post-statement, missing statutory 60-day deadline by 24 hours.",
            "rationale": (
                "Submission on day 61 breaches statutory 60-day filing limits, extinguishing chargeback rights despite valid underlying fraud grounds."
            ),
        },
        # --- Pair 2: Return Authorization & Tracking Evidence ---
        {
            "id_num": 83,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_02_return_tracking",
            "category": "defective_or_not_as_described",
            "label": "APPROVE",
            "amount": 280.00,
            "merchant": "Artisan Coffee Roasters",
            "merchant_category": "appliances",
            "tenure": 18,
            "prior_disputes": 0,
            "delay_days": 15,
            "evidence_signals": [
                "merchant_rma_issued",
                "carrier_proof_of_delivery_to_warehouse",
            ],
            "case_description": (
                "Customer disputes $280.00 for a defective espresso grinder from Artisan Coffee Roasters. The customer obtained a formal Return "
                "Merchandise Authorization (RMA #8491) from merchant support and shipped the grinder back via FedEx. Carrier records prove "
                "delivery to the merchant's dock signed by warehouse staff; merchant failed to credit customer after 14 business days."
            ),
            "evidence_summary": "Item returned with authorized RMA and carrier proof of delivery to warehouse dock; refund withheld.",
            "rationale": (
                "Customer strictly complied with merchant RMA return protocols with verified warehouse delivery; failure to refund warrants approval."
            ),
        },
        {
            "id_num": 84,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_02_return_tracking",
            "category": "defective_or_not_as_described",
            "label": "REJECT",
            "amount": 280.00,
            "merchant": "Artisan Coffee Roasters",
            "merchant_category": "appliances",
            "tenure": 18,
            "prior_disputes": 0,
            "delay_days": 15,
            "evidence_signals": [
                "no_rma_authorization",
                "unverified_carrier_tracking_lost",
            ],
            "case_description": (
                "Customer disputes $280.00 for a defective espresso grinder from Artisan Coffee Roasters. The customer shipped the item back "
                "without requesting an RMA or notifying the merchant, dropped it at an unstaffed parcel drop box with no tracking receipt, and "
                "the merchant has no record or physical intake of the grinder."
            ),
            "evidence_summary": "Item shipped back without RMA authorization or carrier tracking receipt; merchant never received unit.",
            "rationale": (
                "Returning merchandise without required return authorization or verifiable carrier delivery proof fails cardholder return burden."
            ),
        },
        # --- Pair 3: 3D Secure / Biometric Authentication Quality ---
        {
            "id_num": 85,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_03_3ds_authentication",
            "category": "unauthorized_transaction",
            "label": "APPROVE",
            "amount": 450.00,
            "merchant": "Omni Fashion Store",
            "merchant_category": "apparel",
            "tenure": 30,
            "prior_disputes": 0,
            "delay_days": 5,
            "evidence_signals": [
                "no_3ds_verification",
                "unrecognized_device_fingerprint",
            ],
            "case_description": (
                "Customer disputes a $450.00 online clothing purchase at Omni Fashion Store claiming unauthorized use. "
                "The transaction was processed as standard card-not-present without 3D Secure protocol, without biometric challenge, and from "
                "an unrecognized device fingerprint and operating system."
            ),
            "evidence_summary": "Card-not-present transaction processed with zero 3D Secure or biometric authentication verification.",
            "rationale": (
                "Liability remains with merchant on unauthenticated card-not-present transactions lacking 3DS protocol protection."
            ),
        },
        {
            "id_num": 86,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_03_3ds_authentication",
            "category": "unauthorized_transaction",
            "label": "REJECT",
            "amount": 450.00,
            "merchant": "Omni Fashion Store",
            "merchant_category": "apparel",
            "tenure": 30,
            "prior_disputes": 0,
            "delay_days": 5,
            "evidence_signals": [
                "3ds_v2_frictionless_biometric_match",
                "known_trusted_device",
            ],
            "case_description": (
                "Customer disputes a $450.00 online clothing purchase at Omni Fashion Store claiming unauthorized use. "
                "The transaction was fully authenticated via 3D Secure Version 2.2 with hardware token biometrics on the cardholder's "
                "regular personal smartphone and validated trusted Wi-Fi MAC address."
            ),
            "evidence_summary": "Transaction authenticated via 3DS v2.2 with biometric match on customer primary trusted smartphone.",
            "rationale": (
                "Full cryptographic 3DS v2.2 biometric authentication on a registered trusted device transfers liability away from merchant."
            ),
        },
        # --- Pair 4: Delivery Location Specificity ---
        {
            "id_num": 87,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_04_delivery_location",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 190.00,
            "merchant": "HomeComfort Furnishings",
            "merchant_category": "furniture",
            "tenure": 14,
            "prior_disputes": 0,
            "delay_days": 10,
            "evidence_signals": ["carrier_photo_mismatch", "wrong_porch_structure"],
            "case_description": (
                "Customer disputes $190.00 for bed linens from HomeComfort Furnishings. Carrier tracking records delivery with a drop-off photo "
                "showing a package sitting on a red brick porch with black metal railings. Customer's registered billing house has wooden "
                "siding, white porch pillars, and zero brickwork, demonstrating delivery to the wrong house on the block."
            ),
            "evidence_summary": "Carrier delivery photo depicts a red brick porch; customer house is wooden siding with white pillars.",
            "rationale": (
                "Photographic proof of delivery to an architecturally distinct incorrect address proves carrier misdelivery."
            ),
        },
        {
            "id_num": 88,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_04_delivery_location",
            "category": "goods_not_received",
            "label": "REJECT",
            "amount": 190.00,
            "merchant": "HomeComfort Furnishings",
            "merchant_category": "furniture",
            "tenure": 14,
            "prior_disputes": 0,
            "delay_days": 10,
            "evidence_signals": [
                "carrier_photo_exact_match",
                "house_number_clearly_visible",
            ],
            "case_description": (
                "Customer disputes $190.00 for bed linens from HomeComfort Furnishings. Carrier tracking records delivery with a drop-off photo "
                "showing the package on the front porch with wooden siding, white porch pillars, and the customer's house number '742' "
                "clearly visible in the camera frame."
            ),
            "evidence_summary": "Carrier delivery photo clearly captures cardholder house number and matching porch architecture.",
            "rationale": (
                "Unambiguous photographic confirmation showing parcel on cardholder porch with matching house number establishes delivery."
            ),
        },
        # --- Pair 5: Subscription Cancellation Timestamp Timing ---
        {
            "id_num": 89,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_05_cancellation_timing",
            "category": "subscription_cancellation",
            "label": "APPROVE",
            "amount": 55.00,
            "merchant": "CloudSync Enterprise",
            "merchant_category": "saas",
            "tenure": 22,
            "prior_disputes": 0,
            "delay_days": 4,
            "evidence_signals": [
                "canceled_25_minutes_prior_to_renewal",
                "system_billing_race_condition",
            ],
            "case_description": (
                "Customer was charged $55.00 for recurring cloud sync software on November 1st at 12:05 AM. Merchant account audit logs "
                "confirm the customer submitted their cancellation at 11:40 PM on October 31st, 25 minutes prior to the midnight billing cycle. "
                "An automated batch processing race condition executed the charge despite the pending cancellation queue."
            ),
            "evidence_summary": "Cancellation submitted 25 minutes prior to renewal billing cycle; charged due to batch race condition.",
            "rationale": (
                "Cancellation timestamp preceded billing cycle renewal; merchant technical execution lag does not justify unauthorized renewal."
            ),
        },
        {
            "id_num": 90,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_05_cancellation_timing",
            "category": "subscription_cancellation",
            "label": "REJECT",
            "amount": 55.00,
            "merchant": "CloudSync Enterprise",
            "merchant_category": "saas",
            "tenure": 22,
            "prior_disputes": 0,
            "delay_days": 4,
            "evidence_signals": [
                "canceled_45_minutes_after_renewal",
                "valid_renewal_charge",
            ],
            "case_description": (
                "Customer was charged $55.00 for recurring cloud sync software on November 1st at 12:05 AM. Merchant account audit logs "
                "confirm the customer submitted their cancellation at 12:50 AM on November 1st, 45 minutes after the monthly renewal had "
                "successfully executed and renewed the service period."
            ),
            "evidence_summary": "Cancellation submitted 45 minutes after scheduled renewal charge executed.",
            "rationale": (
                "Cancellation initiated after valid contract renewal has completed applies to the subsequent cycle, not the current one."
            ),
        },
        # --- Pair 6: Duplicate Charge vs Multiple Distinct Orders ---
        {
            "id_num": 91,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_06_duplicate_vs_distinct",
            "category": "duplicate_billing",
            "label": "APPROVE",
            "amount": 89.99,
            "merchant": "Beacon Bookshop",
            "merchant_category": "retail",
            "tenure": 36,
            "prior_disputes": 0,
            "delay_days": 3,
            "evidence_signals": [
                "identical_auth_split_second",
                "single_shipping_label",
            ],
            "case_description": (
                "Customer statement displays two identical charges of $89.99 at Beacon Bookshop within two seconds of each other. "
                "Merchant inventory records show only a single order ID (#BK-4921) with one single shipping tracking number created and "
                "dispatched to the cardholder."
            ),
            "evidence_summary": "Two identical authorizations within 2 seconds; merchant records reflect only single order fulfillment.",
            "rationale": (
                "Systemic double-authorization confirmed by identical split-second transactions and single product shipment."
            ),
        },
        {
            "id_num": 92,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_06_duplicate_vs_distinct",
            "category": "duplicate_billing",
            "label": "REJECT",
            "amount": 89.99,
            "merchant": "Beacon Bookshop",
            "merchant_category": "retail",
            "tenure": 36,
            "prior_disputes": 0,
            "delay_days": 3,
            "evidence_signals": [
                "two_distinct_order_ids",
                "two_distinct_parcels_delivered",
            ],
            "case_description": (
                "Customer statement displays two identical charges of $89.99 at Beacon Bookshop separated by 25 minutes. "
                "Merchant records show the customer placed two distinct orders: order #BK-4921 for history books and order #BK-4935 for art "
                "catalogues, resulting in two separate carrier parcels delivered to the customer."
            ),
            "evidence_summary": "Two separate orders placed 25 minutes apart with distinct items and tracking numbers.",
            "rationale": (
                "Independent checkout actions resulting in separate distinct merchandise shipments disprove duplicate billing."
            ),
        },
        # --- Pair 7: Digital Content License Activation ---
        {
            "id_num": 93,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_07_digital_activation",
            "category": "goods_not_received",
            "label": "APPROVE",
            "amount": 150.00,
            "merchant": "VFX Studio Tools",
            "merchant_category": "software",
            "tenure": 16,
            "prior_disputes": 0,
            "delay_days": 2,
            "evidence_signals": ["license_never_activated", "zero_software_downloads"],
            "case_description": (
                "Customer accidentally purchased a $150.00 specialized video rendering plugin from VFX Studio Tools. Within three hours of "
                "purchase, the customer contacted merchant support requesting cancellation. Merchant server logs confirm the product license "
                "key was never activated, zero software downloads occurred, and the key was revoked."
            ),
            "evidence_summary": "License key was never activated, zero downloads occurred, key formally revoked.",
            "rationale": (
                "Prompt accidental purchase cancellation where digital license key was never activated or downloaded justifies refund."
            ),
        },
        {
            "id_num": 94,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_07_digital_activation",
            "category": "goods_not_received",
            "label": "REJECT",
            "amount": 150.00,
            "merchant": "VFX Studio Tools",
            "merchant_category": "software",
            "tenure": 16,
            "prior_disputes": 0,
            "delay_days": 14,
            "evidence_signals": [
                "license_activated_device_id",
                "18_hours_active_plugin_usage",
            ],
            "case_description": (
                "Customer claims a $150.00 specialized video rendering plugin from VFX Studio Tools was purchased by mistake. "
                "Merchant license activation server records demonstrate the license key was activated on the customer's workstation within "
                "ten minutes of purchase, with telemetry tracking eighteen hours of active plugin rendering across two weeks."
            ),
            "evidence_summary": "License activated on workstation with 18 hours of recorded rendering use before dispute.",
            "rationale": (
                "Active cryptographic license activation and extensive verified software execution preclude claims of accidental purchase."
            ),
        },
        # --- Pair 8: Locksmith Service Work Order Documentation ---
        {
            "id_num": 95,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_08_service_work_order",
            "category": "defective_or_not_as_described",
            "label": "APPROVE",
            "amount": 320.00,
            "merchant": "Metro Locksmith 24/7",
            "merchant_category": "home_services",
            "tenure": 28,
            "prior_disputes": 0,
            "delay_days": 8,
            "evidence_signals": [
                "blank_unsigned_work_order",
                "zero_itemization_or_photos",
            ],
            "case_description": (
                "Customer disputes a $320.00 emergency lockout charge from Metro Locksmith 24/7 alleging the technician damaged the door lock "
                "mechanism and failed to install the agreed deadbolt. The merchant provided only a generic debit credit terminal slip without "
                "a work order, customer signature, invoice breakdown, or photographic proof of completed work."
            ),
            "evidence_summary": "Merchant provided generic credit terminal receipt lacking work order, customer signature, or parts itemization.",
            "rationale": (
                "Service merchants must document completed work with signed work orders or itemized contracts; omission warrants approval."
            ),
        },
        {
            "id_num": 96,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_08_service_work_order",
            "category": "defective_or_not_as_described",
            "label": "REJECT",
            "amount": 320.00,
            "merchant": "Metro Locksmith 24/7",
            "merchant_category": "home_services",
            "tenure": 28,
            "prior_disputes": 0,
            "delay_days": 8,
            "evidence_signals": [
                "signed_work_order_with_satisfaction",
                "timestamped_photos_of_new_lock",
            ],
            "case_description": (
                "Customer disputes a $320.00 emergency lockout charge from Metro Locksmith 24/7 alleging the lock was damaged. "
                "The merchant submitted an itemized work order signed by the customer stating 'Work completed satisfactorily and deadbolt "
                "tested', accompanied by timestamped digital photos of the installed, functional lock mechanism."
            ),
            "evidence_summary": "Itemized work order signed by customer certifying satisfaction plus photos of installed deadbolt.",
            "rationale": (
                "Signed customer satisfaction certification and photographic proof of completed installation defeat post-service defect claim."
            ),
        },
        # --- Pair 9: Hotel Free Cancellation Policy Window ---
        {
            "id_num": 97,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_09_hotel_cancellation_window",
            "category": "credit_not_processed",
            "label": "APPROVE",
            "amount": 220.00,
            "merchant": "Grand Pacific Hotel",
            "merchant_category": "lodging",
            "tenure": 25,
            "prior_disputes": 0,
            "delay_days": 7,
            "evidence_signals": [
                "canceled_49h_prior",
                "exceeds_48h_policy_requirement",
            ],
            "case_description": (
                "Customer booked a hotel room at Grand Pacific Hotel for $220.00 with a documented policy allowing free cancellation up to "
                "48 hours before the 3:00 PM check-in date. The customer submitted their cancellation exactly 49 hours prior to check-in and "
                "received an automated cancellation acknowledgment. The hotel billed a one-night penalty anyway."
            ),
            "evidence_summary": "Cancellation submitted 49 hours prior to check-in, exceeding the 48-hour free cancellation requirement.",
            "rationale": (
                "Cancellation occurred within the permitted contractual free cancellation window; merchant improperly withheld penalty."
            ),
        },
        {
            "id_num": 98,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_09_hotel_cancellation_window",
            "category": "credit_not_processed",
            "label": "REJECT",
            "amount": 220.00,
            "merchant": "Grand Pacific Hotel",
            "merchant_category": "lodging",
            "tenure": 25,
            "prior_disputes": 0,
            "delay_days": 7,
            "evidence_signals": ["canceled_23h_prior", "missed_48h_policy_requirement"],
            "case_description": (
                "Customer booked a hotel room at Grand Pacific Hotel for $220.00 with a documented policy allowing free cancellation up to "
                "48 hours before the 3:00 PM check-in date. The customer submitted their cancellation 23 hours prior to check-in due to a "
                "personal schedule change. The hotel billed a one-night penalty in accordance with their terms."
            ),
            "evidence_summary": "Cancellation submitted 23 hours prior to check-in, missing the 48-hour free cancellation deadline.",
            "rationale": (
                "Cancellation missed the agreed contractual deadline; hotel penalty assessment strictly conforms to agreed reservation terms."
            ),
        },
        # --- Pair 10: Restocking Fee Disclosure Visibility ---
        {
            "id_num": 99,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_10_restocking_fee_disclosure",
            "category": "credit_not_processed",
            "label": "APPROVE",
            "amount": 75.00,
            "merchant": "Optima Displays Direct",
            "merchant_category": "electronics",
            "tenure": 21,
            "prior_disputes": 0,
            "delay_days": 12,
            "evidence_signals": [
                "restocking_fee_undisclosed_at_checkout",
                "surprise_deduction",
            ],
            "case_description": (
                "Customer returned an unopened, factory-sealed computer monitor to Optima Displays Direct. The merchant refunded the order "
                "minus a surprise $75.00 (15%) restocking fee. The customer provided checkout screen recordings and receipts demonstrating that "
                "no restocking fee policy was disclosed on the product page, checkout cart, or order confirmation terms."
            ),
            "evidence_summary": "Merchant deducted $75 restocking fee that was nowhere disclosed prior to or at time of sale.",
            "rationale": (
                "Payment network regulations strictly prohibit deduction of undisclosed fees from returned merchandise refunds."
            ),
        },
        {
            "id_num": 100,
            "case_type": "contrastive_pair",
            "contrastive_pair_id": "pair_10_restocking_fee_disclosure",
            "category": "credit_not_processed",
            "label": "REJECT",
            "amount": 75.00,
            "merchant": "Optima Displays Direct",
            "merchant_category": "electronics",
            "tenure": 21,
            "prior_disputes": 0,
            "delay_days": 12,
            "evidence_signals": [
                "restocking_fee_prominently_disclosed",
                "explicit_checkbox_acknowledged",
            ],
            "case_description": (
                "Customer returned an unopened, factory-sealed computer monitor to Optima Displays Direct. The merchant refunded the order "
                "minus a $75.00 (15%) restocking fee. The merchant provided checkout logs proving the customer checked an explicit mandatory "
                "box stating 'I acknowledge that returns of computer monitors are subject to a 15% restocking fee' before payment."
            ),
            "evidence_summary": "Cardholder explicitly checked mandatory disclosure acknowledging 15% restocking fee at checkout.",
            "rationale": (
                "Explicit advance agreement to clearly disclosed return restocking fees makes the deduction contractually valid."
            ),
        },
    ]

    return cases


def generate_dataset() -> dict[str, Any]:
    """Generate the complete 100-example reference dataset with full metadata and template variants."""
    raw_cases = get_raw_seed_data()
    if len(raw_cases) != 100:
        raise ValueError(f"Expected exactly 100 cases, found {len(raw_cases)}")

    examples: list[dict[str, Any]] = []
    approve_count = 0
    reject_count = 0

    for item in raw_cases:
        id_num = item["id_num"]
        example_id = f"ref-example-{id_num:03d}"
        label = item["label"]
        if label == "APPROVE":
            approve_count += 1
        elif label == "REJECT":
            reject_count += 1
        else:
            raise ValueError(f"Invalid label {label} in item {id_num}")

        amount = item["amount"]
        tenure = item["tenure"]
        prior_disputes = item["prior_disputes"]
        delay_days = item["delay_days"]
        merchant = item["merchant"]
        merchant_category = item["merchant_category"]
        evidence_summary = item["evidence_summary"]

        raw_template = build_raw_template(
            amount=amount,
            tenure=tenure,
            prior_disputes=prior_disputes,
            delay_days=delay_days,
            merchant=merchant,
            category=merchant_category,
            evidence_summary=evidence_summary,
        )

        ratio_template = build_ratio_template(
            amount=amount,
            tenure=tenure,
            prior_disputes=prior_disputes,
            delay_days=delay_days,
            merchant=merchant,
            category=merchant_category,
            evidence_summary=evidence_summary,
        )

        ratio_features = compute_ratio_features(
            amount=amount,
            tenure=tenure,
            prior_disputes=prior_disputes,
            delay_days=delay_days,
        )

        metadata: dict[str, Any] = {
            "category": item["category"],
            "case_type": item["case_type"],
            "contrastive_pair_id": item.get("contrastive_pair_id"),
            "transaction_id": f"tx_syn_{id_num:04d}",
            "transaction_amount": amount,
            "currency": "USD",
            "merchant": merchant,
            "merchant_category": merchant_category,
            "customer_tenure_months": tenure,
            "prior_dispute_count": prior_disputes,
            "dispute_filing_delay_days": delay_days,
            "evidence_signals": item["evidence_signals"],
            "raw_value_description": raw_template,
            "ratio_verbalized_description": ratio_template,
            "ratio_features": ratio_features,
        }

        example = {
            "example_id": example_id,
            "tenant_id": TENANT_ID,
            "agent_id": AGENT_ID,
            "case_description": item["case_description"],
            "label": label,
            "source": SOURCE,
            "status": STATUS,
            "metadata": metadata,
            "rationale": item["rationale"],
        }
        examples.append(example)

    dataset: dict[str, Any] = {
        "dataset_name": "arda_reliability_synthetic_seed_v1",
        "dataset_version": "1.0.0",
        "description": (
            "Synthetic seed dataset of 100 reference examples for ARDA Reliability Service "
            "embedding model evaluation, vector retrieval, and reliability scoring."
        ),
        "decision_domain": "Financial Transaction Dispute & Refund Adjudication",
        "tenant_id": TENANT_ID,
        "agent_id": AGENT_ID,
        "total_examples": len(examples),
        "label_distribution": {
            "APPROVE": approve_count,
            "REJECT": reject_count,
        },
        "case_type_distribution": {
            "straightforward": sum(
                1 for e in examples if e["metadata"]["case_type"] == "straightforward"
            ),
            "borderline": sum(
                1 for e in examples if e["metadata"]["case_type"] == "borderline"
            ),
            "incomplete_evidence": sum(
                1
                for e in examples
                if e["metadata"]["case_type"] == "incomplete_evidence"
            ),
            "conflicting_signals": sum(
                1
                for e in examples
                if e["metadata"]["case_type"] == "conflicting_signals"
            ),
            "contrastive_pair": sum(
                1 for e in examples if e["metadata"]["case_type"] == "contrastive_pair"
            ),
        },
        "examples": examples,
    }

    return dataset


def save_dataset(output_path: Path) -> None:
    """Generate and write the dataset JSON file to disk."""
    dataset = generate_dataset()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Dataset successfully generated and saved to: {output_path}")
    print(f"Total examples: {dataset['total_examples']}")
    print(f"Label distribution: {dataset['label_distribution']}")
    print(f"Case type distribution: {dataset['case_type_distribution']}")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    target_file = current_dir / "synthetic_seed_examples.json"
    save_dataset(target_file)
