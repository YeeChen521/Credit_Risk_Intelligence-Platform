"""Pydantic v2 request and response models for the credit risk scoring API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


# ── Input: application ────────────────────────────────────────────────────────

class ApplicationData(BaseModel):
    """Raw application row from application_train.csv. SK_ID_CURR is the only required field."""

    model_config = ConfigDict(extra="ignore")

    SK_ID_CURR: int

    # Loan contract
    NAME_CONTRACT_TYPE: Optional[str] = None

    # Demographics
    CODE_GENDER: Optional[str] = None
    CNT_CHILDREN: Optional[float] = None
    CNT_FAM_MEMBERS: Optional[float] = None
    DAYS_BIRTH: Optional[float] = None
    DAYS_EMPLOYED: Optional[float] = None
    DAYS_REGISTRATION: Optional[float] = None
    DAYS_ID_PUBLISH: Optional[float] = None
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    NAME_INCOME_TYPE: Optional[str] = None
    OCCUPATION_TYPE: Optional[str] = None
    ORGANIZATION_TYPE: Optional[str] = None
    NAME_TYPE_SUITE: Optional[str] = None

    # Income and credit amounts
    AMT_INCOME_TOTAL: Optional[float] = None
    AMT_CREDIT: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None

    # Property ownership — raw Y/N strings; feature pipeline converts to Int8
    FLAG_OWN_CAR: Optional[str] = None
    FLAG_OWN_REALTY: Optional[str] = None
    OWN_CAR_AGE: Optional[float] = None
    NAME_HOUSING_TYPE: Optional[str] = None

    # Region and city mismatch flags
    REGION_POPULATION_RELATIVE: Optional[float] = None
    REGION_RATING_CLIENT: Optional[float] = None
    REGION_RATING_CLIENT_W_CITY: Optional[float] = None
    LIVE_CITY_NOT_WORK_CITY: Optional[float] = None
    LIVE_REGION_NOT_WORK_REGION: Optional[float] = None
    REG_CITY_NOT_LIVE_CITY: Optional[float] = None
    REG_CITY_NOT_WORK_CITY: Optional[float] = None
    REG_REGION_NOT_LIVE_REGION: Optional[float] = None
    REG_REGION_NOT_WORK_REGION: Optional[float] = None

    # Application process
    WEEKDAY_APPR_PROCESS_START: Optional[str] = None
    HOUR_APPR_PROCESS_START: Optional[float] = None

    # External scoring sources
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None

    # Contact flags (0/1 numeric in source CSV)
    FLAG_MOBIL: Optional[float] = None
    FLAG_EMP_PHONE: Optional[float] = None
    FLAG_WORK_PHONE: Optional[float] = None
    FLAG_CONT_MOBILE: Optional[float] = None
    FLAG_PHONE: Optional[float] = None
    FLAG_EMAIL: Optional[float] = None

    # Social circle delinquency
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = None

    # Document flags (0/1 numeric)
    FLAG_DOCUMENT_2: Optional[float] = None
    FLAG_DOCUMENT_3: Optional[float] = None
    FLAG_DOCUMENT_4: Optional[float] = None
    FLAG_DOCUMENT_5: Optional[float] = None
    FLAG_DOCUMENT_6: Optional[float] = None
    FLAG_DOCUMENT_7: Optional[float] = None
    FLAG_DOCUMENT_8: Optional[float] = None
    FLAG_DOCUMENT_9: Optional[float] = None
    FLAG_DOCUMENT_10: Optional[float] = None
    FLAG_DOCUMENT_11: Optional[float] = None
    FLAG_DOCUMENT_12: Optional[float] = None
    FLAG_DOCUMENT_13: Optional[float] = None
    FLAG_DOCUMENT_14: Optional[float] = None
    FLAG_DOCUMENT_15: Optional[float] = None
    FLAG_DOCUMENT_16: Optional[float] = None
    FLAG_DOCUMENT_17: Optional[float] = None
    FLAG_DOCUMENT_18: Optional[float] = None
    FLAG_DOCUMENT_19: Optional[float] = None
    FLAG_DOCUMENT_20: Optional[float] = None
    FLAG_DOCUMENT_21: Optional[float] = None

    # Credit bureau enquiry recency
    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_DAY: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_WEEK: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_MON: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = None

    # Building / property features — avg
    APARTMENTS_AVG: Optional[float] = None
    BASEMENTAREA_AVG: Optional[float] = None
    YEARS_BEGINEXPLUATATION_AVG: Optional[float] = None
    YEARS_BUILD_AVG: Optional[float] = None
    COMMONAREA_AVG: Optional[float] = None
    ELEVATORS_AVG: Optional[float] = None
    ENTRANCES_AVG: Optional[float] = None
    FLOORSMAX_AVG: Optional[float] = None
    FLOORSMIN_AVG: Optional[float] = None
    LANDAREA_AVG: Optional[float] = None
    LIVINGAPARTMENTS_AVG: Optional[float] = None
    LIVINGAREA_AVG: Optional[float] = None
    NONLIVINGAPARTMENTS_AVG: Optional[float] = None
    NONLIVINGAREA_AVG: Optional[float] = None

    # Building / property features — mode
    APARTMENTS_MODE: Optional[float] = None
    BASEMENTAREA_MODE: Optional[float] = None
    YEARS_BEGINEXPLUATATION_MODE: Optional[float] = None
    YEARS_BUILD_MODE: Optional[float] = None
    COMMONAREA_MODE: Optional[float] = None
    ELEVATORS_MODE: Optional[float] = None
    ENTRANCES_MODE: Optional[float] = None
    FLOORSMAX_MODE: Optional[float] = None
    FLOORSMIN_MODE: Optional[float] = None
    LANDAREA_MODE: Optional[float] = None
    LIVINGAPARTMENTS_MODE: Optional[float] = None
    LIVINGAREA_MODE: Optional[float] = None
    NONLIVINGAPARTMENTS_MODE: Optional[float] = None
    NONLIVINGAREA_MODE: Optional[float] = None
    TOTALAREA_MODE: Optional[float] = None

    # Building / property features — median
    APARTMENTS_MEDI: Optional[float] = None
    BASEMENTAREA_MEDI: Optional[float] = None
    YEARS_BEGINEXPLUATATION_MEDI: Optional[float] = None
    YEARS_BUILD_MEDI: Optional[float] = None
    COMMONAREA_MEDI: Optional[float] = None
    ELEVATORS_MEDI: Optional[float] = None
    ENTRANCES_MEDI: Optional[float] = None
    FLOORSMAX_MEDI: Optional[float] = None
    FLOORSMIN_MEDI: Optional[float] = None
    LANDAREA_MEDI: Optional[float] = None
    LIVINGAPARTMENTS_MEDI: Optional[float] = None
    LIVINGAREA_MEDI: Optional[float] = None
    NONLIVINGAPARTMENTS_MEDI: Optional[float] = None
    NONLIVINGAREA_MEDI: Optional[float] = None

    # Building categorical modes
    FONDKAPREMONT_MODE: Optional[str] = None
    HOUSETYPE_MODE: Optional[str] = None
    WALLSMATERIAL_MODE: Optional[str] = None
    EMERGENCYSTATE_MODE: Optional[str] = None


# ── Input: bureau sub-tables ──────────────────────────────────────────────────

class BureauRecord(BaseModel):
    """One row from bureau.csv. SK_ID_CURR is omitted — inferred from the parent request."""

    model_config = ConfigDict(extra="ignore")

    SK_ID_BUREAU: Optional[int] = None
    CREDIT_ACTIVE: Optional[str] = None
    CREDIT_CURRENCY: Optional[str] = None
    DAYS_CREDIT: Optional[float] = None
    CREDIT_DAY_OVERDUE: Optional[float] = None
    DAYS_CREDIT_ENDDATE: Optional[float] = None
    DAYS_ENDDATE_FACT: Optional[float] = None
    AMT_CREDIT_SUM: Optional[float] = None
    AMT_CREDIT_SUM_DEBT: Optional[float] = None
    AMT_CREDIT_SUM_LIMIT: Optional[float] = None
    AMT_CREDIT_SUM_OVERDUE: Optional[float] = None
    CREDIT_TYPE: Optional[str] = None
    DAYS_CREDIT_UPDATE: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    CNT_CREDIT_PROLONG: Optional[float] = None


class BureauBalanceRecord(BaseModel):
    """One row from bureau_balance.csv."""

    model_config = ConfigDict(extra="ignore")

    SK_ID_BUREAU: Optional[int] = None
    MONTHS_BALANCE: Optional[int] = None
    STATUS: Optional[str] = None


# ── Input: previous application sub-tables ────────────────────────────────────

class PreviousApplicationRecord(BaseModel):
    """One row from previous_application.csv."""

    model_config = ConfigDict(extra="ignore")

    SK_ID_PREV: Optional[int] = None
    SK_ID_CURR: Optional[int] = None
    NAME_CONTRACT_TYPE: Optional[str] = None
    AMT_ANNUITY: Optional[float] = None
    AMT_APPLICATION: Optional[float] = None
    AMT_CREDIT: Optional[float] = None
    AMT_DOWN_PAYMENT: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    WEEKDAY_APPR_PROCESS_START: Optional[str] = None
    HOUR_APPR_PROCESS_START: Optional[float] = None
    FLAG_LAST_APPL_PER_CONTRACT: Optional[str] = None
    NFLAG_LAST_APPL_IN_DAY: Optional[float] = None
    RATE_DOWN_PAYMENT: Optional[float] = None
    RATE_INTEREST_PRIMARY: Optional[float] = None
    RATE_INTEREST_PRIVILEGED: Optional[float] = None
    NAME_CASH_LOAN_PURPOSE: Optional[str] = None
    NAME_CONTRACT_STATUS: Optional[str] = None
    DAYS_DECISION: Optional[float] = None
    NAME_PAYMENT_TYPE: Optional[str] = None
    CODE_REJECT_REASON: Optional[str] = None
    NAME_TYPE_SUITE: Optional[str] = None
    NAME_CLIENT_TYPE: Optional[str] = None
    NAME_GOODS_CATEGORY: Optional[str] = None
    NAME_PORTFOLIO: Optional[str] = None
    NAME_PRODUCT_TYPE: Optional[str] = None
    CHANNEL_TYPE: Optional[str] = None
    SELLERPLACE_AREA: Optional[float] = None
    NAME_SELLER_INDUSTRY: Optional[str] = None
    CNT_PAYMENT: Optional[float] = None
    NAME_YIELD_GROUP: Optional[str] = None
    PRODUCT_COMBINATION: Optional[str] = None
    DAYS_FIRST_DRAWING: Optional[float] = None
    DAYS_FIRST_DUE: Optional[float] = None
    DAYS_LAST_DUE_1ST_VERSION: Optional[float] = None
    DAYS_LAST_DUE: Optional[float] = None
    DAYS_TERMINATION: Optional[float] = None
    NFLAG_INSURED_ON_APPROVAL: Optional[float] = None


# ── Input: POS cash, installments, credit card ────────────────────────────────

class PosCashRecord(BaseModel):
    """One row from POS_CASH_balance.csv."""

    model_config = ConfigDict(extra="ignore")

    SK_ID_PREV: Optional[int] = None
    SK_ID_CURR: Optional[int] = None
    MONTHS_BALANCE: Optional[float] = None
    CNT_INSTALMENT: Optional[float] = None
    CNT_INSTALMENT_FUTURE: Optional[float] = None
    NAME_CONTRACT_STATUS: Optional[str] = None
    SK_DPD: Optional[float] = None
    SK_DPD_DEF: Optional[float] = None


class InstallmentsRecord(BaseModel):
    """One row from installments_payments.csv."""

    model_config = ConfigDict(extra="ignore")

    SK_ID_PREV: Optional[int] = None
    SK_ID_CURR: Optional[int] = None
    NUM_INSTALMENT_VERSION: Optional[float] = None
    NUM_INSTALMENT_NUMBER: Optional[float] = None
    DAYS_INSTALMENT: Optional[float] = None
    DAYS_ENTRY_PAYMENT: Optional[float] = None
    AMT_INSTALMENT: Optional[float] = None
    AMT_PAYMENT: Optional[float] = None


class CreditCardRecord(BaseModel):
    """One row from credit_card_balance.csv."""

    model_config = ConfigDict(extra="ignore")

    SK_ID_PREV: Optional[int] = None
    SK_ID_CURR: Optional[int] = None
    MONTHS_BALANCE: Optional[int] = None
    AMT_BALANCE: Optional[float] = None
    AMT_CREDIT_LIMIT_ACTUAL: Optional[float] = None
    AMT_DRAWINGS_ATM_CURRENT: Optional[float] = None
    AMT_DRAWINGS_CURRENT: Optional[float] = None
    AMT_DRAWINGS_OTHER_CURRENT: Optional[float] = None
    AMT_DRAWINGS_POS_CURRENT: Optional[float] = None
    AMT_INST_MIN_REGULARITY: Optional[float] = None
    AMT_PAYMENT_CURRENT: Optional[float] = None
    AMT_PAYMENT_TOTAL_CURRENT: Optional[float] = None
    AMT_RECEIVABLE_PRINCIPAL: Optional[float] = None
    AMT_RECIVABLE: Optional[float] = None
    AMT_TOTAL_RECEIVABLE: Optional[float] = None
    CNT_DRAWINGS_ATM_CURRENT: Optional[float] = None
    CNT_DRAWINGS_CURRENT: Optional[float] = None
    CNT_DRAWINGS_OTHER_CURRENT: Optional[float] = None
    CNT_DRAWINGS_POS_CURRENT: Optional[float] = None
    CNT_INSTALMENT_MATURE_CUM: Optional[float] = None
    SK_DPD: Optional[float] = None
    SK_DPD_DEF: Optional[float] = None


# ── Top-level request ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Full scoring request: one application row plus all available sub-table records."""

    application: ApplicationData
    bureau: list[BureauRecord] = []
    bureau_balance: list[BureauBalanceRecord] = []
    previous_applications: list[PreviousApplicationRecord] = []
    pos_cash: list[PosCashRecord] = []
    installments: list[InstallmentsRecord] = []
    credit_card: list[CreditCardRecord] = []


# ── Output models ─────────────────────────────────────────────────────────────

class SHAPFeature(BaseModel):
    """One SHAP feature contribution entry in the prediction explanation."""

    feature_name: str
    shap_value: float
    feature_value: float
    description: str


class PredictionResponse(BaseModel):
    """Scoring response returned by POST /predict."""

    application_id: int
    default_probability: float
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]
    top_shap_features: list[SHAPFeature]
    model_version: str
    cached: bool
    predicted_at: datetime


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str
    timestamp: datetime


class ModelInfoResponse(BaseModel):
    """Response from GET /model/info — describes the currently loaded production bundle."""

    model_version: str
    mlflow_run_id: str
    model_type: str
    fair: bool
    feature_count: int
    promoted_at: Optional[str] = None
    training_auc: Optional[float] = None


# ── Portfolio analytics ───────────────────────────────────────────────────────

class RiskTierCount(BaseModel):
    """Aggregated count and average probability for one risk tier."""

    risk_tier: str
    count: int
    avg_probability: float


class PortfolioSummaryResponse(BaseModel):
    """High-level portfolio metrics, optionally scoped to a time period."""

    total_predictions: int
    tier_breakdown: list[RiskTierCount]
    high_risk_count: int
    avg_default_probability: float
    period: str


class PredictionRecord(BaseModel):
    """One row from the predictions audit table."""

    id: int
    application_id: int
    default_probability: float
    risk_tier: str
    model_version: str
    predicted_at: datetime
    cached: bool


class PortfolioListResponse(BaseModel):
    """Paginated list of prediction records."""

    predictions: list[PredictionRecord]
    total: int
    period: str


# ── Helper ────────────────────────────────────────────────────────────────────

def compute_risk_tier(prob: float) -> Literal["LOW", "MEDIUM", "HIGH"]:
    """Map a default probability to a three-band risk tier.

    LOW    = prob < 0.30
    MEDIUM = 0.30 <= prob <= 0.60
    HIGH   = prob > 0.60
    """
    if prob < 0.30:
        return "LOW"
    if prob <= 0.60:
        return "MEDIUM"
    return "HIGH"
