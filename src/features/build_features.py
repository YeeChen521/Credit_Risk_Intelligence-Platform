import polars as pl
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
RAW_DIR = PROJECT_ROOT / "projects" / "Credit Risk Intelligence Platform" / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "projects" / "Credit Risk Intelligence Platform"/ "data" / "processed"

DROP_HIGH_MISSING_PREFIXES = (
    "COMMONAREA_",
    "NONLIVINGAPARTMENTS_",
    "NONLIVINGAREA_",
    "FONDKAPREMONT_MODE",
    "LIVINGAPARTMENTS_",
)

CATEGORICAL_PREFIXES = ("NAME_", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "WEEKDAY_")
CATEGORICAL_EXACT = {
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
}

def missing_summary(df: pl.DataFrame, table_name: str) -> pl.DataFrame:
    """Return one row per column with null count and missing rate."""
    n = df.height
    rows = []
    for col in df.columns:
        null_count = df[col].null_count()
        rows.append(
            {
                "table": table_name,
                "column": col,
                "dtype": str(df.schema[col]),
                "null_count": null_count,
                "missing_rate": null_count / n if n else 0.0,
                "n_rows": n,
            }
        )
    return pl.DataFrame(rows).sort("missing_rate", descending=True)

def is_categorical(col: str, dtype: pl.DataType) -> bool:
    if dtype == pl.Utf8:
        return True
    if col in CATEGORICAL_EXACT:
        return True
    return any(col.startswith(p) for p in CATEGORICAL_PREFIXES)


def should_drop_column(col: str, missing_rate: float) -> bool:
    if missing_rate >= 0.70 and any(col.startswith(p) for p in DROP_HIGH_MISSING_PREFIXES):
        return True
    return False


def handle_missing_application(df: pl.DataFrame,filename:str) -> pl.DataFrame:
    """Impute/drop missing values on the application table."""

    miss = missing_summary(df, filename)
    miss_map = dict(zip(miss["column"].to_list(), miss["missing_rate"].to_list()))

    drop_cols = [
        c for c in df.columns
        if should_drop_column(c, miss_map.get(c, 0.0))
    ]

    out = df.drop(drop_cols) if drop_cols else df

    if "OWN_CAR_AGE" in out.columns and "FLAG_OWN_CAR" in out.columns:
        out = out.with_columns(
            pl.when(pl.col("FLAG_OWN_CAR") == "N")
            .then(0.0)
            .otherwise(pl.col("OWN_CAR_AGE"))
            .alias("OWN_CAR_AGE")
        )

    ext_cols = [c for c in out.columns if c.startswith("EXT_SOURCE")]

    ext_exprs = []
    for col in ext_cols:
        median_val = out[col].median()

        ext_exprs.append(
            pl.col(col)
            .is_null()
            .cast(pl.Int8)
            .alias(f"MISSING_{col}")
        )

        ext_exprs.append(
            pl.col(col)
            .fill_null(median_val)
            .alias(col)
        )

    general_exprs = []

    for col in out.columns:
        if col in ("SK_ID_CURR", "TARGET"):
            continue

        if col.startswith("EXT_SOURCE"):
            continue

        dtype = out.schema[col]

        if out[col].null_count() == 0:
            continue

        if is_categorical(col, dtype):
            general_exprs.append(
                pl.col(col).fill_null("Missing").alias(col)
            )

        elif dtype.is_numeric():
            median_val = out[col].median()
            general_exprs.append(
                pl.col(col).fill_null(median_val).alias(col)
            )

        else:
            general_exprs.append(
                pl.col(col).fill_null("Missing").alias(col)
            )

    all_exprs = ext_exprs + general_exprs

    if all_exprs:
        out = out.with_columns(all_exprs)

    for col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        if col in out.columns:
            out = out.with_columns(
                pl.col(col).replace({"Y": 1, "N": 0}).cast(pl.Int8).alias(col)
            )

    return out

def bureau_balance_features(bureau_balance:pl.DataFrame) -> pl.DataFrame:
    bb_numeric = bureau_balance.with_columns([
        pl.col("STATUS")
        .replace(["C", "X"], [0, 0])
        .cast(pl.Int64,strict=False)
        .fill_null(0)
        .alias("STATUS_NUMERICAL")
    ])

    bb_features = (
        bb_numeric
        .group_by("SK_ID_BUREAU")
        .agg([
            pl.len().alias("bb_total_months_tracked"),
            pl.col("STATUS_NUMERICAL").max().alias("bb_worst_status"),
            pl.col("STATUS_NUMERICAL").filter(pl.col("STATUS_NUMERICAL") > 0).len().alias("bb_months_overdue_count"),
            pl.col("STATUS_NUMERICAL").filter(pl.col("MONTHS_BALANCE") >= -6).max().alias("bb_worst_status_last_6m")
        ])
    )
    return bb_features

def bureau_features(bureau:pl.DataFrame,bb_features:pl.DataFrame) -> pl.DataFrame:
    bureau_combined = bureau.join(bb_features, on="SK_ID_BUREAU", how="left")

    bureau_final_features = (
        bureau_combined
        .group_by("SK_ID_CURR")
        .agg([pl.len().alias("bureau_loan_count"),
            pl.col("CREDIT_TYPE").n_unique().alias("bureau_loan_types"),
            
            pl.col("DAYS_CREDIT").max().alias("bureau_days_since_last_loan"),
            pl.col("DAYS_CREDIT").min().alias("bureau_days_since_first_loan"),
            
            pl.col("AMT_CREDIT_SUM").mean().alias("bureau_avg_credit"),
            pl.col("AMT_CREDIT_SUM_DEBT").sum().alias("total_debt"),
            pl.col("AMT_CREDIT_SUM").sum().alias("total_credit"),
            
            pl.col("AMT_CREDIT_SUM_DEBT").filter(pl.col("CREDIT_ACTIVE") == "Active").sum().alias("bureau_active_debt"),
            (pl.col("AMT_CREDIT_SUM_DEBT").sum() / (pl.col("AMT_CREDIT_SUM").sum() + 1e-5)).alias("debt_to_credit_ratio"),
            
            pl.col("CREDIT_DAY_OVERDUE").max().alias("bureau_max_overdue"),
            pl.col("AMT_CREDIT_SUM_OVERDUE").sum().alias("bureau_total_amt_overdue"),
            pl.col("CNT_CREDIT_PROLONG").sum().alias("bureau_total_prolong_count"),
            
            pl.col("bb_worst_status").max().fill_null(0).alias("bureau_global_worst_status"),
            pl.col("bb_months_overdue_count").sum().fill_null(0).alias("bureau_total_months_overdue_across_loans"),
            pl.col("bb_worst_status_last_6m").max().fill_null(0).alias("bureau_global_worst_status_last_6m")])
    )

    bureau_final_features = (
        bureau_final_features
        .with_columns(
            pl.lit(1).alias("HAS_BUREAU_RECORD")
        )
    )
    return bureau_final_features

def installment_payment_feature(installment_payment: pl.DataFrame) -> pl.DataFrame:
    ins_processed = installment_payment.with_columns(
        (pl.col("AMT_PAYMENT") - pl.col("AMT_INSTALMENT")).alias("ins_pay_diff"),
        (pl.col("AMT_PAYMENT") / (pl.col("AMT_INSTALMENT") + 1e-5)).alias("ins_pay_ratio"),
        (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")).alias("ins_day_late")
    )
    
    ins_features = (
        ins_processed
        .group_by("SK_ID_CURR")
        .agg([
            pl.len().alias("ins_total_installment_tracked"),
            
            pl.col("ins_pay_diff").min().alias("ins_worst_underpay"),
            pl.col("ins_pay_diff").mean().alias("ins_mean_payment_diff"),
            pl.col("ins_pay_ratio").mean().alias("ins_mean_payratio"),
            
            pl.col("ins_pay_diff").filter(pl.col("ins_pay_diff") < 0).len().alias("ins_underpayment_count"),
            
            pl.col("ins_day_late").max().alias("ins_worst_day_late"),
            pl.col("ins_day_late").mean().alias("ins_avg_day_late"),
            
            pl.col("ins_day_late").filter(pl.col("ins_day_late") > 0).len().alias("ins_late_count")
        ])
    )
    
    ins_features = (
        ins_features
        .with_columns(
            pl.lit(1).alias("HAS_INSTALMENT_HISTORY")
        )
    )
    return ins_features
    
def credit_card_features(credit_card_balance: pl.DataFrame) -> pl.DataFrame:
    cc_features = (
        credit_card_balance
        .group_by("SK_ID_CURR")
        .agg([
            pl.len().alias("cc_months_tracked"),

            pl.col("AMT_DRAWINGS_CURRENT").sum().alias("cc_total_drawings_amt"),
            pl.col("AMT_DRAWINGS_CURRENT").max().alias("cc_max_single_month_drawing"),
            pl.col("CNT_DRAWINGS_CURRENT").sum().alias("cc_total_drawings_count"),

            pl.col("AMT_DRAWINGS_ATM_CURRENT").sum().alias("cc_total_atm_drawings_amt"),
            (pl.col("AMT_DRAWINGS_ATM_CURRENT").sum() / (pl.col("AMT_DRAWINGS_CURRENT").sum() + 1e-5)).alias("cc_atm_to_total_drawing_ratio"),

            pl.col("AMT_PAYMENT_CURRENT").sum().alias("cc_total_paymemt"),
            (pl.col("AMT_PAYMENT_CURRENT").sum() / (pl.col("AMT_INST_MIN_REGULARITY").sum() + 1e-5)).alias("cc_pay_to_min_inst_ratio"),

            pl.col("AMT_BALANCE").max().alias("cc_max_balance_ever"),
            pl.col("AMT_BALANCE").sort_by("MONTHS_BALANCE").last().alias("cc_current_balance"),
            pl.col("AMT_TOTAL_RECEIVABLE").sort_by("MONTHS_BALANCE").last().alias("cc_current_total_receivable"),

            (pl.col("AMT_BALANCE").sort_by("MONTHS_BALANCE").last() / (pl.col("AMT_CREDIT_LIMIT_ACTUAL").sort_by("MONTHS_BALANCE").last() + 1e-5)).alias("cc_current_utilization_ratio"),

            pl.col("SK_DPD").max().alias("cc_max_days_past_due"),
            pl.col("SK_DPD_DEF").max().alias("cc_max_days_past_due_with_tolerance"),

            pl.col("SK_DPD").filter(pl.col("SK_DPD") > 0).len().alias("cc_month_overdue_count")
        ])
    )

    cc_features = (
        cc_features
        .with_columns(pl.lit(1).alias("HAS_CREDIT_CARD_HISTORY"))
    )
    return cc_features

def pos_cash_balance_features(pos_cash_balance: pl.DataFrame) -> pl.DataFrame:
    pos_features = (
        pos_cash_balance
        .group_by("SK_ID_CURR")
        .agg([
            pl.len().alias("pos_month_tracked"),

            pl.col("CNT_INSTALMENT").max().alias("pos_max_loan_term"),
            pl.col("CNT_INSTALMENT_FUTURE").sort_by("MONTHS_BALANCE").last().alias("pos_current_remaining_installments"),

            (pl.col("CNT_INSTALMENT_FUTURE").sort_by("MONTHS_BALANCE").last() / (pl.col("CNT_INSTALMENT").sort_by("MONTHS_BALANCE").last() + 1e-5)).alias("pos_remaining_installment_ratio"),

            pl.col("SK_DPD").max().alias("pos_max_days_past_due"),
            pl.col("SK_DPD_DEF").max().alias("pos_max_days_past_due_with_tolerance"),

            pl.col("SK_DPD").filter(pl.col("SK_DPD") > 0).len().alias("pos_month_overdue_count")
        ])
    )

    pos_features = (
        pos_features
        .with_columns(pl.lit(1).alias("HAS_POS_HISTORY"))
    )
    return pos_features

PREV_CONTRACT_STATUSES = ["Approved", "Refused", "Canceled", "Unused offer"]

def previous_application_features(previous_application: pl.DataFrame) -> pl.DataFrame:
    prev_sorted = previous_application.sort(["SK_ID_CURR", "DAYS_DECISION"])

    prev_features = (
        prev_sorted
        .group_by("SK_ID_CURR")
        .agg([
            pl.len().alias("prev_total_applications"),

            pl.col("DAYS_DECISION").max().alias("prev_days_since_last_decision"),
            pl.col("DAYS_DECISION").min().alias("prev_days_since_first_decision"),

            pl.col("AMT_APPLICATION").sum().alias("prev_total_amt_asked"),
            pl.col("AMT_CREDIT").sum().alias("prev_total_amt_approved"),
            (pl.col("AMT_CREDIT").sum() / (pl.col("AMT_APPLICATION").sum() + 1e-5)).alias("prev_credit_to_application_ratio"),
            pl.col("AMT_ANNUITY").mean().alias("prev_avg_annuity_amt"),
            pl.col("RATE_DOWN_PAYMENT").mean().alias("prev_avg_down_payment_rate"),

            pl.col("CNT_PAYMENT").mean().alias("prev_avg_loan_term_months"),

            pl.col("NAME_CONTRACT_STATUS").filter(pl.col("NAME_CONTRACT_STATUS") == "Refused").len().alias("prev_count_refused"),
            pl.col("NAME_CONTRACT_STATUS").filter(pl.col("NAME_CONTRACT_STATUS") == "Approved").len().alias("prev_count_approved"),
            pl.col("NAME_CONTRACT_STATUS").filter(pl.col("NAME_CONTRACT_STATUS") == "Canceled").len().alias("prev_count_canceled"),

            (pl.col("NAME_CONTRACT_STATUS").filter(pl.col("NAME_CONTRACT_STATUS") == "Refused").len() / pl.len()).alias("prev_rejection_rate"),
            pl.col("NAME_CONTRACT_STATUS").sort_by("DAYS_DECISION").last().alias("prev_most_recent_status")
        ])
    )

    # Cast to Categorical with a fixed set of categories so dummies are identical
    # across train and test regardless of which statuses happen to appear.
    prev_features = prev_features.with_columns(
        pl.col("prev_most_recent_status")
        .cast(pl.Enum(PREV_CONTRACT_STATUSES))
    )

    prev_features = prev_features.to_dummies(columns=["prev_most_recent_status"])

    # Guarantee every expected dummy column exists (defensive, Enum should handle it)
    for status in PREV_CONTRACT_STATUSES:
        col = f"prev_most_recent_status_{status}"
        if col not in prev_features.columns:
            prev_features = prev_features.with_columns(pl.lit(0).cast(pl.UInt8).alias(col))

    prev_features = prev_features.with_columns(pl.lit(1).alias("HAS_PREVIOUS_APPLICATION"))
    return prev_features

def assemble_master_dataset(
    base_application_df: pl.DataFrame,
    bureau_df: pl.DataFrame,
    ins_df: pl.DataFrame,
    cc_df: pl.DataFrame,
    pos_df: pl.DataFrame,
    prev_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Sequentially joins all engineered feature tables onto a base application dataframe (train or test).
    """
    return (
        base_application_df
        .join(bureau_df, on="SK_ID_CURR", how="left")
        .join(ins_df, on="SK_ID_CURR", how="left")
        .join(cc_df, on="SK_ID_CURR", how="left")
        .join(pos_df, on="SK_ID_CURR", how="left")
        .join(prev_df, on="SK_ID_CURR", how="left")
    )

RATIO_OR_AVG_PREFIXES = (
    "bureau_avg", "bureau_global", "debt_to_credit",
    "ins_mean", "ins_worst", "ins_avg",
    "cc_atm_to", "cc_pay_to", "cc_current_utilization",
    "pos_remaining",
    "prev_credit_to", "prev_avg", "prev_rejection",
)

def fill_joined_feature_nulls(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []

    for col, dtype in df.schema.items():
        if col in ("SK_ID_CURR", "TARGET"):
            continue
        if df[col].null_count() == 0:
            continue

        if dtype.is_numeric():
            # Ratio/average features: use -1 so models can distinguish
            # "no record" from a genuine zero value.
            if any(col.startswith(p) for p in RATIO_OR_AVG_PREFIXES):
                exprs.append(pl.col(col).fill_null(-1))
            else:
                exprs.append(pl.col(col).fill_null(0))

        elif dtype == pl.Utf8:
            exprs.append(pl.col(col).fill_null("Missing"))

    return df.with_columns(exprs) if exprs else df

def save_parquet(df: pl.DataFrame, filename:str) -> None:
    df.write_parquet(
        PROCESSED_DIR / filename,
        compression="snappy",
    )
    
def main():
    bureau = pl.read_csv(RAW_DIR/"bureau.csv")
    bureau_balance = pl.read_csv(RAW_DIR/"bureau_balance.csv")
    installment_payment = pl.read_csv(RAW_DIR/"installments_payments.csv")
    credit_card_balance = pl.read_csv(RAW_DIR/"credit_card_balance.csv")
    pos_cash_balance = pl.read_csv(RAW_DIR/"POS_CASH_balance.csv")
    previous_application = pl.read_csv(RAW_DIR/"previous_application.csv")
    application_train = pl.read_csv(RAW_DIR/"application_train.csv")
    application_test = pl.read_csv(RAW_DIR/"application_test.csv")
    
    application_train_cleaned = handle_missing_application(application_train,"application_train.csv")
    application_test_cleaned = handle_missing_application(application_test,"application_test.csv")
    
    bb_features = bureau_balance_features(bureau_balance)
    bureau_final_features = bureau_features(bureau,bb_features)
    ins_features = installment_payment_feature(installment_payment)
    cc_features = credit_card_features(credit_card_balance)
    pos_features = pos_cash_balance_features(pos_cash_balance)
    prev_features = previous_application_features(previous_application)
    
    assert bureau_final_features["SK_ID_CURR"].is_unique().all()
    assert ins_features["SK_ID_CURR"].is_unique().all()
    assert cc_features["SK_ID_CURR"].is_unique().all()
    assert pos_features["SK_ID_CURR"].is_unique().all()
    assert prev_features["SK_ID_CURR"].is_unique().all()
    
    train_complete = assemble_master_dataset(
        application_train_cleaned, bureau_final_features, ins_features, cc_features, pos_features, prev_features
    )

    test_complete = assemble_master_dataset(
        application_test_cleaned, bureau_final_features, ins_features, cc_features, pos_features, prev_features
    )
    
    train_complete = fill_joined_feature_nulls(train_complete)
    test_complete = fill_joined_feature_nulls(test_complete)
    
    assert train_complete["SK_ID_CURR"].is_unique().all()
    assert test_complete["SK_ID_CURR"].is_unique().all()

    assert train_complete["TARGET"].null_count() == 0

    print(train_complete.shape)
    print(test_complete.shape)
    
    save_parquet(train_complete,"train_complete")
    save_parquet(test_complete,"test_complete")
    
if __name__ == "__main__":
    main()  