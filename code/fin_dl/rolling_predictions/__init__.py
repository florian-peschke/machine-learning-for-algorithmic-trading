import typing as t


class RollingEvaluationConstants:
    MAIN_FOLDER_NAME: str = "rolling_predictions"
    DATA_FOLDER_NAME: str = "inputs"

    HISTORY_FOLDER_NAME: str = "history"
    EVALUATION_FOLDER_NAME: str = "evaluation"
    PERFORMANCE_FOLDER_NAME: str = "performance"
    HESSIAN_FOLDER_NAME: str = "hessians"
    JACOBIAN_FOLDER_NAME: str = "jacobians"
    MISC_FOLDER_NAME: str = "miscellaneous"
    MODEL_WEIGHTS_FILE_NAME: str = "weights.h5"

    LABEL_VAR_NAME: str = "Type"
    LABEL_INDEX: str = "Index"
    LABEL_VALUE: str = "Value"
    LABEL_FEATURE: str = "features"
    LABEL_WINDOW: str = "sequence_len"
    LABEL_CORRELATIONS: str = "correlations"

    LABEL_DIAGONALS: str = "diagonals"
    LABEL_NORMAL: str = "normal"

    PARTITION_NAMES: t.List[str] = ["training", "validation", "testing"]
    CORRELATION_X_LABEL: str = "X"
    CORRELATION_Y_LABEL: str = "Y"
    CORRELATION_LABEL: str = "Corr"
    CORRELATION_TYPE_LABEL: str = "Type"
    CORRELATION_FRAME_COLUMNS: t.List[str] = [CORRELATION_X_LABEL, CORRELATION_Y_LABEL, CORRELATION_LABEL]

    SKIP_DATAFRAME_WITH_LEQ_NANS: float = 0.05
