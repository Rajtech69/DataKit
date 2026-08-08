from datakit.core.exceptions import ConfigError

_DEFAULTS = {
    "missing_critical_threshold": 0.40,
    "missing_warning_threshold": 0.10,
    "high_cardinality_threshold": 50,
    "near_constant_threshold": 0.99,
    "leakage_correlation_threshold": 0.98,
    "id_like_uniqueness_threshold": 0.95,
    "data_loss_warning_threshold": 0.10,
    "large_dataset_mb": 500,
    "outlier_iqr_multiplier": 1.5,
    "outlier_zscore_threshold": 3.0,
    "figsize": (10, 6),
    "dpi": 100,
    "theme": "whitegrid",
    "palette": None,
    "tight_layout": True,
    "pairplot_max_columns": 10,
    "count_plot_max_categories": 30,
    "default_report_format": "html"
}

class Config:
    def __init__(self):
        self._config = _DEFAULTS.copy()

    def set(self, **kwargs):
        for k, v in kwargs.items():
            if k not in _DEFAULTS:
                raise ConfigError(f"Unknown config key: {k}")
            self._config[k] = v

    def get(self, key):
        if key not in _DEFAULTS:
            raise ConfigError(f"Unknown config key: {key}")
        return self._config[key]

    def reset(self):
        self._config = _DEFAULTS.copy()

config = Config()
