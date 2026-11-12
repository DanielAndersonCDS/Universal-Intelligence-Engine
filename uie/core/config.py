"""Every tunable threshold in one place so it can be recorded in the provenance manifest."""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields


@dataclass
class Config:
    # general
    infer_roles: bool = True          # guess amount/counterparty/price/entity/label roles from column names
    hash_data: bool = True            # SHA-256 the input files for the provenance manifest
    # outliers / correlation
    outlier_t: float = 4.0
    corr_threshold: float = 0.7
    corr_max_cols: int = 20
    derived_max_cols: int = 24
    derived_sample_rows: int = 20000
    # per-group analysis
    group_max: int = 12
    group_min_rows: int = 500
    group_top_facts: int = 6
    # stationarity-aware trends
    adf_min_points: int = 20
    drift_t_min: float = 2.0
    jump_z_min: float = 5.0
    # finance
    returns_max_entities: int = 10
    vol_shift_ratio: float = 1.5
    spurious_level_r: float = 0.8
    spurious_return_r: float = 0.3
    return_corr_min: float = 0.5
    benford_min_n: int = 1000
    benford_mad_info: float = 0.012
    benford_mad_notable: float = 0.015
    round_min_share: float = 0.03
    round_min_count: int = 50
    bunch_min_count: int = 20
    bunch_ratio: float = 2.0
    tx_window_days: float = 3.0
    tx_min_pairs: int = 5
    pop_min_change: float = 0.05
    pop_min_rows: int = 30
    gap_min_days: float = 5.0
    gap_factor: float = 3.0
    # AI / ML
    leak_purity: float = 0.97
    leak_lift: float = 0.10
    leak_sample_rows: int = 200000
    leak_bins: int = 20
    imbalance_notable: float = 0.10
    imbalance_warning: float = 0.01
    near_constant: float = 0.99
    high_card: int = 100
    rare_count: int = 5
    group_eta2: float = 0.14
    # drift
    psi_notable: float = 0.10
    psi_warning: float = 0.25
    ks_notable: float = 0.20
    drift_sample_rows: int = 100000
    drift_max_categories: int = 50
    # split check
    near_round_decimals: int = 2
    # reconciliation
    recon_tolerance: float = 0.01
    recon_rel_tolerance: float = 0.0
    recon_max_exceptions: int = 50000
    # triage
    reraise_factor: float = 1.5

    def to_dict(self) -> dict:
        return asdict(self)

    def update(self, **kw):
        names = {f.name for f in fields(self)}
        for k, v in kw.items():
            if k in names and v is not None:
                setattr(self, k, v)
        return self
