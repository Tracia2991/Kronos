"""
从 qlib 行情对齐 `predictions.pkl`（与 qlib_test 一致：截面 score = 预测路径上 close 相对「信号日」收盘的差），
计算论文常用预测指标：IC、RankIC、MAE、R²。

标签构造与 ``qlib_data_preprocess`` 一致：先按 ``Config.qlib_price_neutralize`` 对**收盘价**
做截面/行业内去均值，再算 ``close[t+H]-close[t]``（不再对价差二次 demean，避免与训练输入脱节）。

依赖：pandas、numpy、pyqlib（`pip install pyqlib`）。与 finetune 中 qlib 初始化方式一致。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from price_neutralize import neutralize_close_wide_for_metrics

_FINETUNE_DIR = os.path.dirname(os.path.abspath(__file__))


def _ensure_finetune_on_syspath() -> None:
    """便于 `python finetune/prediction_metrics.py` 与 notebook 中 `import prediction_metrics`。"""
    if _FINETUNE_DIR not in sys.path:
        sys.path.insert(0, _FINETUNE_DIR)


@dataclass
class PredictionTaskMetrics:
    """单条 signal（如 mean / last）在某一 horizon 上的指标。"""

    signal_name: str
    horizon: int
    n_days_ic: int
    n_pairs_mae: int
    ic_mean: float
    ic_std: float
    ic_ir: float
    rank_ic_mean: float
    rank_ic_std: float
    rank_ic_ir: float
    mae: float
    r2: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_predictions_pkl(path: str) -> dict[str, pd.DataFrame]:
    """加载 qlib_test 写出的 predictions.pkl：dict[str, wide DataFrame]。"""
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, Mapping):
        raise TypeError(f"predictions.pkl 根对象应为 dict，实际为 {type(obj)}")
    out: dict[str, pd.DataFrame] = {}
    for k, v in obj.items():
        if not isinstance(v, pd.DataFrame):
            raise TypeError(f"键 {k!r} 的值应为 DataFrame，实际为 {type(v)}")
        out[str(k)] = v.copy()
    return out


def _normalize_pred_index(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.index = pd.to_datetime(d.index).normalize()
    d = d.sort_index()
    d.columns = [str(c) for c in d.columns]
    return d


def _spearman_corr_np(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson on ranks ≈ Spearman；常数列返回 nan。"""
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    sx, sy = np.nanstd(rx), np.nanstd(ry)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson_corr_np(x: np.ndarray, y: np.ndarray) -> float:
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def daily_ic_and_rank_ic(
    pred: pd.DataFrame,
    label: pd.DataFrame,
    min_instruments: int = 10,
) -> tuple[pd.Series, pd.Series]:
    """
    逐日截面：IC = Pearson(pred, label)，RankIC = Spearman(pred, label)。
    仅保留 pred/label 均非 nan 且样本数 >= min_instruments 的交易日。
    """
    pred = _normalize_pred_index(pred)
    label = _normalize_pred_index(label)
    common_idx = pred.index.intersection(label.index)
    common_cols = pred.columns.intersection(label.columns)

    ic_list: list[float] = []
    rank_ic_list: list[float] = []
    dates: list[pd.Timestamp] = []

    for dt in common_idx:
        p = pred.loc[dt, common_cols].astype(float)
        l = label.loc[dt, common_cols].astype(float)
        m = p.notna() & l.notna()
        if int(m.sum()) < min_instruments:
            continue
        pv = p.loc[m].to_numpy(dtype=float)
        lv = l.loc[m].to_numpy(dtype=float)
        ic_list.append(_pearson_corr_np(pv, lv))
        rank_ic_list.append(_spearman_corr_np(pv, lv))
        dates.append(dt)

    ic_s = pd.Series(ic_list, index=pd.DatetimeIndex(dates), name="IC")
    ric_s = pd.Series(rank_ic_list, index=pd.DatetimeIndex(dates), name="RankIC")
    return ic_s, ric_s


def mae_and_r2(pred: pd.DataFrame, label: pd.DataFrame) -> tuple[float, float]:
    """全样本 flatten 后的 MAE 与 R²（以 label 为真值）。"""
    pred = _normalize_pred_index(pred)
    label = _normalize_pred_index(label)
    common_idx = pred.index.intersection(label.index)
    common_cols = pred.columns.intersection(label.columns)
    p = pred.loc[common_idx, common_cols].astype(float)
    l = label.loc[common_idx, common_cols].astype(float)
    mask = p.notna() & l.notna()
    err = (p - l).where(mask)
    flat_p = p.where(mask).to_numpy().ravel()
    flat_l = l.where(mask).to_numpy().ravel()
    m = np.isfinite(flat_p) & np.isfinite(flat_l)
    flat_p, flat_l = flat_p[m], flat_l[m]
    if flat_p.size == 0:
        return float("nan"), float("nan")
    mae = float(np.mean(np.abs(flat_p - flat_l)))
    ss_res = float(np.sum((flat_l - flat_p) ** 2))
    ss_tot = float(np.sum((flat_l - np.mean(flat_l)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    return mae, r2


def _qlib_features_to_close_wide(df: pd.DataFrame) -> pd.DataFrame:
    """将 D.features 的返回统一成 index=datetime, columns=instrument 的收盘价宽表。"""
    if df.empty:
        return pd.DataFrame()

    d = df.copy()

    def _pivot_long(dt_col: str, inst_col: str, val_col: str) -> pd.DataFrame:
        w = d.pivot_table(index=dt_col, columns=inst_col, values=val_col, aggfunc="last")
        w.index = pd.to_datetime(w.index).normalize()
        w.columns = [str(c) for c in w.columns]
        return w.sort_index()

    # 1) MultiIndex 行 + 列 $close（qlib 常见）
    if isinstance(d.index, pd.MultiIndex):
        names = [str(x).lower() if x is not None else "" for x in d.index.names]
        close_col = "$close" if "$close" in d.columns else None
        if close_col is None:
            for c in d.columns:
                if "close" in str(c).lower():
                    close_col = c
                    break
        if close_col is None and d.shape[1] == 1:
            close_col = d.columns[0]
        if close_col is None:
            raise ValueError(f"找不到收盘价列: columns={list(d.columns)}")

        s = d[close_col]
        level_inst = None
        for i, nm in enumerate(names):
            if nm in ("instrument", "instruments", "code"):
                level_inst = i
                break
        if level_inst is None:
            # 常见顺序 (instrument, datetime) 或 (datetime, instrument)：取非 datetime 层
            for i, nm in enumerate(names):
                if nm and "date" not in nm and "time" not in nm:
                    level_inst = i
                    break
        if level_inst is None:
            level_inst = -1
        wide = s.unstack(level=level_inst)
        # 若 unstack 后行索引不是时间，再 swap
        if not isinstance(wide.index, pd.DatetimeIndex):
            wide.index = pd.to_datetime(wide.index).normalize()
        else:
            wide.index = pd.to_datetime(wide.index).normalize()
        wide.columns = [str(c) for c in wide.columns]
        return wide.sort_index()

    # 2) 已展开成长表：datetime, instrument, $close
    if not isinstance(d.index, pd.MultiIndex):
        cols_lower = {str(c).lower(): c for c in d.columns}
        dt_col = next((cols_lower[k] for k in cols_lower if "datetime" in k or k == "date"), None)
        inst_col = next(
            (cols_lower[k] for k in cols_lower if "instrument" in k or k in ("code", "symbol")), None
        )
        close_col = next((c for c in d.columns if str(c) == "$close" or str(c).lower() == "close"), None)
        if close_col is None:
            num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
            close_col = num_cols[0] if len(num_cols) == 1 else None
        if dt_col and inst_col and close_col:
            return _pivot_long(dt_col, inst_col, close_col)

    raise ValueError(
        f"无法解析 D.features 的 DataFrame 结构: index={getattr(df.index, 'names', None)}, "
        f"columns={list(df.columns)[:8]}"
    )


def fetch_close_wide_from_qlib(
    instruments: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    provider_uri: str,
    region: str | None = None,
) -> pd.DataFrame:
    """qlib.init 后拉取日线 $close 宽表。"""
    import qlib
    from qlib.config import REG_CN, REG_US
    from qlib.data import D

    if region is None or region == "cn":
        reg = REG_CN
    elif region == "us":
        reg = REG_US
    else:
        reg = region  # 调用方传入 REG_* 常量
    qlib.init(provider_uri=os.path.abspath(os.path.expanduser(provider_uri)), region=reg)
    inst_list = [str(x) for x in instruments]
    raw = D.features(inst_list, ["$close"], start_time=str(start.date()), end_time=str(end.date()), freq="day")
    return _qlib_features_to_close_wide(raw)


def build_forward_close_excess_label(
    close_wide: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    与 qlib_test 中「last」类信号对齐：label[t] = close[t+horizon] - close[t]（按各自交易日序列 shift）。
    close_wide: index 升序交易日，columns 为股票代码。
    """
    if horizon <= 0:
        raise ValueError("horizon 必须为正整数（通常为 Config.predict_window）")
    c = close_wide.sort_index().copy()
    c.index = pd.to_datetime(c.index).normalize()
    fut = c.shift(-horizon)
    return fut - c


def evaluate_signal_vs_label(
    pred: pd.DataFrame,
    label: pd.DataFrame,
    signal_name: str,
    horizon: int,
    min_instruments: int = 10,
) -> PredictionTaskMetrics:
    ic_s, ric_s = daily_ic_and_rank_ic(pred, label, min_instruments=min_instruments)
    mae, r2 = mae_and_r2(pred, label)

    ic_m, ic_sd = float(np.nanmean(ic_s)), float(np.nanstd(ic_s, ddof=1)) if ic_s.size > 1 else float("nan")
    ric_m, ric_sd = float(np.nanmean(ric_s)), float(np.nanstd(ric_s, ddof=1)) if ric_s.size > 1 else float("nan")
    ic_ir = float(ic_m / (ic_sd + 1e-12)) if np.isfinite(ic_sd) and ic_sd > 0 else float("nan")
    ric_ir = float(ric_m / (ric_sd + 1e-12)) if np.isfinite(ric_sd) and ric_sd > 0 else float("nan")

    # MAE / R2 用有效 (date, inst) 对数
    pred_n = _normalize_pred_index(pred)
    label_n = _normalize_pred_index(label)
    common_idx = pred_n.index.intersection(label_n.index)
    common_cols = pred_n.columns.intersection(label_n.columns)
    m = pred_n.loc[common_idx, common_cols].notna() & label_n.loc[common_idx, common_cols].notna()
    n_pairs = int(m.sum().sum())

    return PredictionTaskMetrics(
        signal_name=signal_name,
        horizon=horizon,
        n_days_ic=int(ic_s.notna().sum()),
        n_pairs_mae=n_pairs,
        ic_mean=ic_m,
        ic_std=ic_sd,
        ic_ir=ic_ir,
        rank_ic_mean=ric_m,
        rank_ic_std=ric_sd,
        rank_ic_ir=ric_ir,
        mae=mae,
        r2=r2,
    )


def evaluate_predictions_pkl(
    predictions_pkl: str,
    provider_uri: str,
    horizon: int | None = None,
    region: str | None = None,
    min_instruments: int = 10,
    extra_calendar_days: int = 80,
    label_neutralize: str | None = None,
) -> dict[str, PredictionTaskMetrics]:
    """
    读取 predictions.pkl，用 qlib 行情构造 horizon 期「收盘价价差」标签并计算各 signal 的指标。
    中性化顺序与 ``qlib_data_preprocess`` 一致：先对日频 close 宽表做截面/行业去均值，再 shift 得价差。

    horizon 默认读取 finetune.config.Config().predict_window。
    """
    _ensure_finetune_on_syspath()
    try:
        from config import Config  # type: ignore
    except ImportError:
        Config = None  # type: ignore

    preds = load_predictions_pkl(predictions_pkl)
    if horizon is None:
        if Config is None:
            raise ValueError("未安装/导入 config 时请显式传入 horizon=")
        horizon = int(Config().predict_window)

    if label_neutralize is None:
        if Config is not None:
            _c = Config()
            label_neutralize = (
                getattr(_c, "qlib_price_neutralize", None)
                or getattr(_c, "prediction_label_neutralize", None)
                or "none"
            )
        else:
            label_neutralize = "none"

    # 合并所有 signal 涉及的日期与股票，一次拉 close
    all_dates = []
    all_insts: set[str] = set()
    for df in preds.values():
        df = _normalize_pred_index(df)
        all_dates.extend(list(df.index))
        all_insts.update(df.columns.tolist())
    if not all_dates or not all_insts:
        raise ValueError("predictions.pkl 中无有效 index/columns")

    start = pd.Timestamp(min(all_dates)) - pd.Timedelta(days=extra_calendar_days)
    end = pd.Timestamp(max(all_dates)) + pd.Timedelta(days=extra_calendar_days)
    instruments = sorted(all_insts)

    close_w = fetch_close_wide_from_qlib(instruments, start, end, provider_uri, region=region)
    close_w = neutralize_close_wide_for_metrics(
        close_w, label_neutralize, provider_uri=provider_uri, region=region
    )
    label_w = build_forward_close_excess_label(close_w, horizon=horizon)

    results: dict[str, PredictionTaskMetrics] = {}
    for name, p in preds.items():
        p = _normalize_pred_index(p)
        results[name] = evaluate_signal_vs_label(
            p,
            label_w,
            signal_name=name,
            horizon=horizon,
            min_instruments=min_instruments,
        )
    return results


def metrics_table(rows: Mapping[str, PredictionTaskMetrics]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in rows.values()]).set_index("signal_name")


def _main_cli() -> None:
    _ensure_finetune_on_syspath()
    parser = argparse.ArgumentParser(description="计算 predictions.pkl 的 IC / RankIC / MAE / R²")
    parser.add_argument(
        "--predictions-pkl",
        required=True,
        help="qlib_test 生成的 predictions.pkl 路径",
    )
    parser.add_argument(
        "--provider-uri",
        default=os.environ.get("QLIB_DATA_PATH", "").strip() or None,
        help="qlib 数据目录（含 calendars/features/...）；默认环境变量 QLIB_DATA_PATH 或 config.Config",
    )
    parser.add_argument("--horizon", type=int, default=None, help="远期 horizon（默认 Config.predict_window）")
    parser.add_argument("--min-instruments", type=int, default=10, help="截面最少股票数才参与该日 IC")
    parser.add_argument("--out-json", type=str, default=None, help="可选：将结果写入 json")
    parser.add_argument("--region", type=str, default="cn", choices=("cn", "us"), help="qlib 区域")
    parser.add_argument(
        "--label-neutralize",
        type=str,
        default=None,
        help="与 preprocess 一致：none | cs_demean | industry（默认 Config.qlib_price_neutralize）",
    )
    args = parser.parse_args()

    provider = args.provider_uri
    if not provider:
        from config import Config  # type: ignore

        provider = Config().qlib_data_path

    res = evaluate_predictions_pkl(
        args.predictions_pkl,
        provider_uri=provider,
        horizon=args.horizon,
        min_instruments=args.min_instruments,
        region=args.region,
        label_neutralize=args.label_neutralize,
    )
    table = metrics_table(res)
    print(table.to_string())
    if args.out_json:
        payload = {k: v.to_dict() for k, v in res.items()}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print("Wrote", args.out_json)


__all__ = [
    "PredictionTaskMetrics",
    "load_predictions_pkl",
    "daily_ic_and_rank_ic",
    "mae_and_r2",
    "fetch_close_wide_from_qlib",
    "neutralize_close_wide_for_metrics",
    "build_forward_close_excess_label",
    "evaluate_signal_vs_label",
    "evaluate_predictions_pkl",
    "metrics_table",
]


if __name__ == "__main__":
    _main_cli()
