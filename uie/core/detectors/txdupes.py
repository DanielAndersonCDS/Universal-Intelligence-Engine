"""Near-duplicate transactions: same amount and counterparty within a time window."""
from ..facts import Fact
from ..util import qident


def detect(ctx):
    t = ctx.time_col
    amts = [c for c in (ctx.col(n) for n in ctx.roles("amount")) if c and c.kind == "numeric"]
    cps = [c for c in (ctx.col(n) for n in ctx.roles("counterparty")) if c]
    if not t or not amts or not cps:
        return []
    a, cp, rel = amts[0], cps[0], ctx.rel
    win = float(ctx.sem.get("dedup_window_days") or ctx.cfg.tx_window_days)
    qa, qc, qt = qident(a.name), qident(cp.name), qident(t)
    base = (f"SELECT {qc} AS counterparty, {qa} AS amount, CAST({qt} AS TIMESTAMP) AS t, "
            f"lag(CAST({qt} AS TIMESTAMP)) OVER (PARTITION BY {qc}, {qa} ORDER BY CAST({qt} AS TIMESTAMP)) AS prev_t "
            f"FROM {rel} WHERE {qa} IS NOT NULL AND {qa} <> 0 AND {qc} IS NOT NULL AND {qt} IS NOT NULL")
    cond = f"prev_t IS NOT NULL AND date_diff('second', prev_t, t) <= {win * 86400!r}"
    row = ctx.con.execute(
        f"WITH s AS ({base}) SELECT count_if({cond}), sum(abs(amount)) FILTER (WHERE {cond}), count_if({cond} AND t = prev_t), "
        f"count(DISTINCT (counterparty, amount)) FILTER (WHERE {cond}), count(*) FROM s").fetchone()
    pairs, stake, same_ts, groups, n = int(row[0] or 0), float(row[1] or 0), int(row[2] or 0), int(row[3] or 0), int(row[4] or 0)
    if pairs < ctx.cfg.tx_min_pairs or not n:
        return []
    share = pairs / n
    return [Fact(type="near_duplicate_tx", title=f"Near-duplicate transactions ({a.name} / {cp.name})", columns=[a.name, cp.name, t],
                 metrics={"amount": a.name, "counterparty": cp.name, "window": win, "pairs": pairs, "share": share,
                          "amount_at_stake": stake, "same_timestamp": same_ts, "groups": groups,
                          "inferred_roles": not (ctx.sem.columns.get(a.name) and ctx.sem.columns[a.name].role)},
                 severity="warning" if share >= 0.02 else "notable", score=min(1.0, share * 10),
                 evidence_sql=(f"WITH s AS ({base}) SELECT counterparty, amount, prev_t, t FROM s WHERE {cond} "
                               f"ORDER BY abs(amount) DESC LIMIT 100"))]
