"""Hybrid SimPy + agent-style boredom model for picker-to-parts order picking.

All behavioural coefficients are placeholders and must be calibrated before inference.
The loss decomposition is path-dependent: direct_only switches on own effects first,
then full adds partner interruption and contagion. A different switching order can
attribute interaction terms differently.
"""
from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simpy


@dataclass(frozen=True)
class Config:
    """All model parameters. Time unit inside SimPy is minutes."""

    shift_hours: float = 8.0  # hours
    team_size: int = 8  # pickers
    demand_profile: Tuple[float, ...] = (22, 34, 50, 64, 60, 36, 26, 20)  # orders/hour by shift hour
    demand_scale: float = 1.0  # multiplier
    lines_min: int = 1  # lines/order
    lines_max: int = 8  # lines/order
    travel_mean_min: float = 0.75  # minutes/line
    travel_cv: float = 0.45  # coefficient of variation
    pick_mean_min: float = 0.55  # minutes/line
    pick_cv: float = 0.35  # coefficient of variation
    handover_min: float = 0.7  # minutes/order
    rework_mean_min: float = 1.5  # minutes/error
    base_error_prob: float = 0.008  # probability/line
    error_cost_eur: float = 18.0  # EUR/error, includes rework/quality handling
    delay_cost_eur_per_order_min: float = 0.04  # EUR/order-minute
    unfinished_penalty_eur: float = 35.0  # EUR/order unfinished at shift end
    wage_eur_per_hour: float = 28.0  # EUR/paid hour
    rolling_window_min: int = 15  # minutes
    boredom_dt_min: float = 1.0  # minutes
    share_bored: float = 0.25  # share of team initially bored
    b0_bored: float = 0.60  # PLACEHOLDER, not empirically calibrated; plausible 0.4-0.8
    b0_low: float = 0.05  # PLACEHOLDER, not empirically calibrated; plausible 0.0-0.2
    a_up_per_min: float = 0.020  # PLACEHOLDER, not empirically calibrated; plausible 0.005-0.05 /min
    a_down_per_min: float = 0.030  # PLACEHOLDER, not empirically calibrated; plausible 0.01-0.08 /min
    contagion_per_min: float = 0.018  # PLACEHOLDER, not empirically calibrated; plausible 0.0-0.05 /min
    slowdown_strength: float = 0.35  # PLACEHOLDER, not empirically calibrated; plausible 0.1-0.7 multiplier at B=1
    error_boredom_slope: float = 0.025  # PLACEHOLDER, not empirically calibrated; plausible 0.005-0.06 probability at B=1
    conversation_lambda0_per_hour: float = 0.05  # PLACEHOLDER, not empirically calibrated; plausible 0.0-0.2 /h
    conversation_lambda1_per_hour: float = 1.0  # PLACEHOLDER, not empirically calibrated; plausible 0.2-2.0 /h at B=1
    conversation_mean_min: float = 2.0  # PLACEHOLDER, not empirically calibrated; plausible 0.5-5 min
    conversation_b_slope: float = 1.0  # PLACEHOLDER, not empirically calibrated; plausible 0.0-2.0 proportional increase
    conversation_cv: float = 0.55  # PLACEHOLDER, not empirically calibrated; plausible 0.2-1.0
    busy_partner_accept_prob: float = 0.55  # PLACEHOLDER, not empirically calibrated; plausible 0.2-0.9
    idle_partner_accept_prob: float = 0.90  # PLACEHOLDER, not empirically calibrated; plausible 0.5-1.0
    contact_mode: str = "random"  # random or neighbours
    contacts_per_update: int = 2  # PLACEHOLDER, not empirically calibrated; plausible 1-4 contacts/min
    intervention_threshold: float = 0.75  # PLACEHOLDER; plausible 0.6-0.9
    intervention_reset_fraction: float = 0.35  # PLACEHOLDER; plausible 0.1-0.6
    enable_intervention: bool = False
    replications: int = 100
    quick_replications: int = 10
    base_seed: int = 20261105
    output_dir: str = "boredom_outputs"


@dataclass
class Order:
    order_id: int
    arrival: float
    lines: int
    base_travel: np.ndarray
    base_pick: np.ndarray
    error_u: np.ndarray
    rework: np.ndarray
    start: Optional[float] = None
    completion: Optional[float] = None
    errors: int = 0


class Picker:
    """Plain-Python agent whose process is scheduled by SimPy."""

    def __init__(self, model: "Simulation", picker_id: int, initial_b: float):
        self.model = model
        self.env = model.env
        self.id = picker_id
        self.b = initial_b
        self.busy = False
        self.in_conversation = False
        self.pending_inviter: Optional[int] = None
        self.pending_duration = 0.0
        self.queue_wait_proc: Optional[simpy.Process] = None
        self.busy_history: Deque[Tuple[float, float]] = deque()
        self.busy_total = 0.0
        self.conv_total = 0.0
        self.slowdown_total = 0.0
        self.lines_done = 0
        self.process = self.env.process(self.run())

    def record_busy(self, duration: float) -> None:
        self.busy_total += duration
        self.busy_history.append((self.env.now, duration))

    def rolling_utilisation(self) -> float:
        cutoff = self.env.now - self.model.cfg.rolling_window_min
        while self.busy_history and self.busy_history[0][0] < cutoff:
            self.busy_history.popleft()
        return min(1.0, sum(x[1] for x in self.busy_history) / self.model.cfg.rolling_window_min)

    def maybe_start_conversation(self) -> Iterable[simpy.Event]:
        if not self.model.conversations_on or self.in_conversation:
            return
        cfg = self.model.cfg
        rate_per_min = (cfg.conversation_lambda0_per_hour + cfg.conversation_lambda1_per_hour * self.b) / 60.0
        if self.model.rng_conv.random() >= 1.0 - math.exp(-rate_per_min):
            return
        candidates = [p for p in self.model.pickers if p.id != self.id and not p.in_conversation and p.pending_inviter is None]
        if not candidates:
            return
        partner = candidates[int(self.model.rng_conv.integers(len(candidates)))]
        mean = cfg.conversation_mean_min * (1.0 + cfg.conversation_b_slope * self.b)
        duration = lognormal_from_mean_cv(self.model.rng_conv, mean, cfg.conversation_cv)
        if self.model.partner_interruptions_on:
            accept = cfg.idle_partner_accept_prob if not partner.busy else cfg.busy_partner_accept_prob
            if self.model.rng_conv.random() < accept:
                if partner.busy:
                    partner.pending_inviter, partner.pending_duration = self.id, duration
                else:
                    partner.pending_inviter, partner.pending_duration = self.id, duration
                    if partner.queue_wait_proc is not None and partner.queue_wait_proc.is_alive:
                        partner.queue_wait_proc.interrupt("conversation")
                yield from self.converse(duration, partner)
                return
        # direct_only: initiator loses time, partner is unaffected.
        yield from self.converse(duration, None)

    def converse(self, duration: float, partner: Optional["Picker"]) -> Iterable[simpy.Event]:
        self.in_conversation = True
        self.conv_total += duration
        if partner is not None:
            self.model.conversation_pairs[self.id, partner.id] = self.model.conversation_pairs.get((self.id, partner.id), 0.0) + duration
        yield self.env.timeout(duration)
        self.in_conversation = False

    def accept_pending(self) -> Iterable[simpy.Event]:
        if self.pending_inviter is None:
            return
        duration = self.pending_duration
        self.pending_inviter, self.pending_duration = None, 0.0
        self.in_conversation = True
        self.conv_total += duration
        yield self.env.timeout(duration)
        self.in_conversation = False

    def run(self) -> Iterable[simpy.Event]:
        while self.env.now < self.model.shift_min:
            yield from self.accept_pending()
            yield from self.maybe_start_conversation()
            get_event = self.model.queue.get()
            self.queue_wait_proc = self.env.active_process
            try:
                order: Order = yield get_event
            except simpy.Interrupt:
                get_event.cancel()
                self.queue_wait_proc = None
                yield from self.accept_pending()
                continue
            self.queue_wait_proc = None
            if order.start is None:
                order.start = self.env.now
            self.busy = True
            for i in range(order.lines):
                yield from self.accept_pending()
                yield from self.maybe_start_conversation()
                b = self.b if self.model.own_effects_on else 0.0
                base = float(order.base_travel[i] + order.base_pick[i])
                duration = base * (1.0 + self.model.cfg.slowdown_strength * b)
                self.slowdown_total += duration - base
                self.record_busy(duration)
                yield self.env.timeout(duration)
                self.lines_done += 1
                p_error = min(0.99, self.model.cfg.base_error_prob + self.model.cfg.error_boredom_slope * b)
                if order.error_u[i] < p_error:
                    order.errors += 1
                    rw = float(order.rework[i]) * (1.0 + self.model.cfg.slowdown_strength * b)
                    self.record_busy(rw)
                    yield self.env.timeout(rw)
            self.record_busy(self.model.cfg.handover_min)
            yield self.env.timeout(self.model.cfg.handover_min)
            self.busy = False
            order.completion = self.env.now
            self.model.completed.append(order)


class Simulation:
    def __init__(self, cfg: Config, scenario: str, seed: int):
        self.cfg, self.scenario = cfg, scenario
        self.env = simpy.Environment()
        self.shift_min = cfg.shift_hours * 60.0
        streams = np.random.SeedSequence(seed).spawn(5)
        self.rng_arrival, self.rng_content, self.rng_pick, self.rng_error, self.rng_conv = [np.random.default_rng(s) for s in streams]
        self.queue = simpy.Store(self.env)
        self.arrived: List[Order] = []
        self.completed: List[Order] = []
        self.b_trace: List[Tuple[float, float]] = []
        self.conversation_pairs: Dict[Tuple[int, int], float] = {}
        self.own_effects_on = scenario != "baseline"
        self.partner_interruptions_on = scenario == "full"
        self.contagion_on = scenario == "full"
        self.conversations_on = scenario != "baseline"
        n_bored = int(round(cfg.team_size * cfg.share_bored))
        self.pickers = [Picker(self, i, cfg.b0_bored if i < n_bored and scenario != "baseline" else (cfg.b0_low if scenario != "baseline" else 0.0)) for i in range(cfg.team_size)]
        self.env.process(self.arrivals())
        self.env.process(self.update_boredom())

    def contacts(self, i: int) -> List[Picker]:
        if self.cfg.team_size <= 1:
            return []
        if self.cfg.contact_mode == "neighbours":
            ids = {(i - 1) % self.cfg.team_size, (i + 1) % self.cfg.team_size}
            return [self.pickers[j] for j in ids if j != i]
        ids = [j for j in range(self.cfg.team_size) if j != i]
        k = min(self.cfg.contacts_per_update, len(ids))
        chosen = self.rng_conv.choice(ids, size=k, replace=False)
        return [self.pickers[int(j)] for j in chosen]

    def arrivals(self) -> Iterable[simpy.Event]:
        order_id = 0
        while self.env.now < self.shift_min:
            hour = min(int(self.env.now // 60), len(self.cfg.demand_profile) - 1)
            rate = self.cfg.demand_profile[hour] * self.cfg.demand_scale / 60.0
            if rate <= 0:
                yield self.env.timeout(1.0)
                continue
            wait = self.rng_arrival.exponential(1.0 / rate)
            next_boundary = (hour + 1) * 60.0
            if self.env.now + wait >= next_boundary:
                yield self.env.timeout(max(0.0, next_boundary - self.env.now))
                continue
            yield self.env.timeout(wait)
            if self.env.now >= self.shift_min:
                break
            lines = int(self.rng_content.integers(self.cfg.lines_min, self.cfg.lines_max + 1))
            order = Order(order_id, self.env.now, lines,
                          lognormal_array(self.rng_pick, self.cfg.travel_mean_min, self.cfg.travel_cv, lines),
                          lognormal_array(self.rng_pick, self.cfg.pick_mean_min, self.cfg.pick_cv, lines),
                          self.rng_error.random(lines),
                          self.rng_error.exponential(self.cfg.rework_mean_min, lines))
            order_id += 1
            self.arrived.append(order)
            yield self.queue.put(order)

    def update_boredom(self) -> Iterable[simpy.Event]:
        dt = self.cfg.boredom_dt_min
        while self.env.now < self.shift_min:
            yield self.env.timeout(dt)
            if self.scenario == "baseline":
                for p in self.pickers:
                    p.b = 0.0
            else:
                old = [p.b for p in self.pickers]
                new = []
                for i, p in enumerate(self.pickers):
                    underload = 1.0 - p.rolling_utilisation()
                    d_b = self.cfg.a_up_per_min * underload * (1.0 - old[i]) - self.cfg.a_down_per_min * (1.0 - underload) * old[i]
                    if self.contagion_on:
                        cs = self.contacts(i)
                        mean_c = float(np.mean([old[x.id] for x in cs])) if cs else old[i]
                        d_b += self.cfg.contagion_per_min * max(0.0, mean_c - old[i]) * (1.0 - old[i])
                    value = float(np.clip(old[i] + dt * d_b, 0.0, 1.0))
                    if self.cfg.enable_intervention and value > self.cfg.intervention_threshold:
                        value *= 1.0 - self.cfg.intervention_reset_fraction
                    new.append(value)
                for p, value in zip(self.pickers, new):
                    p.b = value
            self.b_trace.append((self.env.now, float(np.mean([p.b for p in self.pickers]))))

    def run(self) -> Tuple[Dict[str, float], pd.DataFrame]:
        self.env.run(until=self.shift_min)
        unfinished = [o for o in self.arrived if o.completion is None]
        flow = [o.completion - o.arrival for o in self.completed if o.completion is not None]
        errors = sum(o.errors for o in self.completed)
        conv = sum(p.conv_total for p in self.pickers)
        slowdown = sum(p.slowdown_total for p in self.pickers)
        lines = sum(o.lines for o in self.completed)
        capacity_cost = errors * self.cfg.error_cost_eur + sum(flow) * self.cfg.delay_cost_eur_per_order_min + len(unfinished) * self.cfg.unfinished_penalty_eur
        paid_cost = (conv + slowdown) / 60.0 * self.cfg.wage_eur_per_hour
        metrics = {
            "arrived": len(self.arrived), "completed": len(self.completed), "unfinished": len(unfinished),
            "lines": lines, "team_picks_per_hour": lines / self.cfg.shift_hours,
            "picks_per_picker_hour": lines / self.cfg.shift_hours / self.cfg.team_size,
            "error_rate": errors / max(lines, 1), "errors": errors,
            "idle_minutes": self.cfg.team_size * self.shift_min - sum(p.busy_total for p in self.pickers) - conv,
            "conversation_minutes": conv, "slowdown_minutes": slowdown,
            "mean_flow_time": float(np.mean(flow)) if flow else 0.0,
            "mean_b": float(np.mean([x[1] for x in self.b_trace])) if self.b_trace else 0.0,
            "capacity_cost": capacity_cost, "paid_time_cost": paid_cost,
            "conversation_pair_minutes": sum(self.conversation_pairs.values()),
            "conversation_symmetry_ok": float(conv + 1e-9 >= 2.0 * sum(self.conversation_pairs.values())),
            "conservation_ok": float(len(self.arrived) == len(self.completed) + len(unfinished)),
            "b_bounds_ok": float(all(0.0 <= x[1] <= 1.0 for x in self.b_trace)),
        }
        return metrics, pd.DataFrame(self.b_trace, columns=["minute", "mean_b"])


def lognormal_from_mean_cv(rng: np.random.Generator, mean: float, cv: float) -> float:
    sigma2 = math.log(1.0 + cv * cv)
    return float(rng.lognormal(math.log(mean) - sigma2 / 2.0, math.sqrt(sigma2)))


def lognormal_array(rng: np.random.Generator, mean: float, cv: float, n: int) -> np.ndarray:
    sigma2 = math.log(1.0 + cv * cv)
    return rng.lognormal(math.log(mean) - sigma2 / 2.0, math.sqrt(sigma2), n)


def expected_utilisation(cfg: Config) -> pd.DataFrame:
    mean_lines = (cfg.lines_min + cfg.lines_max) / 2.0
    expected_order_min = mean_lines * (cfg.travel_mean_min + cfg.pick_mean_min + cfg.base_error_prob * cfg.rework_mean_min) + cfg.handover_min
    rows = []
    for h, rate in enumerate(cfg.demand_profile):
        util = rate * cfg.demand_scale * expected_order_min / 60.0 / cfg.team_size
        rows.append({"hour": h + 1, "orders_per_hour": rate * cfg.demand_scale, "expected_utilisation": util})
    return pd.DataFrame(rows)


def run_replications(cfg: Config, scenarios: Iterable[str], replications: int, seed_offset: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records, traces = [], []
    for r in range(replications):
        seed = cfg.base_seed + seed_offset + r
        for scenario in scenarios:
            metrics, trace = Simulation(cfg, scenario, seed).run()
            records.append({"replication": r, "scenario": scenario, **metrics})
            trace["replication"], trace["scenario"] = r, scenario
            traces.append(trace)
    return pd.DataFrame(records), pd.concat(traces, ignore_index=True)


def decomposition(df: pd.DataFrame, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    group_cols = group_cols or []
    idx = group_cols + ["replication"]
    p = df.pivot_table(index=idx, columns="scenario", values="capacity_cost").reset_index()
    p["direct_loss"] = p["direct_only"] - p["baseline"]
    p["indirect_loss"] = p["full"] - p["direct_only"]
    p["total_loss"] = p["full"] - p["baseline"]
    p["spillover_ratio"] = np.where(np.abs(p["total_loss"]) > 1e-9, p["indirect_loss"] / p["total_loss"], np.nan)
    return p


def summarise(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    rows = []
    groups = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for keys, g in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metrics:
            x = g[metric].dropna().to_numpy()
            row[f"{metric}_mean"] = float(np.mean(x)) if len(x) else np.nan
            row[f"{metric}_ci95"] = float(1.96 * np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def sanity_checks(cfg: Config) -> None:
    off = replace(cfg, share_bored=0.0, b0_bored=0.0, b0_low=0.0, a_up_per_min=0.0, contagion_per_min=0.0,
                  slowdown_strength=0.0, error_boredom_slope=0.0, conversation_lambda0_per_hour=0.0,
                  conversation_lambda1_per_hour=0.0)
    vals = {}
    for s in ("baseline", "direct_only", "full"):
        vals[s], _ = Simulation(off, s, cfg.base_seed).run()
    equal = all(vals[s][k] == vals["baseline"][k] for s in ("direct_only", "full") for k in ("completed", "errors", "capacity_cost"))
    m1, _ = Simulation(cfg, "full", cfg.base_seed + 999).run()
    m2, _ = Simulation(cfg, "full", cfg.base_seed + 999).run()
    checks = {
        "effects-off equality": equal,
        "B bounded": bool(m1["b_bounds_ok"]),
        "order conservation": bool(m1["conservation_ok"]),
        "conversation symmetry/accounting": bool(m1["conversation_symmetry_ok"]),
        "same seed reproducible": m1 == m2,
    }
    for name, ok in checks.items():
        print(f"SANITY {name}: {'PASS' if ok else 'FAIL'}")
    assert all(checks.values())


def parameter_sweep(cfg: Config, name: str, values: Iterable[float], reps: int, offset: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = []
    for j, value in enumerate(values):
        c = replace(cfg, **{name: value})
        df, _ = run_replications(c, ("baseline", "direct_only", "full"), reps, offset + j * 10000)
        df[name] = value
        raw.append(df)
    raw_df = pd.concat(raw, ignore_index=True)
    return raw_df, decomposition(raw_df, [name])


def make_outputs(cfg: Config, reps: int) -> None:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sanity_checks(cfg)
    util = expected_utilisation(cfg)
    print("\nExpected utilisation by hour (before boredom):")
    print(util.to_string(index=False, formatters={"expected_utilisation": "{:.1%}".format}))
    util.to_csv(out / "expected_utilisation.csv", index=False)

    main, traces = run_replications(cfg, ("baseline", "direct_only", "full"), reps)
    main.to_csv(out / "main_runs.csv", index=False)
    dec = decomposition(main)
    dec.to_csv(out / "loss_decomposition_runs.csv", index=False)
    summary = main.groupby("scenario").mean(numeric_only=True).reset_index()
    loss_summary = summarise(dec, [], ["direct_loss", "indirect_loss", "total_loss", "spillover_ratio"])
    print("\nScenario means:")
    print(summary[["scenario", "team_picks_per_hour", "error_rate", "conversation_minutes", "mean_flow_time", "unfinished", "mean_b", "capacity_cost", "paid_time_cost"]].round(3).to_string(index=False))
    print("\nLoss decomposition, mean +/- 95% CI:")
    print(loss_summary.round(3).to_string(index=False))
    summary.to_csv(out / "scenario_summary.csv", index=False)
    loss_summary.to_csv(out / "loss_summary.csv", index=False)

    for name, values, offset in [
        ("share_bored", [0, 0.125, 0.25, 0.5, 0.75], 100000),
        ("demand_scale", [0.6, 0.8, 1.0, 1.2], 200000),
        ("team_size", [4, 8, 12, 16], 300000),
    ]:
        raw, d = parameter_sweep(cfg, name, values, reps, offset)
        raw.to_csv(out / f"sweep_{name}_runs.csv", index=False)
        s = summarise(d, [name], ["total_loss", "spillover_ratio"])
        s.to_csv(out / f"sweep_{name}_summary.csv", index=False)

    sensitivity = []
    params = ["slowdown_strength", "error_boredom_slope", "conversation_lambda1_per_hour", "contagion_per_min", "a_up_per_min", "conversation_mean_min"]
    for j, p in enumerate(params):
        base = getattr(cfg, p)
        for level, mult in (("low", 0.5), ("high", 1.5)):
            c = replace(cfg, **{p: base * mult})
            d = decomposition(run_replications(c, ("baseline", "direct_only", "full"), reps, 400000 + j * 20000 + (0 if level == "low" else 10000))[0])
            sensitivity.append({"parameter": p, "level": level, "total_loss": d.total_loss.mean(), "spillover_ratio": d.spillover_ratio.mean()})
    sens = pd.DataFrame(sensitivity)
    sens.to_csv(out / "sensitivity_summary.csv", index=False)

    mean_trace = traces.groupby(["scenario", "minute"], as_index=False).mean(numeric_only=True)
    plt.figure(figsize=(8, 4.5))
    for s, g in mean_trace.groupby("scenario"):
        plt.plot(g.minute / 60, g.mean_b, label=s)
    plt.xlabel("Shift hour"); plt.ylabel("Mean boredom B"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "01_mean_boredom.png", dpi=160); plt.close()

    ls = loss_summary.iloc[0]
    plt.figure(figsize=(5, 4.5)); plt.bar(["Loss"], [ls.direct_loss_mean], label="Direct")
    plt.bar(["Loss"], [ls.indirect_loss_mean], bottom=[ls.direct_loss_mean], label="Indirect")
    plt.ylabel("EUR per shift"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "02_direct_indirect_loss.png", dpi=160); plt.close()

    for fname, x, label in [("sweep_share_bored_summary.csv", "share_bored", "Share initially bored"), ("sweep_demand_scale_summary.csv", "demand_scale", "Demand scale")]:
        d = pd.read_csv(out / fname)
        plt.figure(figsize=(6, 4.5)); plt.errorbar(d[x], d.spillover_ratio_mean, yerr=d.spillover_ratio_ci95, marker="o", capsize=3)
        plt.xlabel(label); plt.ylabel("Spillover ratio"); plt.axhline(0, color="black", linewidth=.7); plt.tight_layout()
        number = "03" if x == "share_bored" else "04"
        plt.savefig(out / f"{number}_spillover_{x}.png", dpi=160); plt.close()

    base_loss = dec.total_loss.mean()
    piv = sens.pivot(index="parameter", columns="level", values="total_loss")
    piv["span"] = (piv.high - piv.low).abs(); piv = piv.sort_values("span")
    plt.figure(figsize=(8, 5)); y = np.arange(len(piv))
    plt.barh(y, piv.high - piv.low, left=piv.low)
    plt.axvline(base_loss, color="black", linestyle="--", label="Base")
    plt.yticks(y, piv.index); plt.xlabel("Total capacity loss, EUR/shift"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "05_tornado_total_loss.png", dpi=160); plt.close()
    print(f"\nSaved CSV and PNG outputs to: {out.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="Use 10 replications per experiment.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.output_dir is not None:
        cfg = replace(cfg, output_dir=args.output_dir)
    if args.seed is not None:
        cfg = replace(cfg, base_seed=args.seed)
    make_outputs(cfg, cfg.quick_replications if args.quick else cfg.replications)


if __name__ == "__main__":
    main()
