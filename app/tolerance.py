"""发布评审：现有排程对单项服务延迟的容忍度。

给定一份排程请求与其基线排程（/api/schedule 首项，按既有规则求出），对每个**未锁定**
服务，保持其他所有服务的基线开始时刻不变，把该服务的部署开始时刻从基线起逐分钟
（整分钟，UTC 纪元秒）向后推迟，求出：

- ``max_feasible_delay_minutes``：从 0 分钟起**连续**仍满足全部既有约束
  （候选窗、依赖、团队禁发、维护冻结、共享资源分时容量）的最大延迟（含 0 分钟本身），
  不超过请求上限；
- ``first_failure_minutes``：上限内首个失效分钟；上限内全部可行时为 null；
- ``blockers``：首个失效分钟触发的约束（沿用阻塞核 code/message 格式）。

锁定服务（pinned_starts）标为不可推迟（``deferrable=false``），不参与分析。

关键语义：判断的是“从 0 起连续可行”，一旦在分钟 d 失效就立即报告 d，**不会**因为
更晚的时刻（如穿过一段冻结之后）重新可行而越过 d；延迟后的部署区间与回滚占用区间
都按原规则参与全部约束。

可行性判定实现为与 scheduler.py 同一套语义的整数秒解析判定（候选窗包含、冻结占用避让、
禁发仅部署避让、依赖不等式、资源分时容量按分段并发计数），并在三个关键分钟
（0、首个失效分钟及其前一分钟；全可行时额外复核上限分钟）用 CP-SAT 固定时刻模型复核：
解析与 CP-SAT 结论一致才采信（首个失效分钟的阻塞核直接取自该次 CP-SAT 求解）；
任一复核点不一致即回退为 0..上限逐分钟 CP-SAT 精确探测，保证结果与求解器完全一致。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from .scheduler import _build_model
from .validator import Normalized

MINUTE = 60

# 固定时刻探测在极端情形下的单次求解时限（秒）；这类模型全部变量为常量等式，
# 通常在毫秒级完成，时限仅用于防止异常输入拖垮请求。
_PROBE_TIME_SECONDS = 5.0


@dataclass
class ProbeOutcome:
    """单个分钟点的 CP-SAT 探测结果。"""

    feasible: bool
    status: str  # feasible | infeasible | unknown
    blockers: list[dict] = field(default_factory=list)


@dataclass
class ServiceToleranceResult:
    service_id: str
    deferrable: bool
    # 未锁定服务：连续可行的最大分钟数；首个失效分钟（全可行时为 None）；
    # 首个失效分钟触发的约束（全可行/不可推迟时为 None）。
    max_feasible_delay_minutes: int | None = None
    first_failure_minutes: int | None = None
    blockers: list[dict] | None = None
    status: str = "ok"  # ok | unknown（探测超时未能判定的分钟，按不可越过处理）


def _point_capacity(norm: Normalized, resource: str, ts: int) -> int:
    """资源 resource 在时刻 ts（左闭右开）生效的容量：临时时段覆盖基准容量。"""
    cap = norm.resource_capacities.get(resource, 1)
    for p in norm.capacity_periods.get(resource, []):
        if p.start <= ts < p.end:
            return p.capacity
    return cap


def _analytic_feasible(norm: Normalized, base: dict[str, int], target: str,
                       delay_seconds: int) -> bool:
    """其他服务固定在基线、target 推迟 delay_seconds 时，是否满足全部既有约束。

    与 scheduler.py 的建模语义一一对应（全部时间为整数纪元秒，区间均左闭右开）。
    """
    services = norm.services
    s = services[target]
    t = base[target] + delay_seconds
    deploy_end = t + s.duration
    occupy_end = t + s.rollback + s.duration

    # ---- 候选时间窗：部署+回滚整个占用区间必须完整落在某个候选窗内 ----
    if not any(w.start <= t and occupy_end <= w.end for w in s.windows):
        return False

    # ---- 维护冻结：占用区间（含回滚）不得与任何冻结重叠 ----
    for f in norm.freezes:
        if t < f.end and occupy_end > f.start:
            return False

    # ---- 团队禁发：仅部署区间不得与所属团队禁发时段重叠（回滚允许进入） ----
    for team in norm.team_members.get(target, []):
        for p in norm.blackouts.get(team, []):
            if t < p.end and deploy_end > p.start:
                return False

    # ---- 依赖先后：target 作为下游 / 上游两个方向（其他服务固定在基线） ----
    for d in norm.dependencies:
        up = services[d.upstream]
        if d.downstream == target:
            if t < base[d.upstream] + up.duration + d.gap:
                return False
        if d.upstream == target:
            if base[d.downstream] < t + s.duration + d.gap:
                return False

    # ---- 共享资源（含分时容量）：target 占用期间任意时刻并发数 ≤ 当时容量 ----
    others_by_resource: dict[str, list[str]] = {}
    for sid, sv in services.items():
        if sid == target:
            continue
        for r in sv.resources:
            others_by_resource.setdefault(r, []).append(sid)

    for r in s.resources:
        others = others_by_resource.get(r, [])
        # 收集 [t, occupy_end) 内所有可能改变“占用集合或生效容量”的边界点；
        # 相邻边界点之间并发数与容量均恒定，故逐点（左闭右开语义）检查即可。
        points = {t}
        for u in others:
            b = base[u]
            e = b + services[u].duration + services[u].rollback
            if e > t and b < occupy_end:
                points.add(max(b, t))
                if e < occupy_end:
                    points.add(e)
        for p in norm.capacity_periods.get(r, []):
            if p.end > t and p.start < occupy_end:
                points.add(max(p.start, t))
                if p.end < occupy_end:
                    points.add(p.end)
        for at in sorted(points):
            if at >= occupy_end:
                continue
            active = 1  # target 自身在 [t, occupy_end) 内恒占用 1 个名额
            for u in others:
                b = base[u]
                e = b + services[u].duration + services[u].rollback
                if b <= at < e:
                    active += 1
            if active > _point_capacity(norm, r, at):
                return False

    return True


def probe_minute(norm: Normalized, base: dict[str, int], target: str,
                 delay_seconds: int, time_limit: float = _PROBE_TIME_SECONDS) -> ProbeOutcome:
    """用 CP-SAT 复核单个分钟点：其他服务硬固定在基线、target 硬固定在推迟时刻。

    业务约束仍挂在各自的假设字面量上，因此 INFEASIBLE 时可取
    sufficient-assumption 阻塞核，即该分钟触发的约束组合。
    """
    services = norm.services
    t = base[target] + delay_seconds
    occupy_end = t + services[target].duration + services[target].rollback

    # 延伸建模轴右端，避免 target 被固定到所有候选窗之后时出现“无关于任何假设”的不可行。
    ctx = _build_model(norm, h1_override=occupy_end)
    m = ctx.m
    for sid, st in base.items():
        m.add(ctx.starts[sid] == (t if sid == target else st))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.01, time_limit)
    solver.parameters.num_search_workers = 8
    status = solver.solve(m)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ProbeOutcome(feasible=True, status="feasible")
    if status == cp_model.INFEASIBLE:
        core = solver.sufficient_assumptions_for_infeasibility()
        if core:
            blockers = [
                {"code": ctx.assumption_map[lit][0],
                 "message": ctx.assumption_map[lit][1]}
                for lit in core
            ]
        else:
            # 理论上延伸时间轴后不会发生（所有矛盾均可由业务假设解释）；
            # 兜底：列出全部业务假设，保持“不静默”的阻塞报告。
            blockers = [{"code": code, "message": msg}
                        for code, msg in ctx.assumption_map.values()]
        return ProbeOutcome(feasible=False, status="infeasible", blockers=blockers)
    return ProbeOutcome(
        feasible=False, status="unknown",
        blockers=[{"code": "solver_unknown",
                  "message": "求解器在时限内未能判定该延迟分钟是否可行，"
                             "按约定不越过该分钟继续判定；可增大 max_solve_seconds 后重试"}],
    )


def _exhaustive(norm: Normalized, base: dict[str, int], target: str, cap: int,
                deadline: float) -> ServiceToleranceResult:
    """回退路径：0..cap 逐分钟用 CP-SAT 精确探测，直到首个失效分钟。"""
    for d in range(cap + 1):
        remaining = max(0.01, min(_PROBE_TIME_SECONDS, deadline - time.monotonic()))
        outcome = probe_minute(norm, base, target, d * MINUTE, remaining)
        if outcome.feasible:
            continue
        return ServiceToleranceResult(
            service_id=target,
            deferrable=True,
            max_feasible_delay_minutes=d - 1,
            first_failure_minutes=d,
            blockers=outcome.blockers,
            status="unknown" if outcome.status == "unknown" else "ok",
        )
    return ServiceToleranceResult(
        service_id=target, deferrable=True,
        max_feasible_delay_minutes=cap,
        first_failure_minutes=None, blockers=None)


def analyze_service(norm: Normalized, base: dict[str, int], target: str,
                    cap_minutes: int, deadline: float) -> ServiceToleranceResult:
    """求单个未锁定服务的延迟容忍度。"""
    # 解析判定：从 1 分钟起找首个失效分钟（0 分钟即基线，必然可行）。
    analytic_first_fail: int | None = None
    for d in range(1, cap_minutes + 1):
        if not _analytic_feasible(norm, base, target, d * MINUTE):
            analytic_first_fail = d
            break

    # CP-SAT 复核点：0 分钟（健全性）、首个失效分钟及其前一分钟；
    # 解析认为全可行时复核上限分钟（应当可行）。
    check_minutes = [0]
    if analytic_first_fail is not None:
        check_minutes.append(analytic_first_fail)
        if analytic_first_fail - 1 >= 1:
            check_minutes.append(analytic_first_fail - 1)
    else:
        check_minutes.append(cap_minutes)

    verified: dict[int, ProbeOutcome] = {}
    for d in sorted(set(check_minutes)):
        remaining = max(0.01, min(_PROBE_TIME_SECONDS, deadline - time.monotonic()))
        verified[d] = probe_minute(norm, base, target, d * MINUTE, remaining)

    # 解析与 CP-SAT 必须在每个复核点一致；任一不一致（含超时）即回退逐分钟精确探测。
    mismatch = False
    for d, outcome in verified.items():
        analytic_says_feasible = d != analytic_first_fail
        if outcome.status == "unknown" or outcome.feasible != analytic_says_feasible:
            mismatch = True
            break
    if mismatch:
        return _exhaustive(norm, base, target, cap_minutes, deadline)

    if analytic_first_fail is None:
        return ServiceToleranceResult(
            service_id=target, deferrable=True,
            max_feasible_delay_minutes=cap_minutes,
            first_failure_minutes=None, blockers=None)

    return ServiceToleranceResult(
        service_id=target, deferrable=True,
        max_feasible_delay_minutes=analytic_first_fail - 1,
        first_failure_minutes=analytic_first_fail,
        blockers=verified[analytic_first_fail].blockers)


def analyze_tolerance(norm: Normalized, base: dict[str, int],
                      cap_minutes: int, max_solve_seconds: float) -> list[ServiceToleranceResult]:
    """对全部服务求容忍度：锁定服务不可推迟；其余服务按基线开始时刻顺序输出（确定可复现）。"""
    # 探测预算：基线求解之外，容忍度分析共享一份 max_solve_seconds 预算。
    deadline = time.monotonic() + max_solve_seconds
    results: list[ServiceToleranceResult] = []
    for sid in sorted(norm.services, key=lambda u: (base[u], u)):
        if sid in norm.pinned_starts:
            results.append(ServiceToleranceResult(
                service_id=sid, deferrable=False,
                max_feasible_delay_minutes=None,
                first_failure_minutes=None, blockers=None))
            continue
        results.append(analyze_service(norm, base, sid, cap_minutes, deadline))
    return results
