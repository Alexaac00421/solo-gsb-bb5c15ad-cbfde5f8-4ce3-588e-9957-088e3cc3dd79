"""FastAPI 入口：HTTP API + 操作页面静态托管。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .models import (
    AlternativePlan,
    AlternativesInfo,
    BlockingConstraint,
    ScheduleDiffRequest,
    ScheduleDiffResponse,
    ScheduleItem,
    ScheduleRequest,
    ScheduleResponse,
    ScheduleToleranceRequest,
    ScheduleToleranceResponse,
    ServiceTolerance,
)
from .diff import build_diff, run_request
from .scheduler import solve_baseline
from .times import to_local_display
from .tolerance import analyze_tolerance

app = FastAPI(
    title="跨时区发布窗口编排",
    description="在时区/DST、依赖先后、禁发、冻结与资源约束下求可执行发布排程",
    version="1.0.0",
)

WEB_DIR = Path(__file__).parent / "web"


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/schedule", response_model=ScheduleResponse)
def schedule(req: ScheduleRequest) -> ScheduleResponse:
    run = run_request(req)

    if run.empty:
        return ScheduleResponse(
            feasible=False, status="infeasible",
            blockers=[BlockingConstraint(code="empty_services",
                                        message="至少需要一个服务")],
        )

    if run.validation_errors is not None:
        # 422：输入本身有问题（DST 重复/不存在、坏时区、依赖环、区间非法等）。
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=422,
                            content={"errors": run.validation_errors})

    assert run.norm is not None and run.result is not None
    norm, result = run.norm, run.result

    if not result.feasible:
        return ScheduleResponse(
            feasible=False,
            status=result.status,
            blockers=[BlockingConstraint(**b) for b in (result.blockers or [])],
        )

    def build_plan(starts: dict[str, int]) -> list[ScheduleItem]:
        return [_build_item(norm, sid, st)
                for sid, st in sorted(starts.items(), key=lambda kv: kv[1])]

    # 备选方案（仅在请求给出 alternative_count 时非空）：plans[0] 即首项，
    # 与顶层 schedule 内容一致，未启用备选时 alternatives 为 null，响应保持兼容。
    alternatives = None
    if result.plans is not None:
        alternatives = AlternativesInfo(
            requested=norm.alternative_count,
            min_gap_minutes=norm.min_gap_seconds // 60,
            status=result.alternatives_status,
            plans=[AlternativePlan(index=i, schedule=build_plan(p))
                   for i, p in enumerate(result.plans)],
        )

    return ScheduleResponse(
        feasible=True, status=result.status,
        schedule=build_plan(result.starts),
        alternatives=alternatives,
    )


def _build_item(norm, sid: str, st: int) -> ScheduleItem:
    s = norm.services[sid]
    end = st + s.duration
    occ = st + s.duration + s.rollback
    used = next(
        (w for w in s.windows if w.start <= st and occ <= w.end),
        None,
    )
    window_text = (
        f"{_iso(used.start)} ~ {_iso(used.end)} UTC（{used.label}）"
        if used else "（窗口信息缺失）"
    )
    return ScheduleItem(
        service_id=sid,
        name=s.name,
        deploy_start_utc=_iso(st),
        deploy_end_utc=_iso(end),
        occupied_until_utc=_iso(occ),
        local_time=to_local_display(datetime.fromtimestamp(st, tz=timezone.utc), s.timezone),
        timezone=s.timezone,
        window_used=window_text,
        resources=s.resources,
    )


@app.post("/api/schedule/diff", response_model=ScheduleDiffResponse)
def schedule_diff(body: ScheduleDiffRequest):
    """变更影响预演：分别求解 current / modified，按服务 ID 对齐后返回时刻差异。

    任一请求输入错误或无解时，只返回对应一侧的完整错误/阻塞约束组合，不输出差异；
    两份请求都无变化时返回空差异（changed/added/removed 均为空列表）。
    """
    current = run_request(body.current)
    modified = run_request(body.modified)

    resp = build_diff(current, modified)
    if not resp.feasible and resp.failures:
        # 只要存在输入错误（422 类）即以 422 返回；纯无解/超时仍为 200。
        if any(f.kind == "validation" for f in resp.failures.values()):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=422,
                content=resp.model_dump(mode="json"),
            )
    return resp


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


@app.post("/api/schedule/tolerance", response_model=ScheduleToleranceResponse)
def schedule_tolerance(body: ScheduleToleranceRequest):
    """发布评审：现有排程对每个未锁定服务单项延迟的容忍度。

    先按现有规则求基线（与 /api/schedule 首项完全一致）；基线无解（infeasible，
    含阻塞组合）或求解未定（unknown）时沿用原有输出、不输出容忍度；输入非法沿用
    现有 422 格式。基线可行时对每个未锁定服务，保持其他服务基线开始时刻不变，
    逐分钟推迟该服务（上限为 max_delay_minutes），给出从 0 起连续可行的最大延迟、
    上限内首个失效分钟及触发的约束；锁定服务标为不可推迟。
    """
    req = body.request

    if not req.services:
        return ScheduleToleranceResponse(
            feasible=False, status="infeasible",
            blockers=[BlockingConstraint(code="empty_services",
                                         message="至少需要一个服务")],
        )

    # 只做输入校验：容忍度始终基于首项基线，故直接求基线一次
    # （忽略请求中的 alternative_count / min_gap_minutes，避免额外求解备选方案）。
    from .validator import validate_and_normalize
    norm, validation_errors = validate_and_normalize(req)
    if validation_errors:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=422,
                            content={"errors": validation_errors})

    # 基线按原规则求解：无解（含阻塞核）或未定（unknown）均沿用原有输出，不输出容忍度。
    result = solve_baseline(norm)
    if not result.feasible:
        return ScheduleToleranceResponse(
            feasible=False,
            status=result.status,
            blockers=[BlockingConstraint(**b) for b in (result.blockers or [])],
        )

    assert result.starts is not None
    base = result.starts

    def build_plan(starts: dict[str, int]) -> list[ScheduleItem]:
        return [_build_item(norm, sid, st)
                for sid, st in sorted(starts.items(), key=lambda kv: (kv[1], kv[0]))]

    raw = analyze_tolerance(norm, base, body.max_delay_minutes,
                            norm.max_solve_seconds)
    tolerance = [
        ServiceTolerance(
            service_id=tr.service_id,
            name=norm.services[tr.service_id].name,
            deferrable=tr.deferrable,
            max_feasible_delay_minutes=tr.max_feasible_delay_minutes,
            first_failure_minutes=tr.first_failure_minutes,
            blockers=([BlockingConstraint(**b) for b in tr.blockers]
                      if tr.blockers is not None else None),
            status=tr.status,
        )
        for tr in raw
    ]

    return ScheduleToleranceResponse(
        feasible=True,
        status=result.status,
        schedule=build_plan(base),
        max_delay_minutes=body.max_delay_minutes,
        tolerance=tolerance,
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
