"""SRT 字幕时间校准工具后端。

核心逻辑：根据用户给出的若干「原字幕时刻 -> 视频实际时刻」锚点，
按原时刻排序后在相邻锚点间做线性插值，两端按最近一段的斜率外推，
统一换算每条字幕的起止时间，并检测负时间 / 起止倒置 / 与下一条重叠。
"""
import csv
import difflib
import io
import math
import re
import unicodedata
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

app = FastAPI(title="SRT 字幕时间校准 / 片段剪辑 / 双语合成 / 阅读速度修复 / 修订差异对照 / 目标词时间标记 / 字幕遮挡避让 / 字幕章节索引 / 字幕空隙提示编排 / 角色抢话分析 / 术语统一替换")

STATIC_DIR = Path(__file__).parent / "static"

TIME_RE = re.compile(r"^\s*(\d{1,3}):([0-5]?\d):([0-5]?\d)[,.](\d{1,3})\s*$")
ARROW_RE = re.compile(r"-->")


# ---------------------------------------------------------------- 时间工具

def parse_time(s: str) -> int:
    """把 'HH:MM:SS,mmm' 解析为毫秒整数，失败抛 ValueError。"""
    m = TIME_RE.match(s or "")
    if not m:
        raise ValueError(f"无法解析时间: {s!r}（应为 HH:MM:SS,mmm）")
    h, mnt, sec, ms = (int(g) for g in m.groups())
    ms = int(str(m.group(4)).ljust(3, "0"))  # 允许 1~3 位毫秒
    return ((h * 60 + mnt) * 60 + sec) * 1000 + ms


def fmt_time(ms: int) -> str:
    """毫秒格式化为 SRT 时间；负数带负号以便预览中识别。"""
    sign = "-" if ms < 0 else ""
    ms = abs(int(round(ms)))
    h, ms = divmod(ms, 3600_000)
    mnt, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{sign}{h:02d}:{mnt:02d}:{sec:02d},{ms:03d}"


# ---------------------------------------------------------------- SRT 解析

def parse_srt(text: str) -> List[dict]:
    """解析 SRT，保留序号与正文。返回 [{index, start, end, lines}]。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n﻿ ")
    if not text:
        raise ValueError("SRT 内容为空")
    entries = []
    for block in re.split(r"\n[ \t]*\n", text):
        lines = [ln for ln in block.strip("\n").split("\n") if ln.strip() != ""]
        if not lines:
            continue
        # 找到包含 --> 的时间行，其上一行（若有）为序号
        ti = next((i for i, ln in enumerate(lines) if ARROW_RE.search(ln)), None)
        if ti is None:
            raise ValueError(f"缺少时间行 '-->'：{lines[0][:40]!r}")
        parts = ARROW_RE.split(lines[ti])
        if len(parts) != 2:
            raise ValueError(f"时间行格式错误：{lines[ti]!r}")
        # 去掉时间行中可能附带的定位信息（如 'X1:.. X2:..'）
        start_s = parts[0].strip()
        end_s = parts[1].strip().split()[0] if parts[1].strip() else ""
        try:
            start, end = parse_time(start_s), parse_time(end_s)
        except ValueError as e:
            raise ValueError(f"第 {len(entries) + 1} 块：{e}")
        index = lines[0].strip() if ti > 0 else str(len(entries) + 1)
        entries.append({
            "index": index,
            "start": start,
            "end": end,
            "lines": lines[ti + 1:],
        })
    if not entries:
        raise ValueError("未解析到任何字幕块")
    return entries


# ---------------------------------------------------------------- 锚点与映射

def build_mapper(anchors_ms: List[tuple]):
    """anchors_ms: [(orig_ms, actual_ms)]，已排序且已校验。
    返回分段线性映射函数：相邻锚点线性插值，两端按最近一段斜率外推。"""
    pts = anchors_ms

    def map_t(t: float) -> float:
        if t <= pts[0][0]:
            (o0, a0), (o1, a1) = pts[0], pts[1]
        elif t >= pts[-1][0]:
            (o0, a0), (o1, a1) = pts[-2], pts[-1]
        else:
            for i in range(len(pts) - 1):
                if pts[i][0] <= t <= pts[i + 1][0]:
                    (o0, a0), (o1, a1) = pts[i], pts[i + 1]
                    break
        slope = (a1 - a0) / (o1 - o0)
        return a0 + (t - o0) * slope

    return map_t


def validate_anchors(raw: List["Anchor"]) -> List[tuple]:
    """校验并排序锚点，返回 [(orig_ms, actual_ms)]。失败抛 ValueError。"""
    if len(raw) < 2:
        raise ValueError("至少需要 2 个锚点")
    pts = []
    for i, a in enumerate(raw, 1):
        try:
            pts.append((parse_time(a.orig), parse_time(a.actual)))
        except ValueError as e:
            raise ValueError(f"锚点 {i}：{e}")
    pts.sort(key=lambda p: p[0])
    for i in range(1, len(pts)):
        if pts[i][0] == pts[i - 1][0]:
            raise ValueError(f"锚点原时刻重复：{fmt_time(pts[i][0])}")
        if pts[i][1] <= pts[i - 1][1]:
            raise ValueError(
                f"映射后的实际时刻必须严格递增：{fmt_time(pts[i - 1][1])} 之后不能是 {fmt_time(pts[i][1])}"
            )
    return pts


def plan_anchor_suggestion(raw: List["Anchor"]) -> dict:
    """锚点校验失败时的异常定位：求需要剔除锚点最少的修复方案。

    按原时刻排序后，选出「原时刻、实际时刻都严格递增」的最长子序列（至少 2 点）。
    多解时决胜顺序：
      1) 最长（剔除行数最少）；
      2) 首尾原时刻（全局最小/最大原时刻）锚点的保留数量最多；
      3) 剔除项的原输入行号序列字典序最小。
    返回规划字典；无法凑出 2 点时 feasible=False（无法预演）。
    """
    n = len(raw)
    if n < 2:
        return {"feasible": False, "reason": "至少需要 2 个锚点，无法预演；请补充锚点后重试。"}

    parsed = []  # (输入行号, 原时刻ms, 实际时刻ms)
    bad = {}     # 无法解析的行号 -> 原因（必然要剔除）
    for i, a in enumerate(raw, 1):
        try:
            parsed.append((i, parse_time(a.orig), parse_time(a.actual)))
        except ValueError as e:
            bad[i] = str(e)
    if len(parsed) < 2:
        return {
            "feasible": False,
            "reason": "时间格式可解析的有效锚点不足 2 个，无法预演；请按 HH:MM:SS,mmm 修正时间格式。",
        }

    # 按原时刻排序（相同原时刻再按实际时刻、行号兜底，保证确定性）；
    # 转移时同时要求 orig、actual 严格小于，故重复原时刻不会串链。
    nodes = sorted(parsed, key=lambda p: (p[1], p[2], p[0]))
    m = len(nodes)
    row_orig = {r: o for r, o, _ in parsed}
    row_actual = {r: c for r, _, c in parsed}
    min_o = min(o for _, o, _ in parsed)
    max_o = max(o for _, o, _ in parsed)
    all_rows = set(range(1, n + 1))
    orig_count: dict = {}
    for _, o, _ in parsed:
        orig_count[o] = orig_count.get(o, 0) + 1

    def endpoint_hits(chain: tuple) -> int:
        """链中属于首/尾原时刻（全局最小、最大原时刻）的锚点数。"""
        return sum(1 for r in chain if row_orig[r] == min_o or row_orig[r] == max_o)

    def better(c1: tuple, c2: tuple) -> bool:
        """完整保留链的优劣比较：长度 > 首尾保留数 > 剔除行号序列字典序。"""
        if len(c1) != len(c2):
            return len(c1) > len(c2)
        e1, e2 = endpoint_hits(c1), endpoint_hits(c2)
        if e1 != e2:
            return e1 > e2
        rem1 = tuple(sorted(all_rows.difference(c1)))
        rem2 = tuple(sorted(all_rows.difference(c2)))
        return rem1 < rem2

    # O(m^2) DP：dp[i] 为以 nodes[i] 结尾的最优保留链（链元素为输入行号）
    dp = []
    for i in range(m):
        ri, oi, ai = nodes[i]
        best = (ri,)
        for j in range(i):
            rj, oj, aj = nodes[j]
            if oj < oi and aj < ai:
                cand = (*dp[j], ri)
                if better(cand, best):
                    best = cand
        dp.append(best)
    chain = dp[0]
    for c in dp[1:]:
        if better(c, chain):
            chain = c

    if len(chain) < 2:
        return {
            "feasible": False,
            "reason": "找不到两个原时刻与实际时刻都严格递增的锚点，无法预演；请手动修正锚点时刻。",
        }

    kept_rows = set(chain)
    removed = []
    for r in sorted(all_rows - kept_rows):
        if r in bad:
            reason = bad[r]
        elif orig_count[row_orig[r]] > 1:
            reason = f"原时刻 {fmt_time(row_orig[r])} 与其他锚点重复"
        else:
            reason = "实际时刻未随原时刻严格递增，剔除后保留锚点整体最优"
        removed.append({
            "row": r,
            "orig": raw[r - 1].orig,
            "actual": raw[r - 1].actual,
            "reason": reason,
        })
    kept = [
        {"row": r, "orig": fmt_time(row_orig[r]), "actual": fmt_time(row_actual[r])}
        for r in chain
    ]
    pts = [(row_orig[r], row_actual[r]) for r in chain]
    return {
        "feasible": True,
        "remove_rows": sorted(all_rows - kept_rows),
        "kept": kept,
        "removed": removed,
        "_pts": pts,
    }


def calibrate(entries: List[dict], map_t) -> List[dict]:
    """换算每条字幕起止时间并打问题标记。"""
    out = []
    for e in entries:
        ns, ne = int(round(map_t(e["start"]))), int(round(map_t(e["end"])))
        flags = []
        if ns < 0 or ne < 0:
            flags.append("负时间")
        if ne <= ns:
            flags.append("起止倒置")
        out.append({**e, "new_start": ns, "new_end": ne, "flags": flags})
    for i in range(len(out) - 1):
        if out[i]["new_end"] > out[i + 1]["new_start"]:
            out[i]["flags"].append("与下一条重叠")
    return out


# ---------------------------------------------------------------- API

class Anchor(BaseModel):
    orig: str    # 原字幕时刻 HH:MM:SS,mmm
    actual: str  # 视频实际时刻 HH:MM:SS,mmm


class CalibrateRequest(BaseModel):
    srt: str
    anchors: List[Anchor]


def _prepare(req: CalibrateRequest):
    """公共预处理，成功返回 (entries, calibrated)，失败抛 400。

    锚点校验类错误 detail 为 {"type": "anchor_invalid", "message": ...}，
    便于前端据此调用 /api/suggest 获取剔除建议；SRT 解析错误仍返回纯文本。
    """
    try:
        pts = validate_anchors(req.anchors)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail={"type": "anchor_invalid", "message": str(e)}
        )
    try:
        entries = parse_srt(req.srt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return calibrate(entries, build_mapper(pts))


def _preview_payload(rows: List[dict]) -> dict:
    return {
        "ok": all(not r["flags"] for r in rows),
        "problem_count": sum(1 for r in rows if r["flags"]),
        "entries": [
            {
                "index": r["index"],
                "old": f"{fmt_time(r['start'])} --> {fmt_time(r['end'])}",
                "new": f"{fmt_time(r['new_start'])} --> {fmt_time(r['new_end'])}",
                "flags": r["flags"],
                "text": "\n".join(r["lines"]),
            }
            for r in rows
        ],
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/preview")
def preview(req: CalibrateRequest):
    rows = _prepare(req)
    return _preview_payload(rows)


@app.post("/api/suggest")
def suggest(req: CalibrateRequest):
    """锚点校验失败时的异常定位建议。

    返回需要剔除的最少锚点；确认前不改动任何输入，确认后由前端按原流程
    （/api/preview、/api/export）继续。无法凑出 2 个有效锚点时提示无法预演。
    """
    plan = plan_anchor_suggestion(req.anchors)
    if not plan["feasible"]:
        return {"feasible": False, "reason": plan["reason"]}

    pts = plan.pop("_pts")
    # 用保留锚点复用现有校准预览，展示受影响字幕；SRT 本身无法解析时仅提示
    try:
        rows = calibrate(parse_srt(req.srt), build_mapper(pts))
        plan["preview"] = _preview_payload(rows)
        plan["preview_error"] = None
    except ValueError as e:
        plan["preview"] = None
        plan["preview_error"] = str(e)
    return plan


# ---------------------------------------------------------------- 剪辑片段

class KeepInterval(BaseModel):
    start: str  # 原视频保留区间起点 HH:MM:SS,mmm
    end: str    # 原视频保留区间终点 HH:MM:SS,mmm


class ClipRequest(BaseModel):
    srt: str
    intervals: List[KeepInterval]


def validate_intervals(raw: List["KeepInterval"]) -> List[tuple]:
    """校验保留区间并按起点排序，返回 [(start_ms, end_ms, 输入行号)]。

    规则：起点须严格早于终点；区间为左闭右开，故相邻区间端点相触
    （前段终点 == 后段起点）不算重叠，严格交叉才报错。
    """
    rows = [r for r in raw if r.start.strip() or r.end.strip()]
    if not rows:
        raise ValueError("请至少填写 1 个保留区间")
    parsed = []
    for i, r in enumerate(rows, 1):
        try:
            s, e = parse_time(r.start), parse_time(r.end)
        except ValueError as ex:
            raise ValueError(f"保留区间 {i}：{ex}")
        if s >= e:
            raise ValueError(
                f"保留区间 {i}：起点 {fmt_time(s)} 须早于终点 {fmt_time(e)}"
            )
        parsed.append((s, e, i))
    parsed.sort(key=lambda x: (x[0], x[1], x[2]))
    # 按起点排序后，相邻区间若不重叠则任意区间都不可能重叠
    for k in range(1, len(parsed)):
        ps, pe, prow = parsed[k - 1]
        s, e, row = parsed[k]
        if s < pe:  # 左闭右开：s == pe 只是端点相触，允许
            raise ValueError(
                f"保留区间重叠：第 {prow} 行（{fmt_time(ps)} – {fmt_time(pe)}）"
                f"与第 {row} 行（{fmt_time(s)} – {fmt_time(e)}）"
            )
    return parsed


def clip_entries(entries: List[dict], intervals: List[tuple]):
    """按左闭右开区间裁剪字幕，返回 (新字幕列表, 保留区间总时长ms)。

    对每条字幕 [start, end) 与每个保留区间求交集：空交集丢弃；跨多个
    区间的字幕按区间拆成多条，正文原样保留。交集时间换算到拼接后的
    连续时间轴：新时间 = 交集时间 - 区间起点 + 此前所有区间时长之和。
    输出按新起点排序（起点相同保持原顺序）并从 1 连续重编号。
    """
    out = []
    offset = 0
    for s, e, _row in intervals:
        for order, entry in enumerate(entries):
            lo, hi = max(entry["start"], s), min(entry["end"], e)
            if lo < hi:  # 左闭右开：lo == hi 为空交集
                out.append({
                    "index": "",
                    "src_index": entry["index"],
                    "src_order": order,
                    "old_start": lo,
                    "old_end": hi,
                    "start": offset + lo - s,
                    "end": offset + hi - s,
                    "lines": entry["lines"],
                })
        offset += e - s
    out.sort(key=lambda r: (r["start"], r["src_order"]))
    for n, r in enumerate(out, 1):
        r["index"] = str(n)
    return out, offset


def _clip_prepare(req: ClipRequest):
    """剪辑功能公共预处理，成功返回 (entries, intervals)，失败抛 400。"""
    try:
        intervals = validate_intervals(req.intervals)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        entries = parse_srt(req.srt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return entries, intervals


@app.post("/api/clip/preview")
def clip_preview(req: ClipRequest):
    entries, intervals = _clip_prepare(req)
    rows, total_ms = clip_entries(entries, intervals)
    return {
        "entry_count": len(rows),
        "interval_count": len(intervals),
        "total_duration": fmt_time(total_ms),
        "entries": [
            {
                "index": r["index"],
                "source": r["src_index"],
                "old": f"{fmt_time(r['old_start'])} --> {fmt_time(r['old_end'])}",
                "new": f"{fmt_time(r['start'])} --> {fmt_time(r['end'])}",
                "text": "\n".join(r["lines"]),
            }
            for r in rows
        ],
    }


@app.post("/api/clip/export")
def clip_export(req: ClipRequest):
    entries, intervals = _clip_prepare(req)
    rows, _ = clip_entries(entries, intervals)
    if not rows:
        raise HTTPException(
            status_code=400, detail="所有字幕均与保留区间无交集，没有可导出的内容"
        )
    blocks = [
        f"{r['index']}\n{fmt_time(r['start'])} --> {fmt_time(r['end'])}\n"
        + "\n".join(r["lines"])
        for r in rows
    ]
    content = "\n\n".join(blocks) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="clipped.srt"'},
    )


# ---------------------------------------------------------------- 双语合成

class MergeRequest(BaseModel):
    main: str         # 主轨 SRT
    translation: str  # 译文轨 SRT


MAX_MERGE_EDGES = 2_000_000  # 正时长交叠候选对上限（含残差边开销）


def _overlap_ms(a: dict, b: dict) -> int:
    """两条字幕的正时长交叠毫秒数；端点相触 / 不交叠返回 0。"""
    lo, hi = max(a["start"], b["start"]), min(a["end"], b["end"])
    return hi - lo if hi > lo else 0


def _validate_track_duration(entries: List[dict], track: str) -> None:
    """任一轨出现空时长（end == start）或起止倒置（end < start）即报错。"""
    for i, e in enumerate(entries, 1):
        if e["end"] <= e["start"]:
            kind = "空时长" if e["end"] == e["start"] else "起止时间倒置"
            raise ValueError(
                f"{track}第 {i} 条字幕{kind}："
                f"{fmt_time(e['start'])} --> {fmt_time(e['end'])}"
            )


def _track_is_ordered(entries: List[dict]) -> bool:
    """轨内条目是否按 (start, end) 均非降排列；正常 SRT 时间轴均满足。"""
    return all(
        entries[i]["start"] <= entries[i + 1]["start"]
        and entries[i]["end"] <= entries[i + 1]["end"]
        for i in range(len(entries) - 1)
    )


def _weight_params(m: int, n: int):
    """三级目标的整数权重参数（详见 pair_entries 文档）。

    每条译文在任一可行配对中至多使用一次（容量 1），故「已用译文序号集合」
    可用无进位的二进制位掩码表示：译文 j（0 起）贡献 2**(n-1-j)，最小序号
    为最高位。等大小集合的升序序号序列字典序越小，该掩码越大；掩码总和
    < 2**n。配对固定奖励取 2**n，2m 份之和仍小于一个交叠毫秒位，从而：
    交叠毫秒数 > 已配数（少漏配）> 序号序列字典序。
    """
    seq_bit = [1 << (n - 1 - j) for j in range(n)]
    shift_seq = n
    match_bonus = 1 << shift_seq
    shift_cnt = shift_seq + (2 * m).bit_length() + 1
    return seq_bit, match_bonus, shift_cnt


def _pair_ordered_dp(mains: List[dict], trans: List[dict]):
    """两轨均 (start,end) 非降时的保序最优配对（带状窗口，典型近线性）。

    f[i][j][k]：第 i 条主轨已配 k∈{0,1,2} 条译文、译文已消费到第 j 条时
    的最优权重。k==0 可跳过译文 j 或闭合当前主轨；k>0 可跳过译文或把译文 j
    配给当前主轨（仅正时长交叠）。两轨有序时交叠矩阵为 Monge 型，保序最优
    即全局最优。

    有序轨上每条主轨的正交叠译文下标是连续区间，且其左/右端点随主轨单调，
    故每行只需在 [L_i, R_i] 窗口内计算状态；轨内互不重叠（正常 SRT）时
    每个译文窗口只被常数个主轨覆盖，总状态 O(m+n)。轨内可有序重叠时窗口变
    宽，最差退化到 O(m*n)，仍正确。
    """
    import bisect

    m, n = len(mains), len(trans)
    seq_bit, bonus, shift_cnt = _weight_params(m, n)
    t_starts = [e["start"] for e in trans]
    t_ends = [e["end"] for e in trans]

    # 主轨 i 的正交叠译文下标区间 first_i .. last_i（TE、TS 均非降）
    first, last = [], []
    for me in mains:
        a = bisect.bisect_right(t_ends, me["start"])      # 第一个 end_j > start_i
        b = bisect.bisect_left(t_starts, me["end"]) - 1  # 最后一个 start_j < end_i
        first.append(min(a, n))
        last.append(b)  # first_i <= last_i 时才存在交叠译文

    # 每行窗口 [L_i, R_i]：左端承接下一主轨首个可能译文，右端覆盖到下一主轨左端
    left = [0] * (m + 1)
    right = [0] * (m + 1)
    for i in range(m):
        if i >= 1:
            left[i] = first[i]
        r = last[i] + 1  # 本行最后交叠译文的下一列（关闭/跳过传播所需）
        if i + 1 < m:
            r = max(r, first[i + 1])
        right[i] = min(r, n)
    left[m] = first[m - 1] if m else 0
    right[m] = n
    for i in range(1, m + 1):  # 窗口左端随主轨单调不减，钳制抖动
        if left[i] < left[i - 1]:
            left[i] = left[i - 1]
    for i in range(m + 1):
        if right[i] < left[i]:
            right[i] = left[i]

    neg = -(1 << 230)
    TAG_SKIP, TAG_MATCH, TAG_ROOT = 0, 1, 254
    TAG_CLOSE0, TAG_INVALID = 2, 255
    tags = {}

    def gix(i, j, k):
        return (i * (n + 1) + j) * 3 + k

    prev = None          # 上一行窗口内的状态 dict: (j,k) -> value
    p_lo = p_hi = 0
    for i in range(m + 1):
        lo, hi = left[i], right[i]
        cur = {}
        si = mains[i]["start"] if i < m else 0
        ei = mains[i]["end"] if i < m else 0

        def pget(j, k):
            if prev is None:
                return 0 if (j == 0 and k == 0) else neg
            if p_lo <= j <= p_hi:
                return prev.get((j, k), neg)
            return neg

        # 窗口首列 k=0：由上一行闭合而来（i==0 且 lo==0 为起点）
        if prev is None and lo == 0:
            cur[(lo, 0)] = 0
            tags[gix(i, lo, 0)] = TAG_ROOT
        else:
            best_v, best_k = neg, 0
            for kp in range(3):
                v = pget(lo, kp)
                if v > best_v:
                    best_v, best_k = v, kp
            cur[(lo, 0)] = best_v
            tags[gix(i, lo, 0)] = TAG_CLOSE0 + best_k
        cur.setdefault((lo, 1), neg)
        cur.setdefault((lo, 2), neg)

        for j in range(lo + 1, hi + 1):
            # k == 0：左邻跳过译文，或上一行在列 j 闭合当前主轨
            best, tag = cur.get((j - 1, 0), neg), TAG_SKIP
            for kp in range(3):
                v = pget(j, kp)
                if v > best:
                    best, tag = v, TAG_CLOSE0 + kp
            cur[(j, 0)] = best
            tags[gix(i, j, 0)] = TAG_INVALID if best == neg else tag

            wc = None
            if i < m:
                ts, te = t_starts[j - 1], t_ends[j - 1]
                xlo = si if si > ts else ts
                xhi = ei if ei < te else te
                if xhi > xlo:
                    wc = ((xhi - xlo) << shift_cnt) + bonus + seq_bit[j - 1]
            for k in (1, 2):
                best, tag = cur.get((j - 1, k), neg), TAG_SKIP
                pv = cur.get((j - 1, k - 1), neg)
                if wc is not None and pv > neg:
                    cand = pv + wc
                    if cand > best:
                        best, tag = cand, TAG_MATCH
                cur[(j, k)] = best
                tags[gix(i, j, k)] = TAG_INVALID if best == neg else tag
        prev, p_lo, p_hi = cur, lo, hi

    assignments = [[] for _ in range(m)]
    i, j, k = m, n, 0
    while i > 0 or j > 0 or k > 0:
        tag = tags.get(gix(i, j, k), TAG_INVALID)
        if k == 0:
            if tag == TAG_SKIP:
                j -= 1
            elif TAG_CLOSE0 <= tag <= TAG_CLOSE0 + 2:
                i -= 1
                k = tag - TAG_CLOSE0
            else:
                raise ValueError("配对回溯失败（内部错误）")
        elif tag == TAG_MATCH:
            assignments[i].append(j - 1)
            j -= 1
            k -= 1
        elif tag == TAG_SKIP:
            j -= 1
        else:
            raise ValueError("配对回溯失败（内部错误）")
    for lst in assignments:
        lst.sort()
    return assignments, {t for lst in assignments for t in lst}


def _pair_mincostflow(mains: List[dict], trans: List[dict]):
    """轨内时间回缩（不有序）时的回退算法：最小费用流求全局最优 b-匹配。

    源 -> 主轨容量 2、主轨 -> 译文容量 1（仅正时长交叠候选，费用为 -权重）、
    译文 -> 汇容量 1；势函数 + Dijkstra 逐次最短路，增广净增益 <= 0 时停止，
    故不强制配满。对任意（含交叉）配对均全局最优。
    """
    import heapq

    m, n = len(mains), len(trans)
    src, sink, nnode = 0, m + n + 1, m + n + 2
    graph = [[] for _ in range(nnode)]
    edge_u, edge_v, cap, edge_cost = [], [], [], []

    def add_edge(u, v, c, w):
        eid = len(edge_u)
        edge_u.extend((u, v)); edge_v.extend((v, u))
        cap.extend((c, 0)); edge_cost.extend((w, -w))
        graph[u].append(eid)
        graph[v].append(eid + 1)
        return eid

    for i in range(m):
        add_edge(src, 1 + i, 2, 0)
    for j in range(n):
        add_edge(1 + m + j, sink, 1, 0)

    seq_bit, bonus, shift_cnt = _weight_params(m, n)
    pair_edges = {}
    edge_count = m + n
    for i, me in enumerate(mains):
        for j, te in enumerate(trans):
            ov = _overlap_ms(me, te)
            if ov <= 0:
                continue
            edge_count += 1
            if edge_count > MAX_MERGE_EDGES:
                raise ValueError(
                    f"两轨正时长交叠候选对过多（已超过 {MAX_MERGE_EDGES}），"
                    "超出配对计算上限，请确认两轨为同一片源后裁短再试"
                )
            weight = (ov << shift_cnt) + bonus + seq_bit[j]
            eid = add_edge(1 + i, 1 + m + j, 1, -weight)
            pair_edges[eid] = (i, j)

    # 初始势函数：初始正向图是 DAG，按节点编号松弛一遍即得自源点最短距
    dist = [None] * nnode
    dist[src] = 0
    for u in range(nnode):
        if dist[u] is None:
            continue
        for eid in graph[u]:
            if cap[eid] and (dist[edge_v[eid]] is None
                             or dist[u] + edge_cost[eid] < dist[edge_v[eid]]):
                dist[edge_v[eid]] = dist[u] + edge_cost[eid]
    potential = [0 if d is None else d for d in dist]

    while True:
        dist = [None] * nnode
        prev_edge = [-1] * nnode
        dist[src] = 0
        heap = [(0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d != dist[u]:
                continue
            for eid in graph[u]:
                if cap[eid] <= 0:
                    continue
                v = edge_v[eid]
                nd = d + edge_cost[eid] + potential[u] - potential[v]
                if dist[v] is None or nd < dist[v]:
                    dist[v] = nd
                    prev_edge[v] = eid
                    heapq.heappush(heap, (nd, v))
        if dist[sink] is None:
            break
        for u in range(nnode):
            if dist[u] is not None:
                potential[u] += dist[u]
        if -potential[sink] <= 0:  # 再增广一条不改善任何一级目标
            break
        v = sink
        while v != src:
            eid = prev_edge[v]
            cap[eid] -= 1
            cap[eid ^ 1] += 1
            v = edge_u[eid]

    assignments = [[] for _ in range(m)]
    matched = set()
    for eid, (i, j) in pair_edges.items():
        if cap[eid] == 0:
            assignments[i].append(j)
            matched.add(j)
    for lst in assignments:
        lst.sort()
    return assignments, matched


def pair_entries(mains: List[dict], trans: List[dict]):
    """全局最优配对（不要求轨内条目互不重叠）。

    每条主轨可配 0~2 条译文、每条译文至多使用一次，仅正时长交叠可配对。
    三级目标依次为：最大化总交叠毫秒数；并列时最大化已配译文数（少漏配）；
    再并列时使已用译文原序号（升序）序列字典序最小。

    三级目标编码为单个整数边权（见 _weight_params）：
        交叠毫秒数 << shift_cnt | 每条配对固定奖励 | 译文序号加权奖励。

    - 两轨均按 (start, end) 非降时（正常 SRT 时间轴），交叠矩阵为 Monge 型，
      保序网格 DP 即全局最优，走 O(m*n) 的 _pair_ordered_dp；
    - 轨内时间回缩（条目相互包含 / 错位）时回退到最小费用流
      _pair_mincostflow，对含交叉的配对仍保证全局最优。

    返回 (assignments, matched_set)：assignments[i] 为配给第 i 条主轨的
    译文下标（升序）。
    """
    if _track_is_ordered(mains) and _track_is_ordered(trans):
        return _pair_ordered_dp(mains, trans)
    return _pair_mincostflow(mains, trans)


def _merge_prepare(req: "MergeRequest"):
    """双语合成公共预处理：解析两轨并校验空时长 / 倒置，失败抛 400。"""
    try:
        mains = parse_srt(req.main)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"主轨解析失败：{e}")
    try:
        trans = parse_srt(req.translation)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"译文轨解析失败：{e}")
    try:
        _validate_track_duration(mains, "主轨")
        _validate_track_duration(trans, "译文轨")
        assignments, matched = pair_entries(mains, trans)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return mains, trans, assignments, matched


def _entry_brief(seq: int, e: dict):
    return {
        "seq": seq,
        "index": e["index"],
        "time": f"{fmt_time(e['start'])} --> {fmt_time(e['end'])}",
        "text": "\n".join(e["lines"]),
    }


@app.post("/api/merge/preview")
def merge_preview(req: MergeRequest):
    mains, trans, assignments, matched = _merge_prepare(req)
    total_overlap = 0
    entries = []
    for i, me in enumerate(mains):
        paired = []
        for jt in assignments[i]:
            ov = _overlap_ms(me, trans[jt])
            total_overlap += ov
            brief = _entry_brief(jt + 1, trans[jt])
            brief["overlap_ms"] = ov
            paired.append(brief)
        item = _entry_brief(i + 1, me)
        item["translations"] = paired
        entries.append(item)
    unmatched = [_entry_brief(j + 1, trans[j]) for j in range(len(trans)) if j not in matched]
    return {
        "main_count": len(mains),
        "trans_count": len(trans),
        "matched_count": len(matched),
        "unmatched_count": len(trans) - len(matched),
        "total_overlap_ms": total_overlap,
        "total_overlap": fmt_time(total_overlap),
        "entries": entries,
        "unmatched": unmatched,
    }


@app.post("/api/merge/export")
def merge_export(req: MergeRequest):
    mains, trans, assignments, _matched = _merge_prepare(req)
    blocks = []
    for i, me in enumerate(mains):
        body = list(me["lines"])
        for jt in assignments[i]:
            body.extend(trans[jt]["lines"])
        if not body:  # 两轨正文均空时仍保留该主轨块
            body = [""]
        blocks.append(
            f"{i + 1}\n{fmt_time(me['start'])} --> {fmt_time(me['end'])}\n"
            + "\n".join(body)
        )
    content = "\n\n".join(blocks) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="bilingual.srt"'},
    )


@app.post("/api/export")
def export(req: CalibrateRequest):
    rows = _prepare(req)
    bad = [r for r in rows if r["flags"]]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"存在 {len(bad)} 条问题字幕（负时间/起止倒置/重叠），已禁止导出",
        )
    blocks = [
        f"{r['index']}\n{fmt_time(r['new_start'])} --> {fmt_time(r['new_end'])}\n"
        + "\n".join(r["lines"])
        for r in rows
    ]
    content = "\n\n".join(blocks) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="calibrated.srt"'},
    )


# ---------------------------------------------------------------- 阅读速度修复

class SpeedRequest(BaseModel):
    srt: str
    max_cps: float           # 最大每秒字数（非空白字符 / 起止跨度秒数）
    max_block_seconds: float # 单块最长秒数


MAX_MERGE_GAP_MS = 500  # 仅允许合并间隔不超过 500ms 的相邻字幕


def _non_ws_chars(lines: List[str]) -> int:
    """正文中的非空白字符数（每个非空白字符计一字）。"""
    return sum(1 for ln in lines for ch in ln if not ch.isspace())


def _speed_params(req: "SpeedRequest"):
    """校验限速参数（均须为正数），返回 (max_cps, max_block_ms)。失败抛 ValueError。"""
    if not math.isfinite(req.max_cps) or req.max_cps <= 0:
        raise ValueError("最大每秒字数须为正数")
    if not math.isfinite(req.max_block_seconds) or req.max_block_seconds <= 0:
        raise ValueError("单块最长秒数须为正数")
    return req.max_cps, req.max_block_seconds * 1000.0


def _validate_speed_track(entries: List[dict]) -> None:
    """空时长（终点=起点）、起止倒置（终点<起点）或轨内重叠即报错。

    逐条保证终点严格大于起点后，再检查相邻两条：前条终点 > 后条起点即重叠。
    相邻不重叠时起点、终点均严格递增，故任意两条都不可能重叠。
    """
    for i, e in enumerate(entries, 1):
        if e["end"] <= e["start"]:
            kind = "空时长" if e["end"] == e["start"] else "起止时间倒置"
            raise ValueError(
                f"第 {i} 条字幕（序号 {e['index']}）{kind}："
                f"{fmt_time(e['start'])} --> {fmt_time(e['end'])}"
            )
    for i in range(len(entries) - 1):
        a, b = entries[i], entries[i + 1]
        if a["end"] > b["start"]:
            raise ValueError(
                f"第 {i + 1} 条（{fmt_time(a['start'])} --> {fmt_time(a['end'])}）"
                f"与第 {i + 2} 条（{fmt_time(b['start'])} --> {fmt_time(b['end'])}）轨内重叠"
            )


def plan_speed_fix(entries: List[dict], max_cps: float, max_block_ms: float):
    """阅读速度修复的全局分块规划。

    把按时间有序的字幕划分为若干连续块，规则：
      - 仅相邻间隔 ≤ 500ms 的字幕可并入同一块（间隔更大处必须切开）；
      - 每块取首条起点、末条终点为起止，正文按原顺序换行拼接；
      - 每块 速度 = 非空白字数 / 起止跨度 ≤ max_cps，且跨度 ≤ max_block_ms。

    目标：合并次数（= 条数 - 块数）最少优先；仍并列时切分位置序列
    （相邻两条之间的切开位置，按条目序号升序）字典序最小。

    后缀 DP：dp[i] = (合并次数, 切分位置序列) 为第 i 条起的最优后缀方案，
    choice[i] 记录首块末条下标。约束逐块独立、后缀最优子结构成立；
    终点严格递增，故跨度随块尾单调，超时长即可停止扩展。
    返回 (blocks, None)；无可行方案时返回 (None, 原因)。
    """
    n = len(entries)
    prefix = [0] * (n + 1)  # 非空白字数前缀和
    for i, e in enumerate(entries):
        prefix[i + 1] = prefix[i] + _non_ws_chars(e["lines"])

    # reach[i]：从第 i 条起允许合并到的最远下标（块内相邻间隔均 ≤ 500ms）
    reach = list(range(n))
    for i in range(n - 2, -1, -1):
        gap = entries[i + 1]["start"] - entries[i]["end"]
        reach[i] = reach[i + 1] if gap <= MAX_MERGE_GAP_MS else i

    dp = [None] * (n + 1)
    dp[n] = (0, ())
    choice = [None] * n
    for i in range(n - 1, -1, -1):
        best = None
        start_i = entries[i]["start"]
        for j in range(i, reach[i] + 1):
            span = entries[j]["end"] - start_i
            if span > max_block_ms:
                break  # 终点严格递增，跨度只会更大
            if dp[j + 1] is None:
                continue
            chars = prefix[j + 1] - prefix[i]
            if chars * 1000.0 / span > max_cps:
                continue
            merges = (j - i) + dp[j + 1][0]
            cuts = ((j + 1,) + dp[j + 1][1]) if j + 1 < n else ()
            if best is None or (merges, cuts) < (best[0], best[1]):
                best = (merges, cuts, j)
        if best is not None:
            dp[i] = best[:2]
            choice[i] = best[2]

    if dp[0] is None:
        return None, _speed_infeasible_reason(entries, dp, reach, max_cps, max_block_ms)

    blocks = []
    i = 0
    while i < n:
        j = choice[i]
        merged = entries[i:j + 1]
        blocks.append({
            "sources": [e["index"] for e in merged],
            "src_first": i + 1,
            "src_last": j + 1,
            "start": entries[i]["start"],
            "end": entries[j]["end"],
            "chars": prefix[j + 1] - prefix[i],
            "lines": [ln for e in merged for ln in e["lines"]],
        })
        i = j + 1
    return blocks, None


def _speed_infeasible_reason(entries: List[dict], dp: list, reach: list,
                             max_cps: float, max_block_ms: float) -> str:
    """定位最靠后的不可行起点，给出无法修复的具体原因。

    bad 之后的位置均可行（bad 是最大的不可行下标），因此从 bad 出发的
    每个候选块若满足速度与时长，dp[bad] 就不会不可行——即候选块必然全部
    超速（时长、间隔已由枚举范围保证），或单条自身即超时长。
    """
    n = len(entries)
    bad = max(i for i in range(n) if dp[i] is None)
    e = entries[bad]
    head = (f"第 {bad + 1} 条字幕（序号 {e['index']}，"
            f"{fmt_time(e['start'])} --> {fmt_time(e['end'])}）")
    own_ms = e["end"] - e["start"]
    if own_ms > max_block_ms:
        return (
            f"{head}自身时长 {own_ms / 1000:.3f} 秒已超过单块最长 "
            f"{max_block_ms / 1000:g} 秒，合并只会更长，无法修复"
        )
    chars = 0
    best_cps = None
    for j in range(bad, reach[bad] + 1):
        chars += _non_ws_chars(entries[j]["lines"])
        span = entries[j]["end"] - e["start"]
        if span > max_block_ms:
            break
        cps = chars * 1000.0 / span
        if best_cps is None or cps < best_cps:
            best_cps = cps
    if best_cps is None:  # 防御：候选为空时（单条超时长已在上面拦截）
        return f"{head}附近不存在可行分块，无法修复"
    return (
        f"{head}起即使尽量向后合并（间隔 ≤ {MAX_MERGE_GAP_MS}ms、单块 ≤ "
        f"{max_block_ms / 1000:g} 秒），速度最低仍为 {best_cps:.2f} 字/秒，"
        f"超过上限 {max_cps:g} 字/秒，无法修复"
    )


def _speed_prepare(req: "SpeedRequest"):
    """阅读速度修复公共预处理：校验参数、解析并校验轨道，失败抛 400。"""
    try:
        max_cps, max_block_ms = _speed_params(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        entries = parse_srt(req.srt)
        _validate_speed_track(entries)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return entries, max_cps, max_block_ms


def _speed_block_payload(b: dict, seq: int) -> dict:
    span = b["end"] - b["start"]
    return {
        "seq": seq,
        "sources": b["sources"],
        "merged": b["src_first"] != b["src_last"],
        "start": fmt_time(b["start"]),
        "end": fmt_time(b["end"]),
        "duration_ms": span,
        "chars": b["chars"],
        "cps": round(b["chars"] * 1000.0 / span, 2),
        "text": "\n".join(b["lines"]),
    }


@app.post("/api/speed/preview")
def speed_preview(req: SpeedRequest):
    entries, max_cps, max_block_ms = _speed_prepare(req)
    blocks, reason = plan_speed_fix(entries, max_cps, max_block_ms)
    if blocks is None:
        return {"feasible": False, "reason": reason, "entry_count": len(entries)}
    return {
        "feasible": True,
        "entry_count": len(entries),
        "block_count": len(blocks),
        "merge_count": len(entries) - len(blocks),
        "max_cps": max_cps,
        "max_block_seconds": req.max_block_seconds,
        "blocks": [_speed_block_payload(b, seq) for seq, b in enumerate(blocks, 1)],
    }


@app.post("/api/speed/export")
def speed_export(req: SpeedRequest):
    entries, max_cps, max_block_ms = _speed_prepare(req)
    blocks, reason = plan_speed_fix(entries, max_cps, max_block_ms)
    if blocks is None:
        raise HTTPException(status_code=400, detail=f"无法修复：{reason}")
    content = "\n\n".join(
        f"{seq}\n{fmt_time(b['start'])} --> {fmt_time(b['end'])}\n"
        + "\n".join(b["lines"] or [""])
        for seq, b in enumerate(blocks, 1)
    ) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="speed_fixed.srt"'},
    )


# ---------------------------------------------------------------- SRT 修订差异对照

class DiffRequest(BaseModel):
    old_srt: str  # 旧版 SRT
    new_srt: str  # 新版 SRT


DIFF_SIM_THRESHOLD = 0.5   # 仅相似度至少 0.5 的组合才允许匹配
SIM_SCALE = 1_000_000_000  # 相似度定点放大倍数（总相似度按整数精确比较）
MAX_DIFF_GRID = 4_000_000  # 两侧条目数乘积（对齐计算规模）上限


def _normalize_text(s: str) -> str:
    """归一化正文：去空白、去标点、小写化（全角字符先兼容分解）。

    兼容分解（NFKD）会把全角字母数字转为半角，故在分解后逐字符按类别
    过滤；日文假名等无兼容分解的字符按原形保留。
    """
    return "".join(
        ch.casefold()
        for ch in unicodedata.normalize("NFKD", s)
        if not ch.isspace() and not _is_punct(ch)
    )


def _is_punct(ch: str) -> bool:
    """是否为标点符号（覆盖中英文常见标点及 Unicode P* 类别）。"""
    if unicodedata.category(ch).startswith("P"):
        return True
    return ch in "，。！？；：、""''（）《》【】…—·「」『』〈〉"


def _entry_body(entries: List[dict], idxs) -> str:
    """按原顺序拼接组内字幕正文（条目之间以换行连接）。"""
    return "\n".join("\n".join(entries[i]["lines"]) for i in idxs)


def _norm_body(entries: List[dict], idxs) -> str:
    return _normalize_text(_entry_body(entries, idxs))


# ---- 相似度 ≥ 0.5 的候选预筛（避免对全网格逐格跑 difflib 精确块匹配）----
#
# SequenceMatcher.ratio = 2M/(|a|+|b|)，M 为等长匹配块总长度，且 M 不超过
# 字符多重集交集 C = Σ_ch min(na(ch), nb(ch))。故 ratio ≥ 0.5 必有
# C ≥ (|a|+|b|)/4；反之可安全剪枝。
# 直接逐字符求 C 仍要字典遍历；改用多组散列桶位掩码得到 C 的廉价上界：
# 对每组独立 256 桶散列，记 A_t/B_t 为落在桶 t 的字符总数，桶级交集
# B = Σ_{共享桶} min(A_t,B_t) ≥ C（哈希碰撞只会合并计数，上界只增不减）；
# 再用 k=共享桶数、D=各自占用桶数放缩为 B ≤ k + min(|a|-D_a, |b|-D_b)，
# 全程仅需缓存的位掩码与 popcount。取多组散列上界的最小值，碰撞同时发生
# 的概率可忽略，无关句子在这一级即被剔除；漏报在数学上不可能。

DIFF_HASH_GROUPS = 4
DIFF_BUCKET_BITS = 8
DIFF_BUCKETS = 1 << DIFF_BUCKET_BITS  # 256


def _bucket_masks(s: str):
    """文本在多组独立散列下的桶占用签名：(长度, 掩码组, 各组占用桶数)。"""
    masks = [0] * DIFF_HASH_GROUPS
    for ch in s:
        h0 = ord(ch)
        for r in range(DIFF_HASH_GROUPS):
            h = (h0 + 0x9e3779b9 * (r + 1)) & 0xFFFFFFFF
            h ^= h >> 16
            h = (h * 0x7feb352d) & 0xFFFFFFFF
            h ^= h >> 15
            h = (h * 0x846ca68b) & 0xFFFFFFFF
            masks[r] |= 1 << ((h ^ (h >> 16)) & (DIFF_BUCKETS - 1))
    return len(s), tuple(masks), tuple(m.bit_count() for m in masks)


def _sig_passes(la, ma, oa, lb, mb, ob, need: int) -> bool:
    """多重集交集上界剪枝：返回 False 时相似度必 < 阈值（绝不漏报）。

    逐组散列独立判断，第一组即剔除时立即返回（绝大多数无关对在此结束）。
    """
    for r in range(DIFF_HASH_GROUPS):
        k = (ma[r] & mb[r]).bit_count()
        # k + min(la-oa[r], lb-ob[r]) 是该组桶级交集的上界
        da, db = la - oa[r], lb - ob[r]
        if k + (da if da < db else db) < need:
            return False
    return True


def _try_similarity(sig_a, sig_b):
    """计算精确相似度；廉价上界判定必不足 DIFF_SIM_THRESHOLD 时返回 None。

    入参为 (归一化正文, 长度, 掩码组, 占用数组)。
    """
    a, la, ma, oa = sig_a
    b, lb, mb, ob = sig_b
    if a == b:
        return 1.0  # 含双方正文均为空：ratio=1.0
    if la == 0 or lb == 0:
        return None
    if min(la, lb) * 3 < max(la, lb):
        return None  # 长度比 < 1/3 时 ratio 必 < 0.5
    need = (la + lb + 3) // 4  # ceil((la+lb)/4)
    if not _sig_passes(la, ma, oa, lb, mb, ob, need):
        return None
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    r = sm.ratio()
    return r if r >= DIFF_SIM_THRESHOLD else None


def _validate_diff_track(entries: List[dict], track: str) -> None:
    """差异对照输入轨：空时长（终点=起点）或起止倒置（终点<起点）即报错。"""
    _validate_track_duration(entries, track)


def align_diff(old_entries: List[dict], new_entries: List[dict]):
    """保序全局动态规划对齐旧、新两版字幕。

    每个对齐组允许以下形态（旧侧条数 → 新侧条数）：
      1:1 改词/未变、1:2 拆分、2:1 合并、1:0 删除、0:1 新增。
    匹配组（前三种）仅在归一化正文相似度 ≥ 0.5 时成立。

    目标按字典序：
      1) 最大化全部匹配组相似度之和；
      2) 总相似度并列时，最小化全部匹配组时间差（中点差毫秒）之和。
    单侧跳过（删除/新增）不参与打分，故只在确实没有更优匹配时出现。

    三级比较编码为单个整数：value = 相似度和 × SIM_SCALE × W − 时间差和，
    其中 W 取（严格大于任何方案时间差和上界的）整数，从而先比相似度、
    再比时间差。相似度以 SIM_SCALE 定点化为整数（误差 < 1e-9/组），
    比较顺序与逐组有理数求和一致。

    返回对齐组列表，每组：
      {kind, old:[seq...], new:[seq...], score:float, gap_ms:float}
    kind 为 raw_11 / raw_split / raw_merge / raw_del / raw_add；
    raw_11 组在展示阶段再按归一化正文是否相同细分为未变 / 改词。
    """
    m, n = len(old_entries), len(new_entries)
    if m * n > MAX_DIFF_GRID:
        raise ValueError(
            f"两版字幕条目数乘积 {m}×{n} 超过对齐计算上限 {MAX_DIFF_GRID}，"
            "请确认两版为同一片源后裁短再试"
        )

    def _rec(s: str):
        length, masks, occ = _bucket_masks(s)
        return (s, length, masks, occ)

    old_rec = [_rec(_normalize_text("\n".join(e["lines"]))) for e in old_entries]
    new_rec = [_rec(_normalize_text("\n".join(e["lines"]))) for e in new_entries]
    # 1:2 / 2:1 组的拼接正文各只需 n-1 / m-1 份，预先构造一次，避免网格内反复拼接
    old_pair_rec = [_rec(a[0] + b[0]) for a, b in zip(old_rec, old_rec[1:])]
    new_pair_rec = [_rec(a[0] + b[0]) for a, b in zip(new_rec, new_rec[1:])]

    # 拍平为并行数组，供热循环按下标直接访问（避免逐格元组拆装）
    ot = [r[0] for r in old_rec]
    ol = [r[1] for r in old_rec]
    om = [[r[2][g] for r in old_rec] for g in range(DIFF_HASH_GROUPS)]
    oo = [[r[3][g] for r in old_rec] for g in range(DIFF_HASH_GROUPS)]
    nt = [r[0] for r in new_rec]
    nl_ = [r[1] for r in new_rec]
    nm = [[r[2][g] for r in new_rec] for g in range(DIFF_HASH_GROUPS)]
    no_ = [[r[3][g] for r in new_rec] for g in range(DIFF_HASH_GROUPS)]
    npt = [r[0] for r in new_pair_rec]
    npl = [r[1] for r in new_pair_rec]
    npm = [[r[2][g] for r in new_pair_rec] for g in range(DIFF_HASH_GROUPS)]
    npo = [[r[3][g] for r in new_pair_rec] for g in range(DIFF_HASH_GROUPS)]
    opt = [r[0] for r in old_pair_rec]
    opl = [r[1] for r in old_pair_rec]
    opm = [[r[2][g] for r in old_pair_rec] for g in range(DIFF_HASH_GROUPS)]
    opo = [[r[3][g] for r in old_pair_rec] for g in range(DIFF_HASH_GROUPS)]
    # 每条新侧条目的 (长度, 各组掩码, 各组占用数) 只组装一次
    new_sig_cols = [
        (nl_[j], tuple(nm[g][j] for g in range(DIFF_HASH_GROUPS)),
         tuple(no_[g][j] for g in range(DIFF_HASH_GROUPS)))
        for j in range(n)
    ]
    new_pair_cols = [
        (npl[j], tuple(npm[g][j] for g in range(DIFF_HASH_GROUPS)),
         tuple(npo[g][j] for g in range(DIFF_HASH_GROUPS)))
        for j in range(n - 1)
    ]

    # W 严格大于任何方案的时间差总和上界：最多 min(m,n) 个匹配组，
    # 每组中点差（2 倍值）不超过两侧中点取值的全局极差，故
    # W = min(m,n)*pmax + 1。单侧为空时没有匹配组，w 取任意正值即可。
    mid_o = [e["start"] + e["end"] for e in old_entries]
    mid_n = [e["start"] + e["end"] for e in new_entries]
    if m > 0 and n > 0:
        w = min(m, n) * (max(max(mid_o), max(mid_n))
                         - min(min(mid_o), min(mid_n))) + 1
    else:
        w = 1

    def _exact(a, b):
        """已通过桶上界剪枝后的精确相似度，不足阈值返回 None。"""
        s = difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()
        return s if s >= DIFF_SIM_THRESHOLD else None

    def _candidate_score(la, am, ao, lb, bm, bo, a, b):
        """热循环中的候选相似度：等串快路 → 长度 → 0 号桶（绝大多数在此剔除）
        → 其余散列组 → 精确块匹配。返回 None 表示不足阈值。"""
        if a == b:
            return 1.0
        if la == 0 or lb == 0 or min(la, lb) * 3 < max(la, lb):
            return None
        need = (la + lb + 3) >> 2
        da0, db0 = la - ao[0], lb - bo[0]
        if (am[0] & bm[0]).bit_count() + (da0 if da0 < db0 else db0) < need:
            return None
        for g in range(1, DIFF_HASH_GROUPS):
            da, db = la - ao[g], lb - bo[g]
            if (am[g] & bm[g]).bit_count() + (da if da < db else db) < need:
                return None
        return _exact(a, b)

    # 网格 dp[i][j]：旧侧消费 i 条、新侧消费 j 条时的最优整数值。
    # 五种转移（i/j 为已消费条数）：
    #   删除旧 i：dp[i][j]   <- dp[i-1][j]   （TAG_DEL）
    #   新增新 j：dp[i][j]   <- dp[i][j-1]   （TAG_ADD）
    #   1:1 匹配：dp[i][j]   <- dp[i-1][j-1]（TAG_M11，相似度≥0.5）
    #   1:2 拆分：dp[i][j]   <- dp[i-1][j-2]（TAG_M12）
    #   2:1 合并：dp[i][j]   <- dp[i-2][j-1]（TAG_M21）
    # dp 值只保留三行滚动；回溯标签存整张网格的 bytearray（每格 1 字节）。
    TAG_DEL, TAG_ADD, TAG_M11, TAG_M12, TAG_M21 = 1, 2, 3, 4, 5
    width = n + 1
    tags = bytearray((m + 1) * width)
    prev2 = None                      # i-2 行
    prev = [0] * width                # i=0 行：只能一路新增（跳过不计分），全为 0
    for jj in range(1, width):
        tags[jj] = TAG_ADD

    for i in range(1, m + 1):
        cur = [0] * width
        base = i * width
        tags[base] = TAG_DEL  # 第 0 列只能一路删除
        oi = i - 1
        a1_text = ot[oi]
        a1_len = ol[oi]
        a1_masks = [gmask[oi] for gmask in om]   # 行常量：旧 oi 在各组散列下的桶掩码
        a1_occ = [gocc[oi] for gocc in oo]
        mo1 = mid_o[oi]
        has_merge = i >= 2
        mo2 = mid_o[oi - 1] if has_merge else 0
        if has_merge:
            a2_len = opl[oi - 1]
            a2_masks = [gmask[oi - 1] for gmask in opm]
            a2_occ = [gocc[oi - 1] for gocc in opo]
            a2_text = opt[oi - 1]
        prev_local, prev2_local = prev, prev2
        for j in range(1, width):
            nj = j - 1
            b_len, b_masks, b_occ = new_sig_cols[nj]
            b_text = nt[nj]
            best = prev_local[j]   # 删除旧 oi
            tag = TAG_DEL

            v = cur[j - 1]        # 新增新 nj
            if v > best:
                best, tag = v, TAG_ADD

            # 1:1 匹配
            s = _candidate_score(a1_len, a1_masks, a1_occ,
                                 b_len, b_masks, b_occ, a1_text, b_text)
            if s is not None:
                cand = prev_local[j - 1] + round(s * SIM_SCALE) * w \
                       - abs(mo1 - mid_n[nj])
                if cand > best:  # 同分保持先到转移，回溯确定
                    best, tag = cand, TAG_M11
            if j >= 2:  # 1:2 拆分（旧 oi -> 新 nj-1、nj）
                pj = nj - 1
                pb_len, pb_masks, pb_occ = new_pair_cols[pj]
                s = _candidate_score(a1_len, a1_masks, a1_occ,
                                     pb_len, pb_masks, pb_occ,
                                     a1_text, npt[pj])
                if s is not None:
                    pen = abs(mo1 - mid_n[pj]) + abs(mo1 - mid_n[nj])
                    cand = prev_local[j - 2] + round(s * SIM_SCALE) * w - pen
                    if cand > best:
                        best, tag = cand, TAG_M12
            if has_merge:  # 2:1 合并（旧 oi-1、oi -> 新 nj）
                s = _candidate_score(a2_len, a2_masks, a2_occ,
                                     b_len, b_masks, b_occ, a2_text, b_text)
                if s is not None:
                    pen = abs(mo2 - mid_n[nj]) + abs(mo1 - mid_n[nj])
                    cand = prev2_local[j - 1] + round(s * SIM_SCALE) * w - pen
                    if cand > best:
                        best, tag = cand, TAG_M21

            cur[j] = best
            tags[base + j] = tag
        prev2, prev = prev, cur

    # 回溯，按对齐顺序反向收集后再翻转
    groups = []
    i, j = m, n
    while i > 0 or j > 0:
        if i == 0:
            groups.append({"kind": "raw_add", "old": [], "new": [j]})
            j -= 1
            continue
        if j == 0:
            groups.append({"kind": "raw_del", "old": [i], "new": []})
            i -= 1
            continue
        tag = tags[i * width + j]
        if tag == TAG_DEL:
            groups.append({"kind": "raw_del", "old": [i], "new": []})
            i -= 1
        elif tag == TAG_ADD:
            groups.append({"kind": "raw_add", "old": [], "new": [j]})
            j -= 1
        elif tag == TAG_M11:
            oi, nj = i - 1, j - 1
            groups.append({
                "kind": "raw_11", "old": [i], "new": [j],
                "score": _try_similarity(old_rec[oi], new_rec[nj]),
                "gap_ms": abs(mid_o[oi] - mid_n[nj]) / 2,
            })
            i -= 1
            j -= 1
        elif tag == TAG_M12:
            oi = i - 1
            pen = abs(mid_o[oi] - mid_n[j - 2]) + abs(mid_o[oi] - mid_n[j - 1])
            groups.append({
                "kind": "raw_split", "old": [i], "new": [j - 1, j],
                "score": _try_similarity(old_rec[oi], new_pair_rec[j - 2]),
                "gap_ms": pen / 4,  # 两组配对的平均中点差
            })
            i -= 1
            j -= 2
        else:  # TAG_M21
            nj = j - 1
            pen = abs(mid_o[i - 2] - mid_n[nj]) + abs(mid_o[i - 1] - mid_n[nj])
            groups.append({
                "kind": "raw_merge", "old": [i - 1, i], "new": [j],
                "score": _try_similarity(old_pair_rec[i - 2], new_rec[nj]),
                "gap_ms": pen / 4,  # 两组配对的平均中点差
            })
            i -= 2
            j -= 1
    groups.reverse()
    return groups


def _classify_group(g: dict, old_entries, new_entries) -> str:
    """根据形态与归一化正文判定展示类型。

    1:1 且拼接后归一化正文完全相同 -> unchanged；否则 changed。
    1:2 / 2:1 一律为 split / merge（组级整体比较，不把拆并后的
    段落逐条对撞而误报改词/删除/新增）。
    """
    kind = g["kind"]
    if kind == "raw_11":
        ob = _norm_body(old_entries, [x - 1 for x in g["old"]])
        nb = _norm_body(new_entries, [x - 1 for x in g["new"]])
        return "unchanged" if ob == nb else "changed"
    return {"raw_split": "split", "raw_merge": "merge",
            "raw_del": "delete", "raw_add": "add"}[kind]


def _diff_entry_side(entries: List[dict], seq: int) -> dict:
    e = entries[seq - 1]
    return {
        "seq": seq,
        "index": e["index"],
        "time": f"{fmt_time(e['start'])} --> {fmt_time(e['end'])}",
        "text": "\n".join(e["lines"]),
    }


def _diff_prepare(req: "DiffRequest"):
    """差异对照公共预处理：解析两轨并校验空时长 / 倒置，失败抛 400。"""
    try:
        old_entries = parse_srt(req.old_srt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"旧版 SRT 解析失败：{e}")
    try:
        new_entries = parse_srt(req.new_srt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"新版 SRT 解析失败：{e}")
    try:
        _validate_diff_track(old_entries, "旧版")
        _validate_diff_track(new_entries, "新版")
        groups = align_diff(old_entries, new_entries)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return old_entries, new_entries, groups


def _diff_group_payload(g: dict, old_entries, new_entries, group_seq: int) -> dict:
    typ = _classify_group(g, old_entries, new_entries)
    old_side = [_diff_entry_side(old_entries, s) for s in g["old"]]
    new_side = [_diff_entry_side(new_entries, s) for s in g["new"]]
    payload = {
        "seq": group_seq,
        "type": typ,
        "old": old_side,
        "new": new_side,
    }
    if "score" in g:
        # 拆分/合并时各组成条与整体正文不同，附带「正文是否已变」便于预览标注
        changed = _norm_body(old_entries, [x - 1 for x in g["old"]]) != \
            _norm_body(new_entries, [x - 1 for x in g["new"]])
        payload["similarity"] = round(g["score"], 4)
        payload["gap_ms"] = int(round(g["gap_ms"]))
        payload["text_changed"] = changed if typ in ("split", "merge") else (typ == "changed")
    return payload


def _diff_payload(old_entries, new_entries, groups) -> dict:
    items = [
        _diff_group_payload(g, old_entries, new_entries, k)
        for k, g in enumerate(groups, 1)
    ]
    counts = {"unchanged": 0, "changed": 0, "split": 0,
              "merge": 0, "delete": 0, "add": 0}
    total_sim = 0.0
    matched_groups = 0
    for it in items:
        counts[it["type"]] += 1
        if "similarity" in it:
            total_sim += it["similarity"]
            matched_groups += 1
    return {
        "old_count": len(old_entries),
        "new_count": len(new_entries),
        "group_count": len(items),
        "counts": counts,
        "matched_group_count": matched_groups,
        "total_similarity": round(total_sim, 4),
        "avg_similarity": round(total_sim / matched_groups, 4) if matched_groups else None,
        "groups": items,
    }


@app.post("/api/diff/preview")
def diff_preview(req: DiffRequest):
    old_entries, new_entries, groups = _diff_prepare(req)
    return _diff_payload(old_entries, new_entries, groups)


_DIFF_CSV_HEADER = ["组号", "差异类型", "相似度", "时间差(ms)",
                    "旧版序号", "旧版时间", "旧版正文",
                    "新版序号", "新版时间", "新版正文"]
_DIFF_TYPE_CN = {
    "unchanged": "未变", "changed": "改词", "split": "拆分",
    "merge": "合并", "delete": "删除", "add": "新增",
}


def _diff_csv_side(side: List[dict], field: str) -> str:
    return "\n----\n".join(str(x[field]) for x in side)


@app.post("/api/diff/export")
def diff_export(req: DiffRequest):
    old_entries, new_entries, groups = _diff_prepare(req)
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")  # RFC 4180：CRLF、按需加引号
    writer.writerow(_DIFF_CSV_HEADER)
    for k, g in enumerate(groups, 1):
        it = _diff_group_payload(g, old_entries, new_entries, k)
        writer.writerow([
            k,
            _DIFF_TYPE_CN[it["type"]],
            it.get("similarity", ""),
            it.get("gap_ms", ""),
            _diff_csv_side(it["old"], "index"),
            _diff_csv_side(it["old"], "time"),
            _diff_csv_side(it["old"], "text"),
            _diff_csv_side(it["new"], "index"),
            _diff_csv_side(it["new"], "time"),
            _diff_csv_side(it["new"], "text"),
        ])
    # UTF-8 BOM，确保 Excel 直接双击打开时中文不乱码
    return Response(
        content=b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="srt_diff.csv"'},
    )


# ---------------------------------------------------------------- 目标词时间标记

WORD_CROSS_GAP_MS = 500  # 仅相邻字幕间隔不超过 500ms 时允许跨块命中


class WordMarkRequest(BaseModel):
    srt: str
    words: str          # 逐行目标词（每行一个，行内空白忽略）
    pre_ms: float = 0   # 向前扩展毫秒（非负）
    post_ms: float = 0  # 向后扩展毫秒（非负）


def _strip_ws(s: str) -> str:
    """去除全部空白字符（Unicode 空白，含换行 / 制表符 / 全角空格）。"""
    return "".join(ch for ch in s if not ch.isspace())


def _parse_target_words(text: str) -> List[tuple]:
    """逐行读取目标词：行内空白去除，空行跳过。返回 [(输入行号, 去空白词)]。"""
    out = []
    for i, line in enumerate(text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 1):
        word = _strip_ws(line)
        if word:
            out.append((i, word))
    if not out:
        raise ValueError("请至少填写 1 个目标词（每行一个）")
    return out


def _build_char_stream(entries: List[dict]) -> List[dict]:
    """按字幕原顺序连接正文的非空白字符。

    返回每个字符的归属信息：{ch, entry}（entry 为所属字幕下标）。
    字幕自身的块内顺序与条目顺序即「原顺序」；空白字符不进入字符流，
    也不占用字幕时长份额。
    """
    stream = []
    for ei, e in enumerate(entries):
        for line in e["lines"]:
            for ch in line:
                if not ch.isspace():
                    stream.append({"ch": ch, "entry": ei})
    return stream


def find_word_hits(entries: List[dict], words: List[tuple]) -> List[dict]:
    """在按原顺序连接的非空白字符流上对每个目标词做大小写敏感的字面匹配。

    - 目标词先去除行内全部空白；匹配为大小写敏感的纯字面子串匹配，
      允许相互重叠的命中；
    - 仅当命中跨度内每一对相邻字幕的间隔（后条起点 − 前条终点）
      ≤ 500ms 时允许跨块命中，否则该跨块出现被忽略（块内命中不受限）；
    - 每个字符按其所在字幕时长等分子区间，命中区间取首字符子区间的
      外侧起点与末字符子区间的外侧终点（跨块不补中间间隔）。
    """
    entry_nonws = [
        sum(1 for ln in e["lines"] for ch in ln if not ch.isspace())
        for e in entries
    ]
    stream = _build_char_stream(entries)
    n = len(stream)
    # 每个字符在其所属字幕内的非空白字符序号（0 起）
    local_pos = [0] * n
    seen = [0] * len(entries)
    for i, c in enumerate(stream):
        local_pos[i] = seen[c["entry"]]
        seen[c["entry"]] += 1
    text = "".join(c["ch"] for c in stream)

    def span_range(p: int, q: int):
        """字符流 [p, q] 的 (起点ms, 终点ms)，端点取字幕边界避免漂移。"""
        ci_first, ci_last = stream[p]["entry"], stream[q]["entry"]
        fe, le = entries[ci_first], entries[ci_last]
        kf, kl = local_pos[p], local_pos[q]
        nf, nl = entry_nonws[ci_first], entry_nonws[ci_last]
        lo = float(fe["start"]) if kf == 0 else fe["start"] + (fe["end"] - fe["start"]) * kf / nf
        hi = float(le["end"]) if kl + 1 == nl else \
            le["start"] + (le["end"] - le["start"]) * (kl + 1) / nl
        return lo, hi

    def gap_ok(p: int, q: int) -> bool:
        """跨度 [p, q] 内每一对相邻字幕间隔均 ≤ 500ms 才允许跨块。

        含中间无正文非空白字符的条目：间隔按相邻条目逐条检查
        （entries 原顺序，空时长 / 倒置 / 轨内重叠已在预处理拦截）。
        """
        a, b = stream[p]["entry"], stream[q]["entry"]
        for k in range(a, b):
            if entries[k + 1]["start"] - entries[k]["end"] > WORD_CROSS_GAP_MS:
                return False
        return True

    hits = []
    for row, word in words:
        L = len(word)
        if L == 0 or n == 0:
            continue
        start = 0
        while True:
            p = text.find(word, start)
            if p < 0:
                break
            q = p + L - 1
            if gap_ok(p, q):
                lo, hi = span_range(p, q)
                fi, li = stream[p]["entry"], stream[q]["entry"]
                # 仅列入实际贡献了字符的字幕（去重、按原顺序）
                idx_e = []
                for pos in range(p, q + 1):
                    idx = entries[stream[pos]["entry"]]["index"]
                    if not idx_e or idx_e[-1] != idx:
                        idx_e.append(idx)
                hits.append({
                    "word": word,
                    "row": row,
                    "start_ms": lo,
                    "end_ms": hi,
                    "cross_block": fi != li,
                    "indices": idx_e,
                })
            start = p + 1  # 允许重叠命中（如 aa 在 aaa 中出现两次）
    hits.sort(key=lambda h: (h["start_ms"], h["end_ms"], h["row"], h["word"]))
    return hits


def merge_hit_intervals(hits: List[dict], pre_ms: float, post_ms: float) -> List[dict]:
    """对命中区间做前后扩展并合并重叠 / 相接区间。

    扩展后起点小于 0 时截为 0（终点不截）；区间按起点排序后，
    后段起点 ≤ 前段终点（含端点相接）即合并为同一区间。
    同一合并区间内去重收集目标词与原字幕序号。
    """
    expanded = []
    for h in hits:
        s = max(0.0, h["start_ms"] - pre_ms)
        e = h["end_ms"] + post_ms
        expanded.append((s, e, h))
    expanded.sort(key=lambda x: (x[0], x[1]))

    merged = []
    for s, e, h in expanded:
        if merged and s <= merged[-1]["end_ms"]:
            seg = merged[-1]
            if e > seg["end_ms"]:
                seg["end_ms"] = e
            if h["word"] not in seg["words"]:
                seg["words"].append(h["word"])
            for idx in h["indices"]:
                if idx not in seg["indices"]:
                    seg["indices"].append(idx)
            seg["hit_count"] += 1
        else:
            merged.append({
                "start_ms": s,
                "end_ms": e,
                "words": [h["word"]],
                "indices": list(h["indices"]),
                "hit_count": 1,
            })
    return merged


def _word_prepare(req: "WordMarkRequest"):
    """目标词标记公共预处理：参数校验、SRT 解析与轨道校验，失败抛 400。"""
    if not math.isfinite(req.pre_ms) or req.pre_ms < 0:
        raise HTTPException(status_code=400, detail="向前扩展毫秒须为非负数")
    if not math.isfinite(req.post_ms) or req.post_ms < 0:
        raise HTTPException(status_code=400, detail="向后扩展毫秒须为非负数")
    try:
        entries = parse_srt(req.srt)
        _validate_speed_track(entries)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        words = _parse_target_words(req.words)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return entries, words, req.pre_ms, req.post_ms


def _word_hit_payload(h: dict) -> dict:
    return {
        "word": h["word"],
        "row": h["row"],
        "start": fmt_time(int(round(h["start_ms"]))),
        "end": fmt_time(int(round(h["end_ms"]))),
        "start_ms": int(round(h["start_ms"])),
        "end_ms": int(round(h["end_ms"])),
        "cross_block": h["cross_block"],
        "indices": h["indices"],
    }


def _word_segment_payload(seq: int, seg: dict) -> dict:
    start_ms = int(round(seg["start_ms"]))
    end_ms = int(round(seg["end_ms"]))
    return {
        "seq": seq,
        "start": fmt_time(start_ms),
        "end": fmt_time(end_ms),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "words": seg["words"],
        "indices": seg["indices"],
        "hit_count": seg["hit_count"],
    }


@app.post("/api/wordmark/preview")
def wordmark_preview(req: WordMarkRequest):
    entries, words, pre_ms, post_ms = _word_prepare(req)
    hits = find_word_hits(entries, words)
    segments = merge_hit_intervals(hits, pre_ms, post_ms)
    hit_rows = {h["row"] for h in hits}
    unmatched = [
        {"row": row, "word": word}
        for row, word in words if row not in hit_rows
    ]
    return {
        "entry_count": len(entries),
        "word_count": len(words),
        "pre_ms": int(round(pre_ms)),
        "post_ms": int(round(post_ms)),
        "hit_count": len(hits),
        "segment_count": len(segments),
        "hits": [_word_hit_payload(h) for h in hits],
        "segments": [_word_segment_payload(k, s) for k, s in enumerate(segments, 1)],
        "unmatched": unmatched,
    }


_WORD_CSV_HEADER = ["序号", "起点(ms)", "终点(ms)", "时长(ms)",
                    "起点", "终点", "命中词", "原字幕序号", "命中次数"]


@app.post("/api/wordmark/export")
def wordmark_export(req: WordMarkRequest):
    entries, words, pre_ms, post_ms = _word_prepare(req)
    hits = find_word_hits(entries, words)
    segments = merge_hit_intervals(hits, pre_ms, post_ms)
    if not segments:
        raise HTTPException(status_code=400, detail="没有任何目标词命中，没有可导出的区间")
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")  # RFC 4180：CRLF、按需加引号
    writer.writerow(_WORD_CSV_HEADER)
    for k, seg in enumerate(segments, 1):
        s_ms = int(round(seg["start_ms"]))
        e_ms = int(round(seg["end_ms"]))
        writer.writerow([
            k,
            s_ms,
            e_ms,
            e_ms - s_ms,
            fmt_time(s_ms),
            fmt_time(e_ms),
            "、".join(seg["words"]),
            "、".join(seg["indices"]),
            seg["hit_count"],
        ])
    # UTF-8 BOM，确保 Excel 直接双击打开时中文不乱码
    return Response(
        content=b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="word_mark_intervals.csv"'},
    )


# ---------------------------------------------------------------- 字幕遮挡避让

SIDE_TOP = "top"        # 上方：WebVTT line:10%
SIDE_BOTTOM = "bottom"  # 下方：WebVTT line:90%


class OccludeInterval(BaseModel):
    start: str  # 遮挡区间起点 HH:MM:SS,mmm
    end: str    # 遮挡区间终点 HH:MM:SS,mmm
    side: str   # 遮挡一侧：top（上方）/ bottom（下方）


class PositionRequest(BaseModel):
    srt: str
    intervals: List[OccludeInterval]


def validate_occlude_intervals(raw: List[OccludeInterval]) -> List[tuple]:
    """校验遮挡区间，返回 [(start_ms, end_ms, side, 输入行号)]。

    规则：起点须严格早于终点；side 仅允许 top / bottom。同侧区间允许
    重叠（与剪辑功能的保留区间不同），故不做重叠校验；区间按起点排序
    仅为展示稳定，左闭右开交叠判断本身不依赖顺序。
    """
    rows = [r for r in raw if r.start.strip() or r.end.strip()]
    if not rows:
        raise ValueError("请至少填写 1 个遮挡区间")
    parsed = []
    for i, r in enumerate(rows, 1):
        try:
            s, e = parse_time(r.start), parse_time(r.end)
        except ValueError as ex:
            raise ValueError(f"遮挡区间 {i}：{ex}")
        if s >= e:
            raise ValueError(
                f"遮挡区间 {i}：起点 {fmt_time(s)} 须早于终点 {fmt_time(e)}"
            )
        side = r.side.strip().lower()
        if side not in (SIDE_TOP, SIDE_BOTTOM):
            raise ValueError(f"遮挡区间 {i}：遮挡一侧须为「上方」或「下方」")
        parsed.append((s, e, side, i))
    parsed.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return parsed


def plan_positions(entries: List[dict], intervals: List[tuple]):
    """逐条字幕选上方 / 下方的动态规划。

    字幕与遮挡区间按左闭右开 [start, end) 求交叠：仅正时长交叠
    （max(起点) < min(终点)，端点相触不算）才视为遮挡；同侧区间可重叠，
    交叠的一侧不可放置。

    某条字幕上、下两侧都存在交叠区间时无解，记录其原序号与两侧冲突区间。
    其余情况用 DP 逐条选侧，目标依次为：
      1) 相邻字幕换位次数（相邻选侧不同即一次换位）最少；
      2) 位于下方的条数最多；
      3) 仍并列时，按位置序列字典序取「下方优先」者（从上到下逐条比较，
         首次不同处取下方）。

    状态编码为 (换位数, 负的下方条数, -掩码)：掩码把第 i 条（0 起）选下方
    置为高位 bit（位 n-1-i），故掩码越大位置序列越靠下方；元组最小即
    三级目标依次最优。

    返回 (sides, blocked, conflicts)：sides[i] 为 SIDE_TOP / SIDE_BOTTOM；
    blocked[i] 为 {"top": [区间...], "bottom": [区间...]}（区间为
    (start_ms, end_ms, 输入行号)）；存在两侧皆遮挡时 sides 为 None，
    conflicts 为逐组冲突字典。
    """
    blocked = [{SIDE_TOP: [], SIDE_BOTTOM: []} for _ in entries]
    conflicts = []
    for ei, entry in enumerate(entries):
        for iv in intervals:
            s, e, side, row = iv
            lo, hi = max(entry["start"], s), min(entry["end"], e)
            if lo < hi:  # 左闭右开：lo == hi 只是端点相触，不算交叠
                blocked[ei][side].append(iv)
        if blocked[ei][SIDE_TOP] and blocked[ei][SIDE_BOTTOM]:
            conflicts.append({
                "seq": ei + 1,
                "index": entry["index"],
                "time": f"{fmt_time(entry['start'])} --> {fmt_time(entry['end'])}",
                "text": "\n".join(entry["lines"]),
                "top": blocked[ei][SIDE_TOP],
                "bottom": blocked[ei][SIDE_BOTTOM],
            })
    if conflicts:
        return None, blocked, conflicts

    n = len(entries)
    NEG = None  # 占位，下方用显式 None 判定不可达

    def better(a, b):
        """三级目标元组比较：换位少 > 下方多 > 位置序列下方优先。"""
        return b is None or (a[0] < b[0]
                             or (a[0] == b[0] and (a[1] < b[1]
                                 or (a[1] == b[1] and a[2] > b[2]))))

    # dp 仅保留上一条在各选侧下的最优状态；choice 回溯整条选侧
    prev = {SIDE_TOP: NEG, SIDE_BOTTOM: NEG}
    choice = []
    for i, entry in enumerate(entries):
        cur = {SIDE_TOP: NEG, SIDE_BOTTOM: NEG}
        pick = {}
        bit = 1 << (n - 1 - i)
        for side in (SIDE_TOP, SIDE_BOTTOM):
            if blocked[i][side]:
                continue
            down = 1 if side == SIDE_BOTTOM else 0
            if i == 0:
                cur[side] = (0, -down, down * bit)
                pick[side] = None
                continue
            for prev_side, state in prev.items():
                if state is NEG:
                    continue
                cand = (state[0] + (1 if prev_side != side else 0),
                        state[1] - down, state[2] + down * bit)
                if better(cand, cur[side]):
                    cur[side] = cand
                    pick[side] = prev_side
        choice.append(pick)
        prev = cur

    last_side = min((s for s in (SIDE_TOP, SIDE_BOTTOM) if prev[s] is not NEG),
                    key=lambda s: (prev[s][0], prev[s][1], -prev[s][2]),
                    default=None)
    if last_side is None:
        # 理论上不可达：无两侧皆遮挡时逐条至少有一侧可选
        raise ValueError("位置规划失败（内部错误）")

    sides = [None] * n
    side = last_side
    for i in range(n - 1, -1, -1):
        sides[i] = side
        side = choice[i][side]
    return sides, blocked, []


def _occlude_iv_payload(iv: tuple) -> dict:
    s, e, _side, row = iv
    return {
        "row": row,
        "start": fmt_time(s),
        "end": fmt_time(e),
        "start_ms": s,
        "end_ms": e,
    }


def _position_prepare(req: PositionRequest):
    """遮挡避让公共预处理：校验区间、解析并校验轨道，失败抛 400。"""
    try:
        intervals = validate_occlude_intervals(req.intervals)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        entries = parse_srt(req.srt)
        _validate_speed_track(entries)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return entries, intervals


def _position_conflict_payload(c: dict) -> dict:
    return {
        "seq": c["seq"],
        "index": c["index"],
        "time": c["time"],
        "text": c["text"],
        "top": [_occlude_iv_payload(iv) for iv in c["top"]],
        "bottom": [_occlude_iv_payload(iv) for iv in c["bottom"]],
    }


@app.post("/api/position/preview")
def position_preview(req: PositionRequest):
    entries, intervals = _position_prepare(req)
    sides, blocked, conflicts = plan_positions(entries, intervals)
    if sides is None:
        return {
            "feasible": False,
            "entry_count": len(entries),
            "interval_count": len(intervals),
            "conflict_count": len(conflicts),
            "conflicts": [_position_conflict_payload(c) for c in conflicts],
        }
    switches = 0
    items = []
    for i, entry in enumerate(entries):
        if i > 0 and sides[i] != sides[i - 1]:
            switches += 1
        items.append({
            "seq": i + 1,
            "index": entry["index"],
            "time": f"{fmt_time(entry['start'])} --> {fmt_time(entry['end'])}",
            "text": "\n".join(entry["lines"]),
            "side": sides[i],
            "available": {
                SIDE_TOP: not blocked[i][SIDE_TOP],
                SIDE_BOTTOM: not blocked[i][SIDE_BOTTOM],
            },
            "blocked": {
                SIDE_TOP: [_occlude_iv_payload(iv) for iv in blocked[i][SIDE_TOP]],
                SIDE_BOTTOM: [_occlude_iv_payload(iv) for iv in blocked[i][SIDE_BOTTOM]],
            },
            "switches": switches,
        })
    return {
        "feasible": True,
        "entry_count": len(entries),
        "interval_count": len(intervals),
        "switch_count": switches,
        "bottom_count": sum(1 for s in sides if s == SIDE_BOTTOM),
        "top_count": sum(1 for s in sides if s == SIDE_TOP),
        "entries": items,
    }


def _fmt_vtt_time(ms: int) -> str:
    """毫秒格式化为 WebVTT 时间 HH:MM:SS.mmm（毫秒分隔符为点）。"""
    return fmt_time(ms).replace(",", ".")


@app.post("/api/position/export")
def position_export(req: PositionRequest):
    entries, intervals = _position_prepare(req)
    sides, _blocked, conflicts = plan_positions(entries, intervals)
    if sides is None:
        detail = "；".join(
            f"第 {c['seq']} 条字幕（原序号 {c['index']}）上、下两侧均被遮挡"
            for c in conflicts
        )
        raise HTTPException(
            status_code=400,
            detail=f"存在 {len(conflicts)} 条两侧均被遮挡的字幕，已禁止下载：{detail}",
        )
    blocks = []
    for i, entry in enumerate(entries):
        line = "10%" if sides[i] == SIDE_TOP else "90%"
        blocks.append(
            f"{i + 1}\n"
            f"{_fmt_vtt_time(entry['start'])} --> {_fmt_vtt_time(entry['end'])} "
            f"line:{line}\n"
            + "\n".join(entry["lines"] or [""])
        )
    content = "WEBVTT\n\n" + "\n\n".join(blocks) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="text/vtt; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="positioned.vtt"'},
    )


# ---------------------------------------------------------------- 字幕章节索引

class ChapterRequest(BaseModel):
    srt: str
    target_seconds: float  # 目标章节时长（秒，正数）
    min_seconds: float     # 最短章节时长（秒，正数）
    max_seconds: float     # 最长章节时长（秒，正数）


def _chapter_params(req: "ChapterRequest"):
    """校验章节时长参数：均为正数且 最短 ≤ 目标 ≤ 最长。返回毫秒三元组。"""
    for name, v in (("目标", req.target_seconds), ("最短", req.min_seconds),
                    ("最长", req.max_seconds)):
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"{name}章节时长须为正数")
    if not (req.min_seconds <= req.target_seconds <= req.max_seconds):
        raise ValueError("章节时长须满足 最短 ≤ 目标 ≤ 最长")
    return (req.target_seconds * 1000.0, req.min_seconds * 1000.0,
            req.max_seconds * 1000.0)


def plan_chapters(entries: List[dict], target_ms: float, min_ms: float,
                  max_ms: float):
    """章节切分的全局规划。

    把按时间有序的字幕划分为若干连续章节：仅允许在相邻字幕之间切分，
    章节覆盖全部字幕；每章跨度 = 章内首条起点 → 末条终点，且必须落在
    [min_ms, max_ms] 内。目标依次为：
      1) 各章跨度与目标时长的绝对偏差之和最小；
      2) 切点处字幕间隙（后条起点 − 前条终点）总时长最大；
      3) 切点序列（第 2 章起各章首条的下标，0 起，升序）字典序最小。

    后缀 DP：dp[i] = (偏差和, 负间隙和, 切点序列) 为第 i 条起的最优后缀
    方案，choice[i] 记录首章末条下标。终点严格递增，故跨度随章末单调，
    超最长即可停止扩展。返回 (chapters, None)；无可行切分时 (None, 原因)。
    """
    n = len(entries)
    starts = [e["start"] for e in entries]
    ends = [e["end"] for e in entries]

    dp = [None] * (n + 1)
    dp[n] = (0.0, 0, ())
    choice = [None] * n
    for i in range(n - 1, -1, -1):
        best = None
        for j in range(i, n):
            span = ends[j] - starts[i]
            if span > max_ms:
                break  # 终点严格递增，跨度只会更大
            if span < min_ms or dp[j + 1] is None:
                continue
            dev = abs(span - target_ms) + dp[j + 1][0]
            neg_gap = dp[j + 1][1] - (starts[j + 1] - ends[j] if j + 1 < n else 0)
            if best is not None and (dev, neg_gap) > best[:2]:
                continue  # 前两级已劣，无需构造切点序列
            cuts = ((j + 1,) + dp[j + 1][2]) if j + 1 < n else ()
            if best is None or (dev, neg_gap, cuts) < best[:3]:
                best = (dev, neg_gap, cuts, j)
        if best is not None:
            dp[i] = best[:3]
            choice[i] = best[3]

    if dp[0] is None:
        return None, _chapter_infeasible_reason(entries, dp, min_ms, max_ms)

    chapters = []
    i = 0
    while i < n:
        j = choice[i]
        first_line = next(
            (ln for e in entries[i:j + 1] for ln in e["lines"] if ln.strip()), ""
        )
        chapters.append({
            "src_first": i + 1,
            "src_last": j + 1,
            "first_index": entries[i]["index"],
            "last_index": entries[j]["index"],
            "start": starts[i],
            "end": ends[j],
            "first_line": first_line,
        })
        i = j + 1
    return chapters, None


def _chapter_infeasible_reason(entries: List[dict], dp: list,
                               min_ms: float, max_ms: float) -> str:
    """定位最靠后的不可行章首，给出无法切分的具体原因。

    bad 之后的位置均可行（bad 是最大的不可行下标），故从 bad 出发若存在
    跨度落在 [min_ms, max_ms] 的章节，dp[bad] 就不会不可行——即不存在
    任何章末使跨度落在范围内：要么单条自身已超最长，要么剩余总跨度不足
    最短，要么跨度从不足最短直接跳过最长。
    """
    n = len(entries)
    bad = max(i for i in range(n) if dp[i] is None)
    e = entries[bad]
    head = (f"第 {bad + 1} 条字幕（序号 {e['index']}，"
            f"{fmt_time(e['start'])} --> {fmt_time(e['end'])}）")
    own_ms = e["end"] - e["start"]
    if own_ms > max_ms:
        return (
            f"{head}自身时长 {own_ms / 1000:.3f} 秒已超过最长章节时长 "
            f"{max_ms / 1000:g} 秒，无法切分"
        )
    rest_ms = entries[-1]["end"] - e["start"]
    if rest_ms < min_ms:
        return (
            f"{head}起至末尾的剩余总跨度 {rest_ms / 1000:.3f} 秒已不足最短章节时长 "
            f"{min_ms / 1000:g} 秒，无法切分"
        )
    # 跨度随章末单调递增：找到最后一个仍不足最短的章末，下一条即超过最长
    j = bad
    while entries[j + 1]["end"] - e["start"] < min_ms:
        j += 1
    short_ms = entries[j]["end"] - e["start"]
    long_ms = entries[j + 1]["end"] - e["start"]
    return (
        f"{head}起：章末止于第 {j + 1} 条时跨度 {short_ms / 1000:.3f} 秒不足最短 "
        f"{min_ms / 1000:g} 秒，延至第 {j + 2} 条时跨度 {long_ms / 1000:.3f} 秒"
        f"超过最长 {max_ms / 1000:g} 秒，不存在落在范围内的切分"
    )


def _chapter_prepare(req: "ChapterRequest"):
    """章节索引公共预处理：校验参数、解析并校验轨道，失败抛 400。"""
    try:
        target_ms, min_ms, max_ms = _chapter_params(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        entries = parse_srt(req.srt)
        _validate_speed_track(entries)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return entries, target_ms, min_ms, max_ms


def _chapter_payload(c: dict, seq: int, target_ms: float) -> dict:
    span = c["end"] - c["start"]
    return {
        "seq": seq,
        "start": fmt_time(c["start"]),
        "end": fmt_time(c["end"]),
        "start_ms": c["start"],
        "end_ms": c["end"],
        "duration_ms": span,
        "deviation_ms": int(round(abs(span - target_ms))),
        "entry_count": c["src_last"] - c["src_first"] + 1,
        "first_index": c["first_index"],
        "last_index": c["last_index"],
        "first_line": c["first_line"],
    }


@app.post("/api/chapter/preview")
def chapter_preview(req: ChapterRequest):
    entries, target_ms, min_ms, max_ms = _chapter_prepare(req)
    chapters, reason = plan_chapters(entries, target_ms, min_ms, max_ms)
    if chapters is None:
        return {"feasible": False, "reason": reason, "entry_count": len(entries)}
    total_dev = sum(abs(c["end"] - c["start"] - target_ms) for c in chapters)
    total_gap = sum(chapters[k]["start"] - chapters[k - 1]["end"]
                    for k in range(1, len(chapters)))
    return {
        "feasible": True,
        "entry_count": len(entries),
        "chapter_count": len(chapters),
        "target_seconds": req.target_seconds,
        "min_seconds": req.min_seconds,
        "max_seconds": req.max_seconds,
        "total_deviation_ms": int(round(total_dev)),
        "total_gap_ms": total_gap,
        "cuts": [c["src_first"] for c in chapters[1:]],
        "chapters": [
            _chapter_payload(c, seq, target_ms)
            for seq, c in enumerate(chapters, 1)
        ],
    }


_CHAPTER_CSV_HEADER = ["章号", "起点(ms)", "终点(ms)", "时长(ms)",
                       "起点", "终点", "字幕条数", "首句"]


@app.post("/api/chapter/export")
def chapter_export(req: ChapterRequest):
    entries, target_ms, min_ms, max_ms = _chapter_prepare(req)
    chapters, reason = plan_chapters(entries, target_ms, min_ms, max_ms)
    if chapters is None:
        raise HTTPException(status_code=400, detail=f"无法切分：{reason}")
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")  # RFC 4180：CRLF、按需加引号
    writer.writerow(_CHAPTER_CSV_HEADER)
    for k, c in enumerate(chapters, 1):
        writer.writerow([
            k,
            c["start"],
            c["end"],
            c["end"] - c["start"],
            fmt_time(c["start"]),
            fmt_time(c["end"]),
            c["src_last"] - c["src_first"] + 1,
            c["first_line"],
        ])
    # UTF-8 BOM，确保 Excel 直接双击打开时中文不乱码
    return Response(
        content=b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="chapters.csv"'},
    )


# ---------------------------------------------------------------- 字幕空隙提示编排

class HintRequest(BaseModel):
    srt: str
    hints: str  # 逐行提示：目标起点 持续毫秒 正文


MAX_HINT_GRID = 2_000_000  # 提示条数 × 空隙数 的匹配计算规模上限


def _parse_hint_lines(text: str) -> List[dict]:
    """逐行解析提示：每行「目标起点 持续毫秒 正文」三段（空白分隔）。

    空行跳过；时间复用 parse_time（HH:MM:SS,mmm），持续毫秒须为正整数，
    正文为行内剩余部分、去首尾空白后不得为空。返回
    [{row, target, duration, text}]（row 为输入行号，1 起）。
    """
    out = []
    for i, line in enumerate(
        text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 1
    ):
        if not line.strip():
            continue
        parts = line.split(None, 2)  # 目标起点 / 持续毫秒 / 正文（可含空白）
        if len(parts) < 3 or not parts[2].strip():
            raise ValueError(
                f"提示第 {i} 行：须为「目标起点 持续毫秒 正文」三段，且正文不能为空"
            )
        try:
            target = parse_time(parts[0])
        except ValueError as e:
            raise ValueError(f"提示第 {i} 行：{e}")
        if not parts[1].isdigit() or int(parts[1]) <= 0:
            raise ValueError(f"提示第 {i} 行：持续毫秒须为正整数")
        out.append({
            "row": i,
            "target": target,
            "duration": int(parts[1]),
            "text": parts[2].strip(),
        })
    if not out:
        raise ValueError("请至少填写 1 条提示（每行：目标起点 持续毫秒 正文）")
    return out


def plan_hint_placement(entries: List[dict], hints: List[dict]):
    """空隙提示编排的有序全局匹配规划。

    空隙 = 相邻原字幕之间的正时长间隔（轨道校验已保证无重叠，故间隔非负，
    仅取严格为正者），每个空隙最多放入一条提示。提示按 (目标起点, 输入行号)
    排序后与空隙做保序全局匹配；提示 i 可放入空隙 j 当且仅当
    空隙时长 ≥ 提示持续毫秒，此时实际起点把目标起点夹到
    [空隙起点, 空隙终点 - 持续] 范围，偏差 = |实际起点 - 目标起点|。
    三级目标依次为：
      1) 放入条数最多；
      2) 已放入提示的偏差之和最小；
      3) 已用空隙序号（1 起，升序）序列字典序最小。

    与双语合成相同，三级目标编码为单个整数：放入一对的权重为
        CNT_UNIT | (DMAX - 偏差) << G | 空隙序号位（首个空隙为最高位），
    其中 DMAX 为偏差总和的严格上界、G 为空隙数；等条数时偏差位严格优先于
    序号位，等条数等偏差时权重比较即空隙集合指示向量的比较，等价于空隙
    序号序列字典序最小。保序网格 DP 求最大权重，值相等时解的空隙集合
    相同（保序匹配下空隙集合唯一决定分配），回溯取任意最大转移即可。

    返回 (gaps, placed, unplaced)：gaps 为全部正时长空隙（含使用情况）；
    placed / unplaced 均按提示排序顺序，placed 含实际起止、偏差与所在空隙。
    """
    gaps = []
    for k in range(len(entries) - 1):
        gs, ge = entries[k]["end"], entries[k + 1]["start"]
        if ge > gs:  # 仅正时长空隙可用
            gaps.append({
                "seq": len(gaps) + 1,
                "start": gs,
                "end": ge,
                "prev_index": entries[k]["index"],
                "next_index": entries[k + 1]["index"],
                "used": False,
            })
    ordered = sorted(hints, key=lambda h: (h["target"], h["row"]))
    h_n, g_n = len(ordered), len(gaps)
    if g_n == 0:
        return gaps, [], ordered
    if h_n * g_n > MAX_HINT_GRID:
        raise ValueError(
            f"提示条数 × 空隙数（{h_n}×{g_n}）超过匹配计算上限 "
            f"{MAX_HINT_GRID}，请确认输入规模后裁短再试"
        )

    t_max = max([h["target"] for h in ordered]
                + [g["end"] for g in gaps] + [1])
    dmax = h_n * t_max + 1            # 偏差总和的严格上界
    dev_unit = 1 << g_n               # 序号位掩码占 G 个二进制位
    cnt_unit = (dmax + 1) << g_n      # 条数位严格高于偏差位
    gap_bit = [1 << (g_n - 1 - j) for j in range(g_n)]

    TAG_UP, TAG_LEFT, TAG_DIAG = 0, 1, 2  # 跳过提示 / 跳过空隙 / 放入
    width = g_n + 1
    tags = bytearray((h_n + 1) * width)
    prev = [0] * width
    for i in range(1, h_n + 1):
        cur = [0] * width
        t_i = ordered[i - 1]["target"]
        d_i = ordered[i - 1]["duration"]
        base = i * width
        for j in range(1, g_n + 1):
            best = prev[j]  # 跳过提示 i-1
            tag = TAG_UP
            if cur[j - 1] > best:  # 跳过空隙 j-1
                best, tag = cur[j - 1], TAG_LEFT
            g = gaps[j - 1]
            if g["end"] - g["start"] >= d_i:
                # 实际起点夹到 [空隙起点, 空隙终点 - 持续]
                actual = t_i if t_i > g["start"] else g["start"]
                latest = g["end"] - d_i
                if actual > latest:
                    actual = latest
                dev = actual - t_i
                if dev < 0:
                    dev = -dev
                cand = prev[j - 1] + cnt_unit + (dmax - dev) * dev_unit \
                    + gap_bit[j - 1]
                if cand > best:
                    best, tag = cand, TAG_DIAG
            cur[j] = best
            tags[base + j] = tag
        prev = cur

    placement = {}  # 排序后提示下标 -> (空隙下标, 实际起点, 偏差)
    i, j = h_n, g_n
    while i > 0:
        tag = tags[i * width + j] if j > 0 else TAG_UP
        if tag == TAG_DIAG:
            g = gaps[j - 1]
            t_i = ordered[i - 1]["target"]
            d_i = ordered[i - 1]["duration"]
            actual = t_i if t_i > g["start"] else g["start"]
            latest = g["end"] - d_i
            if actual > latest:
                actual = latest
            placement[i - 1] = (j - 1, actual, abs(actual - t_i))
            i -= 1
            j -= 1
        elif tag == TAG_LEFT:
            j -= 1
        else:
            i -= 1

    placed, unplaced = [], []
    for k, h in enumerate(ordered):
        if k in placement:
            gj, actual, dev = placement[k]
            gaps[gj]["used"] = True
            placed.append({
                **h,
                "start": actual,
                "end": actual + h["duration"],
                "deviation_ms": dev,
                "gap": gaps[gj]["seq"],
            })
        else:
            unplaced.append(h)
    return gaps, placed, unplaced


def _hint_prepare(req: "HintRequest"):
    """空隙提示编排公共预处理：解析并校验轨道与提示行、求编排方案，失败抛 400。"""
    try:
        entries = parse_srt(req.srt)
        _validate_speed_track(entries)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        hints = _parse_hint_lines(req.hints)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        gaps, placed, unplaced = plan_hint_placement(entries, hints)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return entries, hints, gaps, placed, unplaced


def _hint_gap_payload(g: dict) -> dict:
    return {
        "seq": g["seq"],
        "start": fmt_time(g["start"]),
        "end": fmt_time(g["end"]),
        "start_ms": g["start"],
        "end_ms": g["end"],
        "duration_ms": g["end"] - g["start"],
        "prev_index": g["prev_index"],
        "next_index": g["next_index"],
        "used": g["used"],
    }


def _hint_placed_payload(p: dict) -> dict:
    return {
        "row": p["row"],
        "target": fmt_time(p["target"]),
        "target_ms": p["target"],
        "duration_ms": p["duration"],
        "start": fmt_time(p["start"]),
        "end": fmt_time(p["end"]),
        "start_ms": p["start"],
        "end_ms": p["end"],
        "deviation_ms": p["deviation_ms"],
        "gap": p["gap"],
        "text": p["text"],
    }


def _hint_unplaced_payload(h: dict) -> dict:
    return {
        "row": h["row"],
        "target": fmt_time(h["target"]),
        "target_ms": h["target"],
        "duration_ms": h["duration"],
        "text": h["text"],
    }


@app.post("/api/hint/preview")
def hint_preview(req: HintRequest):
    entries, hints, gaps, placed, unplaced = _hint_prepare(req)
    return {
        "entry_count": len(entries),
        "gap_count": len(gaps),
        "hint_count": len(hints),
        "placed_count": len(placed),
        "unplaced_count": len(unplaced),
        "total_deviation_ms": sum(p["deviation_ms"] for p in placed),
        "gaps": [_hint_gap_payload(g) for g in gaps],
        "placed": [_hint_placed_payload(p) for p in placed],
        "unplaced": [_hint_unplaced_payload(h) for h in unplaced],
    }


@app.post("/api/hint/export")
def hint_export(req: HintRequest):
    entries, _hints, _gaps, placed, _unplaced = _hint_prepare(req)
    blocks = [
        {"start": e["start"], "end": e["end"], "lines": e["lines"], "hint": 0}
        for e in entries
    ] + [
        {"start": p["start"], "end": p["end"], "lines": [p["text"]], "hint": 1}
        for p in placed
    ]
    # 提示严格落在空隙内，与原字幕互不重叠；按时间排序后从 1 连续重编号
    blocks.sort(key=lambda b: (b["start"], b["end"], b["hint"]))
    content = "\n\n".join(
        f"{n}\n{fmt_time(b['start'])} --> {fmt_time(b['end'])}\n"
        + "\n".join(b["lines"] or [""])
        for n, b in enumerate(blocks, 1)
    ) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="hinted.srt"'},
    )


# ---------------------------------------------------------------- 角色抢话分析

class TalkoverRequest(BaseModel):
    srt: str  # 每条字幕正文首行须以「[角色名]」开头


ROLE_TAG_RE = re.compile(r"^\[([^\[\]]*)\]")


def _entry_role(e: dict):
    """取字幕正文首行的「[角色名]」标记。

    返回 (角色名, None)；缺失标记或角色名去首尾空白后为空时返回
    (None, 原因)。标记必须位于正文首行的最开头，前面不得有空白。
    """
    if not e["lines"]:
        return None, "正文为空，缺少「[角色名]」标记"
    m = ROLE_TAG_RE.match(e["lines"][0])
    if not m:
        return None, "正文首行缺少「[角色名]」标记"
    name = m.group(1).strip()
    if not name:
        return None, "「[角色名]」中的角色名为空"
    return name, None


def _union_intervals(intervals: List[tuple]) -> List[tuple]:
    """左闭右开区间求并集。

    按起点排序后单趟合并：后段起点 <= 前段终点（含端点相触）即并入前段；
    空时长 / 起止倒置区间不占时间轴，直接丢弃。
    """
    valid = sorted((s, e) for s, e in intervals if e > s)
    if not valid:
        return []
    merged = [list(valid[0])]
    for s, e in valid[1:]:
        if s <= merged[-1][1]:  # 左闭右开：端点相触也合并为同一并集
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def analyze_talkover(entries: List[dict]) -> dict:
    """角色抢话分析主逻辑。

    1. 每条字幕正文首行须以「[角色名]」开头，角色名去首尾空白后同名合并；
       缺失标记或空角色名时指出原字幕序号并拒绝分析（ValueError）；
    2. 先对同一角色的字幕时间按左闭右开区间 [start, end) 求并集；
    3. 再以各角色并集的全部边界切分时间轴，逐段记录活跃角色集合
       （端点相触不算同时活跃：同一时刻先离场后入场）；
    4. 仅有两名及以上角色同时活跃的片段计入抢话；相邻且活跃角色集合
       相同的抢话片段合并；
    5. 汇总每名角色独占毫秒数（仅自己活跃）与每对角色共同发言毫秒数
       （两人同时活跃即计入，允许有第三人在场）。

    返回 {names, overlaps, solo, pairs}：overlaps 为 [{start,end,names}]，
    names 为按首次出现顺序排列的角色列表；solo 为 {角色: ms}；pairs 为
    [((角色A, 角色B), ms)]（按角色首次出现顺序排列，含 0 毫秒对）。
    """
    problems = []
    raw = {}   # 角色名 -> [(start_ms, end_ms)]
    order = []  # 角色首次出现顺序（兼作展示与排序次序）
    for i, e in enumerate(entries, 1):
        name, err = _entry_role(e)
        if err is not None:
            problems.append(f"第 {i} 条字幕（原序号 {e['index']}）{err}")
            continue
        if name not in raw:
            raw[name] = []
            order.append(name)
        raw[name].append((e["start"], e["end"]))
    if problems:
        raise ValueError(
            "；".join(problems) + "，已拒绝分析，请为这些字幕补全角色标记后重试"
        )

    # 同一角色的字幕区间先求并集
    unions = {name: _union_intervals(raw[name]) for name in order}

    # 以各角色并集边界切分时间轴：起点 +1 活跃、终点 -1 离场，
    # 同一时刻的事件先成组结算区间再统一改集合（左闭右开，端点相触不算）
    events = []
    for name in order:
        for s, e in unions[name]:
            events.append((s, 1, name))
            events.append((e, -1, name))
    events.sort(key=lambda ev: (ev[0], ev[1]))

    segments = []  # (start_ms, end_ms, frozenset(角色))
    active = set()
    cur_t = None
    k = 0
    while k < len(events):
        t = events[k][0]
        if cur_t is not None and active and t > cur_t:
            segments.append((cur_t, t, frozenset(active)))
        while k < len(events) and events[k][0] == t:
            _, delta, name = events[k]
            if delta == 1:
                active.add(name)
            else:
                active.discard(name)
            k += 1
        cur_t = t

    # 抢话片段：活跃角色 >= 2；相邻且活跃集合相同的片段合并
    overlaps = []
    for s, e, names in segments:
        if len(names) < 2:
            continue
        if overlaps and overlaps[-1]["end"] == s and overlaps[-1]["names"] == names:
            overlaps[-1]["end"] = e
        else:
            overlaps.append({"start": s, "end": e, "names": names})

    # 汇总：角色独占 / 角色对共同发言（按首次出现顺序定序）
    rank = {name: i for i, name in enumerate(order)}
    solo = {name: 0 for name in order}
    pair_ms = {
        (order[i], order[j]): 0
        for i in range(len(order)) for j in range(i + 1, len(order))
    }
    for s, e, names in segments:
        dur = e - s
        if len(names) == 1:
            solo[next(iter(names))] += dur
        else:
            ns = sorted(names, key=lambda n: rank[n])
            for x in range(len(ns)):
                for y in range(x + 1, len(ns)):
                    pair_ms[(ns[x], ns[y])] += dur
    pairs = sorted(pair_ms.items(), key=lambda kv: (rank[kv[0][0]], rank[kv[0][1]]))
    return {"names": order, "overlaps": overlaps, "solo": solo, "pairs": pairs}


def _talkover_prepare(req: "TalkoverRequest"):
    """角色抢话分析公共预处理：解析 SRT 并做角色标记 / 抢话分析，失败抛 400。"""
    try:
        entries = parse_srt(req.srt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        result = analyze_talkover(entries)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return entries, result


def _talkover_overlap_payload(seq: int, o: dict, rank: dict) -> dict:
    return {
        "seq": seq,
        "start": fmt_time(o["start"]),
        "end": fmt_time(o["end"]),
        "start_ms": o["start"],
        "end_ms": o["end"],
        "duration_ms": o["end"] - o["start"],
        "roles": sorted(o["names"], key=rank.get),
    }


@app.post("/api/talkover/preview")
def talkover_preview(req: TalkoverRequest):
    entries, r = _talkover_prepare(req)
    names = r["names"]
    rank = {n: i for i, n in enumerate(names)}
    overlaps = [
        _talkover_overlap_payload(k, o, rank)
        for k, o in enumerate(r["overlaps"], 1)
    ]
    return {
        "entry_count": len(entries),
        "role_count": len(names),
        "roles": names,
        "overlap_count": len(overlaps),
        "overlap_total_ms": sum(o["end"] - o["start"] for o in r["overlaps"]),
        "overlaps": overlaps,
        "solo": [
            {"role": n, "ms": r["solo"][n], "time": fmt_time(r["solo"][n])}
            for n in names
        ],
        "pairs": [
            {"role_a": a, "role_b": b, "ms": ms, "time": fmt_time(ms)}
            for (a, b), ms in r["pairs"]
        ],
    }


_TALKOVER_SEG_HEADER = ["序号", "起点(ms)", "终点(ms)", "时长(ms)",
                        "起点", "终点", "同时发言角色"]
_TALKOVER_SOLO_HEADER = ["角色", "独占(ms)", "独占时长"]
_TALKOVER_PAIR_HEADER = ["角色A", "角色B", "共同发言(ms)", "共同发言时长"]


@app.post("/api/talkover/export")
def talkover_export(req: TalkoverRequest):
    _entries, r = _talkover_prepare(req)
    rank = {n: i for i, n in enumerate(r["names"])}
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")  # RFC 4180：CRLF、按需加引号
    writer.writerow(["# 抢话片段（两名及以上角色同时活跃，端点相触不算）"])
    writer.writerow(_TALKOVER_SEG_HEADER)
    for k, o in enumerate(r["overlaps"], 1):
        writer.writerow([
            k,
            o["start"],
            o["end"],
            o["end"] - o["start"],
            fmt_time(o["start"]),
            fmt_time(o["end"]),
            "、".join(sorted(o["names"], key=rank.get)),
        ])
    writer.writerow([])
    writer.writerow(["# 每名角色独占时长（仅该角色一人活跃）"])
    writer.writerow(_TALKOVER_SOLO_HEADER)
    for n in r["names"]:
        writer.writerow([n, r["solo"][n], fmt_time(r["solo"][n])])
    writer.writerow([])
    writer.writerow(["# 每对角色共同发言时长（两人同时活跃即计入，允许有第三人在场）"])
    writer.writerow(_TALKOVER_PAIR_HEADER)
    for (a, b), ms in r["pairs"]:
        writer.writerow([a, b, ms, fmt_time(ms)])
    # UTF-8 BOM，确保 Excel 直接双击打开时中文不乱码
    return Response(
        content=b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="talkover.csv"'},
    )


# ---------------------------------------------------------------- 术语统一替换

class TermRequest(BaseModel):
    srt: str
    terms: str  # 逐行术语：标准词=别名1|别名2


def _ascii_lower(s: str) -> str:
    """仅把 ASCII 大写字母转为小写，其余字符（含非 ASCII）原样保留。"""
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in s)


def _parse_term_lines(text: str) -> List[dict]:
    """逐行解析术语「标准词=别名1|别名2」。

    空行跳过；标准词与每个别名去首尾空白后均不得为空；同一别名
    （忽略 ASCII 大小写）不得对应不同标准词，同一标准词重复登记同一
    别名时去重。返回 [{row, std, aliases:[{alias, key}]}]，key 为归一化
    别名，aliases 按行内顺序去重。
    """
    out = []
    owner = {}  # 归一化别名 -> (标准词, 首次登记行号)
    for i, line in enumerate(
        text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 1
    ):
        if not line.strip():
            continue
        std, sep, alias_part = line.partition("=")
        std = std.strip()
        if not sep or not std:
            raise ValueError(
                f"术语第 {i} 行：须为「标准词=别名1|别名2」格式，且标准词不能为空"
            )
        aliases = []
        seen_in_line = set()
        for piece in alias_part.split("|"):
            alias = piece.strip()
            if not alias:
                raise ValueError(f"术语第 {i} 行：别名不能为空")
            key = _ascii_lower(alias)
            if key in seen_in_line:
                continue
            seen_in_line.add(key)
            if key in owner and owner[key][0] != std:
                raise ValueError(
                    f"术语第 {i} 行：别名「{alias}」已对应标准词「{owner[key][0]}」"
                    f"（第 {owner[key][1]} 行），不能同时对应「{std}」"
                )
            owner.setdefault(key, (std, i))
            aliases.append({"alias": alias, "key": key})
        out.append({"row": i, "std": std, "aliases": aliases})
    if not out:
        raise ValueError("请至少填写 1 行术语（每行：标准词=别名1|别名2）")
    return out


def _collect_term_aliases(terms: List[dict]) -> List[dict]:
    """汇总全部别名：按归一化别名去重，保留首次登记的标准词 / 行号 / 写法。

    返回 [{key, std, row, alias, order}]，order 为登记顺序，用于同一跨度
    完全并列时的确定性兜底（先登记者优先）。
    """
    uniq = {}
    for t in terms:
        for a in t["aliases"]:
            if a["key"] not in uniq:
                uniq[a["key"]] = {
                    "key": a["key"], "std": t["std"],
                    "row": t["row"], "alias": a["alias"],
                }
    aliases = list(uniq.values())
    for k, a in enumerate(aliases):
        a["order"] = k
    return aliases


def find_term_hits(text: str, aliases: List[dict]) -> List[dict]:
    """在原文 text 上一次性查找全部别名（忽略 ASCII 大小写，允许重叠命中）。

    位置为 text 的字符偏移（左闭右开）；命中全部在原文上定位，后续不在
    修订文本上继续匹配。返回按 (起点, 终点, 登记顺序) 排序的命中列表。
    """
    norm = _ascii_lower(text)
    hits = []
    for a in aliases:
        key = a["key"]
        size = len(key)
        start = 0
        while True:
            p = norm.find(key, start)
            if p < 0:
                break
            hits.append({
                "start": p,
                "end": p + size,
                "surface": text[p:p + size],
                "alias": a["alias"],
                "std": a["std"],
                "row": a["row"],
                "order": a["order"],
            })
            start = p + 1  # 允许重叠命中（如 aa 在 aaa 中命中两次）
    hits.sort(key=lambda h: (h["start"], h["end"], h["order"]))
    return hits


def plan_term_replacements(text: str, hits: List[dict]):
    """从同条字幕的全部命中中规划互不重叠的采用组合。

    目标依次：最大化被替换的原文字符数、最小化替换次数、采用命中的起点
    序列（升序）字典序最小；完全并列（同一跨度命中不同术语）时按别名登记
    顺序取先者。命中全部在原文上定位，规划确定后一次性改写，不能边替换
    边继续匹配。

    按字符位置自右向左 DP：dp[pos] 为后缀 text[pos:] 的最优
    (替换字符数, 替换次数, 起点序列)，choice[pos] 记录选中的命中
    （None 表示该字符不替换）。同一起点的命中已按 (终点, 登记顺序) 排序，
    严格更优才替换，故完全并列时先登记者胜出。字幕正文规模下起点序列
    元组比较开销可忽略。返回 (adopted, discarded)，均按起点升序。
    """
    L = len(text)
    by_start = {}
    for h in hits:
        by_start.setdefault(h["start"], []).append(h)
    dp = [None] * (L + 1)
    dp[L] = (0, 0, ())
    choice = [None] * (L + 1)
    for pos in range(L - 1, -1, -1):
        best = dp[pos + 1]  # 不替换当前字符
        pick = None
        for h in by_start.get(pos, ()):
            sub = dp[h["end"]]
            cand = (sub[0] + h["end"] - pos, sub[1] + 1, (pos,) + sub[2])
            if (cand[0] > best[0]
                    or (cand[0] == best[0] and cand[1] < best[1])
                    or (cand[0] == best[0] and cand[1] == best[1]
                        and cand[2] < best[2])):
                best = cand
                pick = h
        dp[pos] = best
        choice[pos] = pick
    adopted = []
    pos = 0
    while pos < L:
        h = choice[pos]
        if h is None:
            pos += 1
        else:
            adopted.append(h)
            pos = h["end"]
    chosen = {id(h) for h in adopted}
    discarded = [h for h in hits if id(h) not in chosen]
    return adopted, discarded


def _apply_term_hits(text: str, adopted: List[dict]) -> str:
    """按采用命中一次性改写原文（adopted 互不重叠且按起点升序）。"""
    out = []
    pos = 0
    for h in adopted:
        out.append(text[pos:h["start"]])
        out.append(h["std"])
        pos = h["end"]
    out.append(text[pos:])
    return "".join(out)


def _term_prepare(req: "TermRequest"):
    """术语统一替换公共预处理：解析 SRT 与逐行术语并汇总别名，失败抛 400。"""
    try:
        entries = parse_srt(req.srt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        terms = _parse_term_lines(req.terms)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return entries, terms, _collect_term_aliases(terms)


def _term_entry_result(entry: dict, aliases: List[dict]) -> dict:
    """对单条字幕做命中查找、替换规划与一次性改写。

    返回 {text, revised, changed, hits, segments}：hits 为全部命中（含采用
    标志，按起点排序）；segments 为修订正文分段（rep=True 为替换片段），
    供预览高亮。正文多行按换行连接后查找，命中位置为连接后文本的字符
    偏移；别名不含换行，故命中不会跨行，改写后行结构不变。
    """
    text = "\n".join(entry["lines"])
    hits = find_term_hits(text, aliases)
    adopted, _discarded = plan_term_replacements(text, hits)
    chosen = {id(h) for h in adopted}
    revised = _apply_term_hits(text, adopted)
    segments = []
    pos = 0
    for h in adopted:
        if h["start"] > pos:
            segments.append({"text": text[pos:h["start"]], "rep": False})
        segments.append({"text": h["std"], "rep": True})
        pos = h["end"]
    if pos < len(text):
        segments.append({"text": text[pos:], "rep": False})
    return {
        "text": text,
        "revised": revised,
        "changed": revised != text,
        "hits": [
            {
                "start": h["start"],
                "end": h["end"],
                "surface": h["surface"],
                "alias": h["alias"],
                "std": h["std"],
                "row": h["row"],
                "adopted": id(h) in chosen,
            }
            for h in hits
        ],
        "segments": segments,
    }


@app.post("/api/terms/preview")
def terms_preview(req: TermRequest):
    entries, terms, aliases = _term_prepare(req)
    items = []
    hit_count = adopted_count = changed_count = 0
    for i, e in enumerate(entries, 1):
        r = _term_entry_result(e, aliases)
        hit_count += len(r["hits"])
        adopted_count += sum(1 for h in r["hits"] if h["adopted"])
        changed_count += 1 if r["changed"] else 0
        items.append({
            "seq": i,
            "index": e["index"],
            "time": f"{fmt_time(e['start'])} --> {fmt_time(e['end'])}",
            **r,
        })
    return {
        "entry_count": len(entries),
        "term_count": len(terms),
        "alias_count": len(aliases),
        "hit_count": hit_count,
        "adopted_count": adopted_count,
        "discarded_count": hit_count - adopted_count,
        "changed_count": changed_count,
        "entries": items,
    }


@app.post("/api/terms/export")
def terms_export(req: TermRequest):
    entries, _terms, aliases = _term_prepare(req)
    blocks = []
    for e in entries:
        r = _term_entry_result(e, aliases)
        blocks.append(
            f"{e['index']}\n{fmt_time(e['start'])} --> {fmt_time(e['end'])}\n"
            + (r["revised"] if r["revised"] else "")
        )
    content = "\n\n".join(blocks) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="terms_replaced.srt"'},
    )
