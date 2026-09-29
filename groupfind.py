#!/usr/bin/env python3
"""社区发现：读无向无权图，按 README 第 2 节口径分组、算指标、出报告与页面。

用法：
    python3 groupfind.py --graph <图文件> [--members <节点表>] [--groups <组统计>] [--page <页面>]

只依赖标准库。分组过程完全确定：不依赖 hash 随机盐、集合遍历顺序或随机数。
"""

import argparse
import html
import math
import os
import sys
from decimal import Decimal, ROUND_HALF_UP, getcontext

getcontext().prec = 60

MAX_ROUNDS = 20

PALETTE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
)


class GraphError(Exception):
    """图文件内容不合法。"""


def _parse_node(token, lineno, name):
    if not token.isascii() or not token.isdigit():
        raise GraphError(f"{name}: 第 {lineno} 行不合法: {token!r} 不是非负十进制整数")
    return int(token)


def parse_graph(path):
    """读图文件，返回 (nodes, adj, selfloops, deg, m)。

    nodes: 升序节点编号列表；adj: 节点 -> {邻居: 重数}（不含自环）；
    selfloops: 节点 -> 自环条数；deg: 节点 -> 度数（自环计 2）；m: 边数（自环计 1）。
    """
    name = os.path.basename(path)
    nodeset = set()
    adj = {}
    selfloops = {}
    deg = {}
    m = 0

    def ensure(v):
        if v not in nodeset:
            nodeset.add(v)
            adj[v] = {}
            deg[v] = 0

    try:
        f = open(path, "r", encoding="utf-8", newline=None)
    except OSError as exc:
        raise GraphError(f"读不到图文件 {path}: {exc.strerror or exc}") from exc
    with f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 1:
                ensure(_parse_node(parts[0], lineno, name))
            elif len(parts) == 2:
                u = _parse_node(parts[0], lineno, name)
                v = _parse_node(parts[1], lineno, name)
                ensure(u)
                ensure(v)
                m += 1
                if u == v:
                    selfloops[u] = selfloops.get(u, 0) + 1
                    deg[u] += 2
                else:
                    adj[u][v] = adj[u].get(v, 0) + 1
                    adj[v][u] = adj[v].get(u, 0) + 1
                    deg[u] += 1
                    deg[v] += 1
            else:
                raise GraphError(f"{name}: 第 {lineno} 行不合法: 每行只能有 1 或 2 个整数")
    return sorted(nodeset), adj, selfloops, deg, m


def find_groups(nodes, adj, selfloops, deg, m, max_rounds=MAX_ROUNDS):
    """按 README 第 2 节口径分组。

    返回 (gid_of, groups, rounds, converged)：
    gid_of: 节点 -> 组编号；groups: 按组编号升序的 (组编号, 规模, e, d) 列表；
    rounds: 实际轮数；converged: 是否收敛。
    """
    # 内部用 token 标识组，组编号 = 组内最小节点编号，单独维护，避免反复改名。
    token_of = {}
    members = []   # token -> set，空组置 None
    dsum = []      # token -> d(C)
    esum = []      # token -> e(C)
    gid = []       # token -> 组内最小节点编号
    for v in nodes:
        token_of[v] = len(members)
        members.append({v})
        dsum.append(deg[v])
        esum.append(selfloops.get(v, 0))
        gid.append(v)

    two_m = 2 * m
    rounds = 0
    converged = False
    for rnd in range(1, max_rounds + 1):
        rounds = rnd
        moved = False
        for v in nodes:
            tv = token_of[v]
            counts = {}
            for u, w in adj[v].items():
                tu = token_of[u]
                counts[tu] = counts.get(tu, 0) + w
            k_in_a = counts.get(tv, 0)          # k_in(v, A \ {v})，自环本就不在邻接表里
            d_a_prime = dsum[tv] - deg[v]
            deg_v = deg[v]
            best_tok = -1
            best_n = 0
            best_gid = 0
            for tok, k in counts.items():
                if tok == tv:
                    continue
                gain = two_m * (k - k_in_a) - deg_v * (dsum[tok] - d_a_prime)
                if gain > 0 and (
                    best_tok < 0 or gain > best_n or (gain == best_n and gid[tok] < best_gid)
                ):
                    best_tok = tok
                    best_n = gain
                    best_gid = gid[tok]
            if best_tok < 0:
                continue
            loops = selfloops.get(v, 0)
            old = members[tv]
            old.discard(v)
            esum[tv] -= k_in_a + loops
            dsum[tv] -= deg_v
            if not old:
                members[tv] = None
            elif gid[tv] == v:
                gid[tv] = min(old)
            members[best_tok].add(v)
            esum[best_tok] += counts[best_tok] + loops
            dsum[best_tok] += deg_v
            if v < gid[best_tok]:
                gid[best_tok] = v
            token_of[v] = best_tok
            moved = True
        if not moved:
            converged = True
            break

    groups = []
    for tok, mem in enumerate(members):
        if mem:
            groups.append((gid[tok], len(mem), esum[tok], dsum[tok]))
    groups.sort()
    gid_of = {v: gid[token_of[v]] for v in nodes}
    return gid_of, groups, rounds, converged


def _fmt(num, den, places):
    """num/den 四舍五入到 places 位小数；den 为 0 时记 0。"""
    if den == 0:
        value = Decimal(0)
    else:
        value = Decimal(num) / Decimal(den)
    q = value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    if q == 0:
        q = abs(q)
    return format(q, "f")


def group_stats(groups, m):
    """每组 (组编号, 规模, e, d) -> (组编号, 规模, ρ 字符串)。"""
    return [(g, size, _fmt(2 * e, d, 3)) for g, size, e, d in groups]


def modularity(groups, m):
    """Q 的分子、分母：Q = (4m·Σe − Σd²) / (4m²)。"""
    sum_e = sum(e for _, _, e, _ in groups)
    sum_d2 = sum(d * d for _, _, _, d in groups)
    return 4 * m * sum_e - sum_d2, 4 * m * m


def internal_ratio(groups, m):
    """R = Σe / m 的分子、分母。"""
    return sum(e for _, _, e, _ in groups), m


def build_report(basename, n, m, gid_of, groups, rounds, converged):
    q_num, q_den = modularity(groups, m)
    r_num, r_den = internal_ratio(groups, m)
    lines = [
        f"图\t{basename}",
        f"节点\t{n}",
        f"边\t{m}",
        f"分组\t{len(groups)}",
        f"模块度\t{_fmt(q_num, q_den, 4)}",
        f"内部边占比\t{_fmt(r_num, r_den, 4)}",
        f"轮数\t{rounds}",
        f"收敛\t{'yes' if converged else 'no'}",
        "分组统计",
    ]
    for g, size, rho in group_stats(groups, m):
        lines.append(f"{g}\t{size}\t{rho}")
    lines.append("节点归属")
    for v in sorted(gid_of):
        lines.append(f"{v}\t{gid_of[v]}")
    return "\n".join(lines) + "\n"


def build_members(gid_of):
    return "".join(f"{v}\t{gid_of[v]}\n" for v in sorted(gid_of))


def build_groups(groups, m):
    return "".join(f"{g}\t{size}\t{rho}\n" for g, size, rho in group_stats(groups, m))


def render_page(basename, nodes, adj, selfloops, gid_of, groups, rounds, converged, m):
    """生成自包含 HTML：内联样式 + 内联 SVG，无脚本、无外部引用。

    所有分组与数字都来自入参（同一次引擎运行的结果），页面不重算。
    """
    title = html.escape(basename)
    order = [g for g, _, _, _ in groups]
    color_of = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(order)}

    # 布局：组排在中心大圆上，组内节点排在小圆上，完全由编号决定，重复运行一致。
    width, height = 760, 680
    cx, cy, big_r = 360.0, 340.0, 250.0
    pos = {}
    for i, (g, size, _, _) in enumerate(groups):
        ga = 2.0 * math.pi * i / len(groups) - math.pi / 2.0
        gx, gy = cx + big_r * math.cos(ga), cy + big_r * math.sin(ga)
        small_r = 0.0 if size == 1 else 16.0 + 4.0 * math.sqrt(size)
        mem = sorted(v for v in nodes if gid_of[v] == g)
        for j, v in enumerate(mem):
            va = 2.0 * math.pi * j / size - math.pi / 2.0
            pos[v] = (gx + small_r * math.cos(va), gy + small_r * math.sin(va))

    parts = []
    for u in nodes:
        for v in adj[u]:
            if v <= u:
                continue
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            if gid_of[u] == gid_of[v]:
                parts.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
                    f' class="ein"/>'
                )
            else:
                parts.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
                    f' class="eout"/>'
                )
    for v in nodes:
        if selfloops.get(v):
            x, y = pos[v]
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y - 14:.1f}" r="7" class="eself"/>'
            )
    for v in nodes:
        x, y = pos[v]
        c = color_of[gid_of[v]]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{c}" class="node"/>')
        parts.append(f'<text x="{x:.1f}" y="{y + 3.5:.1f}" class="label">{v}</text>')
    svg = (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}"'
        f' xmlns="http://www.w3.org/2000/svg" role="img">{"".join(parts)}</svg>'
    )

    q_num, q_den = modularity(groups, m)
    r_num, r_den = internal_ratio(groups, m)
    rows = "".join(
        f'<tr><td><span class="sw" style="background:{color_of[g]}"></span>{g}</td>'
        f"<td>{size}</td><td>{rho}</td></tr>"
        for g, size, rho in group_stats(groups, m)
    )
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>分组结果 - {title}</title>
<style>
body {{ font-family: sans-serif; margin: 24px; color: #222; }}
h1 {{ font-size: 20px; }}
.wrap {{ display: flex; gap: 32px; align-items: flex-start; flex-wrap: wrap; }}
svg {{ background: #fafafa; border: 1px solid #ddd; }}
.ein {{ stroke: #666; stroke-width: 2.2; stroke-opacity: 0.75; }}
.eout {{ stroke: #bbb; stroke-width: 0.7; stroke-opacity: 0.6; }}
.eself {{ fill: none; stroke: #666; stroke-width: 2.2; stroke-opacity: 0.75; }}
.node {{ stroke: #fff; stroke-width: 1.5; }}
.label {{ font-size: 9px; text-anchor: middle; fill: #fff; pointer-events: none; }}
table {{ border-collapse: collapse; font-size: 14px; }}
th, td {{ border: 1px solid #ccc; padding: 4px 12px; text-align: right; }}
th {{ background: #f0f0f0; }}
.sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 5px; margin-right: 6px; }}
.summary {{ margin: 12px 0 20px; font-size: 14px; }}
.summary span {{ margin-right: 18px; }}
</style>
</head>
<body>
<h1>分组结果：{title}</h1>
<div class="summary">
<span>节点 {len(nodes)}</span><span>边 {m}</span><span>分组 {len(groups)}</span>
<span>模块度 {_fmt(q_num, q_den, 4)}</span><span>内部边占比 {_fmt(r_num, r_den, 4)}</span>
<span>轮数 {rounds}</span><span>收敛 {"yes" if converged else "no"}</span>
</div>
<div class="wrap">
{svg}
<table>
<tr><th>组编号</th><th>规模</th><th>内部边比例</th></tr>
{rows}
</table>
</div>
</body>
</html>
"""


def _write_text(path, text):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description="图的社区发现（确定性口径，见 README 第 2 节）")
    parser.add_argument("--graph", required=True, help="图文件路径")
    parser.add_argument("--members", help="节点归属输出路径，默认 var/<图名>.members.txt")
    parser.add_argument("--groups", help="组统计输出路径，默认 var/<图名>.groups.txt")
    parser.add_argument("--page", help="分组页面输出路径，仅显式传入时生成")
    args = parser.parse_args(argv)

    try:
        nodes, adj, selfloops, deg, m = parse_graph(args.graph)
    except GraphError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    gid_of, groups, rounds, converged = find_groups(nodes, adj, selfloops, deg, m)

    basename = os.path.basename(args.graph)
    stem = os.path.splitext(basename)[0]
    members_path = args.members or os.path.join("var", f"{stem}.members.txt")
    groups_path = args.groups or os.path.join("var", f"{stem}.groups.txt")

    _write_text(members_path, build_members(gid_of))
    _write_text(groups_path, build_groups(groups, m))
    if args.page:
        _write_text(
            args.page,
            render_page(basename, nodes, adj, selfloops, gid_of, groups, rounds, converged, m),
        )

    sys.stdout.write(
        build_report(basename, len(nodes), m, gid_of, groups, rounds, converged)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
