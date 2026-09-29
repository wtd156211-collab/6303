#!/usr/bin/env python3
"""groupfind：确定性社区发现。

给一张无向无权图，按 README 第 2 节写死的口径分组：
节点升序逐轮局部移动、增益 N 最大者胜出（并列取组编号最小者）、
一轮无人移动即收敛、最多 20 轮。输出每个节点的组编号、每组的规模与
内部边比例，以及模块度 Q、内部边占比 R、轮数与收敛标记。

命令行：
    python3 groupfind.py --graph <图文件> [--members <节点表>]
                         [--groups <组统计>] [--page <页面>]

只用标准库；结果不依赖 hash() 随机盐、集合遍历顺序或随机数。
"""

import argparse
import html
import math
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

MAX_ROUNDS = 20

# 页面用的固定色板，按组编号顺序循环取色。
PALETTE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
)


class GraphError(Exception):
    """图文件读不到或某行不合法。"""


class Graph:
    """无向无权图：带重数邻接表（不含自环），自环单独计数。"""

    __slots__ = ("nodes", "adj", "loops", "deg", "m")

    def __init__(self):
        self.nodes = []          # 升序节点表
        self.adj = {}            # adj[u][v] = 重数（u != v）
        self.loops = {}          # loops[u] = 自环条数
        self.deg = {}            # deg[v]：普通边计 1，自环计 2
        self.m = 0               # 边数，自环计 1


def _parse_node(token, path, lineno):
    if not token or any(ch < "0" or ch > "9" for ch in token):
        raise GraphError("%s: 第 %d 行不合法: %r" % (path, lineno, token))
    return int(token)


def parse_graph(path):
    """读图文件，返回 Graph。读不到或行不合法时抛 GraphError。"""
    graph = Graph()
    adj, loops, deg = graph.adj, graph.loops, graph.deg
    try:
        fh = open(path, "r", encoding="utf-8-sig", newline=None)
    except OSError as exc:
        raise GraphError("读不到图文件 %s: %s" % (path, exc.strerror or exc))
    with fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 1:
                u = _parse_node(parts[0], path, lineno)
                adj.setdefault(u, {})
                deg.setdefault(u, 0)
            elif len(parts) == 2:
                u = _parse_node(parts[0], path, lineno)
                v = _parse_node(parts[1], path, lineno)
                adj.setdefault(u, {})
                adj.setdefault(v, {})
                deg.setdefault(u, 0)
                deg.setdefault(v, 0)
                if u == v:
                    loops[u] = loops.get(u, 0) + 1
                    deg[u] += 2
                else:
                    adj[u][v] = adj[u].get(v, 0) + 1
                    adj[v][u] = adj[v].get(u, 0) + 1
                    deg[u] += 1
                    deg[v] += 1
                graph.m += 1
            else:
                raise GraphError("%s: 第 %d 行不合法: %r" % (path, lineno, line))
    graph.nodes = sorted(adj)
    return graph


def _move_nodes(graph):
    """按口径做局部移动，返回 (node_group, members, gmin, gsum, 轮数, 收敛)。

    组的内部键（rep）创建后不变；对外组编号 = 组内最小节点编号（gmin），
    每次移动后立即重算。
    """
    node_group = {}
    members = {}
    gmin = {}
    gsum = {}
    for v in graph.nodes:
        node_group[v] = v
        members[v] = {v}
        gmin[v] = v
        gsum[v] = graph.deg[v]
    two_m = 2 * graph.m
    rounds = 0
    converged = False
    for rnd in range(1, MAX_ROUNDS + 1):
        rounds = rnd
        moved = False
        for v in graph.nodes:
            rep_a = node_group[v]
            dv = graph.deg[v]
            cnt = {}
            for u, w in graph.adj[v].items():
                rep_u = node_group[u]
                cnt[rep_u] = cnt.get(rep_u, 0) + w
            kin_a = cnt.get(rep_a, 0)          # k_in(v, A')，自环本就不在邻接表里
            d_a_prime = gsum[rep_a] - dv       # d(A')，A' 为空时 gsum 恰为 dv，得 0
            best_rep = None
            best_n = 0
            best_gid = 0
            for rep_c, k in cnt.items():
                if rep_c == rep_a:
                    continue
                n = two_m * (k - kin_a) - dv * (gsum[rep_c] - d_a_prime)
                if n <= 0:
                    continue
                gid = gmin[rep_c]
                if best_rep is None or n > best_n or (n == best_n and gid < best_gid):
                    best_rep, best_n, best_gid = rep_c, n, gid
            if best_rep is None:
                continue
            moved = True
            node_group[v] = best_rep
            rest = members[rep_a]
            rest.discard(v)
            gsum[rep_a] -= dv
            if not rest:
                del members[rep_a], gmin[rep_a], gsum[rep_a]
            elif gmin[rep_a] == v:
                gmin[rep_a] = min(rest)
            joined = members[best_rep]
            joined.add(v)
            gsum[best_rep] += dv
            if v < gmin[best_rep]:
                gmin[best_rep] = v
        if not moved:
            converged = True
            break
    return node_group, members, gmin, gsum, rounds, converged


def _fmt_ratio(num, den, places):
    """num/den 四舍五入到 places 位小数，返回定点字符串。"""
    if den == 0:
        num, den = 0, 1
    val = (Decimal(num) / Decimal(den)).quantize(
        Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    if val == 0:
        val = val.copy_abs()
    return str(val)


class Result:
    """一次运行的全部产出：分组、指标、轮数。"""

    __slots__ = ("membership", "groups", "q_str", "r_str", "rounds", "converged")

    def __init__(self, membership, groups, q_str, r_str, rounds, converged):
        self.membership = membership  # [(节点, 组编号)]，按节点升序
        self.groups = groups          # [(组编号, 规模, 内部边比例串)]，按组编号升序
        self.q_str = q_str
        self.r_str = r_str
        self.rounds = rounds
        self.converged = converged


def analyze(graph):
    """分组并算指标，返回 Result。"""
    node_group, members, gmin, gsum, rounds, converged = _move_nodes(graph)
    m = graph.m
    internal = {rep: 0 for rep in members}
    for u, nbrs in graph.adj.items():
        rep_u = node_group[u]
        for v, w in nbrs.items():
            if u < v and node_group[v] == rep_u:
                internal[rep_u] += w
    for u, c in graph.loops.items():
        internal[node_group[u]] += c
    sum_e = sum(internal.values())
    sum_d2 = sum(d * d for d in gsum.values())
    q_str = _fmt_ratio(4 * m * sum_e - sum_d2, 4 * m * m, 4) if m else "0.0000"
    r_str = _fmt_ratio(sum_e, m, 4) if m else "0.0000"
    groups = []
    for rep in sorted(members, key=gmin.__getitem__):
        d = gsum[rep]
        rho = _fmt_ratio(2 * internal[rep], d, 3) if d else "0.000"
        groups.append((gmin[rep], len(members[rep]), rho))
    membership = [(v, gmin[node_group[v]]) for v in graph.nodes]
    return Result(membership, groups, q_str, r_str, rounds, converged)


def build_report(name, graph, result):
    """stdout 报告：字段顺序、TAB 分隔、小数位数都按 README 写死。"""
    lines = [
        "图\t" + name,
        "节点\t%d" % len(graph.nodes),
        "边\t%d" % graph.m,
        "分组\t%d" % len(result.groups),
        "模块度\t" + result.q_str,
        "内部边占比\t" + result.r_str,
        "轮数\t%d" % result.rounds,
        "收敛\t" + ("yes" if result.converged else "no"),
        "分组统计",
    ]
    lines.extend("%d\t%d\t%s" % row for row in result.groups)
    lines.append("节点归属")
    lines.extend("%d\t%d" % row for row in result.membership)
    return "\n".join(lines) + "\n"


def members_text(result):
    return "".join("%d\t%d\n" % row for row in result.membership)


def groups_text(result):
    return "".join("%d\t%d\t%s\n" % row for row in result.groups)


def render_page(result, graph, name):
    """自包含分组页面：内联样式 + 内联 SVG，数字全部来自 result，不重算。"""
    groups = result.groups
    gid_of = dict(result.membership)
    index_of = {gid: i for i, (gid, _size, _rho) in enumerate(groups)}
    color_of = {gid: PALETTE[i % len(PALETTE)] for gid, i in index_of.items()}
    members_of = {gid: [] for gid, _size, _rho in groups}
    for v, gid in result.membership:
        members_of[gid].append(v)

    count = len(groups)
    positions = {}
    small_max = 0.0
    if count:
        big_r = max(180.0, 60.0 * count / math.pi)
        for i, (gid, size, _rho) in enumerate(groups):
            ang = 2.0 * math.pi * i / count - math.pi / 2.0
            cx, cy = big_r * math.cos(ang), big_r * math.sin(ang)
            small_r = 14.0 + 4.0 * math.sqrt(size)
            small_max = max(small_max, small_r)
            for j, v in enumerate(members_of[gid]):
                a = 2.0 * math.pi * j / size
                positions[v] = (cx + small_r * math.cos(a),
                                cy + small_r * math.sin(a))
    else:
        big_r = 0.0
    extent = big_r + small_max + 40.0
    side = 2.0 * extent

    cross, inner, loops_svg, dots = [], [], [], []
    for u, nbrs in graph.adj.items():
        x1, y1 = positions[u]
        for v in nbrs:
            if u < v:
                x2, y2 = positions[v]
                line = '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (
                    x1, y1, x2, y2)
                (inner if gid_of[u] == gid_of[v] else cross).append(line)
    for u in graph.loops:
        x, y = positions[u]
        loops_svg.append(
            '<circle cx="%.1f" cy="%.1f" r="7" fill="none" stroke="%s" '
            'stroke-width="1.5"/>' % (x, y - 10, color_of[gid_of[u]]))
    for v, gid in result.membership:
        x, y = positions[v]
        dots.append(
            '<circle cx="%.1f" cy="%.1f" r="5" fill="%s">'
            "<title>节点 %d / 组 %d</title></circle>" % (
                x, y, color_of[gid], v, gid))

    svg = (
        '<svg viewBox="%.1f %.1f %.1f %.1f" width="%d" height="%d" '
        'role="img" aria-label="分组图">'
        '<g stroke="#cccccc" stroke-width="0.6">%s</g>'
        '<g stroke="#888888" stroke-width="2">%s</g>'
        "%s%s</svg>"
    ) % (-extent, -extent, side, side, int(math.ceil(side)), int(math.ceil(side)),
         "".join(cross), "".join(inner), "".join(loops_svg), "".join(dots))

    rows = "".join(
        '<tr><td><span class="sw" style="background:%s"></span></td>'
        "<td>%d</td><td>%d</td><td>%s</td></tr>" % (
            color_of[gid], gid, size, rho)
        for gid, size, rho in groups)

    title = html.escape(name)
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>分组结果 - %s</title>
<style>
body { font-family: "Helvetica Neue", Arial, "PingFang SC", sans-serif;
       margin: 24px; color: #222; }
h1 { font-size: 20px; }
.summary span { margin-right: 18px; }
.wrap { display: flex; gap: 32px; align-items: flex-start; flex-wrap: wrap; }
svg { background: #fafafa; border: 1px solid #e0e0e0; }
table { border-collapse: collapse; }
th, td { border: 1px solid #ddd; padding: 4px 12px; text-align: right;
         font-variant-numeric: tabular-nums; }
th { background: #f2f2f2; }
.sw { display: inline-block; width: 12px; height: 12px; border-radius: 6px; }
</style>
</head>
<body>
<h1>分组结果：%s</h1>
<p class="summary">
<span>节点 <b>%d</b></span><span>边 <b>%d</b></span>
<span>分组 <b>%d</b></span><span>模块度 <b>%s</b></span>
<span>内部边占比 <b>%s</b></span><span>轮数 <b>%d</b></span>
<span>收敛 <b>%s</b></span>
</p>
<div class="wrap">
%s
<aside>
<h2>分组统计</h2>
<table>
<tr><th></th><th>组编号</th><th>规模</th><th>内部边比例</th></tr>
%s
</table>
</aside>
</div>
</body>
</html>
""" % (title, title, len(graph.nodes), graph.m, len(groups),
       result.q_str, result.r_str, result.rounds,
       "yes" if result.converged else "no", svg, rows)


def _write_text(path, text):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="groupfind.py", description="确定性社区发现：分组、指标、报告与分组页面")
    parser.add_argument("--graph", required=True, help="图文件路径")
    parser.add_argument("--members", help="节点归属输出（默认 var/<图名>.members.txt）")
    parser.add_argument("--groups", help="组统计输出（默认 var/<图名>.groups.txt）")
    parser.add_argument("--page", help="分组页面输出（仅显式传入时生成）")
    args = parser.parse_args(argv)

    try:
        graph = parse_graph(args.graph)
    except GraphError as exc:
        print(exc, file=sys.stderr)
        return 1

    result = analyze(graph)
    name = os.path.basename(args.graph)
    stem = os.path.splitext(name)[0]

    _write_text(args.members or os.path.join("var", stem + ".members.txt"),
                members_text(result))
    _write_text(args.groups or os.path.join("var", stem + ".groups.txt"),
                groups_text(result))
    if args.page:
        _write_text(args.page, render_page(result, graph, name))

    sys.stdout.write(build_report(name, graph, result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
