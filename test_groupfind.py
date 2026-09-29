#!/usr/bin/env python3
"""groupfind 的 unittest 测试：样例逐行比对、确定性、边角情形、页面、CLI。"""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest

import groupfind

ROOT = os.path.dirname(os.path.abspath(__file__))
GRAPHS = os.path.join(ROOT, "samples", "graphs")
EXPECTED = os.path.join(ROOT, "samples", "expected")

SAMPLE_NAMES = [
    "01-two-communities.txt",
    "02-nested-communities.txt",
    "03-bridge-node.txt",
    "04-isolated-nodes.txt",
    "05-self-loops.txt",
    "06-near-complete.txt",
    "07-single-node.txt",
    "08-scale-100k.txt",
]

Q_THRESHOLDS = {
    "01-two-communities.txt": 0.30,
    "02-nested-communities.txt": 0.30,
    "03-bridge-node.txt": 0.30,
    "04-isolated-nodes.txt": 0.30,
    "05-self-loops.txt": 0.30,
    "06-near-complete.txt": 0.00,
    "07-single-node.txt": 0.00,
    "08-scale-100k.txt": 0.30,
}


def run_cli(graph_path, *extra):
    """进程内跑 CLI，返回 (退出码, stdout, 临时目录)。"""
    tmp = tempfile.TemporaryDirectory()
    members = os.path.join(tmp.name, "out.members.txt")
    groups = os.path.join(tmp.name, "out.groups.txt")
    argv = ["--graph", graph_path, "--members", members, "--groups", groups, *extra]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = groupfind.main(argv)
    return code, out.getvalue(), tmp, members, groups


def parse_report(report):
    """把报告拆成 (头部键值 dict, 分组统计行, 节点归属行)。"""
    lines = report.splitlines()
    head = {}
    i = 0
    while lines[i] != "分组统计":
        key, value = lines[i].split("\t")
        head[key] = value
        i += 1
    j = lines.index("节点归属")
    stats = lines[i + 1:j]
    membership = lines[j + 1:]
    return head, stats, membership


class TestSamples(unittest.TestCase):
    def test_expected_output_line_by_line(self):
        for name in SAMPLE_NAMES:
            with self.subTest(graph=name):
                code, out, tmp, _, _ = run_cli(os.path.join(GRAPHS, name))
                self.assertEqual(code, 0)
                with open(os.path.join(EXPECTED, name), encoding="utf-8") as f:
                    self.assertEqual(out, f.read())
                tmp.cleanup()

    def test_modularity_thresholds(self):
        for name, threshold in Q_THRESHOLDS.items():
            with self.subTest(graph=name):
                code, out, tmp, _, _ = run_cli(os.path.join(GRAPHS, name))
                self.assertEqual(code, 0)
                head, _, _ = parse_report(out)
                self.assertGreaterEqual(float(head["模块度"]), threshold)
                tmp.cleanup()

    def test_members_and_groups_files_match_report(self):
        for name in SAMPLE_NAMES:
            with self.subTest(graph=name):
                code, out, tmp, members, groups = run_cli(os.path.join(GRAPHS, name))
                self.assertEqual(code, 0)
                _, stats, membership = parse_report(out)
                with open(members, encoding="utf-8") as f:
                    self.assertEqual(f.read().splitlines(), membership)
                with open(groups, encoding="utf-8") as f:
                    self.assertEqual(f.read().splitlines(), stats)
                tmp.cleanup()

    def test_determinism_across_hash_seeds(self):
        graph = os.path.join(GRAPHS, "02-nested-communities.txt")
        outputs = []
        for seed in ("0", "1", "123"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            proc = subprocess.run(
                [sys.executable, "groupfind.py", "--graph", graph,
                 "--members", os.path.join(tempfile.gettempdir(), "gf_m.txt"),
                 "--groups", os.path.join(tempfile.gettempdir(), "gf_g.txt")],
                capture_output=True, text=True, cwd=ROOT, env=env,
            )
            self.assertEqual(proc.returncode, 0)
            outputs.append(proc.stdout)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])

    def test_repeated_runs_identical(self):
        graph = os.path.join(GRAPHS, "03-bridge-node.txt")
        code1, out1, tmp1, m1, g1 = run_cli(graph)
        code2, out2, tmp2, m2, g2 = run_cli(graph)
        self.assertEqual((code1, out1), (code2, out2))
        with open(m1) as f1, open(m2) as f2:
            self.assertEqual(f1.read(), f2.read())
        with open(g1) as f1, open(g2) as f2:
            self.assertEqual(f1.read(), f2.read())
        tmp1.cleanup()
        tmp2.cleanup()


class TestParser(unittest.TestCase):
    def _write(self, text):
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        self.addCleanup(os.unlink, path)
        return path

    def test_comments_blanks_crlf_and_declarations(self):
        path = self._write("# 注释\r\n\r\n0 1\r\n1 2\n\n5\n")
        nodes, adj, selfloops, deg, m = groupfind.parse_graph(path)
        self.assertEqual(nodes, [0, 1, 2, 5])
        self.assertEqual(m, 2)
        self.assertEqual(deg[5], 0)

    def test_duplicate_edges_counted(self):
        path = self._write("0 1\n0 1\n1 0\n")
        _, adj, _, deg, m = groupfind.parse_graph(path)
        self.assertEqual(m, 3)
        self.assertEqual(adj[0][1], 3)
        self.assertEqual(deg[0], 3)

    def test_self_loop_counts(self):
        path = self._write("2 2\n")
        _, adj, selfloops, deg, m = groupfind.parse_graph(path)
        self.assertEqual(m, 1)
        self.assertEqual(selfloops[2], 1)
        self.assertEqual(deg[2], 2)
        self.assertEqual(adj[2], {})

    def test_invalid_lines_rejected(self):
        for bad in ("0 1 2\n", "a b\n", "-1 2\n", "1.5 2\n"):
            path = self._write(bad)
            with self.assertRaises(groupfind.GraphError):
                groupfind.parse_graph(path)

    def test_missing_file(self):
        with self.assertRaises(groupfind.GraphError):
            groupfind.parse_graph("/nonexistent/graph.txt")


class TestEngine(unittest.TestCase):
    def _run(self, text):
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        self.addCleanup(os.unlink, path)
        return groupfind.find_groups(*groupfind.parse_graph(path))

    def test_single_node(self):
        gid_of, groups, rounds, converged = self._run("0\n")
        self.assertEqual(gid_of, {0: 0})
        self.assertEqual(groups, [(0, 1, 0, 0)])
        self.assertEqual((rounds, converged), (1, True))

    def test_isolated_node_is_own_group(self):
        gid_of, groups, _, _ = self._run("0 1\n5\n")
        self.assertEqual(gid_of[5], 5)
        self.assertIn((5, 1, 0, 0), groups)

    def test_self_loop_only_node_stays_alone(self):
        gid_of, groups, _, _ = self._run("0 1\n1 2\n0 2\n9 9\n")
        self.assertEqual(gid_of[9], 9)
        self.assertIn((9, 1, 1, 2), groups)  # e=1, d=2 -> ρ=1.000

    def test_group_id_is_min_member(self):
        gid_of, groups, _, _ = self._run("0 1\n1 2\n0 2\n")
        self.assertEqual(set(gid_of.values()), {0})
        self.assertEqual(groups[0][0], 0)

    def test_max_rounds_cap(self):
        # 收敛不了的图不好构造，这里直接验证 max_rounds 参数生效。
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write("0 1\n")
        self.addCleanup(os.unlink, path)
        parsed = groupfind.parse_graph(path)
        _, _, rounds, _ = groupfind.find_groups(*parsed, max_rounds=1)
        self.assertEqual(rounds, 1)


class TestMetrics(unittest.TestCase):
    def test_fmt_rounding(self):
        self.assertEqual(groupfind._fmt(1, 3, 4), "0.3333")
        self.assertEqual(groupfind._fmt(20, 21, 4), "0.9524")
        self.assertEqual(groupfind._fmt(1, 8, 3), "0.125")
        self.assertEqual(groupfind._fmt(0, 0, 4), "0.0000")
        self.assertEqual(groupfind._fmt(5, 0, 3), "0.000")

    def test_zero_edges(self):
        self.assertEqual(groupfind._fmt(0, 0, 4), "0.0000")
        q_num, q_den = groupfind.modularity([(0, 1, 0, 0)], 0)
        self.assertEqual(groupfind._fmt(q_num, q_den, 4), "0.0000")


class TestCli(unittest.TestCase):
    def test_default_output_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = os.path.join(tmp, "t.txt")
            with open(graph, "w") as f:
                f.write("0 1\n")
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    code = groupfind.main(["--graph", graph])
            finally:
                os.chdir(cwd)
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(os.path.join(tmp, "var", "t.members.txt")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "var", "t.groups.txt")))

    def test_exit_code_on_bad_graph(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = groupfind.main(["--graph", "/nonexistent/x.txt"])
        self.assertNotEqual(code, 0)
        self.assertTrue(err.getvalue().strip())

    def test_report_ends_with_newline_and_lf_only(self):
        code, out, tmp, _, _ = run_cli(os.path.join(GRAPHS, "01-two-communities.txt"))
        self.assertTrue(out.endswith("\n"))
        self.assertNotIn("\r", out)
        tmp.cleanup()


class TestPage(unittest.TestCase):
    def test_page_self_contained_and_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = os.path.join(tmp, "p.html")
            code, out, tmp2, _, _ = run_cli(
                os.path.join(GRAPHS, "01-two-communities.txt"), "--page", page
            )
            self.assertEqual(code, 0)
            with open(page, encoding="utf-8") as f:
                doc = f.read()
            self.assertNotIn("<script", doc)
            self.assertNotIn("http://", doc.replace("http://www.w3.org/2000/svg", ""))
            self.assertNotIn("https://", doc)
            self.assertNotIn("<link", doc)
            # 组内边比跨组边粗
            self.assertIn("stroke-width: 2.2", doc)
            self.assertIn("stroke-width: 0.7", doc)
            # 页面数字与报告一致
            _, stats, _ = parse_report(out)
            for line in stats:
                g, size, rho = line.split("\t")
                self.assertIn(f"<td>{size}</td><td>{rho}</td>", doc)
            tmp2.cleanup()

    def test_page_deterministic(self):
        graph = os.path.join(GRAPHS, "05-self-loops.txt")
        with tempfile.TemporaryDirectory() as tmp:
            p1, p2 = os.path.join(tmp, "1.html"), os.path.join(tmp, "2.html")
            for p in (p1, p2):
                code, _, tmp2, _, _ = run_cli(graph, "--page", p)
                self.assertEqual(code, 0)
                tmp2.cleanup()
            with open(p1, "rb") as f1, open(p2, "rb") as f2:
                self.assertEqual(f1.read(), f2.read())


if __name__ == "__main__":
    unittest.main()
