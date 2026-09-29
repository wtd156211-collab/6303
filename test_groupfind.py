"""groupfind 的 unittest 测试。

运行：python3 -m unittest
"""

import os
import subprocess
import sys
import tempfile
import unittest

import groupfind

ROOT = os.path.dirname(os.path.abspath(__file__))
GRAPHS = os.path.join(ROOT, "samples", "graphs")
EXPECTED = os.path.join(ROOT, "samples", "expected")
SCRIPT = os.path.join(ROOT, "groupfind.py")


def analyze_text(text):
    """把图内容写进临时文件，跑引擎，返回 (graph, result)。"""
    with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write(text)
        path = fh.name
    try:
        graph = groupfind.parse_graph(path)
        return graph, groupfind.analyze(graph)
    finally:
        os.unlink(path)


class ParseTest(unittest.TestCase):
    def parse(self, text):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write(text)
            path = fh.name
        try:
            return groupfind.parse_graph(path)
        finally:
            os.unlink(path)

    def test_comments_blanks_crlf(self):
        graph = self.parse("# 注释\r\n\r\n0 1\r\n1 2\n")
        self.assertEqual(graph.nodes, [0, 1, 2])
        self.assertEqual(graph.m, 2)
        self.assertEqual(graph.adj[0], {1: 1})

    def test_node_declaration_and_isolated(self):
        graph = self.parse("0 1\n5\n")
        self.assertEqual(graph.nodes, [0, 1, 5])
        self.assertEqual(graph.deg[5], 0)
        self.assertEqual(graph.m, 1)

    def test_duplicate_edges_count_by_multiplicity(self):
        graph = self.parse("0 1\n1 0\n0 1\n")
        self.assertEqual(graph.m, 3)
        self.assertEqual(graph.adj[0][1], 3)
        self.assertEqual(graph.deg[0], 3)

    def test_self_loop(self):
        graph = self.parse("2 2\n")
        self.assertEqual(graph.m, 1)
        self.assertEqual(graph.loops[2], 1)
        self.assertEqual(graph.deg[2], 2)
        self.assertEqual(graph.adj[2], {})

    def test_missing_file(self):
        with self.assertRaises(groupfind.GraphError):
            groupfind.parse_graph("/nonexistent/graph.txt")

    def test_invalid_lines(self):
        for bad in ("0 1 2\n", "-1 2\n", "a b\n", "1.5 2\n", "0 x\n"):
            with self.assertRaises(groupfind.GraphError, msg=bad):
                self.parse(bad)


class EngineTest(unittest.TestCase):
    def test_two_cliques_with_bridge(self):
        _graph, result = analyze_text(
            "0 1\n0 2\n1 2\n3 4\n3 5\n4 5\n2 3\n")
        self.assertEqual(dict(result.membership),
                         {0: 0, 1: 0, 2: 0, 3: 3, 4: 3, 5: 3})
        self.assertTrue(result.converged)

    def test_isolated_node_is_own_group(self):
        _graph, result = analyze_text("0 1\n2\n")
        self.assertEqual(dict(result.membership), {0: 0, 1: 0, 2: 2})
        self.assertEqual(result.groups[-1], (2, 1, "0.000"))

    def test_self_loop_only_node(self):
        _graph, result = analyze_text("0 0\n")
        self.assertEqual(result.membership, [(0, 0)])
        self.assertEqual(result.groups, [(0, 1, "1.000")])

    def test_single_node_graph(self):
        graph, result = analyze_text("0\n")
        self.assertEqual((result.rounds, result.converged), (1, True))
        self.assertEqual((result.q_str, result.r_str), ("0.0000", "0.0000"))
        report = groupfind.build_report("g.txt", graph, result)
        self.assertIn("分组\t1\n", report)

    def test_empty_graph(self):
        graph, result = analyze_text("# 只有注释\n\n")
        self.assertEqual(graph.nodes, [])
        self.assertEqual((result.rounds, result.converged), (1, True))
        self.assertEqual(result.groups, [])

    def test_multiplicity_affects_grouping(self):
        # 0-1 三重边、2-3 单重边、1-2 单重桥边：0/1 与 2/3 应各自成组。
        _graph, result = analyze_text(
            "0 1\n0 1\n0 1\n2 3\n1 2\n")
        self.assertEqual(dict(result.membership),
                         {0: 0, 1: 0, 2: 2, 3: 2})

    def test_deterministic_repeated_runs(self):
        text = "0 1\n0 2\n1 2\n3 4\n3 5\n4 5\n2 3\n6\n"
        first = analyze_text(text)[1]
        for _ in range(3):
            again = analyze_text(text)[1]
            self.assertEqual(first.membership, again.membership)
            self.assertEqual(first.groups, again.groups)
            self.assertEqual(first.rounds, again.rounds)

    def test_rounding_half_up(self):
        self.assertEqual(groupfind._fmt_ratio(1, 8, 3), "0.125")
        self.assertEqual(groupfind._fmt_ratio(5, 40, 3), "0.125")
        self.assertEqual(groupfind._fmt_ratio(3, 40, 3), "0.075")
        self.assertEqual(groupfind._fmt_ratio(1, 16, 4), "0.0625")
        self.assertEqual(groupfind._fmt_ratio(0, 7, 4), "0.0000")
        self.assertEqual(groupfind._fmt_ratio(1, 3, 4), "0.3333")


class SampleTest(unittest.TestCase):
    """逐图对照 samples/expected 逐字节校验（samples 只读）。"""

    def test_all_samples(self):
        for name in sorted(os.listdir(GRAPHS)):
            with self.subTest(graph=name):
                graph = groupfind.parse_graph(os.path.join(GRAPHS, name))
                result = groupfind.analyze(graph)
                report = groupfind.build_report(name, graph, result)
                with open(os.path.join(EXPECTED, name),
                          encoding="utf-8") as fh:
                    self.assertEqual(report, fh.read())

    def test_modularity_thresholds(self):
        thresholds = {"01": 0.30, "02": 0.30, "03": 0.30, "04": 0.30,
                      "05": 0.30, "06": 0.00, "07": 0.00, "08": 0.30}
        for name in sorted(os.listdir(GRAPHS)):
            with self.subTest(graph=name):
                graph = groupfind.parse_graph(os.path.join(GRAPHS, name))
                result = groupfind.analyze(graph)
                self.assertGreaterEqual(
                    float(result.q_str), thresholds[name[:2]])


class CliTest(unittest.TestCase):
    def run_cli(self, args, cwd, env=None):
        full_env = dict(os.environ)
        if env:
            full_env.update(env)
        return subprocess.run(
            [sys.executable, SCRIPT] + args, cwd=cwd, env=full_env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_report_and_default_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = os.path.join(GRAPHS, "01-two-communities.txt")
            proc = self.run_cli(["--graph", graph], cwd=tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(os.path.join(
                    EXPECTED, "01-two-communities.txt"), "rb") as fh:
                self.assertEqual(proc.stdout, fh.read())
            with open(os.path.join(
                    tmp, "var", "01-two-communities.members.txt"),
                    encoding="utf-8") as fh:
                members = fh.read()
            with open(os.path.join(
                    tmp, "var", "01-two-communities.groups.txt"),
                    encoding="utf-8") as fh:
                groups = fh.read()
            report = proc.stdout.decode("utf-8")
            block = report.split("分组统计\n")[1]
            group_lines, member_lines = block.split("节点归属\n")
            self.assertEqual(groups, group_lines)
            self.assertEqual(members, member_lines)

    def test_same_result_from_other_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = os.path.join(GRAPHS, "02-nested-communities.txt")
            proc = self.run_cli(["--graph", graph], cwd=tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(os.path.join(
                    EXPECTED, "02-nested-communities.txt"), "rb") as fh:
                self.assertEqual(proc.stdout, fh.read())

    def test_hash_seed_independence(self):
        outputs = []
        with tempfile.TemporaryDirectory() as tmp:
            for seed in ("0", "1", "12345"):
                proc = self.run_cli(
                    ["--graph", os.path.join(GRAPHS, "05-self-loops.txt")],
                    cwd=tmp, env={"PYTHONHASHSEED": seed})
                self.assertEqual(proc.returncode, 0, proc.stderr)
                with open(os.path.join(
                        tmp, "var", "05-self-loops.members.txt"), "rb") as fh:
                    outputs.append((proc.stdout, fh.read()))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])

    def test_bad_graph_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.txt")
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write("0 1 2\n")
            proc = self.run_cli(["--graph", bad], cwd=tmp)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(proc.stdout, b"")
            self.assertEqual(len(proc.stderr.decode("utf-8").splitlines()), 1)

    def test_missing_graph_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self.run_cli(["--graph", "nope.txt"], cwd=tmp)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(len(proc.stderr.decode("utf-8").splitlines()), 1)


class PageTest(unittest.TestCase):
    def setUp(self):
        graph = groupfind.parse_graph(
            os.path.join(GRAPHS, "01-two-communities.txt"))
        self.result = groupfind.analyze(graph)
        self.page = groupfind.render_page(
            self.result, graph, "01-two-communities.txt")

    def test_self_contained(self):
        self.assertNotIn("http://", self.page)
        self.assertNotIn("https://", self.page)
        self.assertNotIn("<script", self.page)
        self.assertNotIn("<link", self.page)
        self.assertIn("<svg", self.page)
        self.assertIn("<style>", self.page)

    def test_numbers_come_from_engine(self):
        for gid, size, rho in self.result.groups:
            self.assertIn(
                "<td>%d</td><td>%d</td><td>%s</td>" % (gid, size, rho),
                self.page)
        self.assertIn("模块度 <b>%s</b>" % self.result.q_str, self.page)

    def test_edge_weights(self):
        self.assertIn('stroke-width="2"', self.page)    # 组内边加粗
        self.assertIn('stroke-width="0.6"', self.page)  # 跨组边细

    def test_deterministic(self):
        graph = groupfind.parse_graph(
            os.path.join(GRAPHS, "01-two-communities.txt"))
        again = groupfind.render_page(
            groupfind.analyze(graph), graph, "01-two-communities.txt")
        self.assertEqual(self.page, again)


if __name__ == "__main__":
    unittest.main()
