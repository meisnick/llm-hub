"""Unit tests for the proxy's pure logic: think-tag splitting, effort mapping, GBNF, tps guards.

    python3 -m unittest discover -s tests
"""
import os, sys, unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hub"))
import llm_proxy as P  # noqa: E402
import llm_router as R  # noqa: E402


def run_splitter(chunks, cls):
    s, out = cls(), []
    for c in chunks:
        out += s.feed(c)
    out += s.flush()
    merged = {"reasoning": "", "content": ""}
    for ch, txt in out:
        merged[ch] += txt
    return merged


class ThinkSplitterTest(unittest.TestCase):
    def test_basic(self):
        m = run_splitter(["<think>\nplan it", "</think>", "The answer."], P.ThinkSplitter)
        self.assertEqual(m["reasoning"], "plan it")
        self.assertEqual(m["content"], "The answer.")

    def test_tag_split_across_chunks(self):
        m = run_splitter(["<thi", "nk>abc</th", "ink>xyz"], P.ThinkSplitter)
        self.assertEqual(m["reasoning"], "abc")
        self.assertEqual(m["content"], "xyz")

    def test_primed_think_without_opener(self):
        # templates that pre-open the think block stream only the closer
        m = run_splitter(["reasoning here</ifm|think>answer"], P.ThinkSplitter)
        self.assertEqual(m["reasoning"], "reasoning here")
        self.assertEqual(m["content"], "answer")


class ContentThinkRouterTest(unittest.TestCase):
    def test_block_mid_content(self):
        m = run_splitter(["\n\n", "<think>secret", " plan</think>", "\nHello"], P.ContentThinkRouter)
        self.assertEqual(m["reasoning"], "secret plan")
        self.assertEqual(m["content"], "Hello")

    def test_empty_template_block_vanishes(self):
        m = run_splitter(["<think>\n\n</think>\n\nHi there"], P.ContentThinkRouter)
        self.assertEqual(m["reasoning"], "")
        self.assertEqual(m["content"], "Hi there")

    def test_plain_content_untouched(self):
        m = run_splitter(["a < b and ", "c > d"], P.ContentThinkRouter)
        self.assertEqual(m["content"], "a < b and c > d")


class EffortTest(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(P.map_effort("medium", {"low", "medium", "high"}), "medium")

    def test_ties_go_up(self):
        self.assertEqual(P.map_effort("high", {"low", "medium", "xhigh"}), "xhigh")

    def test_clamp_down(self):
        self.assertEqual(P.map_effort("xhigh", {"low", "medium", "high"}), "high")

    def test_unknown_template(self):
        self.assertEqual(P.map_effort("high", None), "high")


class GbnfTest(unittest.TestCase):
    def test_simple_object(self):
        g = P.schema_to_gbnf({"type": "object", "properties": {"title": {"type": "string"}}})
        self.assertIn("root ::=", g)
        self.assertIn('\\"title\\"', g)

    def test_nested_falls_back(self):
        self.assertIsNone(P.schema_to_gbnf({"type": "object", "properties": {"a": {"type": "array"}}}))


class TpsTest(unittest.TestCase):
    def test_guards(self):
        self.assertIsNone(P.tps(2, 1.0))          # too few tokens
        self.assertIsNone(P.tps(100, 0.05))       # window too short
        self.assertIsNone(P.tps(5000, 1.0))       # physically implausible
        self.assertEqual(P.tps(50, 1.0), 50.0)

    def test_split_small_phase_uses_decode(self):
        self.assertEqual(P.split_tps(5, 100.0, 42.0), 42.0)
        self.assertIsNone(P.split_tps(0, 100.0, 42.0))


class RouterTokenCountTest(unittest.TestCase):
    def test_max_of_cumulative_sources(self):
        tail = (b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
                b'data: {"usage":{"completion_tokens":120}}\n\n'
                b'data: {"timings":{"predicted_n":131}}\n\n'
                b'data: [DONE]\n\n')
        self.assertEqual(R.extract_gen_tokens(tail), 131)

    def test_non_streaming_body(self):
        body = b'{"choices":[{"message":{"content":"4"}}],"usage":{"completion_tokens":9}}'
        self.assertEqual(R.extract_gen_tokens(body), 9)

    def test_session_fingerprint_stable_as_chat_grows(self):
        a = [{"role": "system", "content": "s"}, {"role": "user", "content": "hello"}]
        b = a + [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "more"}]
        self.assertEqual(R._first_user_fp(a), R._first_user_fp(b))
        self.assertNotEqual(R._first_user_fp(a), R._first_user_fp([{"role": "user", "content": "other"}]))


if __name__ == "__main__":
    unittest.main()
