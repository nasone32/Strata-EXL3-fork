"""serve/test_think_roundtrip.py - the thinking round trip, on CPU (mock engine, no GPU, no pack).

Two contracts, both engine-anchored (the prefix the engine can reuse is an exact id prefix, not a mock's say-so):

  * the server returns reasoning_content as the reasoning text WITHOUT the structural <think> / </think> - both
    APIs, streamed and whole - whatever the family: a template that pre-writes the opening marker (Qwen's) gets a
    parser no-op, one that leaves the model to open its own thinking (MiMo's) has that first tag consumed;
  * a history built from what the server returned - the faithful client that sends reasoning_content back beside
    the answer, the bundled web app's new payload, and an older stored copy that still carries the wrapper this
    server used to return - renders back to the very tokens the engine committed (prompt + generated, the
    end-of-turn token excluded as the engine's serial arm does), so the next turn reuses the whole prefix instead
    of reading the conversation again.

    python -m unittest serve.test_think_roundtrip -v
"""
from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import (ChatTemplate, LITERAL_TAGS, OutputParser, THINK_OPEN,  # noqa: E402
                            mark_think_literals, openai_to_messages)
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096
SELF_OPENING = ROOT / "serve/fixtures/self_opening_think_template.jinja"      # MiMo-style: no pre-written <think>
PRE_OPENED = ROOT / "serve/chat_template.jinja"                              # Qwen-style: the prompt opens it


class ThinkTokenizer(ByteTokenizer):
    """The byte stand-in with <think> / </think> as single special ids, as the packs carry them (the real MiMo
    tokenizer: 151667 / 151668)."""

    SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>"]


class ScriptedEngine(MockEngine):
    """One script per request (the last repeats); records every prompt it was given and every id it yielded (the
    server closes the generator at the stop token, so the record is kept in a finally)."""

    def __init__(self, tok, scripts, max_context=CTX):
        super().__init__(tok, list(scripts), max_context=max_context)
        self.prompts, self.generated = [], []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        out = []
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                out.append(t)
                yield t
        finally:
            self.generated.append(out)


def engine_criterion(live: list[int], ids: list[int]) -> bool:
    """mimo_engine.cu's all-or-nothing rule: the whole live vector must be a proper prefix of the new prompt."""
    return len(live) < len(ids) and ids[:len(live)] == live


class RoundTripCase:
    """Shared helpers for the per-template test cases below."""

    def setUp(self):
        self.tok = ThinkTokenizer()
        self.engine = ScriptedEngine(self.tok, ["ok"])          # replaced by each test's scripts
        self.svc = Service(self.engine, self.tok, ChatTemplate(self.TEMPLATE))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.shutdown)
        self.addCleanup(self.httpd.server_close)

    # ------------------------------------------------------------ what the engine holds
    def live(self, turn: int) -> list[int]:
        """The engine's committed tokens after `turn`: its prompt plus what it generated, without the end-of-turn
        token (the serial emission arm stops before live.push_back; a template's own <|im_end|> follows there)."""
        eos = self.tok.encode("<|im_end|>", parse_special=True)[0]
        gen = list(self.engine.generated[turn])
        if gen and gen[-1] == eos:
            gen = gen[:-1]
        return self.engine.prompts[turn] + gen

    def assertTurnReuses(self, turn: int, note: str):
        """Request `turn`'s prompt must start with all of the previous turn's live tokens (and continue)."""
        live, ids = self.live(turn - 1), self.engine.prompts[turn]
        common = next((i for i, (a, b) in enumerate(zip(live, ids)) if a != b), min(len(live), len(ids)))
        self.assertTrue(engine_criterion(live, ids),
                        f"{note}: the re-rendered history is not the engine's prefix "
                        f"(live {len(live)}, common {common}, prompt {len(ids)})")

    # ------------------------------------------------------------ the APIs
    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def stream(self, body):
        body = {**body, "stream": True}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        deltas, usage = [], None
        with urllib.request.urlopen(req, timeout=30) as r:
            buf = b""
            for chunk in iter(lambda: r.read(1), b""):
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    j = json.loads(data)
                    if j.get("usage"):
                        usage = j["usage"]
                    d = j["choices"][0]["delta"]
                    if d.get("reasoning_content"):
                        deltas.append(("reasoning", d["reasoning_content"]))
                    if d.get("content"):
                        deltas.append(("content", d["content"]))
        return deltas, usage

    def chat(self, messages, **extra):
        return self.post("/v1/chat/completions", {"model": "m", "messages": messages, **extra})["choices"][0]

    def set_scripts(self, *texts):
        """What the model writes on each following request (the last repeats)."""
        eos = self.tok.encode("<|im_end|>", parse_special=True)
        self.engine.scripts = [self.tok.encode(t, parse_special=True) + eos for t in texts]
        self.engine.script, self.engine.turns = self.engine.scripts[0], 0

    def messages_api(self, body):
        req = urllib.request.Request(self.base + "/v1/messages", data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())


class SelfOpeningTemplate(RoundTripCase, unittest.TestCase):
    """The MiMo-style family: the generation prompt does not pre-write the <think>, the model opens its own
    thinking with it, and the history renders the markers around reasoning_content itself."""

    TEMPLATE = SELF_OPENING
    THOUGHT = "The user asks for two plus two. That is four."
    ANSWER = "Four it is."
    SCRIPT = f"<think>{THOUGHT}</think>{ANSWER}"          # what the model writes, marker and all
    ASK = [{"role": "user", "content": "2+2?"}]

    def setUp(self):
        super().setUp()
        self.set_scripts(self.SCRIPT)

    def test_turn1_reasoning_has_no_markers_whole_and_streamed(self):
        c = self.chat(self.ASK)
        self.assertEqual((c["message"]["reasoning_content"], c["message"]["content"], c["finish_reason"]),
                         (self.THOUGHT, self.ANSWER, "stop"))
        deltas, usage = self.stream({"model": "m", "messages": self.ASK})
        self.assertEqual("".join(t for k, t in deltas if k == "reasoning"), self.THOUGHT)
        self.assertEqual("".join(t for k, t in deltas if k == "content"), self.ANSWER)
        self.assertEqual(usage["completion_tokens"], len(self.tok.encode(self.SCRIPT, parse_special=True)) + 1)

    def test_a_faithful_client_gets_the_whole_prefix_back(self):
        """The new web-app payload: reasoning_content beside the answer.  The engine's committed tokens are the
        turn's prompt plus its generation; request 2's prompt must start with all of them - including a turn
        that ends with trailing newlines before the end-of-turn token (they are part of the answer the model
        wrote, and the template renders them back)."""
        for answer in (self.ANSWER, self.ANSWER + "\n\n"):
            with self.subTest(answer=answer[:20]):
                self.set_scripts(f"<think>{self.THOUGHT}</think>{answer}")
                first = self.chat(self.ASK)["message"]
                self.assertEqual(first["content"], answer)
                self.set_scripts(self.SCRIPT)
                second = self.chat([*self.ASK, {"role": "assistant", "content": first["content"],
                                                "reasoning_content": first["reasoning_content"]},
                                    {"role": "user", "content": "and 2+3?"}])
                self.assertEqual(second["finish_reason"], "stop")
                self.assertTurnReuses(1, "reasoning beside the answer")

    def test_a_three_turn_conversation_reuses_every_turn(self):
        """The audit's failing shape: by the second follow-up the history carries TWO completed assistant turns,
        each re-rendered from its own reasoning_content and content."""
        history = list(self.ASK)
        for turn in range(3):
            c = self.chat(history)                     # sends request `turn`
            if turn:
                self.assertTurnReuses(turn, f"turn {turn + 1} of three")
            history += [{"role": "assistant", "content": c["message"]["content"],
                         "reasoning_content": c["message"]["reasoning_content"]},
                        {"role": "user", "content": f"follow-up {turn}?"}]
            self.set_scripts(self.SCRIPT)

    def test_an_old_stored_wrapper_roundtrips_too(self):
        """reasoning_content as this server used to return it (the opening marker inside the field; another
        server's copy may close it too) is unwrapped at render: same prefix, no doubled <think>."""
        first = self.chat(self.ASK)["message"]
        for wrapper in (THINK_OPEN + first["reasoning_content"],
                        THINK_OPEN + first["reasoning_content"] + "</think>",
                        first["reasoning_content"] + "</think>"):
            with self.subTest(wrapper=wrapper[:len(THINK_OPEN)] + "..." + wrapper[-len("</think>"):]):
                self.set_scripts(self.SCRIPT)
                self.chat([*self.ASK, {"role": "assistant", "content": first["content"],
                                       "reasoning_content": wrapper},
                           {"role": "user", "content": "and 2+3?"}])
                self.assertTurnReuses(1, "the stored wrapper")

    def test_dropping_the_reasoning_still_answers(self):
        """A client that omits the reasoning (the old web app) still gets its answer - the history just no longer
        matches the engine's tokens, so that conversation reads again (the honest fallback, not a wrong reuse)."""
        first = self.chat(self.ASK)["message"]
        self.set_scripts(self.SCRIPT)
        c = self.chat([*self.ASK, {"role": "assistant", "content": first["content"]},
                       {"role": "user", "content": "and 2+3?"}])
        self.assertEqual(c["message"]["content"], self.ANSWER)
        self.assertFalse(engine_criterion(self.live(0), self.engine.prompts[1]))

    def test_thinking_off_roundtrips(self):
        off = {"chat_template_kwargs": {"enable_thinking": False}}
        self.set_scripts("The answer is 4.")
        c = self.chat(self.ASK, **off)
        self.assertEqual((c["message"]["content"], c["finish_reason"]), ("The answer is 4.", "stop"))
        self.set_scripts("Five.")
        self.chat([*self.ASK, {"role": "assistant", "content": "The answer is 4."},
                   {"role": "user", "content": "and 2+3?"}], **off)
        self.assertTurnReuses(1, "thinking off")

    def test_reasoning_content_and_tool_calls_roundtrip(self):
        """A turn that thinks, then calls a tool: the reply carries reasoning_content and tool_calls, and the
        client's turn-2 history (the answer's reasoning, the call, the tool's result) re-renders to the engine's
        tokens."""
        tools = [{"type": "function", "function": {"name": "get_time", "description": "the time in a timezone",
                                                   "parameters": {"type": "object",
                                                                  "properties": {"tz": {"type": "string"}}}}}]
        call = "<tool_call><function=get_time><parameter=tz>Europe/Berlin</parameter></function></tool_call>"
        self.set_scripts(f"<think>Need the clock.</think>{call}", "<think>Done.</think>It is 12:00.")
        c = self.chat(self.ASK, tools=tools)
        self.assertEqual(c["finish_reason"], "tool_calls")
        self.assertEqual(c["message"]["reasoning_content"], "Need the clock.")
        self.assertEqual(c["message"]["tool_calls"][0]["function"]["name"], "get_time")
        turn = [{"role": "assistant", "content": c["message"]["content"],
                 "reasoning_content": c["message"]["reasoning_content"],
                 "tool_calls": c["message"]["tool_calls"]},
                {"role": "tool", "tool_call_id": c["message"]["tool_calls"][0]["id"], "content": "12:00"}]
        self.chat([*self.ASK, *turn, {"role": "user", "content": "thanks"}], tools=tools)
        self.assertTurnReuses(1, "reasoning + tool call + result")

    def test_anthropic_thinking_blocks_roundtrip(self):
        m = self.messages_api({"model": "m", "max_tokens": 200,
                               "messages": [{"role": "user", "content": "2+2?"}]})
        kinds = {b["type"]: b.get("thinking", b.get("text")) for b in m["content"]}
        self.assertEqual((kinds.get("thinking"), kinds.get("text")), (self.THOUGHT, self.ANSWER))
        self.set_scripts(self.SCRIPT)
        self.messages_api({"model": "m", "max_tokens": 200, "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": self.THOUGHT},
                                              {"type": "text", "text": self.ANSWER}]},
            {"role": "user", "content": "and 2+3?"}]})
        self.assertTurnReuses(1, "anthropic thinking blocks")

    def test_quoted_tags_stay_text(self):
        """A <think> the user quotes, and one the model wrote later inside its reasoning, are text - the round-trip
        fix consumes only the stream's structural opening marker."""
        user = 'what does "<think>" mean?'
        ids, _, _ = self.svc.prepare(openai_to_messages({"messages": [{"role": "user", "content": user}]})[0],
                                     None, {})
        self.assertIn(user, self.tok.decode(ids))               # the quoted tag is all there, as text
        open_id = self.tok.encode(THINK_OPEN, parse_special=True)[0]
        self.assertEqual(ids.count(open_id), 0)                 # none as the special: this template pre-writes none
        # mid-reasoning markers are reasoning text, on both sides of the API: a <think> anywhere, and a
        # #537-quoted </think> (quote or backtick before it, no newline after)
        p = OutputParser(thinking=True)
        evs = []
        for piece in ("Start. ", "the tag <think> opens", ", and it wrote `</think>` in its text. ",
                      "Done", "</think>", "Answer"):
            evs += p.feed(piece)
        evs += p.finish()
        self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"),
                         "Start. the tag <think> opens, and it wrote `</think>` in its text. Done")
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "Answer")


class PreOpenedTemplate(RoundTripCase, unittest.TestCase):
    """The Qwen-style family: the prompt itself ends with the opening <think>, so the model's output starts with
    the reasoning - the parser's marker rule must be a no-op here and the round trip keep working."""

    TEMPLATE = PRE_OPENED
    THOUGHT = "Two and two make four.\n"
    ANSWER = "The answer is 4."
    SCRIPT = f"{THOUGHT}</think>\n\n{ANSWER}"                   # no leading marker: the prompt pre-wrote it
    ASK = [{"role": "user", "content": "2+2?"}]

    def setUp(self):
        super().setUp()
        self.set_scripts(self.SCRIPT)

    def test_turn1_reasoning_has_no_markers(self):
        c = self.chat(self.ASK)
        self.assertEqual((c["message"]["reasoning_content"], c["message"]["content"]),
                         (self.THOUGHT, self.ANSWER))

    def test_the_history_roundtrips(self):
        first = self.chat(self.ASK)["message"]
        self.set_scripts(self.SCRIPT)
        self.chat([*self.ASK, {"role": "assistant", "content": first["content"],
                               "reasoning_content": first["reasoning_content"]},
                   {"role": "user", "content": "and 2+3?"}])
        self.assertTurnReuses(1, "pre-opened template")


class ParserOpeningMarker(unittest.TestCase):
    """The structural opening marker, at every streaming boundary."""

    REASON = "The user wants an explanation."
    RAW = THINK_OPEN + REASON + "</think>" + "Because."

    def parser(self, chunks, thinking=True, tools=None):
        p = OutputParser(thinking=thinking, tools=tools)
        out = []
        for c in chunks:
            out += p.feed(c)
        out += p.finish()
        return p, out

    def events(self, chunks, thinking=True, tools=None):
        return [(e.kind, e.text) for e in self.parser(chunks, thinking, tools)[1]]

    def joined(self, chunks, thinking=True, tools=None):
        p, evs = self.parser(chunks, thinking, tools)
        return ({k: "".join(e.text for e in evs if e.kind == k) for k in ("reasoning", "content")},
                [(e.kind, e.call.name) for e in evs if e.call],
                p.self_opened)

    def test_the_marker_is_consumed_at_every_split(self):
        for at in range(1, len(THINK_OPEN) + 1):               # the marker split at every boundary...
            for step in (1, 2, 3):                             # ...and the rest fed in varying pieces
                chunks = [self.RAW[:at]]
                rest = self.RAW[at:]
                chunks += [rest[i:i + step] for i in range(0, len(rest), step)]
                with self.subTest(at=at, step=step):
                    text, calls, self_opened = self.joined(chunks)
                    self.assertEqual((text, calls, self_opened),
                                     ({"reasoning": self.REASON, "content": "Because."}, [], True))

    def test_marker_only_then_close(self):
        self.assertEqual(self.joined([THINK_OPEN, "</think>", "Because."])[0],
                         {"reasoning": "", "content": "Because."})
        self.assertEqual(self.joined([THINK_OPEN])[0], {"reasoning": "", "content": ""})  # an empty thinking

    def test_a_partial_marker_at_the_end_is_reasoning_text(self):
        self.assertEqual(self.events(["<thi"]), [("reasoning", "<thi")])
        self.assertEqual(self.events(["<", "thi"]), [("reasoning", "<thi")])

    def test_a_quoted_or_later_marker_is_text(self):
        quoted = '`<think>` opens the reasoning'
        self.assertEqual(self.joined([quoted])[0], {"reasoning": quoted, "content": ""})
        later = f"first {THINK_OPEN} then more"
        self.assertEqual(self.joined([later])[0], {"reasoning": later, "content": ""})

    def test_thinking_off_is_untouched(self):
        self.assertEqual(self.joined(["<think>hi</think>ok"], thinking=False)[0],
                         {"reasoning": "", "content": "<think>hi</think>ok"})

    def test_plain_reasoning_is_unchanged(self):
        text, calls, self_opened = self.joined([self.REASON + "</think>Because."])
        self.assertEqual((text, calls, self_opened),
                         ({"reasoning": self.REASON, "content": "Because."}, [], False))


class SelfOpenedNewlines(unittest.TestCase):
    """The model's own newlines around the answer.  A stream that opened its own thinking (its template renders
    the answer straight after </think>, nothing before a call) keeps them as the answer's text - the round trip
    gives the template back exactly what the model wrote.  A template that pre-wrote the marker re-writes that
    blank line itself, and for those streams the parser drops it exactly as it always did."""

    CALL = "<tool_call><function=f><parameter=q>x</parameter></function></tool_call>"
    TOOLS = [{"name": "f"}]

    def parser(self, chunks, thinking=True, tools=None):
        p = OutputParser(thinking=thinking, tools=tools)
        out = []
        for c in chunks:
            out += p.feed(c)
        out += p.finish()
        return p, out

    def chunks_of(self, raw, step):
        return [raw[i:i + step] for i in range(0, len(raw), step)]

    def test_the_answers_own_leading_newlines_are_kept_at_every_split(self):
        for nl in ("\n\n", "\n", "\n\n\n"):
            raw = f"<think>Reason.</think>{nl}Answer."
            for step in (1, 2, 5, len(raw)):
                with self.subTest(nl=repr(nl), step=step):
                    p, evs = self.parser(self.chunks_of(raw, step))
                    joined = {k: "".join(e.text for e in evs if e.kind == k) for k in ("reasoning", "content")}
                    self.assertEqual(joined, {"reasoning": "Reason.", "content": f"{nl}Answer."})
                    self.assertTrue(p.self_opened)
                    # the events reconstruct the raw stream: marker + reasoning + close + the answer as written
                    self.assertEqual(THINK_OPEN + joined["reasoning"] + "</think>" + joined["content"], raw)

    def test_a_preopened_stream_still_drops_them(self):
        for nl in ("\n\n", "\n"):
            p, evs = self.parser([f"Reason.</think>{nl}Answer."])
            joined = {k: "".join(e.text for e in evs if e.kind == k) for k in ("reasoning", "content")}
            self.assertEqual(joined, {"reasoning": "Reason.", "content": "Answer."})
            self.assertFalse(p.self_opened)

    def test_the_newlines_before_a_call_are_kept_at_every_split(self):
        raw = f"<think>Reason.</think>\n\n{self.CALL}After."
        for step in (1, 3, 7, len(raw)):
            with self.subTest(step=step):
                p, evs = self.parser(self.chunks_of(raw, step), tools=self.TOOLS)
                joined = {k: "".join(e.text for e in evs if e.kind == k) for k in ("reasoning", "content")}
                calls = [e.call for e in evs if e.kind == "tool_call"]
                self.assertEqual(joined, {"reasoning": "Reason.", "content": "\n\nAfter."})
                self.assertEqual([(c.name, c.arguments) for c in calls], [("f", {"q": "x"})])

    def test_a_single_newline_before_a_call(self):
        raw = f"<think>Reason.</think>\n{self.CALL}After."
        p, evs = self.parser([raw], tools=self.TOOLS)
        joined = {k: "".join(e.text for e in evs if e.kind == k) for k in ("reasoning", "content")}
        self.assertEqual(joined, {"reasoning": "Reason.", "content": "\nAfter."})

    def test_a_preopened_stream_still_drops_them_before_a_call(self):
        raw = f"Reason.</think>\n\npre\n{self.CALL}After."
        p, evs = self.parser([raw], tools=self.TOOLS)
        joined = {k: "".join(e.text for e in evs if e.kind == k) for k in ("reasoning", "content")}
        self.assertEqual(joined, {"reasoning": "Reason.", "content": "preAfter."})

    def test_an_empty_answer_after_the_close(self):
        p, evs = self.parser(["<think>Reason.</think>"])
        joined = {k: "".join(e.text for e in evs if e.kind == k) for k in ("reasoning", "content")}
        self.assertEqual(joined, {"reasoning": "Reason.", "content": ""})
        self.assertTrue(p.self_opened)

    def test_newlines_only_then_the_end_of_the_stream(self):
        # the answer never got past its blank lines: they are the answer the model wrote
        p, evs = self.parser(["<think>R.</think>", "\n", "\n"])
        joined = {k: "".join(e.text for e in evs if e.kind == k) for k in ("reasoning", "content")}
        self.assertEqual(joined, {"reasoning": "R.", "content": "\n\n"})


class UnwrapStoredReasoning(unittest.TestCase):
    """A stored wrapper comes off the reasoning_content field only - never the text inside it, never another
    field."""

    def marked(self, messages):
        out, _, _ = mark_think_literals(messages, None, LITERAL_TAGS)
        return out

    def test_the_wrapper_comes_off(self):
        for rc, want in (("<think>plan", "plan"), ("<think>plan</think>", "plan"), ("plan</think>", "plan"),
                         ("<think>", ""), ("<think></think>", "")):
            with self.subTest(rc=rc):
                out = self.marked([{"role": "assistant", "content": "ok", "reasoning_content": rc}])
                self.assertEqual(out[0]["reasoning_content"], want)

    def test_tags_inside_the_reasoning_stay(self):
        rc = 'the tag <think> opens and </think> ends'
        out = self.marked([{"role": "assistant", "content": "ok", "reasoning_content": THINK_OPEN + rc}])
        # the wrapper is gone; the tags inside are kept as the literals they are (marked, rendered as text)
        self.assertEqual(out[0]["reasoning_content"],
                         'the tag \U000F0E01 opens and \U000F0E02 ends')
        self.assertNotIn(THINK_OPEN, out[0]["reasoning_content"])
        self.assertNotIn("</think>", out[0]["reasoning_content"])

    def test_no_wrapper_no_change(self):
        msgs = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "a", "reasoning_content": "plain reasoning"},
                {"role": "user", "content": "b"}]
        self.assertEqual(self.marked(msgs), msgs)

    def test_user_text_and_content_are_not_unwrapped(self):
        msgs = [{"role": "user", "content": "<think>I wrote this</think>"},
                {"role": "assistant", "content": "<think>\nplan\n</think>\n\nok"}]      # inline block, as #537 keeps it
        out = self.marked(msgs)
        self.assertEqual(out[0]["content"], "\U000F0E01I wrote this\U000F0E02")   # marked as a literal, not removed
        self.assertEqual(out[1]["content"], msgs[1]["content"])                    # its own block keeps its markers


def mimo_pack():
    """The real MiMo tokenizer pack, as serve/server.py loads it (its own template and tokenizer), for the
    exact-id checks; None when it is not on this machine (STRATA_MIMO_TOKENIZER names it)."""
    import os
    cands = [os.environ.get("STRATA_MIMO_TOKENIZER", ""),
             "/home/stephan/mimo-tokenizer-pack/tokenizer"]
    cands += [str(p) for p in ROOT.glob("packs/*/tokenizer")]
    for c in cands:
        if c and (Path(c) / "chat_template.jinja").exists() and (Path(c) / "vocab.json").exists():
            return Path(c)
    return None


@unittest.skipIf(mimo_pack() is None, "no MiMo tokenizer pack here (set STRATA_MIMO_TOKENIZER to run this)")
class RealMimoTemplate(RoundTripCase, unittest.TestCase):
    """The production path against the REAL MiMo pack - its chat_template.jinja and its tokenizer (the very
    ids the engine sees) - not a fixture: the turn the server returns, re-rendered from a client's history,
    must be the exact token prefix the engine committed, newlines and all."""

    TEMPLATE = mimo_pack() / "chat_template.jinja"
    THOUGHT = "The user asks for two plus two. That is four.\n"
    ANSWER = "\n\nThe answer is 4."                            # the model's own blank line after </think>
    SCRIPT = f"<think>{THOUGHT}</think>{ANSWER}"
    ASK = [{"role": "user", "content": "2+2?"}]

    def setUp(self):
        pack = mimo_pack()
        sys.path.insert(0, str(ROOT / "tools"))
        import strata_tokenizer as ST
        vocab = json.loads((pack / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        merges = (pack / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((pack / "token_type.json").read_text(encoding="utf-8"))
        tcfg = json.loads((pack / "tokenizer.json").read_text(encoding="utf-8"))
        self.tok = ST.Tokenizer(tokens, merges, types, pre_pattern=tcfg.get("pre_pattern"),
                                normalizer=tcfg.get("normalizer"))
        self.engine = ScriptedEngine(self.tok, ["ok"], max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(self.TEMPLATE))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.shutdown)
        self.addCleanup(self.httpd.server_close)
        self.set_scripts(self.SCRIPT)

    def test_the_whole_turn_comes_back_as_the_exact_token_prefix(self):
        first = self.chat(self.ASK)["message"]
        self.assertEqual((first["reasoning_content"], first["content"]), (self.THOUGHT, self.ANSWER))
        self.set_scripts(self.SCRIPT)
        self.chat([*self.ASK, {"role": "assistant", "content": first["content"],
                               "reasoning_content": first["reasoning_content"]},
                   {"role": "user", "content": "and 2+3?"}])
        self.assertTurnReuses(1, "the real pack, thinking on, blank line kept")
        # and the engine's prompt is not just any prefix: the shared part is the model's own ids verbatim
        live, ids = self.live(0), self.engine.prompts[1]
        self.assertEqual(ids[:len(live)], live)

    def test_a_tool_call_with_the_models_own_whitespace(self):
        """the model thinks, writes its blank line, then calls: the reply's reasoning, the whitespace-only
        content and the call re-render to the engine's tokens."""
        tools = [{"type": "function", "function": {"name": "get_time", "description": "the time in a timezone",
                                                   "parameters": {"type": "object",
                                                                  "properties": {"tz": {"type": "string"}}}}}]
        call = "<tool_call><function=get_time><parameter=tz>Europe/Berlin</parameter></function></tool_call>"
        self.set_scripts(f"<think>Need the clock.\n</think>\n\n{call}",
                         "<think>Done.\n</think>It is 12:00.")
        c = self.chat(self.ASK, tools=tools)
        self.assertEqual(c["finish_reason"], "tool_calls")
        self.assertEqual((c["message"]["reasoning_content"], c["message"]["content"]),
                         ("Need the clock.\n", "\n\n"))
        turn = [{"role": "assistant", "content": c["message"]["content"],
                 "reasoning_content": c["message"]["reasoning_content"],
                 "tool_calls": c["message"]["tool_calls"]},
                {"role": "tool", "tool_call_id": c["message"]["tool_calls"][0]["id"], "content": "12:00"}]
        self.chat([*self.ASK, *turn, {"role": "user", "content": "thanks"}], tools=tools)
        self.assertTurnReuses(1, "the real pack, reasoning + blank line + call")

    def test_thinking_off_roundtrips(self):
        off = {"chat_template_kwargs": {"enable_thinking": False}}
        self.set_scripts("The answer is 4.")
        c = self.chat(self.ASK, **off)
        self.assertEqual(c["message"]["content"], "The answer is 4.")
        self.set_scripts("Five.")
        self.chat([*self.ASK, {"role": "assistant", "content": "The answer is 4."},
                   {"role": "user", "content": "and 2+3?"}], **off)
        self.assertTurnReuses(1, "the real pack, thinking off")


class McpLoopReasoning(unittest.TestCase):
    """run_with_mcp re-renders the round it just ran: the reasoning must go back as the model wrote it, or the
    next round's prompt no longer starts with the tokens the engine holds."""

    def test_round2_reuses_round1s_tokens(self):
        from serve.server import run_with_mcp

        tok = ThinkTokenizer()
        # the shapes that used to be lost on the way back: the reasoning opens with a newline (models that
        # write "<think>\n"), and the round's text carries the model's own blank lines around it - a stripped
        # or trimmed copy would move the re-rendered turn off the engine's prefix at the very next token
        call = "<tool_call><function=get_time><parameter=tz>Europe/Berlin</parameter></function></tool_call>"
        engine = ScriptedEngine(tok, [f"<think>\nNeed the clock.</think>\n\nLet me check.\n{call}",
                                      "<think>Done.</think>It is 12:00."])
        svc = Service(engine, tok, ChatTemplate(SELF_OPENING))
        hub = SimpleNamespace(settings={"max_rounds": 2},
                              routes=lambda: {"get_time": (None, "get_time")},
                              call=lambda name, args, cancel: {"ok": True, "text": "12:00", "chars": 5,
                                                               "truncated": False, "ms": 1})
        messages, tools, kw = openai_to_messages({"messages": [{"role": "user", "content": "the time?"}]})
        ids, thinking, _ = svc.prepare(messages, tools, kw)
        cancel = threading.Event()
        rounds = [x for k, x in run_with_mcp(svc, hub, messages, tools, kw, ids, thinking, 200, 400, {}, cancel,
                                             ["get_time"]) if k == "done"]
        self.assertEqual(rounds[-1]["finish"], "stop")
        eos = tok.encode("<|im_end|>", parse_special=True)[0]
        gen = engine.generated[0]
        live = engine.prompts[0] + (gen[:-1] if gen and gen[-1] == eos else gen)
        with self.subTest(turn=1):                       # round 2 continues on round 1's committed tokens
            self.assertTrue(engine_criterion(live, engine.prompts[1]),
                            "round 2 re-reads the conversation (the engine's prefix is not kept)")


if __name__ == "__main__":
    unittest.main()
