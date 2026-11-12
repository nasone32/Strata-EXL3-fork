// serve/test_web_history.js - the web app's request construction, on the real app.js (node, no browser, no
// server).  Loads serve/web/app.js in a sandbox with a minimal DOM and localStorage, seeds a saved chat, and
// checks what apiMessages()/assistantMessages() put in the history the server re-renders:
//
//   * a completed answer carries its reasoning_content BESIDE the content (it used to be dropped whenever there
//     was answer text, which moved the re-rendered turn off the engine's tokens - zero conversation-cache reuse);
//   * a turn that used MCP tools sends each round's own thinking with that round's message, and the final
//     answer's thinking with the closing text - not the whole conversation's thinking on one turn;
//   * a turn without thinking, or an old saved chat, keeps working.
//
//   node serve/test_web_history.js   (exit 0 = pass; serve/test_web_history.py runs it when node is there)
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const APP = path.join(__dirname, "web", "app.js");

// --------------------------------------------------------------------- the smallest DOM that loads app.js
function element(tag) {
  const el = {
    tagName: (tag || "div").toUpperCase(), children: [], dataset: {}, style: {}, hidden: false,
    value: "", textContent: "", className: "", open: false, disabled: false, scrollTop: 0,
    scrollHeight: 0, clientHeight: 0, attributes: {},
    classList: {add() {}, remove() {}, toggle() {}, contains: () => false},
    setAttribute(k, v) { this.attributes[k] = String(v); },
    getAttribute(k) { return k in this.attributes ? this.attributes[k] : null; },
    removeAttribute(k) { delete this.attributes[k]; },
    appendChild(c) { this.children.push(c); return c; },
    append(...cs) { this.children.push(...cs); },
    remove() {}, closest() { return null; }, querySelector() { return element("span"); },
    querySelectorAll() { return []; }, addEventListener() {}, focus() {}, select() {}, click() {},
    get firstChild() { return element("span"); },
  };
  return el;
}

const storage = {};                       // what localStorage would still hold between page loads
storage["strata.chat"] = JSON.stringify([      // a saved chat, as the app restores it on load
  {role: "user", text: "2+2?", time: 1},
  {role: "assistant", text: "Four it is.", reasoning: "Two and two.", time: 2},
  {role: "user", text: "thanks", time: 3},
]);
const sandbox = {
  console, Math, JSON, Number, String, Object, Array, Date, RegExp, Promise, Error, Uint8Array, TextDecoder,
  setTimeout, clearTimeout, setInterval, clearInterval, performance, URL, URLSearchParams, AbortController,
  innerHeight: 800, innerWidth: 1280,
  addEventListener() {},
  localStorage: {
    getItem: (k) => (k in storage ? storage[k] : null),
    setItem: (k, v) => { storage[k] = String(v); },
    removeItem: (k) => { delete storage[k]; },
  },
  history: {replaceState() {}},
  location: {hash: "", pathname: "/", search: "", origin: "http://127.0.0.1:8095", href: "http://127.0.0.1:8095/"},
  matchMedia: () => ({addEventListener() {}}),
  navigator: {},
  requestAnimationFrame: () => 0,
  fetch: () => new Promise(() => {}),     // never resolves: the health/metrics polls stay quiet
  document: {
    documentElement: element("html"), body: element("body"),
    getElementById: (id) => (sandbox.document.ids[id] = sandbox.document.ids[id] || element("div")),
    createElement: (tag) => element(tag),
    querySelectorAll: () => [],
    addEventListener() {},
    execCommand: () => true,
    ids: {},
  },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(APP, "utf8"), sandbox, {filename: APP});

// app.js's top-level function declarations land on the context; the seeded "chat" became its `messages`
const api = sandbox.apiMessages, assistant = sandbox.assistantMessages;
if (typeof api !== "function" || typeof assistant !== "function") {
  console.error("app.js did not load (apiMessages/assistantMessages are missing)");
  process.exit(1);
}

let failed = 0;
function eq(name, got, want) {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g === w) { console.log(`ok   ${name}`); return; }
  failed++;
  console.error(`FAIL ${name}\n  got  ${g}\n  want ${w}`);
}

// ----------------------------------------------------------------- a completed answer keeps its reasoning
eq("reasoning rides beside the answer", assistant({role: "assistant", text: "Four it is.", reasoning: "Two and two."}),
   [{role: "assistant", content: "Four it is.", reasoning_content: "Two and two."}]);
eq("no reasoning, no field", assistant({role: "assistant", text: "ok", reasoning: ""}),
   [{role: "assistant", content: "ok"}]);
eq("a thinking-only turn still goes back (#1392)", assistant({role: "assistant", text: "", reasoning: "cut off"}),
   [{role: "assistant", content: "", reasoning_content: "cut off"}]);
eq("an old saved turn without a reasoning key", assistant({role: "assistant", text: "ok"}),
   [{role: "assistant", content: "ok"}]);

// ------------------------------------------------------------------------------ tool rounds keep their own
function tool(id, name, round, at, rat, result, extra) {
  return Object.assign({id, name, round, at, rat, state: "done", ok: true, result, arguments: {a: 1}}, extra);
}
const R1 = "Need the clock.", SEP = "\n\n", R2 = "Done reading it.";
// what a real stream leaves behind: rat = the thinking's length when the call started, rnext = where the next
// round's thinking began (recorded when the "\n\n" separator was inserted between rounds)
const toolTurn = {
  role: "assistant", text: "Checking now.It is 12:00.", reasoning: R1 + SEP + R2,
  tools: [Object.assign(tool("c1", "get_time", 1, "Checking now.".length, R1.length, "12:00"),
                        {rnext: R1.length + SEP.length}),
          tool("c2", "read_file", 2, "Checking now.".length, (R1 + SEP).length + R2.length, "x")],
};
eq("each round carries its own thinking", assistant(toolTurn), [
  {role: "assistant", content: "Checking now.", tool_calls: [
    {id: "c1", type: "function", function: {name: "get_time", arguments: "{\"a\":1}"}}], reasoning_content: R1},
  {role: "tool", tool_call_id: "c1", content: "12:00"},
  {role: "assistant", content: "", tool_calls: [
    {id: "c2", type: "function", function: {name: "read_file", arguments: "{\"a\":1}"}}], reasoning_content: R2},
  {role: "tool", tool_call_id: "c2", content: "x"},
  {role: "assistant", content: "It is 12:00."},
]);

// the same turn as an older saved chat saved it: rat but no rnext - the separator is still not history
const oldToolTurn = JSON.parse(JSON.stringify(toolTurn));
delete oldToolTurn.tools[0].rnext;
eq("an old saved tool chat skips its separator", assistant(oldToolTurn), assistant(toolTurn));

// a round with no thinking of its own (the model called straight from the results)
eq("a round without thinking", assistant({
  role: "assistant", text: "Go.", reasoning: R1,
  tools: [tool("c1", "get_time", 1, "Go.".length, R1.length, "12:00"),
          tool("c2", "read_file", 1, "Go.".length, R1.length, "x")],
}), [
  {role: "assistant", content: "Go.", tool_calls: [
    {id: "c1", type: "function", function: {name: "get_time", arguments: "{\"a\":1}"}},
    {id: "c2", type: "function", function: {name: "read_file", arguments: "{\"a\":1}"}}], reasoning_content: R1},
  {role: "tool", tool_call_id: "c1", content: "12:00"},
  {role: "tool", tool_call_id: "c2", content: "x"},
]);

// skipped tools (the MCP round limit) are not replayed, and do not take the thinking with them
eq("skipped rounds are not history", assistant({
  role: "assistant", text: "Go.", reasoning: R1,
  tools: [tool("c1", "get_time", 1, "Go.".length, R1.length, "12:00"),
          tool("c2", "read_file", 2, "Go.".length, R1.length, null, {state: "skipped", result: null})],
}), [
  {role: "assistant", content: "Go.", tool_calls: [
    {id: "c1", type: "function", function: {name: "get_time", arguments: "{\"a\":1}"}}], reasoning_content: R1},
  {role: "tool", tool_call_id: "c1", content: "12:00"},
]);

// the model's own whitespace is history too: a template that trims (the default one's) trims its own copy,
// one that does not must get the blank lines back - .trim() is only ever the emptiness check
eq("the segments keep the model's whitespace", assistant({
  role: "assistant", text: "Intro.\n\nIt is 12:00.\n\n", reasoning: R1 + SEP + R2,
  tools: [Object.assign(tool("c1", "get_time", 1, "Intro.\n\n".length, R1.length, "12:00"),
                        {rnext: R1.length + SEP.length})],
}), [
  {role: "assistant", content: "Intro.\n\n", tool_calls: [
    {id: "c1", type: "function", function: {name: "get_time", arguments: "{\"a\":1}"}}], reasoning_content: R1},
  {role: "tool", tool_call_id: "c1", content: "12:00"},
  {role: "assistant", content: "It is 12:00.\n\n", reasoning_content: R2},
]);

eq("a whitespace-only rest sends no closing turn", assistant({
  role: "assistant", text: "Go.\n\n", reasoning: R1,
  tools: [tool("c1", "get_time", 1, "Go.".length, R1.length, "12:00")],
}), [
  {role: "assistant", content: "Go.", tool_calls: [
    {id: "c1", type: "function", function: {name: "get_time", arguments: "{\"a\":1}"}}], reasoning_content: R1},
  {role: "tool", tool_call_id: "c1", content: "12:00"},
]);

// an older saved chat can miss the text, or carry marks that overshoot: nothing throws, everything clamps
eq("an old chat without text does not throw", assistant({
  role: "assistant", reasoning: R1,
  tools: [tool("c1", "get_time", 1, undefined, R1.length, "12:00")],
}), [
  {role: "assistant", content: "", tool_calls: [
    {id: "c1", type: "function", function: {name: "get_time", arguments: "{\"a\":1}"}}], reasoning_content: R1},
  {role: "tool", tool_call_id: "c1", content: "12:00"},
]);
eq("a mark that overshoots clamps to the reasoning", assistant({
  role: "assistant", text: "Go.", reasoning: R1,
  tools: [tool("c1", "get_time", 1, 2, R1.length + 50, "12:00")],
})[0].reasoning_content, R1);

// --------------------------------------------------------------- apiMessages over a saved chat, end to end
// (seeded before app.js loads, so its `messages` is the saved chat a browser would restore)
eq("a saved chat round-trips through apiMessages", api().map(
  (m) => ({role: m.role, content: m.content, reasoning_content: m.reasoning_content})), [
  {role: "user", content: "2+2?"},
  {role: "assistant", content: "Four it is.", reasoning_content: "Two and two."},
  {role: "user", content: "thanks"},
]);

process.exit(failed ? 1 : 0);
