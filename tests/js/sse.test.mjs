// Unit tests for the demo page's SSE parser: node --test tests/js/
import assert from "node:assert/strict";
import test from "node:test";

import { SSEParser, readEvents } from "../../app/static/demo/sse.js";

function parseAll(...chunks) {
  const parser = new SSEParser();
  const events = chunks.flatMap((chunk) => parser.push(chunk));
  return [...events, ...parser.end()];
}

function streamOf(...chunks) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(typeof chunk === "string" ? encoder.encode(chunk) : chunk);
      }
      controller.close();
    },
  });
}

async function collect(iterator) {
  const items = [];
  for await (const item of iterator) items.push(item);
  return items;
}

test("parses the stream /ask/stream sends", () => {
  const events = parseAll(
    'event: meta\ndata: {"conversation_id":"c","message_id":"m"}\n\n' +
      'event: token\ndata: {"text":"Hello "}\n\n' +
      'event: token\ndata: {"text":"world [1]."}\n\n' +
      'event: done\ndata: {"answer":"Hello world [1].","grounded":true,"citations":[]}\n\n',
  );
  assert.deepEqual(
    events.map((e) => e.event),
    ["meta", "token", "token", "done"],
  );
  assert.equal(JSON.parse(events[1].data).text, "Hello ");
  assert.equal(JSON.parse(events[3].data).answer, "Hello world [1].");
});

test("an event split across chunks at every possible position", () => {
  const whole = 'event: token\ndata: {"text":"split me"}\n\n';
  for (let cut = 0; cut <= whole.length; cut++) {
    const events = parseAll(whole.slice(0, cut), whole.slice(cut));
    assert.deepEqual(events, [{ event: "token", data: '{"text":"split me"}' }], `cut at ${cut}`);
  }
});

test("one character per chunk", () => {
  const whole = "event: a\ndata: 1\n\nevent: b\ndata: 2\n\n";
  assert.deepEqual(parseAll(...whole.split("")), [
    { event: "a", data: "1" },
    { event: "b", data: "2" },
  ]);
});

test("CRLF split between \\r and \\n is one line break, not two", () => {
  const events = parseAll("event: token\r", "\ndata: x\r", "\n\r", "\n");
  assert.deepEqual(events, [{ event: "token", data: "x" }]);
});

test("CR alone and LF alone also end lines", () => {
  assert.deepEqual(parseAll("event: a\rdata: 1\r\r"), [{ event: "a", data: "1" }]);
  assert.deepEqual(parseAll("event: b\ndata: 2\n\n"), [{ event: "b", data: "2" }]);
});

test("multi-line data is joined with newlines", () => {
  const events = parseAll("event: token\ndata: first\ndata: second\ndata:\ndata: fourth\n\n");
  assert.deepEqual(events, [{ event: "token", data: "first\nsecond\n\nfourth" }]);
});

test("heartbeats and other comments are ignored", () => {
  const events = parseAll(
    ": heartbeat\n\n",
    'event: token\n: a comment inside an event\ndata: {"text":"a"}\n\n',
    ": heartbeat\n\n",
  );
  assert.deepEqual(events, [{ event: "token", data: '{"text":"a"}' }]);
});

test("an error event carries status and retry_after", () => {
  const [error] = parseAll(
    'event: error\ndata: {"message":"Daily limit reached","status":503,"retry_after":3600}\n\n',
  );
  assert.equal(error.event, "error");
  assert.deepEqual(JSON.parse(error.data), {
    message: "Daily limit reached",
    status: 503,
    retry_after: 3600,
  });
});

test("field parsing: one optional space, no colon, unknown fields, default type", () => {
  const events = parseAll(
    "data:no space\n\n",
    "data:  two spaces\n\n",
    "data\n\n", // a field with no colon: empty value
    "id: 7\nretry: 1000\nfoo: bar\ndata: kept\n\n",
  );
  assert.deepEqual(events, [
    { event: "message", data: "no space" },
    { event: "message", data: " two spaces" },
    { event: "message", data: "" },
    { event: "message", data: "kept" },
  ]);
});

test("an event with no data is not dispatched, and its type doesn't leak", () => {
  assert.deepEqual(parseAll("event: lonely\n\ndata: next\n\n"), [
    { event: "message", data: "next" },
  ]);
});

test("an event cut off by the end of the stream is dropped", () => {
  assert.deepEqual(parseAll('event: done\ndata: {"answer":"partial'), []);
  assert.deepEqual(parseAll("data: x\n"), []);
});

test("a leading byte order mark is skipped", () => {
  assert.deepEqual(parseAll("﻿data: x\n\n"), [{ event: "message", data: "x" }]);
});

test("readEvents decodes bytes, including a UTF-8 character split across chunks", async () => {
  const bytes = new TextEncoder().encode('event: token\ndata: {"text":"café ☕"}\n\n');
  const cut = bytes.indexOf(0xe2) + 1; // inside the 3-byte "☕"
  const events = await collect(readEvents(streamOf(bytes.slice(0, cut), bytes.slice(cut))));
  assert.deepEqual(events, [{ event: "token", data: '{"text":"café ☕"}' }]);
});

test("readEvents handles heartbeats between events", async () => {
  const events = await collect(
    readEvents(streamOf("event: meta\ndata: {}\n\n", ": heartbeat\n\n", "event: done\ndata: {}\n\n")),
  );
  assert.deepEqual(
    events.map((e) => e.event),
    ["meta", "done"],
  );
});
