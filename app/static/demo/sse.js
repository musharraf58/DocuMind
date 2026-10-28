// Server-sent events parser, for streams read with fetch().
//
// EventSource can't send a POST with a JSON body or an Authorization
// header, so the demo reads /ask/stream with fetch and parses the
// text/event-stream format itself, following the HTML standard:
//
//   event: token          <- the event's type
//   data: {"text":"Hi"}   <- one or more data lines, joined with "\n"
//                         <- a blank line ends (dispatches) the event
//   : heartbeat           <- a comment: ignored
//
// Network chunks don't respect line or event boundaries: one chunk can end
// in the middle of a line, or between the "\r" and "\n" of a line break, or
// hold several events. The parser keeps the unfinished part until the next
// chunk arrives.

export class SSEParser {
  constructor() {
    this.buffer = "";
    this.started = false;
    this.reset();
  }

  reset() {
    this.type = "";
    this.data = "";
  }

  // Feed decoded text; returns the events completed by it: [{event, data}].
  push(text) {
    if (!this.started && text.length > 0) {
      this.started = true;
      if (text.charCodeAt(0) === 0xfeff) text = text.slice(1); // byte order mark
    }
    this.buffer += text;
    const events = [];
    let start = 0;
    for (let i = 0; i < this.buffer.length; i++) {
      const char = this.buffer[i];
      if (char !== "\n" && char !== "\r") continue;
      if (char === "\r" && i === this.buffer.length - 1) break; // "\n" may follow in the next chunk
      this.line(this.buffer.slice(start, i), events);
      if (char === "\r" && this.buffer[i + 1] === "\n") i++;
      start = i + 1;
    }
    this.buffer = this.buffer.slice(start);
    return events;
  }

  // The stream ended. A trailing "\r" still ends its line; an event without
  // its closing blank line is incomplete and dropped, as the standard says.
  end() {
    const events = [];
    if (this.buffer.endsWith("\r")) this.line(this.buffer.slice(0, -1), events);
    this.buffer = "";
    this.reset();
    return events;
  }

  line(line, events) {
    if (line === "") {
      if (this.data !== "") {
        events.push({ event: this.type || "message", data: this.data.slice(0, -1) });
      }
      this.reset();
      return;
    }
    if (line.startsWith(":")) return; // a comment, e.g. the server's heartbeat
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") this.type = value;
    else if (field === "data") this.data += value + "\n";
    // "id" and "retry" only matter to EventSource's reconnection: ignored.
  }
}

// Yields {event, data} from a fetch() response body as the bytes arrive.
export async function* readEvents(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder(); // stream: true keeps split UTF-8 characters intact
  const parser = new SSEParser();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      yield* parser.push(decoder.decode(value, { stream: true }));
    }
    yield* parser.push(decoder.decode());
    yield* parser.end();
  } finally {
    reader.releaseLock();
  }
}
