// DOM helpers. Every piece of text that comes from the server, a document
// or the model goes in through textContent, so the browser shows it as
// text and never interprets it as HTML. Nothing here parses a string as
// markup; tests/test_demo_ui.py fails if an HTML-parsing API appears.

// el("p", {class: "note", text: "Hi", attrs: {"aria-live": "polite"}}, child, ...)
export function el(tag, options = {}, ...children) {
  const node = document.createElement(tag);
  if (options.class) node.className = options.class;
  if (options.text !== undefined) node.textContent = options.text;
  for (const [name, value] of Object.entries(options.attrs ?? {})) {
    node.setAttribute(name, value);
  }
  for (const child of children) node.append(child);
  return node;
}

export function clear(node) {
  node.replaceChildren();
}

export function show(node, visible = true) {
  node.hidden = !visible;
}

export function setText(node, text) {
  node.textContent = text;
}

function wait(seconds) {
  if (seconds === null || seconds === undefined) return "a moment";
  if (seconds < 90) return `${Math.ceil(seconds)} seconds`;
  if (seconds < 5400) return `about ${Math.round(seconds / 60)} minutes`;
  return `about ${Math.round(seconds / 3600)} hours`;
}

// A user-facing sentence for an ApiError (or a stream's error event).
export function describe(error) {
  const status = error.status;
  if (status === 0) return "Can't reach the server. Check your connection and try again.";
  if (status === 401) return "Your session has ended. Please log in again.";
  if (status === 403 && error.code === "demo_read_only") {
    return "The demo account is read-only: it can ask, search and read.";
  }
  if (status === 429) {
    return `Too many requests. Please try again in ${wait(error.retryAfter)}.`;
  }
  if (status === 503 && error.code === "daily_capacity_reached") {
    return `Today's answer capacity for this demo is used up. Please try again in ${wait(error.retryAfter)}.`;
  }
  if (status === 503 && error.message) {
    // The server's own short, fixed messages (never provider text).
    const message = error.message.replace(/\.$/, "");
    return `${message}. Please try again in ${wait(error.retryAfter ?? 60)}.`;
  }
  if (status === 503) {
    return "The service is temporarily unavailable. Please try again in a minute.";
  }
  if (error.message) return error.message;
  return `Something went wrong (error ${status}). Please try again.`;
}

// "[1] handbook.pdf" chips; each expands to document, page, section, snippet.
export function citationList(citations) {
  const list = el("ul", { class: "citations", attrs: { "aria-label": "Sources" } });
  for (const citation of citations) {
    const details = el(
      "div",
      { class: "citation-details" },
      el("p", { class: "citation-document", text: citation.document_name }),
    );
    const where = [];
    if (citation.page_number !== null && citation.page_number !== undefined) {
      where.push(`Page ${citation.page_number}`);
    }
    if (citation.section_title) where.push(citation.section_title);
    if (where.length) details.append(el("p", { class: "citation-where", text: where.join(" · ") }));
    details.append(el("blockquote", { class: "citation-snippet", text: citation.snippet }));
    details.hidden = true;
    details.id = `citation-${crypto.randomUUID()}`;

    const chip = el("button", {
      class: "chip",
      text: `[${citation.number}] ${citation.document_name}`,
      attrs: { type: "button", "aria-expanded": "false", "aria-controls": details.id },
    });
    chip.addEventListener("click", () => {
      const open = chip.getAttribute("aria-expanded") === "true";
      chip.setAttribute("aria-expanded", String(!open));
      details.hidden = open;
    });
    list.append(el("li", {}, chip, details));
  }
  return list;
}
