// The demo page: wake the server, log in, pick a collection, ask with a
// streamed answer, browse conversations; upload for non-demo accounts.

import * as api from "./api.js";
import { citationList, clear, describe, el, setText, show } from "./ui.js";

const $ = (id) => document.getElementById(id);

const state = {
  user: null,
  collection: null,
  conversationId: null,
  streaming: null, // the AbortController of the answer being streamed
  poll: null, // timer refreshing documents that are still processing
};

// --- Start: wait for the server, then show the login form -------------------

async function start() {
  const awake = await api.waitForServer(() => show($("waking")));
  if (!awake) {
    setText(
      $("waking").querySelector(".waking"),
      "The server didn't wake up. Please reload the page to try again.",
    );
    return;
  }
  show($("waking"), false);
  showLogin();
  try {
    const { hint } = await api.demoConfig();
    if (hint) {
      setText($("hint"), hint);
      show($("hint"));
    }
  } catch {
    // No hint: the form works without it.
  }
}

function showLogin(message = "") {
  show($("workspace"), false);
  show($("account"), false);
  show($("login"));
  setText($("login-error"), message);
  $("password").value = "";
  $("email").focus();
}

$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.submitter;
  button.disabled = true;
  setText($("login-error"), "");
  try {
    await api.login($("email").value.trim(), $("password").value);
    state.user = await api.me();
    await enterWorkspace();
  } catch (error) {
    api.setToken(null);
    setText(
      $("login-error"),
      error.status === 401 ? "Incorrect email or password." : describe(error),
    );
  } finally {
    button.disabled = false;
    $("password").value = "";
  }
});

function logout(message = "") {
  state.streaming?.abort();
  clearTimeout(state.poll);
  api.setToken(null);
  Object.assign(state, { user: null, collection: null, conversationId: null });
  clear($("messages"));
  showLogin(message);
}

$("logout").addEventListener("click", () => logout());

// Any request: a 401 means the token expired or was revoked.
async function guarded(action, errorNode) {
  try {
    return await action();
  } catch (error) {
    if (error.status === 401) logout(describe(error));
    else if (errorNode) setText(errorNode, describe(error));
    throw error;
  }
}

// --- Collections and conversations ------------------------------------------

async function enterWorkspace() {
  show($("login"), false);
  setText($("account-email"), state.user.email);
  show($("account"));
  const owner = !state.user.is_demo;
  for (const node of document.querySelectorAll(".owner-only")) show(node, owner);
  show($("workspace"));
  await loadCollections();
}

async function loadCollections(selectId = null) {
  const items = await guarded(api.collections, $("chat-error"));
  const list = $("collections");
  clear(list);
  if (items.length === 0) {
    list.append(el("li", { class: "muted", text: "No collections yet." }));
    return;
  }
  for (const collection of items) {
    const button = el("button", { class: "link", text: collection.name, attrs: { type: "button" } });
    button.addEventListener("click", () => selectCollection(collection));
    list.append(el("li", {}, button));
  }
  const chosen = items.find((c) => c.id === selectId) ?? items[0];
  await selectCollection(chosen);
}

async function selectCollection(collection) {
  state.collection = collection;
  for (const button of $("collections").querySelectorAll("button")) {
    const current = button.textContent === collection.name;
    if (current) button.setAttribute("aria-current", "true");
    else button.removeAttribute("aria-current");
  }
  setText($("chat-context"), `Collection: ${collection.name}`);
  newConversation(false);
  await loadConversations();
  if (!state.user.is_demo) await loadDocuments();
}

async function loadConversations() {
  const items = await guarded(() => api.conversations(state.collection.id), $("chat-error"));
  const list = $("conversations");
  clear(list);
  if (items.length === 0) {
    list.append(el("li", { class: "muted", text: "None yet: ask a question." }));
  }
  for (const conversation of items) {
    const button = el(
      "button",
      { class: "link", attrs: { type: "button" } },
      el("span", { text: conversation.title }),
      el("span", {
        class: "muted small",
        text: new Date(conversation.updated_at).toLocaleString(),
      }),
    );
    if (conversation.id === state.conversationId) button.setAttribute("aria-current", "true");
    button.addEventListener("click", () => openConversation(conversation.id));
    list.append(el("li", {}, button));
  }
}

async function openConversation(conversationId) {
  if (state.streaming) return;
  const detail = await guarded(() => api.conversation(conversationId), $("chat-error"));
  state.conversationId = detail.id;
  setText($("chat-error"), "");
  clear($("messages"));
  for (const message of detail.messages) {
    if (message.role === "user") addQuestion(message.content);
    else {
      const answer = addAnswer();
      finishAnswer(answer, message.content, message.citations, message.grounded);
      if (message.status === "cancelled") note(answer, "Stopped before the end.");
      if (message.status === "failed") note(answer, "This answer failed.");
    }
  }
  await loadConversations();
  $("question").focus();
}

function newConversation(focus = true) {
  if (state.streaming) return;
  state.conversationId = null;
  clear($("messages"));
  setText($("chat-error"), "");
  for (const button of $("conversations").querySelectorAll("button")) {
    button.removeAttribute("aria-current");
  }
  if (focus) $("question").focus();
}

$("new-conversation").addEventListener("click", () => newConversation());

// --- Asking, with the answer streamed ---------------------------------------

function addQuestion(text) {
  $("messages").append(el("li", { class: "message question" }, el("p", { text })));
}

function addAnswer() {
  const text = el("p", { class: "answer-text" });
  const item = el("li", { class: "message answer" }, text);
  $("messages").append(item);
  return item;
}

function note(answer, text, kind = "note") {
  answer.append(el("p", { class: kind, text }));
}

// The final answer replaces the streamed text: only the server's checked
// version has its invalid citation markers removed.
function finishAnswer(answer, text, citations, grounded) {
  setText(answer.querySelector(".answer-text"), text);
  if (citations?.length) answer.append(citationList(citations));
  if (grounded === false) note(answer, "Not grounded in the documents: no source was cited.");
}

$("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question || state.streaming || !state.collection) return;
  setText($("chat-error"), "");
  addQuestion(question);
  $("question").value = "";
  const answer = addAnswer();
  const text = answer.querySelector(".answer-text");
  const messages = $("messages");
  const controller = new AbortController();
  state.streaming = controller;
  messages.setAttribute("aria-busy", "true");
  $("ask").disabled = true;
  show($("stop"));
  setText(text, "…");
  let started = false;
  try {
    await api.ask(
      state.collection.id,
      question,
      state.conversationId,
      (type, data) => {
        if (type === "meta") {
          state.conversationId = data.conversation_id;
        } else if (type === "token") {
          if (!started) setText(text, "");
          started = true;
          text.append(data.text); // a text node: never parsed as HTML
        } else if (type === "done") {
          finishAnswer(answer, data.answer, data.citations, data.grounded);
        } else if (type === "error") {
          note(answer, describe({ status: data.status, message: data.message, retryAfter: data.retry_after }), "error");
        }
      },
      controller.signal,
    );
  } catch (error) {
    if (error.name === "AbortError") {
      note(answer, "Stopped. The text so far is saved in the conversation.");
    } else {
      answer.remove();
      if (error.status === 401) logout(describe(error));
      else setText($("chat-error"), describe(error));
    }
  } finally {
    state.streaming = null;
    messages.setAttribute("aria-busy", "false");
    $("ask").disabled = false;
    show($("stop"), false);
    if (api.loggedIn()) await loadConversations().catch(() => {});
  }
});

$("stop").addEventListener("click", () => state.streaming?.abort());

$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    $("ask-form").requestSubmit();
  }
});

// --- Owners only: collections and uploads -----------------------------------

$("new-collection").addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = $("collection-name").value.trim();
  if (!name) return;
  try {
    const created = await guarded(() => api.createCollection(name), $("chat-error"));
    $("collection-name").value = "";
    await loadCollections(created.id);
  } catch {
    // Shown by guarded().
  }
});

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("file").files[0];
  if (!file || !state.collection) return;
  setText($("upload-error"), "");
  const button = event.submitter;
  button.disabled = true;
  try {
    await guarded(() => api.upload(state.collection.id, file), $("upload-error"));
    $("file").value = "";
    await loadDocuments();
  } catch {
    // Shown by guarded().
  } finally {
    button.disabled = false;
  }
});

async function loadDocuments() {
  clearTimeout(state.poll);
  const collection = state.collection;
  let items;
  try {
    items = await guarded(() => api.documents(collection.id), $("upload-error"));
  } catch {
    return;
  }
  if (state.collection !== collection) return; // switched meanwhile
  const list = $("documents");
  clear(list);
  if (items.length === 0) list.append(el("li", { class: "muted", text: "No documents yet." }));
  for (const document_ of items) {
    const item = el(
      "li",
      {},
      el("span", { text: document_.original_filename }),
      el("span", { class: `status status-${document_.status}`, text: document_.status }),
    );
    if (document_.error_message) {
      item.append(el("p", { class: "muted small", text: document_.error_message }));
    }
    list.append(item);
  }
  // Refresh while something is still being processed.
  if (items.some((d) => d.status === "pending" || d.status === "processing")) {
    state.poll = setTimeout(loadDocuments, 3000);
  }
}

start();
