// Talking to the DocuMind API from the demo page.
//
// The access token lives in this module's memory only: never in
// localStorage, sessionStorage or a cookie. Closing or reloading the tab
// logs out; a script injected into the page could not find it stored.

import { readEvents } from "./sse.js";

let token = null;

export function setToken(value) {
  token = value;
}

export function loggedIn() {
  return token !== null;
}

// An API error: status, code (e.g. "rate_limited", "demo_read_only"), the
// server's short message, and seconds to wait when the server said so.
export class ApiError extends Error {
  constructor(status, code, message, retryAfter = null) {
    super(message);
    this.status = status;
    this.code = code;
    this.retryAfter = retryAfter;
  }
}

async function toError(response) {
  let body = {};
  try {
    body = await response.json();
  } catch {
    // Not JSON (a proxy's error page, say): keep the status only.
  }
  const header = Number(response.headers.get("Retry-After"));
  const retryAfter = Number.isFinite(header) && header > 0 ? header : (body.retry_after ?? null);
  const detail = typeof body.detail === "string" ? body.detail : "";
  return new ApiError(response.status, body.code ?? "", detail, retryAfter);
}

async function send(method, path, { json, form, signal } = {}) {
  const headers = {};
  let body;
  if (token) headers.Authorization = `Bearer ${token}`;
  if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  } else if (form !== undefined) {
    body = form; // FormData or URLSearchParams: the browser sets Content-Type
  }
  let response;
  try {
    response = await fetch(path, { method, headers, body, signal });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError(0, "network", "");
  }
  if (!response.ok) throw await toError(response);
  return response;
}

async function request(method, path, options) {
  const response = await send(method, path, options);
  return response.status === 204 ? null : response.json();
}

export async function login(email, password) {
  const form = new URLSearchParams({ username: email, password });
  const { access_token: accessToken } = await request("POST", "/auth/login", { form });
  setToken(accessToken);
}

export const me = () => request("GET", "/auth/me");
export const demoConfig = () => request("GET", "/demo/config.json");
export const collections = () => request("GET", "/collections");
export const createCollection = (name) => request("POST", "/collections", { json: { name } });
export const documents = (collectionId) =>
  request("GET", `/collections/${encodeURIComponent(collectionId)}/documents?limit=100`);
export const conversations = (collectionId) =>
  request("GET", `/conversations?collection_id=${encodeURIComponent(collectionId)}&limit=20`);
export const conversation = (conversationId) =>
  request("GET", `/conversations/${encodeURIComponent(conversationId)}?limit=100`);

export function upload(collectionId, file) {
  const form = new FormData();
  form.append("file", file);
  return request("POST", `/collections/${encodeURIComponent(collectionId)}/documents`, { form });
}

// Streams an answer: calls onEvent(type, data) for meta, token, done and
// error events. Errors before the stream starts (401, 404, 409, 429, 503)
// are thrown as ApiError, like any other request.
export async function ask(collectionId, question, conversationId, onEvent, signal) {
  const json = { question };
  if (conversationId) json.conversation_id = conversationId;
  const response = await send(
    "POST",
    `/collections/${encodeURIComponent(collectionId)}/ask/stream`,
    { json, signal },
  );
  for await (const { event, data } of readEvents(response.body)) {
    let parsed;
    try {
      parsed = JSON.parse(data);
    } catch {
      continue; // not ours: ignore it rather than break the answer
    }
    onEvent(event, parsed);
  }
}

// Free instances sleep when idle and take about a minute to wake. Polls
// /health/live (which costs no database or Redis work) until it answers,
// calling onWaiting() once if the first try is slow or fails. Returns
// false if the server hasn't answered within `giveUpMs`.
export async function waitForServer(onWaiting, giveUpMs = 180_000) {
  const deadline = Date.now() + giveUpMs;
  let warned = false;
  const warn = () => {
    if (!warned) {
      warned = true;
      onWaiting();
    }
  };
  while (Date.now() < deadline) {
    const slow = setTimeout(warn, 1500);
    try {
      const response = await fetch("/health/live", {
        cache: "no-store",
        signal: AbortSignal.timeout(20_000),
      });
      if (response.ok) return true;
    } catch {
      // Not reachable yet (or timed out): keep waiting.
    } finally {
      clearTimeout(slow);
    }
    warn();
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }
  return false;
}
