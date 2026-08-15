(() => {
  "use strict";

  const API = Object.freeze({
    overview: "/dashboard/api/overview?window=24h",
    users: "/dashboard/api/users?search=",
    sessions: "/dashboard/api/sessions?user_id=",
    session: "/dashboard/api/session?user_id=",
    logs: "/dashboard/api/logs?level=",
    traces: "/dashboard/api/traces?status=",
    trace: "/dashboard/api/trace?trace_id=",
  });

  const state = {
    view: "overview",
    selectedUser: "",
    selectedSession: "",
    selectedTrace: "",
    users: [],
    sessions: [],
    traces: [],
    lastUpdated: null,
  };

  const byId = (id) => document.getElementById(id);

  const el = (tag, className, content) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (content !== undefined && content !== null) element.textContent = String(content);
    return element;
  };

  const setText = (id, content) => {
    const target = byId(id);
    if (target) target.textContent = content === undefined || content === null ? "—" : String(content);
  };

  const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);

  const rootOf = (payload) => {
    if (isObject(payload) && isObject(payload.data)) return payload.data;
    return payload;
  };

  const first = (object, keys, fallback = undefined) => {
    if (!isObject(object)) return fallback;
    for (const key of keys) {
      const value = object[key];
      if (value !== undefined && value !== null && value !== "") return value;
    }
    return fallback;
  };

  const listFrom = (payload, keys = []) => {
    if (Array.isArray(payload)) return payload;
    if (!isObject(payload)) return [];
    for (const key of keys) {
      if (Array.isArray(payload[key])) return payload[key];
    }
    if (isObject(payload.data)) return listFrom(payload.data, keys);
    return [];
  };

  const formatCount = (value) => {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return new Intl.NumberFormat().format(number);
  };

  const formatPercent = (value) => {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    const normalized = number <= 1 ? number * 100 : number;
    return `${normalized.toFixed(normalized >= 10 ? 0 : 1)}%`;
  };

  const formatMs = (value) => {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return `${number >= 100 ? Math.round(number) : number.toFixed(1)} ms`;
  };

  const formatDate = (value) => {
    if (value === undefined || value === null || value === "") return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
  };

  const normalizeStatus = (value) => String(value ?? "unknown").trim().toLowerCase();

  const badgeFor = (value, kind = "status") => {
    const status = normalizeStatus(value);
    const map = {
      ok: ["badge-success", "OK"],
      success: ["badge-success", "Success"],
      healthy: ["badge-success", "Healthy"],
      running: ["badge-info", "Running"],
      active: ["badge-info", "Active"],
      info: ["badge-info", "Info"],
      warning: ["badge-warning", "Warning"],
      degraded: ["badge-warning", "Degraded"],
      error: ["badge-error", "Error"],
      failed: ["badge-error", "Failed"],
      unavailable: ["badge-error", "Unavailable"],
      debug: ["badge-ghost", "Debug"],
      unknown: ["badge-ghost", kind === "source" ? "Unknown" : "Unknown"],
    };
    const [className, label] = map[status] || ["badge-ghost", String(value ?? "Unknown")];
    return el("span", `badge badge-sm ${className}`, label);
  };

  const showEmpty = (container, title, description) => {
    container.replaceChildren();
    const box = el("div", "rounded-xl border border-dashed border-base-300 bg-base-200/40 p-6 text-center");
    box.append(
      el("p", "font-semibold", title),
      el("p", "mt-1 text-sm leading-5 text-base-content/60", description),
    );
    container.append(box);
  };

  const showInlineError = (container, description) => {
    container.replaceChildren();
    const box = el("div", "alert alert-error border-error/20 bg-error/10 text-sm", description);
    container.append(box);
  };

  const setConnection = (message, stateName = "ok") => {
    setText("connection-state", message);
    const status = document.querySelector("#connection-state")?.previousElementSibling;
    if (!status) return;
    status.className = `status ${stateName === "error" ? "status-error" : stateName === "pending" ? "status-warning" : "status-success"}`;
  };

  const setBusy = (busy) => {
    const button = byId("refresh-button");
    const loading = byId("loading-state");
    if (button) {
      button.disabled = busy;
      button.setAttribute("aria-busy", String(busy));
    }
    if (loading && state.lastUpdated === null) loading.hidden = !busy;
  };

  const requestJson = async (url) => {
    const response = await fetch(url, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`Request returned HTTP ${response.status}`);
    return response.json();
  };

  const clearGlobalError = () => {
    const error = byId("global-error");
    if (error) {
      error.hidden = true;
      error.replaceChildren();
    }
  };

  const showGlobalError = (error) => {
    const target = byId("global-error");
    if (!target) return;
    target.replaceChildren();
    const box = el("div", "alert alert-error border-error/20 bg-error/10 shadow-sm");
    box.append(
      el("span", "text-lg", "!"),
      el("div", "min-w-0", "Unable to read the selected local records. Check the server logs, then try Refresh."),
    );
    const detail = error instanceof Error ? error.message : "Unknown dashboard error";
    const detailNode = el("p", "mt-1 break-words text-xs text-base-content/65", detail);
    box.lastElementChild.append(detailNode);
    target.append(box);
    target.hidden = false;
  };

  const renderOverview = (payload) => {
    const root = rootOf(payload);
    const metrics = isObject(root?.metrics) ? root.metrics : isObject(root?.summary) ? root.summary : root;
    const requestCount = first(metrics, ["requests", "request_count", "invocations", "invocation_count", "events", "event_count"]);
    const errorCount = first(metrics, ["errors", "error_count"]);
    const errorRate = first(metrics, ["error_rate", "errors_percent", "failure_rate"],
      requestCount && Number(requestCount) > 0 && errorCount !== undefined
        ? Number(errorCount) / Number(requestCount)
        : undefined,
    );
    const activeUsers = first(metrics, ["active_users", "users", "user_count", "unique_users"]);
    const sessions = first(metrics, ["sessions", "session_count", "total_sessions"]);
    const traces = first(metrics, ["traces", "trace_count", "total_traces"]);

    setText("metric-users", formatCount(activeUsers));
    setText("metric-sessions", formatCount(sessions));
    setText("metric-traces", formatCount(traces));
    setText("metric-errors", formatPercent(errorRate));
    setText("metric-users-desc", requestCount === undefined ? "Across the selected window" : `${formatCount(requestCount)} requests`);
    setText("metric-sessions-desc", first(metrics, ["reset_sessions", "session_versions"]) === undefined ? "All retained versions" : `${formatCount(first(metrics, ["reset_sessions", "session_versions"]))} new versions`);
    setText("metric-traces-desc", first(metrics, ["completed_traces"]) === undefined ? "Completed and in-flight" : `${formatCount(first(metrics, ["completed_traces"]))} completed`);
    setText("metric-errors-desc", errorCount === undefined ? "Based on available logs" : `${formatCount(errorCount)} records marked error`);

    setText("metric-tokens", formatCount(first(metrics, ["tokens", "tokens_processed", "total_tokens", "token_count"])));
    const latency = isObject(first(metrics, ["latency_ms", "latency"])) ? first(metrics, ["latency_ms", "latency"]) : {};
    setText("metric-latency", formatMs(first(metrics, ["avg_latency_ms", "average_latency_ms", "mean_latency_ms"], first(latency, ["average", "avg"]))));
    setText("metric-p95", formatMs(first(metrics, ["p95_latency_ms", "latency_p95_ms", "p95_ms"], first(latency, ["p95", "percentile95"]))));
    const note = first(root, ["note", "performance_note"]);
    if (note !== undefined) setText("performance-note", note);
    const window = first(root, ["window", "time_window"]);
    if (window !== undefined) setText("activity-window", String(window).replace(/^24h$/, "Last 24 hours"));

    renderActivity(listFrom(root, ["activity", "activity_buckets", "timeline", "series", "buckets"]));
    renderSources(root);
  };

  const renderActivity = (items) => {
    const target = byId("activity-chart");
    if (!target) return;
    if (!items.length) {
      showEmpty(target, "No activity in this window", "New records will appear here as the bot receives traffic.");
      return;
    }
    target.replaceChildren();
    const rows = items.slice(0, 14).map((item) => ({
      label: first(item, ["label", "bucket", "hour", "timestamp", "time"], "—"),
      count: Number(first(item, ["count", "requests", "events", "messages", "invocations", "sessions", "value"], 0)) || 0,
    }));
    const max = Math.max(...rows.map((row) => row.count), 1);
    const chart = el("div", "space-y-3");
    rows.forEach((row) => {
      const line = el("div", "grid grid-cols-[5rem_minmax(0,1fr)_3.5rem] items-center gap-3 text-xs");
      line.append(el("span", "truncate text-base-content/60", row.label));
      const progress = el("progress", "progress progress-info h-2.5 w-full", "");
      progress.max = max;
      progress.value = row.count;
      progress.setAttribute("aria-label", `${row.label}: ${formatCount(row.count)} events`);
      line.append(progress, el("span", "text-right font-semibold tabular-nums", formatCount(row.count)));
      chart.append(line);
    });
    target.append(chart);
  };

  const sourceEntries = (root) => {
    const sources = first(root, ["sources", "data_sources", "health", "source_health"]);
    if (Array.isArray(sources)) return sources;
    if (isObject(sources)) return Object.entries(sources).map(([name, value]) => (isObject(value) ? { name, ...value } : { name, status: value }));
    const warnings = Array.isArray(root?.warnings) ? root.warnings.map((warning) => String(warning).toLowerCase()) : [];
    const sourceDefinitions = [
      ["Sessions", "session"],
      ["Telemetry log", "telemetry"],
      ["Trace log", "trace"],
    ];
    return sourceDefinitions.map(([name, keyword]) => {
      const warning = warnings.find((item) => item.includes(keyword));
      return {
        name,
        status: warning ? "degraded" : "healthy",
        detail: warning || "Available to the dashboard",
      };
    });
  };

  const renderSources = (root) => {
    const target = byId("source-health");
    if (!target) return;
    const items = sourceEntries(root);
    if (!items.length) {
      showEmpty(target, "Health data is not reported", "The API did not return source-level health details.");
      return;
    }
    target.replaceChildren();
    items.forEach((item) => {
      const name = first(item, ["name", "source", "id"], "Local source");
      const status = first(item, ["status", "state", "health"], "unknown");
      const card = el("div", "rounded-xl border border-base-300 bg-base-200/50 p-4");
      const head = el("div", "flex items-center justify-between gap-3");
      head.append(el("p", "truncate font-semibold", name), badgeFor(status, "source"));
      card.append(head);
      const detail = first(item, ["detail", "message", "path", "description"]);
      if (detail !== undefined) card.append(el("p", "mt-2 break-words text-xs leading-5 text-base-content/60", detail));
      target.append(card);
    });
  };

  const renderUsers = () => {
    const target = byId("users-list");
    const summary = byId("users-summary");
    if (!target) return;
    if (summary) summary.textContent = state.users.length ? `${formatCount(state.users.length)} user records` : "No users found";
    if (!state.users.length) {
      showEmpty(target, "No matching users", "Try a different stored user or chat ID.");
      return;
    }
    target.replaceChildren();
    state.users.forEach((item) => {
      const userId = String(first(item, ["user_id", "userId", "chat_id", "chatId", "id"], "—"));
      const button = el("button", "btn btn-ghost h-auto min-h-0 w-full justify-between gap-3 rounded-xl px-3 py-3 text-left normal-case hover:bg-base-200");
      button.type = "button";
      button.setAttribute("aria-label", `Inspect user ${userId}`);
      const copy = el("span", "min-w-0");
      copy.append(el("span", "block break-all font-mono text-xs font-semibold", userId));
      const lastSeen = first(item, ["last_seen", "lastSeen", "updated_at", "updatedAt", "latest_update_at"]);
      copy.append(el("span", "mt-1 block text-xs text-base-content/55", lastSeen === undefined ? "Stored user ID" : `Seen ${formatDate(lastSeen)}`));
      const count = first(item, ["sessions", "session_count", "message_count", "messages"]);
      button.append(copy, el("span", "badge badge-ghost shrink-0", count === undefined ? "View" : formatCount(count)));
      button.addEventListener("click", () => selectUser(userId));
      target.append(button);
    });
  };

  const renderSessions = () => {
    const target = byId("sessions-list");
    const summary = byId("sessions-summary");
    if (!target) return;
    if (summary) summary.textContent = state.sessions.length ? `${formatCount(state.sessions.length)} retained` : "No sessions";
    if (!state.sessions.length) {
      showEmpty(target, "No sessions for this user", "A /reset creates a new version; it does not remove older history.");
      return;
    }
    target.replaceChildren();
    const wrapper = el("div", "overflow-x-auto rounded-xl border border-base-300");
    const table = el("table", "table table-sm");
    const head = el("thead");
    const headerRow = el("tr");
    ["Session", "Version", "Messages", "Last activity"].forEach((label) => headerRow.append(el("th", "whitespace-nowrap", label)));
    head.append(headerRow);
    const body = el("tbody");
    state.sessions.forEach((item) => {
      const sessionId = String(first(item, ["session_id", "sessionId", "id"], "—"));
      const version = first(item, ["version", "session_version", "revision"]);
      const row = el("tr", "hover");
      const sessionCell = el("th", "max-w-56");
      const button = el("button", "link link-hover break-all text-left font-mono text-xs font-medium");
      button.type = "button";
      button.textContent = sessionId;
      button.setAttribute("aria-label", `Inspect session ${sessionId}`);
      button.addEventListener("click", () => selectSession(sessionId));
      sessionCell.append(button);
      const versionCell = el("td", "whitespace-nowrap");
      if (version !== undefined) versionCell.append(el("span", "badge badge-ghost badge-sm", `v${version}`));
      const reset = first(item, ["reset", "is_reset", "was_reset", "reset_command"]);
      const generation = Number(first(item, ["generation", "reset_generation"], 0));
      if (reset === true || String(reset).toLowerCase() === "true" || Number(version) > 1 || generation > 0) versionCell.append(el("span", "badge badge-info badge-sm ml-1", "reset"));
      row.append(sessionCell, versionCell);
      row.append(el("td", "tabular-nums", formatCount(first(item, ["message_count", "messages", "turns"]))));
      row.append(el("td", "whitespace-nowrap text-xs text-base-content/60", formatDate(first(item, ["updated_at", "updatedAt", "last_seen", "lastSeen"]))));
      body.append(row);
    });
    table.append(head, body);
    wrapper.append(table);
    target.append(wrapper);
  };

  const attachmentValueIsPresent = (value) => {
    if (value === undefined || value === null || value === false || value === "") return false;
    if (Array.isArray(value)) return value.length > 0;
    return true;
  };

  const attachmentLabel = (value) => {
    if (Array.isArray(value)) return value.length === 1 ? "Attachment" : `${value.length} attachments`;
    if (isObject(value)) {
      const name = first(value, ["name", "filename", "file_name", "mime_type", "mimeType"]);
      return name === undefined ? "Attachment" : `Attachment · ${name}`;
    }
    if (typeof value === "string" && value !== "true") return value;
    return "Attachment";
  };

  const metadataRow = (label, detail, status, timestamp) => {
    const row = el("div", "flex flex-wrap items-center justify-center gap-2 rounded-lg border border-base-300 bg-base-100/70 px-3 py-2 text-xs");
    row.setAttribute("role", "note");
    row.append(el("span", "badge badge-ghost badge-sm", label));
    row.append(el("span", "break-all font-mono text-base-content/70", detail));
    if (status !== undefined) row.append(badgeFor(status));
    if (timestamp !== undefined) row.append(el("span", "text-base-content/45", formatDate(timestamp)));
    return row;
  };

  const renderChat = (payload) => {
    const target = byId("chat-view");
    if (!target) return;
    const root = rootOf(payload);
    const messages = listFrom(root, ["messages", "turns", "events", "conversation"]);
    if (!messages.length) {
      showEmpty(target, "No messages recorded", "This session has no chat events available on disk.");
      return;
    }
    target.replaceChildren();
    messages.forEach((item) => {
      const type = normalizeStatus(first(item, ["type", "kind"], ""));
      const role = normalizeStatus(first(item, ["role", "author", "sender"], "assistant"));
      const timestamp = first(item, ["timestamp", "created_at", "createdAt", "time"]);
      const toolName = first(item, ["name", "tool_name", "toolName"]);
      const toolStatus = first(item, ["status", "state", "outcome"]);
      const isTool = type === "tool" || role === "tool";
      if (isTool) {
        target.append(metadataRow("Tool", toolName ?? "Unnamed tool", toolStatus, timestamp));
        return;
      }

      let attachment = first(item, ["attachment", "attachments", "attachment_count", "has_attachment", "inline_data", "inlineData", "file", "image"]);
      if (!attachmentValueIsPresent(attachment) && ["attachment", "file", "image", "photo", "document"].includes(type)) attachment = first(item, ["name", "filename", "file_name", "mime_type", "mimeType"], type);
      if (attachmentValueIsPresent(attachment)) target.append(metadataRow("Attachment", attachmentLabel(attachment), "info", timestamp));

      const isUser = role === "user" || role === "human" || role === "customer";
      const rawContent = first(item, ["content", "text", "message", "body"]);
      const content = isObject(rawContent) || Array.isArray(rawContent) ? JSON.stringify(rawContent) : rawContent === undefined || rawContent === null ? "" : String(rawContent);
      if (!content) return;
      const chat = el("div", `chat ${isUser ? "chat-end" : "chat-start"}`);
      const header = el("div", "chat-header text-xs text-base-content/55", isUser ? "User" : "Assistant");
      const bubble = el("div", `chat-bubble ${isUser ? "chat-bubble-neutral" : "chat-bubble-info"} max-w-[min(42rem,88vw)] whitespace-pre-wrap break-words text-sm leading-6`, content);
      const footer = el("div", "chat-footer mt-1 text-[0.68rem] text-base-content/45", formatDate(timestamp));
      chat.append(header, bubble, footer);
      target.append(chat);
    });
    target.scrollTop = target.scrollHeight;
  };

  const renderTraces = () => {
    const target = byId("traces-list");
    if (!target) return;
    if (!state.traces.length) {
      showEmpty(target, "No matching traces", "Try another status or search term.");
      return;
    }
    target.replaceChildren();
    const wrapper = el("div", "overflow-x-auto rounded-xl border border-base-300");
    const table = el("table", "table table-zebra table-sm");
    const head = el("thead");
    const headerRow = el("tr");
    ["Trace", "Status", "Duration", "Spans", "Started"].forEach((label) => headerRow.append(el("th", "whitespace-nowrap", label)));
    head.append(headerRow);
    const body = el("tbody");
    state.traces.forEach((item) => {
      const traceId = String(first(item, ["trace_id", "traceId", "id"], "—"));
      const row = el("tr", "hover");
      const traceCell = el("th", "max-w-64");
      const button = el("button", "link link-hover break-all text-left font-mono text-xs font-medium");
      button.type = "button";
      button.textContent = traceId;
      button.setAttribute("aria-label", `Inspect trace ${traceId}`);
      button.addEventListener("click", () => selectTrace(traceId));
      traceCell.append(button);
      row.append(traceCell);
      row.append(el("td", "whitespace-nowrap", ""));
      row.cells[1].append(badgeFor(first(item, ["status", "state", "outcome"], "unknown")));
      row.append(el("td", "whitespace-nowrap tabular-nums", formatMs(first(item, ["duration_ms", "latency_ms", "duration"]))));
      row.append(el("td", "tabular-nums", formatCount(first(item, ["span_count", "spans", "events"]))));
      row.append(el("td", "whitespace-nowrap text-xs text-base-content/60", formatDate(first(item, ["start_time", "started_at", "timestamp", "created_at"]))));
      body.append(row);
    });
    table.append(head, body);
    wrapper.append(table);
    target.append(wrapper);
  };

  const renderTraceDetail = (payload) => {
    const target = byId("trace-detail");
    if (!target) return;
    const root = rootOf(payload);
    const spans = listFrom(root, ["spans", "span_events", "events", "timeline"]);
    if (!spans.length && isObject(root) && (root.trace_id || root.span_id)) spans.push(root);
    if (!spans.length) {
      showEmpty(target, "No spans recorded", "The selected trace has no child spans available.");
      return;
    }
    target.replaceChildren();
    const timeline = el("ul", "timeline timeline-vertical timeline-compact timeline-snap-icon");
    const knownParents = new Set(spans.map((span) => String(first(span, ["span_id", "spanId", "id"], ""))));
    spans.forEach((span) => {
      const spanId = String(first(span, ["span_id", "spanId", "id"], "—"));
      const parentId = String(first(span, ["parent_span_id", "parentSpanId", "parent_id"], ""));
      const depth = parentId && knownParents.has(parentId) ? 1 : 0;
      const item = el("li");
      const start = el("div", "timeline-start w-24 text-right text-[0.68rem] text-base-content/55", formatDate(first(span, ["start_time", "started_at", "timestamp"])));
      const middle = el("div", "timeline-middle");
      middle.append(el("span", "size-2 rounded-full bg-info", ""));
      const end = el("div", "timeline-end w-full pb-5 pl-4");
      const card = el("div", "rounded-xl border border-base-300 bg-base-200/50 p-4");
      card.dataset.depth = String(depth);
      const title = el("div", "flex flex-wrap items-center gap-2");
      title.append(el("p", "break-all font-mono text-xs font-semibold", first(span, ["name", "operation", "span_name"], spanId)));
      title.append(badgeFor(first(span, ["status", "state", "outcome"], "unknown")));
      card.append(title);
      const details = el("div", "mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-base-content/60");
      details.append(el("span", "font-mono", spanId), el("span", "tabular-nums", formatMs(first(span, ["duration_ms", "latency_ms", "duration"]))));
      const service = first(span, ["service", "component", "source"]);
      if (service !== undefined) details.append(el("span", "break-all", service));
      card.append(details);
      const error = first(span, ["error", "error_message", "message"]);
      if (error !== undefined && error !== "") card.append(el("p", "mt-2 break-words text-xs text-error", error));
      end.append(card);
      item.append(start, middle, end);
      timeline.append(item);
    });
    target.append(timeline);
  };

  const renderLogs = (payload) => {
    const target = byId("logs-list");
    if (!target) return;
    const logs = listFrom(rootOf(payload), ["logs", "records", "entries", "items"]);
    if (!logs.length) {
      showEmpty(target, "No matching logs", "Try a different level or search term.");
      return;
    }
    target.replaceChildren();
    const wrapper = el("div", "overflow-x-auto rounded-xl border border-base-300");
    const table = el("table", "table table-zebra table-sm");
    const head = el("thead");
    const headerRow = el("tr");
    ["Time", "Level", "Source", "Message", "Fields"].forEach((label) => headerRow.append(el("th", "whitespace-nowrap", label)));
    head.append(headerRow);
    const body = el("tbody");
    logs.forEach((item) => {
      const row = el("tr", "align-top hover");
      row.append(el("td", "whitespace-nowrap text-xs text-base-content/60", formatDate(first(item, ["timestamp", "time", "created_at", "createdAt"]))));
      const levelCell = el("td", "whitespace-nowrap");
      levelCell.append(badgeFor(first(item, ["level", "severity", "status"], "unknown")));
      row.append(levelCell);
      row.append(el("td", "max-w-40 break-all font-mono text-xs", first(item, ["source", "logger", "module", "component"], "—")));
      row.append(el("td", "min-w-64 max-w-xl whitespace-pre-wrap break-words text-sm leading-5", first(item, ["message", "text", "body"], "—")));
      const fields = first(item, ["attributes", "fields", "metadata", "context"]);
      const fieldsText = isObject(fields) || Array.isArray(fields) ? JSON.stringify(fields) : fields;
      row.append(el("td", "max-w-72 whitespace-pre-wrap break-all font-mono text-[0.68rem] text-base-content/55", fieldsText ?? "—"));
      body.append(row);
    });
    table.append(head, body);
    wrapper.append(table);
    target.append(wrapper);
  };

  const loadOverview = async () => {
    const payload = await requestJson(API.overview);
    renderOverview(payload);
  };

  const loadUsers = async () => {
    const search = byId("user-search")?.value.trim() || "";
    const payload = await requestJson(`${API.users}${encodeURIComponent(search)}&limit=50&offset=0`);
    state.users = listFrom(rootOf(payload), ["users", "items", "results", "records"]);
    renderUsers();
    if (state.selectedUser && !state.users.some((item) => String(first(item, ["user_id", "userId", "chat_id", "chatId", "id"], "")) === state.selectedUser)) {
      state.selectedUser = "";
      state.selectedSession = "";
      setText("selected-user", "Choose a user");
      setText("selected-session", "Select a session to inspect messages.");
      showEmpty(byId("sessions-list"), "Select a user", "Choose a stored user or chat ID to see retained sessions.");
      showEmpty(byId("chat-view"), "No session selected", "Choose a session to inspect its conversation.");
    }
  };

  const loadSessions = async () => {
    if (!state.selectedUser) return;
    const payload = await requestJson(`${API.sessions}${encodeURIComponent(state.selectedUser)}&limit=50&offset=0`);
    state.sessions = listFrom(rootOf(payload), ["sessions", "items", "results", "records"]);
    renderSessions();
  };

  const loadSessionDetail = async () => {
    if (!state.selectedUser || !state.selectedSession) return;
    const payload = await requestJson(`${API.session}${encodeURIComponent(state.selectedUser)}&session_id=${encodeURIComponent(state.selectedSession)}`);
    renderChat(payload);
  };

  const loadTraces = async () => {
    const status = byId("trace-status")?.value || "";
    const search = byId("trace-search")?.value.trim() || "";
    const payload = await requestJson(`${API.traces}${encodeURIComponent(status)}&search=${encodeURIComponent(search)}&limit=80`);
    state.traces = listFrom(rootOf(payload), ["traces", "items", "results", "records"]);
    renderTraces();
  };

  const loadTraceDetail = async () => {
    if (!state.selectedTrace) return;
    const payload = await requestJson(`${API.trace}${encodeURIComponent(state.selectedTrace)}`);
    renderTraceDetail(payload);
  };

  const loadLogs = async () => {
    const level = byId("log-level")?.value || "";
    const search = byId("log-search")?.value.trim() || "";
    const payload = await requestJson(`${API.logs}${encodeURIComponent(level)}&search=${encodeURIComponent(search)}&limit=100`);
    renderLogs(payload);
  };

  const refreshActiveView = async (quiet = false) => {
    if (!quiet) setBusy(true);
    clearGlobalError();
    setConnection("Refreshing local data", "pending");
    try {
      if (state.view === "overview") await loadOverview();
      if (state.view === "users") {
        await loadUsers();
        if (state.selectedUser) await loadSessions();
        if (state.selectedUser && state.selectedSession) await loadSessionDetail();
      }
      if (state.view === "traces") {
        await loadTraces();
        if (state.selectedTrace) await loadTraceDetail();
      }
      if (state.view === "logs") await loadLogs();
      state.lastUpdated = new Date();
      setText("last-updated", state.lastUpdated.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
      setConnection("Local data connected", "ok");
    } catch (error) {
      showGlobalError(error);
      setConnection("Local data unavailable", "error");
      const panelTarget = state.view === "users" ? byId("users-list") : state.view === "traces" ? byId("traces-list") : state.view === "logs" ? byId("logs-list") : byId("activity-chart");
      if (panelTarget) showInlineError(panelTarget, "This view could not load. Try Refresh after checking the server.");
    } finally {
      setBusy(false);
      const loading = byId("loading-state");
      if (loading) loading.hidden = true;
    }
  };

  const selectUser = async (userId) => {
    state.selectedUser = String(userId);
    state.selectedSession = "";
    setText("selected-user", state.selectedUser);
    setText("selected-session", "Select a session to inspect messages.");
    showEmpty(byId("chat-view"), "No session selected", "Choose a session to inspect its conversation.");
    try {
      setConnection("Loading user sessions", "pending");
      await loadSessions();
      setConnection("Local data connected", "ok");
    } catch (error) {
      showGlobalError(error);
      showInlineError(byId("sessions-list"), "Sessions could not load for this user.");
      setConnection("Local data unavailable", "error");
    }
  };

  const selectSession = async (sessionId) => {
    state.selectedSession = String(sessionId);
    setText("selected-session", state.selectedSession);
    showEmpty(byId("chat-view"), "Loading conversation", "Reading this session's messages…");
    try {
      await loadSessionDetail();
    } catch (error) {
      showGlobalError(error);
      showInlineError(byId("chat-view"), "This session's conversation could not load.");
    }
  };

  const selectTrace = async (traceId) => {
    state.selectedTrace = String(traceId);
    setText("selected-trace", state.selectedTrace);
    showEmpty(byId("trace-detail"), "Loading trace", "Reading the span timeline…");
    try {
      await loadTraceDetail();
    } catch (error) {
      showGlobalError(error);
      showInlineError(byId("trace-detail"), "This trace's span detail could not load.");
    }
  };

  const setView = (view, focus = false) => {
    state.view = view;
    document.querySelectorAll("[role=tab]").forEach((tab) => {
      const active = tab.dataset.view === view;
      tab.classList.toggle("tab-active", active);
      tab.setAttribute("aria-selected", String(active));
      tab.setAttribute("tabindex", active ? "0" : "-1");
      if (active && focus) tab.focus();
    });
    document.querySelectorAll("[data-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.panel !== view;
    });
    refreshActiveView();
  };

  const setupTabs = () => {
    const tabs = Array.from(document.querySelectorAll("[role=tab]"));
    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => setView(tab.dataset.view || "overview"));
      tab.addEventListener("keydown", (event) => {
        let nextIndex = index;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") nextIndex = (index + 1) % tabs.length;
        if (event.key === "ArrowLeft" || event.key === "ArrowUp") nextIndex = (index - 1 + tabs.length) % tabs.length;
        if (event.key === "Home") nextIndex = 0;
        if (event.key === "End") nextIndex = tabs.length - 1;
        if (nextIndex !== index || event.key === "Home" || event.key === "End") {
          event.preventDefault();
          setView(tabs[nextIndex].dataset.view || "overview", true);
        }
      });
    });
  };

  const setupFilters = () => {
    byId("user-filter-form")?.addEventListener("submit", (event) => {
      event.preventDefault();
      state.selectedUser = "";
      state.selectedSession = "";
      setText("selected-user", "Choose a user");
      setText("selected-session", "Select a session to inspect messages.");
      showEmpty(byId("sessions-list"), "Select a user", "Choose a stored user or chat ID to see retained sessions.");
      showEmpty(byId("chat-view"), "No session selected", "Choose a session to inspect its conversation.");
      refreshActiveView();
    });
    byId("user-clear")?.addEventListener("click", () => {
      const input = byId("user-search");
      if (input) input.value = "";
      byId("user-filter-form")?.requestSubmit();
    });
    byId("trace-filter-form")?.addEventListener("submit", (event) => {
      event.preventDefault();
      refreshActiveView();
    });
    byId("log-filter-form")?.addEventListener("submit", (event) => {
      event.preventDefault();
      refreshActiveView();
    });
  };

  const setupTheme = () => {
    const toggle = byId("theme-toggle");
    const prefersDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = prefersDark ? "dark" : "light";
    const update = () => {
      const dark = document.documentElement.dataset.theme === "dark";
      toggle?.setAttribute("aria-pressed", String(dark));
      toggle?.setAttribute("aria-label", dark ? "Use light theme" : "Use dark theme");
    };
    toggle?.addEventListener("click", () => {
      document.documentElement.dataset.theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      update();
    });
    update();
  };

  const setupRefresh = () => {
    byId("refresh-button")?.addEventListener("click", () => refreshActiveView());
    window.setInterval(() => {
      const autoRefresh = byId("auto-refresh");
      if (autoRefresh?.checked && !document.hidden) refreshActiveView(true);
    }, 60_000);
  };

  const boot = () => {
    setupTheme();
    setupTabs();
    setupFilters();
    setupRefresh();
    showEmpty(byId("users-list"), "No users loaded", "Open the Users & sessions view to read stored user IDs.");
    showEmpty(byId("sessions-list"), "Select a user", "Choose a stored user or chat ID to see retained sessions.");
    showEmpty(byId("chat-view"), "No session selected", "Choose a session to inspect its conversation.");
    showEmpty(byId("traces-list"), "No traces loaded", "Open the Traces view to read span records.");
    showEmpty(byId("trace-detail"), "No trace selected", "Choose a trace to inspect its span timeline.");
    showEmpty(byId("logs-list"), "No logs loaded", "Open the Logs view to read structured records.");
    refreshActiveView();
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot, { once: true });
  else boot();
})();
