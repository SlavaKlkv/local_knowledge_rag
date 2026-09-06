/*
 * Интерфейс к API платформы.
 *
 * Отдаётся тем же приложением, что и API, поэтому запросы идут на тот же
 * origin: отдельный адрес бэкенда настраивать не нужно, CORS не участвует.
 */

const TOKEN_KEY = "lkr.token";
const THEME_KEY = "lkr.theme";

const state = {
  token: localStorage.getItem(TOKEN_KEY),
  knowledgeBaseId: null,
  conversationId: null,
  documentPolls: new Map(),
};

const $ = (id) => document.getElementById(id);

/* --- Тема --- */

/*
 * Выбор темы живёт в localStorage и переживает перезагрузку. Значение
 * "system" снимает атрибут с <html>, и оформление снова следует за
 * настройкой операционной системы.
 */
function applyTheme(theme) {
  if (theme === "system") {
    document.documentElement.removeAttribute("data-theme");
  } else {
    document.documentElement.dataset.theme = theme;
  }
  for (const button of document.querySelectorAll("#theme-switch button")) {
    button.setAttribute("aria-pressed", String(button.dataset.theme === theme));
  }
}

function setTheme(theme) {
  localStorage.setItem(THEME_KEY, theme);
  applyTheme(theme);
}

/* --- Сетевой слой --- */

async function api(path, { method = "GET", body, isForm = false } = {}) {
  const headers = {};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (body && !isForm) headers["Content-Type"] = "application/json";

  const response = await fetch(path, {
    method,
    headers,
    body: isForm ? body : body ? JSON.stringify(body) : undefined,
  });

  if (response.status === 401) {
    signOut();
    throw new Error("Сессия истекла, войдите заново");
  }
  if (response.status === 204) return null;

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    // Приложение отдаёт ошибки единым конвертом {error: {code, message}}.
    const message =
      payload?.error?.message ||
      payload?.detail?.[0]?.msg ||
      payload?.detail ||
      `Ошибка ${response.status}`;
    throw new Error(message);
  }
  return payload;
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.hidden = true), 4000);
}

/* --- Вход --- */

function signOut() {
  state.token = null;
  localStorage.removeItem(TOKEN_KEY);
  render();
}

async function signIn(register) {
  const email = $("auth-email").value.trim();
  const password = $("auth-password").value;
  const error = $("auth-error");
  error.hidden = true;

  try {
    if (register) {
      await api("/auth/register", {
        method: "POST",
        body: { email, password },
      });
    }
    const token = await api("/auth/login", {
      method: "POST",
      body: { email, password },
    });
    state.token = token.access_token;
    localStorage.setItem(TOKEN_KEY, state.token);
    render();
  } catch (err) {
    error.textContent = err.message;
    error.hidden = false;
  }
}

/* --- Базы знаний --- */

async function loadKnowledgeBases() {
  const bases = await api("/knowledge-bases");
  const list = $("kb-list");
  list.innerHTML = "";

  if (!bases.length) {
    list.innerHTML = '<li class="muted tiny">Пока ни одной базы</li>';
    return;
  }

  if (!bases.some((kb) => kb.id === state.knowledgeBaseId)) {
    state.knowledgeBaseId = bases[0].id;
  }

  for (const kb of bases) {
    const li = document.createElement("li");
    if (kb.id === state.knowledgeBaseId) li.className = "active";
    const button = document.createElement("button");
    button.textContent = kb.name;
    button.onclick = () => selectKnowledgeBase(kb.id);
    li.append(button);
    list.append(li);
  }
}

function selectKnowledgeBase(id) {
  state.knowledgeBaseId = id;
  state.conversationId = null;
  resetChat();
  loadKnowledgeBases();
  loadDocuments();
}

async function createKnowledgeBase() {
  const name = prompt("Название базы знаний");
  if (!name) return;
  const kb = await api("/knowledge-bases", { method: "POST", body: { name } });
  state.knowledgeBaseId = kb.id;
  await loadKnowledgeBases();
  await loadDocuments();
}

/* --- Документы --- */

const STATUS_LABELS = {
  uploaded: ["pending", "в очереди"],
  parsing: ["indexing", "разбор"],
  chunking: ["indexing", "чанкинг"],
  embedding: ["indexing", "эмбеддинги"],
  indexing: ["indexing", "индексация"],
  ready: ["indexed", "готов"],
  failed: ["failed", "ошибка"],
};

async function loadDocuments() {
  if (!state.knowledgeBaseId) return;
  const documents = await api(
    `/documents?knowledge_base_id=${state.knowledgeBaseId}`,
  );
  const list = $("doc-list");
  list.innerHTML = "";

  if (!documents.length) {
    list.innerHTML = '<li class="muted tiny">Документов пока нет</li>';
    return;
  }

  for (const doc of documents) {
    const [kind, label] = STATUS_LABELS[doc.status] || ["pending", doc.status];
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="doc-name" title="${escapeHtml(doc.filename)}">` +
      `${escapeHtml(doc.filename)}</span>` +
      `<span class="status status-${kind}">${label}</span>`;
    list.append(li);
  }

  // Пока хоть один документ индексируется — обновляем список: индексация
  // идёт в воркере, и её окончание приходит не в ответе на загрузку.
  const pending = documents.some((doc) => !["ready", "failed"].includes(doc.status));
  if (pending) setTimeout(loadDocuments, 1500);
}

async function uploadDocument(file) {
  if (!state.knowledgeBaseId) {
    toast("Сначала создайте базу знаний");
    return;
  }
  // knowledge_base_id — query-параметр: в теле запроса едет только файл.
  const form = new FormData();
  form.append("file", file);
  try {
    await api(`/documents?knowledge_base_id=${state.knowledgeBaseId}`, {
      method: "POST",
      body: form,
      isForm: true,
    });
    toast(`${file.name} принят, идёт индексация`);
    loadDocuments();
  } catch (err) {
    toast(err.message);
  }
}

/* --- Кольцо моделей --- */

const HEALTH_LABELS = {
  healthy: "здорова",
  degraded: "деградация",
  cooldown: "остывает",
  unavailable: "не установлена",
};

async function loadRing() {
  let status;
  try {
    status = await api("/inference/status");
  } catch {
    return;
  }

  const installed = new Map(status.models.map((m) => [m.model, m.installed]));
  const list = $("ring-list");
  list.innerHTML = "";

  for (const member of status.ring) {
    const health = installed.get(member.model) === false
      ? "unavailable"
      : member.health;
    const li = document.createElement("li");
    li.innerHTML =
      `<span>${escapeHtml(member.model)}</span>` +
      `<span class="health health-${health}">${HEALTH_LABELS[health] || health}</span>`;
    list.append(li);
  }

  $("ring-summary").textContent =
    `профиль ${status.profile} · ${status.provider}`;
}

/* --- Вопрос-ответ --- */

const NO_ANSWER_REASONS = {
  empty_context: "Поиск не нашёл ни одного подходящего фрагмента.",
  below_threshold: "Найденное не прошло порог релевантности.",
  model_declined: "Модель сообщила, что данных в документах недостаточно.",
  no_citations:
    "Ответ не сослался ни на один реальный фрагмент — проверить его нечем.",
};

function resetChat() {
  $("messages").innerHTML = "";
  renderEmptyChat();
}

function renderEmptyChat() {
  const messages = $("messages");
  messages.innerHTML = `
    <div class="empty">
      <h3>Задайте вопрос по документам</h3>
      <p class="muted">
        Ответ собирается только из найденных фрагментов. Если данных не хватает,
        система скажет об этом прямо, а не придумает ответ.
      </p>
      <div id="suggestions" class="suggestions"></div>
    </div>`;
  for (const text of SUGGESTIONS) {
    const button = document.createElement("button");
    button.textContent = text;
    button.onclick = () => {
      $("chat-input").value = text;
      $("chat-form").requestSubmit();
    };
    $("suggestions").append(button);
  }
}

// Подсказки — только вопросы по документам. Вопрос без ответа в базе
// пользователь задаёт сам: подсказка с заранее известным отказом выглядела
// бы постановочной.
const SUGGESTIONS = [
  "Как компенсируется дежурство в выходной?",
  "Какие лимиты на проживание в командировке?",
  "Сколько длится испытательный срок?",
];

function appendUserMessage(text) {
  const empty = $("messages").querySelector(".empty");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = "msg msg-user";
  div.textContent = text;
  $("messages").append(div);
  scrollMessages();
}

function appendPending() {
  const div = document.createElement("div");
  div.className = "msg msg-bot";
  div.innerHTML = '<span class="muted dots">Модель отвечает</span>';
  $("messages").append(div);
  scrollMessages();
  return div;
}

function renderAnswer(container, data) {
  const parts = [];

  if (data.has_answer) {
    parts.push(`<div class="answer">${escapeHtml(data.answer)}</div>`);
  } else {
    const reason =
      NO_ANSWER_REASONS[data.no_answer_reason] || data.no_answer_reason || "";
    parts.push(
      '<div class="no-answer"><div>' +
        "<strong>Ответа в документах нет</strong>" +
        `<span class="muted">${escapeHtml(reason)}</span>` +
        "</div></div>",
    );
  }

  if (data.citations?.length) {
    const items = data.citations
      .map(
        (c) =>
          `<div class="citation"><span class="ref">[${c.ref}]</span>` +
          `${escapeHtml(c.document_name || c.document_id)}` +
          (c.page ? `, с. ${c.page}` : "") +
          (c.section ? ` — ${escapeHtml(c.section)}` : "") +
          "</div>",
      )
      .join("");
    parts.push(`<div class="citations"><h4>Источники</h4>${items}</div>`);
  }

  const meta = [];
  if (data.model) meta.push(data.model);
  if (data.provider) meta.push(data.provider);
  if (data.latency_ms) meta.push(`${(data.latency_ms / 1000).toFixed(1)} с`);
  if (data.rewritten_query && data.rewritten_query !== data.question) {
    meta.push(`переписанный запрос: ${data.rewritten_query}`);
  }
  if (meta.length) {
    parts.push(
      `<div class="meta">${meta
        .map((m) => `<span>${escapeHtml(m)}</span>`)
        .join("")}</div>`,
    );
  }

  container.innerHTML = parts.join("");
  scrollMessages();
}

async function ask(question) {
  appendUserMessage(question);
  const pending = appendPending();
  const button = $("chat-form").querySelector("button");
  button.disabled = true;

  try {
    const data = await api("/chat", {
      method: "POST",
      body: {
        question,
        knowledge_base_id: state.knowledgeBaseId,
        conversation_id: state.conversationId,
      },
    });
    state.conversationId = data.conversation_id || state.conversationId;
    renderAnswer(pending, data);
  } catch (err) {
    pending.innerHTML = `<span class="error">${escapeHtml(err.message)}</span>`;
  } finally {
    button.disabled = false;
    loadRing();
  }
}

/* --- Поиск --- */

async function search(query) {
  const results = $("search-results");
  results.innerHTML = '<span class="muted dots">Ищем</span>';
  try {
    const data = await api("/search", {
      method: "POST",
      body: { query, knowledge_base_id: state.knowledgeBaseId },
    });
    if (!data.hits.length) {
      results.innerHTML = '<span class="muted">Ничего не найдено</span>';
      return;
    }
    results.innerHTML = data.hits
      .map(
        (hit) =>
          '<div class="hit"><div class="hit-head">' +
          `<span>${escapeHtml(hit.document_name || hit.document_id)}` +
          (hit.section ? ` — ${escapeHtml(hit.section)}` : "") +
          "</span>" +
          `<span class="score">${hit.score.toFixed(3)}</span></div>` +
          `<div>${escapeHtml(hit.text)}</div></div>`,
      )
      .join("");
  } catch (err) {
    results.innerHTML = `<span class="error">${escapeHtml(err.message)}</span>`;
  }
}

/* --- Служебное --- */

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value ?? "";
  return div.innerHTML;
}

function scrollMessages() {
  const messages = $("messages");
  messages.scrollTop = messages.scrollHeight;
}

async function render() {
  const authorized = Boolean(state.token);
  $("auth-view").hidden = authorized;
  $("main-view").hidden = !authorized;
  $("logout").hidden = !authorized;

  if (!authorized) return;

  loadRing();
  try {
    await loadKnowledgeBases();
    await loadDocuments();
  } catch (err) {
    toast(err.message);
  }
  renderEmptyChat();
}

/* --- Подписки --- */

$("auth-form").addEventListener("submit", (event) => {
  event.preventDefault();
  signIn(false);
});
$("register").addEventListener("click", () => signIn(true));
$("logout").addEventListener("click", signOut);
$("kb-new").addEventListener("click", createKnowledgeBase);

$("upload-input").addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) uploadDocument(file);
  event.target.value = "";
});

$("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = $("chat-input").value.trim();
  if (!question) return;
  $("chat-input").value = "";
  ask(question);
});

$("search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  search($("search-input").value.trim());
});

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    for (const other of document.querySelectorAll(".tab")) {
      other.classList.toggle("active", other === tab);
    }
    $("tab-chat").hidden = tab.dataset.tab !== "chat";
    $("tab-search").hidden = tab.dataset.tab !== "search";
  });
}

for (const button of document.querySelectorAll("#theme-switch button")) {
  button.addEventListener("click", () => setTheme(button.dataset.theme));
}

applyTheme(localStorage.getItem(THEME_KEY) || "system");
setInterval(loadRing, 15000);
render();
