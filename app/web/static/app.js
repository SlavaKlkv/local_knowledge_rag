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
  // Строка списка, переведённая в режим переименования (id базы или null).
  renamingKnowledgeBaseId: null,
  conversationId: null,
  documentPolls: new Map(),
  // Последний известный состав кольца: по нему видно, какая модель отвечает
  // на текущий вопрос и куда кольцо ушло при отказе.
  ring: [],
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
  // Имя нужно подсказкам: они у каждой базы свои.
  state.knowledgeBaseName =
    bases.find((kb) => kb.id === state.knowledgeBaseId)?.name ?? "";

  for (const kb of bases) {
    const li = document.createElement("li");
    if (kb.id === state.knowledgeBaseId) li.className = "active";
    if (kb.id === state.renamingKnowledgeBaseId) {
      list.append(renameRow(li, kb));
      continue;
    }
    const button = document.createElement("button");
    button.textContent = kb.name;
    button.onclick = () => selectKnowledgeBase(kb);
    li.append(button);

    const rename = document.createElement("button");
    rename.className = "icon-button rename";
    rename.type = "button";
    rename.textContent = "\u270e";
    rename.title = "Переименовать базу знаний";
    rename.setAttribute("aria-label", `Переименовать базу ${kb.name}`);
    rename.onclick = () => {
      state.renamingKnowledgeBaseId = kb.id;
      loadKnowledgeBases();
    };
    li.append(rename);

    const remove = document.createElement("button");
    remove.className = "icon-button";
    remove.type = "button";
    remove.textContent = "\u00d7";
    remove.title = "Удалить базу знаний";
    remove.setAttribute("aria-label", `Удалить базу ${kb.name}`);
    remove.onclick = () => deleteKnowledgeBase(kb);
    li.append(remove);

    list.append(li);
  }
}

/*
 * Переименование правится на месте, в самой строке списка: так видно, какую
 * базу переименовываешь, и не нужен отдельный диалог ради одного поля.
 */
function renameRow(li, kb) {
  const form = document.createElement("form");
  form.className = "kb-rename";

  const input = document.createElement("input");
  input.type = "text";
  input.value = kb.name;
  input.maxLength = 120;
  input.autocomplete = "off";
  input.setAttribute("aria-label", `Новое название базы ${kb.name}`);
  form.append(input);

  const save = document.createElement("button");
  save.type = "submit";
  save.className = "ghost small";
  save.textContent = "OK";
  form.append(save);

  form.onsubmit = (event) => {
    event.preventDefault();
    renameKnowledgeBase(kb, input.value.trim());
  };
  // Уход фокуса и Escape отменяют правку: строка не должна залипать в режиме
  // редактирования, если о ней передумали.
  input.onkeydown = (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      cancelRename();
    }
  };

  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "icon-button";
  cancel.textContent = "\u00d7";
  cancel.title = "Отменить переименование";
  cancel.setAttribute("aria-label", "Отменить переименование");
  cancel.onclick = cancelRename;
  form.append(cancel);

  li.append(form);
  // Список перерисовывается целиком, поэтому фокус ставим после вставки.
  queueMicrotask(() => {
    input.focus();
    input.select();
  });
  return li;
}

function cancelRename() {
  state.renamingKnowledgeBaseId = null;
  loadKnowledgeBases();
}

async function renameKnowledgeBase(kb, name) {
  if (!name || name === kb.name) {
    cancelRename();
    return;
  }
  try {
    const updated = await api(`/knowledge-bases/${kb.id}`, {
      method: "PATCH",
      body: { name },
    });
    if (kb.id === state.knowledgeBaseId) {
      // Имя активной базы держат подсказки чата — обновляем и его.
      state.knowledgeBaseName = updated.name;
    }
    toast(`База переименована в «${updated.name}»`);
  } catch (err) {
    toast(err.message);
  }
  state.renamingKnowledgeBaseId = null;
  await loadKnowledgeBases();
}

async function deleteKnowledgeBase(kb) {
  // Вместе с базой уходят её документы и векторы — предупреждаем об этом
  // прямо в вопросе, иначе цена клика по крестику неочевидна.
  const ok = await askConfirm(
    `Удалить базу «${kb.name}» вместе со всеми документами?`,
  );
  if (!ok) return;
  try {
    await api(`/knowledge-bases/${kb.id}`, { method: "DELETE" });
    toast(`База «${kb.name}» удалена`);
    if (kb.id === state.knowledgeBaseId) {
      // Активную базу выбирает loadKnowledgeBases; чат и поиск принадлежали
      // удалённой, поэтому очищаем их до перезагрузки списка.
      state.knowledgeBaseId = null;
      state.knowledgeBaseName = "";
      state.conversationId = null;
      resetChat();
      resetSearch();
    }
    await loadKnowledgeBases();
    await loadDocuments();
  } catch (err) {
    toast(err.message);
  }
}

function selectKnowledgeBase(kb) {
  state.knowledgeBaseId = kb.id;
  state.knowledgeBaseName = kb.name;
  state.conversationId = null;
  resetChat();
  resetSearch();
  loadKnowledgeBases();
  loadDocuments();
}

/*
 * Создание базы — инлайн-поле, а не prompt(): нативный диалог выпадает из
 * оформления приложения, а на узком экране ещё и перекрывает список, ради
 * которого его открыли.
 */
function toggleKnowledgeBaseForm() {
  const form = $("kb-form");
  form.hidden = !form.hidden;
  if (form.hidden) $("kb-name").value = "";
  else $("kb-name").focus();
}

function closeKnowledgeBaseForm() {
  $("kb-form").hidden = true;
  $("kb-name").value = "";
}

async function createKnowledgeBase() {
  const name = $("kb-name").value.trim();
  if (!name) return;
  const kb = await api("/knowledge-bases", { method: "POST", body: { name } });
  closeKnowledgeBaseForm();
  state.knowledgeBaseId = kb.id;
  state.knowledgeBaseName = kb.name;
  // Чат принадлежал прежней базе — и история, и подсказки под её документы.
  state.conversationId = null;
  resetChat();
  resetSearch();
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
  if (!state.knowledgeBaseId) {
    // Баз не осталось (последнюю удалили) — список документов должен уйти
    // вместе с ней, иначе на экране висят чужие файлы.
    hideFullText();
    $("doc-list").innerHTML = '<li class="muted tiny">Документов пока нет</li>';
    return;
  }
  const documents = await api(
    `/documents?knowledge_base_id=${state.knowledgeBaseId}`,
  );
  const list = $("doc-list");
  hideFullText();
  list.innerHTML = "";

  if (!documents.length) {
    list.innerHTML = '<li class="muted tiny">Документов пока нет</li>';
    return;
  }

  for (const doc of documents) {
    const [kind, label] = STATUS_LABELS[doc.status] || ["pending", doc.status];
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="doc-name">${escapeHtml(doc.filename)}</span>` +
      `<span class="status status-${kind}">${label}</span>`;
    showFullTextOnHover(li.querySelector(".doc-name"));

    const remove = document.createElement("button");
    remove.className = "icon-button";
    remove.type = "button";
    remove.textContent = "\u00d7";
    remove.title = "Удалить документ";
    remove.setAttribute("aria-label", `Удалить ${doc.filename}`);
    remove.onclick = () => deleteDocument(doc);
    li.append(remove);

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

/*
 * Подтверждение — своя модалка, а не confirm(): нативный диалог рисуется
 * браузером поверх страницы, называет адрес сервиса вместо приложения и
 * блокирует поток на время показа.
 */
let confirmResolve = null;

function askConfirm(text, okLabel = "Удалить") {
  $("confirm-text").textContent = text;
  $("confirm-ok").textContent = okLabel;
  $("confirm-dialog").hidden = false;
  $("confirm-ok").focus();
  return new Promise((resolve) => {
    confirmResolve = resolve;
  });
}

function closeConfirm(answer) {
  if (!confirmResolve) return;
  $("confirm-dialog").hidden = true;
  const resolve = confirmResolve;
  confirmResolve = null;
  resolve(answer);
}

async function deleteDocument(doc) {
  // Удаление необратимо: вместе со строкой в БД уходят и векторы фрагментов,
  // поэтому спрашиваем подтверждение и называем файл.
  const ok = await askConfirm(`Удалить «${doc.filename}» из базы знаний?`);
  if (!ok) return;
  try {
    await api(`/documents/${doc.id}`, { method: "DELETE" });
    toast(`${doc.filename} удалён`);
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
  state.ring = status.ring.map((member) => ({
    model: member.model,
    health: installed.get(member.model) === false ? "unavailable" : member.health,
  }));

  const list = $("ring-list");
  list.innerHTML = "";

  for (const member of state.ring) {
    const health = member.health;
    const li = document.createElement("li");
    li.innerHTML =
      `<span>${escapeHtml(member.model)}</span>` +
      `<span class="health health-${health}">${HEALTH_LABELS[health] || health}</span>`;
    list.append(li);
  }

  renderRuntimePill(status);
  $("ring-summary").textContent = `профиль ${status.profile}`;
}

/*
 * Отметка о том, где считается ответ. Имя runtime'а (ollama, vllm) само по
 * себе ничего не обещает тому, кто эти названия видит впервые, поэтому оно
 * стоит вплотную к слову, которое его объясняет. Локальность берётся из
 * ответа сервера, а не из списка имён здесь: появится провайдер к удалённому
 * API — отметка переключится сама, а не останется зелёной неправдой.
 */
function renderRuntimePill(status) {
  const pill = $("runtime-pill");
  const local = status.provider_is_local;
  pill.textContent = `${status.provider} · ${local ? "локально" : "облако"}`;
  pill.className = `pill ${local ? "pill-local" : "pill-remote"}`;
  pill.title = local
    ? "Модель работает на этой машине: документы и вопросы никуда не отправляются"
    : "Запросы уходят внешнему сервису: документы и вопросы покидают эту машину";
  pill.hidden = false;
}

/* --- Настройка кольца --- */

/*
 * Панель настройки показывает, из чего кольцо собрано на этой машине:
 * профиль железа, runtime и модели плана. Единственное действие, которое
 * меняет состояние, — загрузка весов: она идёт в фоне на сервере, поэтому
 * прогресс опрашивается отдельно, пока хоть одна загрузка не закончилась.
 */

const DOWNLOAD_LABELS = {
  pending: "в очереди",
  downloading: "загрузка",
  completed: "загружена",
  failed: "ошибка",
  cancelled: "отменена",
};

const ACTIVE_DOWNLOAD_STATES = ["pending", "downloading"];

function trackRingPointer() {
  $("ring-config").addEventListener("pointerdown", () => {
    ringPointerDown = true;
  });
  // Отпустить кнопку можно и за пределами панели — слушаем на окне,
  // иначе флаг залипнет и опрос остановится навсегда.
  window.addEventListener("pointerup", () => {
    ringPointerDown = false;
  });
  window.addEventListener("pointercancel", () => {
    ringPointerDown = false;
  });
}

function openRingSetup() {
  clearRingNotice();
  $("ring-dialog").hidden = false;
  ringDraft.models = null;
  loadRingSetup();
}

function closeRingSetup() {
  $("ring-dialog").hidden = true;
  clearTimeout(loadRingSetup.timer);
}

/*
 * Пока идёт загрузка весов, состояние опрашивается каждые две секунды и
 * панель перерисовывается целиком. Две вещи из-за этого ломались, и обе
 * лечатся здесь: ответ опроса, отправленного до действия пользователя,
 * возвращался позже и затирал результат (кнопка «Вернуть состав профиля»
 * выглядела нерабочей), а перерисовка между нажатием и отпусканием кнопки
 * убирала её из DOM, и клик не доходил вовсе.
 */
let ringPointerDown = false;

async function loadRingSetup() {
  const box = $("ring-config");
  clearTimeout(loadRingSetup.timer);
  const token = (loadRingSetup.token = (loadRingSetup.token || 0) + 1);
  if (!box.childElementCount) {
    box.innerHTML = '<span class="muted dots">Собираем состояние</span>';
  }

  let status;
  let runtimes;
  let downloads;
  let config;
  try {
    [status, runtimes, downloads, config] = await Promise.all([
      api("/inference/status"),
      api("/inference/runtimes"),
      api("/inference/downloads"),
      api("/inference/ring"),
    ]);
  } catch (err) {
    if (token !== loadRingSetup.token) return;
    box.innerHTML = `<span class="error">${escapeHtml(err.message)}</span>`;
    return;
  }

  // Ответ устаревшего запроса не должен возвращать на экран состав,
  // который пользователь уже сбросил или сохранил.
  if (token !== loadRingSetup.token) return;

  // Перерисовка во время нажатия отменила бы сам клик: элемент исчезнет
  // раньше, чем браузер успеет его засчитать.
  if (ringPointerDown) {
    loadRingSetup.timer = setTimeout(loadRingSetup, 300);
    return;
  }

  // Черновик состава живёт до сохранения: пока пользователь двигает
  // модели, опрос прогресса не должен затирать его правки.
  if (!ringDraft.models) ringDraft.models = config.models.map((m) => m.model);
  ringDraft.available = config.available;
  ringDraft.catalog = new Map(
    [...config.models, ...config.available].map((m) => [m.model, m]),
  );
  ringDraft.source = config.source;

  ringDraft.status = status;
  ringDraft.progress = new Map(downloads.map((d) => [d.model, d]));

  // Если модель скачивали из чернового состава, завершение загрузки не
  // должно выбрасывать её обратно в список кандидатов после закрытия окна.
  // Как только весь черновик установлен, закрепляем именно тот порядок,
  // который пользователь собрал перед нажатием «Скачать».
  const completedDraftDownload = ringDraft.models.some(
    (model) =>
      ringDraft.downloadsToPersist.has(model) &&
      ringDraft.catalog.get(model)?.installed,
  );
  const draftIsInstalled = ringDraft.models.every(
    (model) => ringDraft.catalog.get(model)?.installed,
  );
  if (completedDraftDownload && draftIsInstalled) {
    try {
      await api("/inference/ring", {
        method: "PUT",
        body: { models: ringDraft.models },
      });
      if (token !== loadRingSetup.token) return;
      ringNotice("Загруженная модель добавлена в состав кольца", "ok");
      ringDraft.downloadsToPersist.clear();
      ringDraft.models = null;
      loadRingSetup();
      loadRing();
      return;
    } catch (err) {
      if (token !== loadRingSetup.token) return;
      ringNotice(`Модель загружена, но состав не сохранён. ${err.message}`);
    }
  }

  box.innerHTML = "";
  box.append(renderRingOverview(status));
  box.append(renderRingComposition());
  box.append(renderRuntimes(runtimes));

  // Пока что-то качается, состояние обновляется само: загрузка идёт в
  // фоновой задаче сервера и о завершении не сообщает.
  const active = downloads.some((d) =>
    ACTIVE_DOWNLOAD_STATES.includes(d.state),
  );
  if (active) loadRingSetup.timer = setTimeout(loadRingSetup, 2000);
}

/*
 * Сколько ГБ предстоит скачать под черновой состав кольца: суммируем веса
 * тех его моделей, которых ещё нет в runtime'е. Размер модели, которой нет
 * в каталоге профиля, неизвестен — она добавляет к плану ноль.
 */
function draftRequiredDiskGb() {
  const total = (ringDraft.models || []).reduce((sum, model) => {
    const slot = ringDraft.catalog.get(model);
    return slot && !slot.installed ? sum + slot.download_size_gb : sum;
  }, 0);
  return Math.round(total * 10) / 10;
}

function renderRingOverview(status) {
  const section = document.createElement("section");
  section.className = "ring-section ring-overview";
  // План загрузки считаем по составу на экране, а не по сохранённому:
  // модель добавляют в кольцо до того, как у неё появятся веса, и строка
  // «Диск» должна отвечать на вопрос «сколько качать под то, что я собрал».
  const requiredGb = draftRequiredDiskGb();
  const enoughSpace = status.free_disk_gb >= requiredGb;
  const rows = [
    ["Профиль железа", `${status.profile} — ${status.profile_description}`],
    ["Runtime", status.provider],
    ["Runtime отвечает", status.provider_healthy ? "да" : "нет"],
    ["Кольцо включено", status.ring_enabled ? "да" : "нет"],
    [
      "Диск",
      `нужно ${requiredGb.toFixed(1)} ГБ, свободно ` +
        `${status.free_disk_gb.toFixed(1)} ГБ`,
    ],
  ];
  section.innerHTML =
    '<div class="ring-section-head"><div>' +
    '<h3>Текущее состояние</h3>' +
    '<p class="ring-section-note">Что сейчас использует приложение</p>' +
    "</div>" +
    `<span class="health health-${status.provider_healthy ? "healthy" : "unavailable"}">` +
    `${status.provider_healthy ? "работает" : "недоступно"}</span></div>` +
    '<dl class="config-list">' +
    rows
      .map(
        ([term, value]) =>
          `<dt>${escapeHtml(term)}</dt><dd>${escapeHtml(value)}</dd>`,
      )
      .join("") +
    "</dl>" +
    (enoughSpace
      ? ""
      : '<p class="notice">Свободного места меньше, чем нужно плану — ' +
        "часть моделей скачать не получится.</p>");
  return section;
}

/*
 * Состав и порядок кольца. Порядок — это порядок обхода при fallback,
 * поэтому порядок задаётся вручную перетаскиванием, а не сортируется сам.
 */
const ringDraft = {
  models: null,
  available: [],
  catalog: new Map(),
  source: "profile",
  status: null,
  progress: new Map(),
  // Модели, загрузка которых началась прямо из чернового состава. После
  // завершения их нужно закрепить в кольце, а не вернуть в дропдаун.
  downloadsToPersist: new Set(),
  // Модель, выбранная в списке кандидатов, — её и добавит «Добавить».
  candidate: null,
  // Раскрыт ли список кандидатов. Панель перерисовывается целиком каждые
  // две секунды, пока идёт загрузка весов, и без этого флага дропдаун
  // схлопывался бы прямо под курсором.
  pickerOpen: false,
};

function renderRingComposition() {
  const section = document.createElement("section");
  section.className = "ring-section ring-composition";

  const heading = document.createElement("div");
  heading.className = "ring-section-head";
  const headingText = document.createElement("div");
  const head = document.createElement("h3");
  head.textContent = "Состав кольца";
  const source = document.createElement("span");
  source.className = "ring-source";
  source.textContent = ringDraft.source === "custom" ? "изменён вручную" : "из профиля";
  headingText.append(head);
  heading.append(headingText, source);
  section.append(heading);

  const hint = document.createElement("p");
  hint.className = "muted tiny";
  hint.textContent =
    ringDraft.source === "custom"
      ? "Состав задан вручную. Порядок сверху вниз — порядок обхода при отказе " +
        "модели. Порядок можно изменить перетаскиванием."
      : "Состав взят из профиля железа. Порядок сверху вниз — порядок обхода при " +
        "отказе модели. Порядок можно изменить перетаскиванием.";
  section.append(hint);

  const list = document.createElement("ul");
  list.className = "config-items";

  ringDraft.models.forEach((model, index) => {
    const info = ringDraft.catalog.get(model);
    const li = document.createElement("li");
    makeRingRowDraggable(li, index);

    const row = document.createElement("div");
    row.className = "config-row";
    row.innerHTML =
      `<span class="order" title="Позиция в порядке обхода">${index + 1}</span>` +
      '<span class="model-identity">' +
      `<strong class="model-name">${escapeHtml(model)}</strong>` +
      `<span class="model-role">${index === 0 ? "Основная модель" : `Резерв ${index}`}</span></span>` +
      '<span class="model-badges">' + originTag(info) +
      `<span class="health health-${info?.installed ? "healthy" : "unavailable"}">` +
      `${info?.installed ? "установлена" : "не установлена"}</span></span>`;

    const controls = document.createElement("span");
    controls.className = "order-controls";
    controls.append(
      ringButton("×", "Убрать из кольца", ringDraft.models.length === 1, () =>
        removeRingModel(index),
      ),
    );
    // Кнопка внутри перетаскиваемой строки: без этого нажатие на крестик
    // начинает drag вместо клика.
    controls.addEventListener("pointerdown", () => {
      li.draggable = false;
    });
    controls.addEventListener("pointerup", () => {
      li.draggable = true;
    });
    row.append(controls);
    li.append(row);
    appendModelDetails(li, info);
    list.append(li);
  });

  section.append(list);

  if (ringDraft.available.length) {
    const addLabel = document.createElement("p");
    addLabel.className = "field-label";
    addLabel.textContent = "Добавить модель в кольцо";
    section.append(addLabel);
    const add = document.createElement("div");
    add.className = "config-row add-row";
    const picker = renderCandidatePicker(ringDraft.available);
    // Одна кнопка: в состав попадает любая модель из каталога, в том числе
    // неустановленная. Скачивание живёт в строке добавленной модели —
    // там видно и размер весов, и прогресс, а сохранить состав по-прежнему
    // можно только когда все модели установлены.
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ghost small";
    button.textContent = "Добавить";
    button.onclick = () => {
      ringDraft.models = [...ringDraft.models, ringDraft.candidate];
      redrawRingSetup();
    };
    add.append(picker, button);
    section.append(add);
  }

  const actions = document.createElement("div");
  actions.className = "row ring-actions";
  const save = document.createElement("button");
  save.type = "button";
  save.className = "primary";
  save.textContent = "Сохранить состав";
  save.onclick = () => saveRingComposition();
  const reset = document.createElement("button");
  reset.type = "button";
  reset.className = "ghost";
  reset.textContent = "Вернуть состав профиля";
  reset.onclick = () => resetRingComposition();
  actions.append(save, reset);
  section.append(actions);

  return section;
}

/*
 * Уведомления настройки кольца живут в самом окне: модалка перекрывает
 * главный экран, и тост об отказе сохранения пользователь там просто не
 * увидит — он решит, что состав сохранён.
 */
function ringNotice(text, kind = "error") {
  const box = $("ring-message");
  clearRingNotice();
  $("ring-message-text").textContent = text;
  box.className = `ring-message ring-message-${kind}`;
  box.hidden = false;
  if (kind === "ok") {
    ringNotice.remaining = 5000;
    scheduleRingNoticeDismissal();
  }
}

function clearRingNotice() {
  const box = $("ring-message");
  clearTimeout(ringNotice.timer);
  clearTimeout(ringNotice.dismissTimer);
  ringNotice.timer = null;
  ringNotice.dismissTimer = null;
  ringNotice.remaining = 0;
  box.classList.remove("ring-message-hiding");
  box.hidden = true;
  $("ring-message-text").textContent = "";
}

function scheduleRingNoticeDismissal() {
  clearTimeout(ringNotice.timer);
  ringNotice.startedAt = Date.now();
  ringNotice.timer = setTimeout(() => {
    const box = $("ring-message");
    box.classList.add("ring-message-hiding");
    ringNotice.dismissTimer = setTimeout(clearRingNotice, 250);
  }, ringNotice.remaining);
}

function pauseRingNoticeDismissal() {
  if (!ringNotice.timer) return;
  ringNotice.remaining = Math.max(
    0,
    ringNotice.remaining - (Date.now() - ringNotice.startedAt),
  );
  clearTimeout(ringNotice.timer);
  ringNotice.timer = null;
}

function resumeRingNoticeDismissal() {
  const box = $("ring-message");
  if (
    box.hidden ||
    !box.classList.contains("ring-message-ok") ||
    box.classList.contains("ring-message-hiding") ||
    ringNotice.timer
  ) {
    return;
  }
  scheduleRingNoticeDismissal();
}


function ringButton(label, title, disabled, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "icon-button";
  button.textContent = label;
  button.title = title;
  button.disabled = disabled;
  button.onclick = onClick;
  return button;
}

/*
 * Перетаскивание строки — единственный способ поменять порядок обхода.
 * Индекс строки хранится в состоянии драга:
 * DOM после каждой перестановки перерисовывается заново.
 */
let ringDragIndex = null;

function makeRingRowDraggable(li, index) {
  li.draggable = true;
  li.classList.add("draggable-row");

  li.addEventListener("dragstart", (event) => {
    ringDragIndex = index;
    li.classList.add("dragging");
    event.dataTransfer.effectAllowed = "move";
    // Firefox не начинает drag без данных в буфере.
    event.dataTransfer.setData("text/plain", String(index));
  });

  li.addEventListener("dragend", () => {
    ringDragIndex = null;
    li.classList.remove("dragging");
  });

  li.addEventListener("dragover", (event) => {
    if (ringDragIndex === null || ringDragIndex === index) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    li.classList.add("drop-target");
  });

  // `dragleave` всплывает и от внутренних элементов строки, поэтому переход
  // курсора с названия на кнопку выглядел бы как уход со строки: подсветка
  // гасла и тут же зажигалась снова — строка мигала под курсором.
  li.addEventListener("dragleave", (event) => {
    if (event.relatedTarget && li.contains(event.relatedTarget)) return;
    li.classList.remove("drop-target");
  });

  li.addEventListener("drop", (event) => {
    event.preventDefault();
    li.classList.remove("drop-target");
    dropRingModel(index);
  });
}

function dropRingModel(target) {
  if (ringDragIndex === null || ringDragIndex === target) return;
  const models = [...ringDraft.models];
  const [moved] = models.splice(ringDragIndex, 1);
  models.splice(target, 0, moved);
  ringDragIndex = null;
  ringDraft.models = models;
  redrawRingSetup();
}

function removeRingModel(index) {
  ringDraft.downloadsToPersist.delete(ringDraft.models[index]);
  ringDraft.models = ringDraft.models.filter((_, i) => i !== index);
  redrawRingSetup();
}

/*
 * Модель профиля и остальные установленные ведут себя по-разному: первую
 * приложение готово скачать и знает её цену, вторая работает только потому,
 * что уже стоит в runtime. Разница видна меткой в составе и
 * разделением списка кандидатов на группы.
 */
function originTag(info) {
  if (!info) return "";
  return info.in_profile
    ? '<span class="tag tag-profile" title="Модель кольца этого профиля ' +
        'железа">профиль</span>'
    : '<span class="tag tag-off-profile" title="Не из кольца этого профиля ' +
        'железа — доступна, потому что установлена в runtime">вне профиля</span>';
}

/*
 * Список кандидатов — свой, а не нативный <select>: раскрытый список
 * оформляется как остальное приложение, показывает метку «вне профиля» и
 * состояние установки у каждой строки, и одинаково выглядит в любом
 * браузере. Нативный список рисует операционная система, и уместить в него
 * что-то кроме плоского текста нельзя.
 */
function renderCandidatePicker(available) {
  if (!available.some((m) => m.model === ringDraft.candidate)) {
    ringDraft.candidate = available[0].model;
  }

  const picker = document.createElement("div");
  picker.className = "picker";

  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "picker-trigger";
  trigger.setAttribute("aria-haspopup", "listbox");
  trigger.setAttribute("aria-expanded", "false");
  trigger.append(candidateLabel(ringDraft.catalog.get(ringDraft.candidate)));

  const menu = document.createElement("div");
  menu.className = "picker-menu";
  menu.setAttribute("role", "listbox");
  menu.hidden = !ringDraft.pickerOpen;
  trigger.setAttribute("aria-expanded", String(ringDraft.pickerOpen));

  const groups = [
    ["Кольцо профиля", available.filter((m) => m.in_profile)],
    ["Остальные модели", available.filter((m) => !m.in_profile)],
  ];
  for (const [label, models] of groups) {
    if (!models.length) continue;
    const head = document.createElement("p");
    head.className = "picker-group";
    head.textContent = label;
    menu.append(head);
    for (const info of models) {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "picker-option";
      option.setAttribute("role", "option");
      option.setAttribute(
        "aria-selected",
        String(info.model === ringDraft.candidate),
      );
      option.dataset.value = info.model;
      option.append(candidateLabel(info));
      option.onclick = () => {
        ringDraft.candidate = info.model;
        closeCandidatePicker();
        redrawRingSetup();
      };
      menu.append(option);
    }
  }

  trigger.onclick = (event) => {
    event.stopPropagation();
    const opening = menu.hidden;
    closeCandidatePicker();
    ringDraft.pickerOpen = opening;
    menu.hidden = !opening;
    trigger.setAttribute("aria-expanded", String(opening));
  };

  picker.append(trigger, menu);
  return picker;
}

function candidateLabel(info) {
  const box = document.createElement("span");
  box.className = "picker-label";
  const name = document.createElement("span");
  name.className = "grow";
  name.textContent = info?.model ?? "";
  box.append(name);
  if (info && !info.in_profile) {
    const tag = document.createElement("span");
    tag.className = "tag tag-off-profile";
    tag.textContent = "вне профиля";
    box.append(tag);
  }
  if (info && !info.installed) {
    const state = document.createElement("span");
    state.className = "health health-unavailable";
    state.textContent = "не установлена";
    box.append(state);
  }
  return box;
}

function closeCandidatePicker() {
  ringDraft.pickerOpen = false;
  for (const menu of document.querySelectorAll(".picker-menu")) {
    menu.hidden = true;
  }
  for (const trigger of document.querySelectorAll(".picker-trigger")) {
    trigger.setAttribute("aria-expanded", "false");
  }
}

/*
 * Перерисовка только секции состава: перезапрашивать всё состояние ради
 * перестановки строки не нужно, а список кандидатов пересобирается здесь же.
 */
function redrawRingSetup() {
  const chosen = new Set(ringDraft.models);
  ringDraft.available = [...ringDraft.catalog.values()].filter(
    (m) => !chosen.has(m.model),
  );
  const sections = $("ring-config").querySelectorAll("section");
  // Состав меняет и план загрузки, поэтому строка «Диск» в обзоре
  // перерисовывается вместе с ним.
  sections[0].replaceWith(renderRingOverview(ringDraft.status));
  sections[1].replaceWith(renderRingComposition());
}

async function saveRingComposition() {
  // Кольцо с неустановленной моделью тратило бы на неё попытку при каждом
  // обходе, поэтому сервер такой состав отклоняет. Проверка здесь — чтобы
  // объяснить это до запроса и не терять черновик.
  const missing = ringDraft.models.filter(
    (model) => !ringDraft.catalog.get(model)?.installed,
  );
  if (missing.length) {
    ringNotice(
      `Состав не сохранён. Не установлены: ${missing.join(", ")}. ` +
        "Скачайте их или уберите из состава — кольцо принимает только " +
        "установленные модели.",
    );
    return;
  }
  try {
    await api("/inference/ring", {
      method: "PUT",
      body: { models: ringDraft.models },
    });
    ringNotice("Состав кольца сохранён", "ok");
  } catch (err) {
    ringNotice(`Состав не сохранён. ${err.message}`);
    return;
  }
  ringDraft.downloadsToPersist.clear();
  ringDraft.models = null;
  loadRingSetup();
  loadRing();
}

async function resetRingComposition() {
  try {
    await api("/inference/ring", { method: "DELETE" });
    ringNotice("Кольцо вернулось к составу профиля", "ok");
  } catch (err) {
    ringNotice(`Состав не изменён. ${err.message}`);
    return;
  }
  ringDraft.downloadsToPersist.clear();
  ringDraft.models = null;
  loadRingSetup();
  loadRing();
}

function renderRuntimes(runtimes) {
  const section = document.createElement("section");
  section.className = "ring-section ring-runtimes";
  const items = runtimes.runtimes
    .map((runtime) => {
      const offer = runtime.installation_offer;
      const command = offer?.manual_command
        ? `<pre class="command">${escapeHtml(offer.manual_command)}</pre>`
        : "";
      return (
        '<li><div class="config-row">' +
        `<span>${escapeHtml(runtime.runtime)}</span>` +
        `<span class="health health-${runtime.available ? "healthy" : "unavailable"}">` +
        `${runtime.available ? "доступен" : "не найден"}</span></div>` +
        (runtime.detail
          ? `<p class="muted tiny">${escapeHtml(runtime.detail)}</p>`
          : "") +
        command +
        "</li>"
      );
    })
    .join("");
  section.innerHTML =
    '<div class="ring-section-head"><div><h3>Runtime</h3>' +
    '<p class="ring-section-note">Техническая информация</p></div></div>' +
    `<p class="muted tiny">Выбран ${escapeHtml(runtimes.selected)}, ` +
    `рекомендуется ${escapeHtml(runtimes.recommended)}. Смена runtime — ` +
    "через переменные окружения, перезапуск обязателен.</p>" +
    `<ul class="config-items">${items}</ul>`;
  return section;
}

/*
 * Цена модели и её загрузка живут в той же строке, что и порядок обхода:
 * отдельный список «Модели плана» показывал ровно тот же состав кольца и
 * расходился с ним, пока черновик не сохранён.
 */
function appendModelDetails(li, info) {
  if (!info) return;

  const note = document.createElement("p");
  note.className = "muted tiny";
  note.textContent = info.in_profile
    ? `${info.family} · ${info.download_size_gb} ГБ · ` +
      `минимум ${info.min_ram_gb} ГБ RAM`
    : `${info.family} · не оценивалась под это железо`;
  li.append(note);

  const download = ringDraft.progress.get(info.model);
  const active = download && ACTIVE_DOWNLOAD_STATES.includes(download.state);
  if (download && download.state !== "completed") {
    const line = document.createElement("p");
    line.className = download.state === "failed" ? "error" : "muted tiny";
    line.textContent =
      `${DOWNLOAD_LABELS[download.state] || download.state} · ` +
      `${download.percent.toFixed(0)}%` +
      (download.error ? ` — ${download.error}` : "");
    li.append(line);
  }

  if (active) {
    // Один подтверждённый запрос останавливает pull и удаляет уже скачанные
    // partial-блоки: пользователь не должен нажимать вторую кнопку следом.
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "ghost small";
    remove.textContent = "Остановить и удалить";
    remove.title = "Остановить скачивание и удалить уже скачанные части";
    remove.onclick = () => deleteModelWeights(info.model, remove);
    li.append(remove);
  } else {
    const actions = document.createElement("div");
    actions.className = "row";

    if (!info.installed) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "ghost small";
      button.textContent = "Скачать";
      // Место проверяем под саму эту модель: общий план кольца может не
      // помещаться на диск целиком, но скачать одну модель из него это не
      // мешает — их ставят по очереди.
      const free = ringDraft.status?.free_disk_gb;
      button.disabled = free !== undefined && free < info.download_size_gb;
      button.onclick = () => downloadModel(info.model, button);
      actions.append(button);
    }

    // Удаление предлагается и для прерванной загрузки: скачанное остаётся
    // на диске, но в списке моделей runtime'а не появляется, и убрать его
    // иначе как вручную было нельзя.
    if (info.installed || (download && download.state !== "completed")) {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "ghost small";
      remove.textContent = "Удалить веса";
      remove.title = info.installed
        ? `Стереть веса ${info.model} из runtime`
        : "Убрать недокачанные веса прерванной загрузки";
      remove.onclick = () => deleteModelWeights(info.model, remove);
      actions.append(remove);
    }

    if (actions.childElementCount) li.append(actions);
  }
}

/*
 * Удаление весов необратимо и стоит времени повторной загрузки, поэтому
 * подтверждается явно, а не срабатывает по одному нажатию.
 */
async function deleteModelWeights(model, button) {
  const ok = await askConfirm(
    `Удалить веса ${model}? Вернуть их можно только повторной загрузкой.`,
  );
  if (!ok) return;
  button.disabled = true;
  let result;
  try {
    result = await api(`/inference/models/${encodeURIComponent(model)}`, {
      method: "DELETE",
    });
  } catch (err) {
    ringNotice(`${model}: удалить не удалось. ${err.message}`);
    button.disabled = false;
    return;
  }
  ringNotice(
    `${model}: ${result.detail}`,
    result.deleted || result.partials_deleted ? "ok" : "muted",
  );
  loadRingSetup();
  loadRing();
}

async function downloadModel(model, button) {
  button.disabled = true;
  try {
    await api(`/inference/models/${encodeURIComponent(model)}/download`, {
      method: "POST",
    });
    ringDraft.downloadsToPersist.add(model);
    ringNotice(`${model}: загрузка запущена`, "ok");
  } catch (err) {
    ringNotice(`${model}: загрузка не запущена. ${err.message}`);
    button.disabled = false;
    return;
  }
  loadRingSetup();
}

async function cancelDownload(model, button) {
  button.disabled = true;
  try {
    await api(
      `/inference/models/${encodeURIComponent(model)}/download/cancel`,
      { method: "POST" },
    );
    ringNotice(
      `${model}: загрузка продолжается в фоне`,
      "ok",
    );
  } catch (err) {
    ringNotice(`${model}: не удалось продолжить в фоне. ${err.message}`);
    button.disabled = false;
    return;
  }
  loadRingSetup();
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
  for (const text of SUGGESTIONS[state.knowledgeBaseName] ?? []) {
    const button = document.createElement("button");
    button.textContent = text;
    button.onclick = () => {
      $("chat-input").value = text;
      $("chat-form").requestSubmit();
    };
    $("suggestions").append(button);
  }
}

// Подсказки — только вопросы по документам, и потому свои у каждой базы:
// вопрос про командировки в базе о доступах вернул бы честный отказ и
// выглядел бы поломкой. База, которой здесь нет (создана пользователем),
// обходится без подсказок — придумать их за него не из чего.
const SUGGESTIONS = {
  "Внутренние документы": [
    "Как компенсируется дежурство в выходной?",
    "Какие лимиты на проживание в командировке?",
    "Сколько длится испытательный срок?",
  ],
  "Безопасность и доступы": [
    "Какой минимальной длины должен быть пароль?",
    "Кому доступна продовая база данных?",
    "Как часто пересматриваются выданные доступы?",
  ],
};

function appendUserMessage(text) {
  const empty = $("messages").querySelector(".empty");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = "msg msg-user";
  div.textContent = text;
  $("messages").append(div);
  scrollMessages();
}

/*
 * Кто отвечает прямо сейчас: первая модель кольца, которая не остывает и
 * не отсутствует. Точного курсора сервер наружу не отдаёт, но порядок
 * обхода и health-состояния известны — этого хватает, чтобы показать имя
 * и заметить переход к следующей модели.
 */
function answeringModel() {
  const member = state.ring.find(
    (m) => m.health !== "cooldown" && m.health !== "unavailable",
  );
  return member?.model || null;
}

function appendPending() {
  const div = document.createElement("div");
  div.className = "msg msg-bot";
  div.append(pendingLine([answeringModel()].filter(Boolean)));
  $("messages").append(div);
  scrollMessages();
  return div;
}

/*
 * Пока модель думает, имя видно целиком, а при отказе строка превращается
 * в цепочку: без неё fallback выглядел бы как необъяснимо долгий ответ
 * чужой модели.
 */
function pendingLine(chain) {
  const span = document.createElement("span");
  span.className = "muted dots";
  if (!chain.length) {
    span.textContent = "Модель отвечает";
    return span;
  }
  const current = chain[chain.length - 1];
  const failed = chain.slice(0, -1);
  span.textContent = failed.length
    ? `${failed.join(", ")} не ответила — отвечает ${current}`
    : `${current} отвечает`;
  return span;
}

/*
 * Ответ идёт десятки секунд, и за это время кольцо может уйти к следующей
 * модели. Опрос состояния во время ожидания показывает это сразу, а не
 * задним числом в строке метаданных под ответом.
 */
function watchAnsweringModel(pending) {
  const chain = [answeringModel()].filter(Boolean);
  const timer = setInterval(async () => {
    await loadRing();
    const current = answeringModel();
    if (!current || current === chain[chain.length - 1]) return;
    chain.push(current);
    pending.replaceChildren(pendingLine(chain));
    scrollMessages();
  }, 3000);
  return () => clearInterval(timer);
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
  const stopWatching = watchAnsweringModel(pending);
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
    stopWatching();
    button.disabled = false;
    loadRing();
  }
}

/* --- Поиск --- */

function resetSearch() {
  // Выдача и ошибка относятся к прежней базе знаний — после переключения
  // они вводят в заблуждение, поэтому очищаются вместе с чатом.
  $("search-input").value = "";
  $("search-results").innerHTML = "";
}

async function search(query) {
  const results = $("search-results");
  results.innerHTML = '<span class="muted dots">Ищем</span>';
  try {
    const data = await api("/search", {
      method: "POST",
      body: { query, knowledge_base_id: state.knowledgeBaseId },
    });
    if (!data.hits.length) {
      results.innerHTML =
        '<span class="muted">В этой базе знаний нет фрагментов по запросу</span>';
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

/*
 * Подсказка с полным текстом для строк, обрезанных многоточием. Нативный
 * title здесь не годится: он появляется с задержкой в секунду и только над
 * самим текстом, а не над всей строкой — длинное имя файла остаётся
 * нечитаемым. Подсказка вешается на любую строку, но показывается только
 * когда текст действительно не поместился.
 */
function showFullTextOnHover(element) {
  const show = () => {
    if (element.scrollWidth <= element.clientWidth) return;
    const tip = document.createElement("div");
    tip.className = "tooltip";
    tip.textContent = element.textContent;
    document.body.append(tip);

    const box = element.getBoundingClientRect();
    tip.style.left = `${Math.max(8, Math.min(box.left, window.innerWidth - tip.offsetWidth - 8))}px`;
    // Над строкой, а если сверху не помещается — под ней.
    const above = box.top - tip.offsetHeight - 6;
    tip.style.top = `${above >= 8 ? above : box.bottom + 6}px`;
    hideFullText.current = tip;
  };

  element.addEventListener("mouseenter", show);
  element.addEventListener("focus", show);
  element.addEventListener("mouseleave", hideFullText);
  element.addEventListener("blur", hideFullText);
  // Список перерисовывается на опросе статусов: подсказку от исчезнувшей
  // строки нужно убрать вместе с ней.
  element.addEventListener("click", hideFullText);
}

function hideFullText() {
  hideFullText.current?.remove();
  hideFullText.current = null;
}

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
  // Состояние runtime'а известно только авторизованному: до входа отметка
  // не показывается, чтобы не обещать локальность непроверенной.
  $("runtime-pill").hidden = !authorized;

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
$("kb-new").addEventListener("click", toggleKnowledgeBaseForm);
$("kb-form").addEventListener("submit", (event) => {
  event.preventDefault();
  createKnowledgeBase();
});
$("ring-setup").addEventListener("click", openRingSetup);
$("ring-close").addEventListener("click", closeRingSetup);
$("ring-message-close").addEventListener("click", clearRingNotice);
$("ring-message").addEventListener("mouseenter", pauseRingNoticeDismissal);
$("ring-message").addEventListener("mouseleave", resumeRingNoticeDismissal);
document.addEventListener("click", (event) => {
  if (!event.target.closest(".picker")) closeCandidatePicker();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeCandidatePicker();
});
$("ring-dialog").addEventListener("click", (event) => {
  // Клик по затемнению закрывает панель, клик внутри карточки — нет.
  if (event.target === $("ring-dialog")) closeRingSetup();
});
$("confirm-ok").addEventListener("click", () => closeConfirm(true));
$("confirm-cancel").addEventListener("click", () => closeConfirm(false));
$("confirm-dialog").addEventListener("click", (event) => {
  if (event.target === $("confirm-dialog")) closeConfirm(false);
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  // Открытое подтверждение перехватывает Escape: иначе за ним закрылась бы
  // и панель кольца, которая может стоять под ним.
  if (confirmResolve) closeConfirm(false);
  else if (!$("ring-dialog").hidden) closeRingSetup();
  else if (!$("kb-form").hidden) closeKnowledgeBaseForm();
});

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
  const query = $("search-input").value.trim();
  if (!query) return;
  search(query);
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

trackRingPointer();
applyTheme(localStorage.getItem(THEME_KEY) || "system");
setInterval(loadRing, 15000);
render();
