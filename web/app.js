/* AI 求职助手 —— 单页前端逻辑（原生 JS，无框架、无构建） */

const $ = (id) => document.getElementById(id);
const state = {
  readiness: null,
  providers: [],
  jobs: [],
  interview: { session: null, history: [], index: 0, rounds: 5, busy: false },
  resumeUploaded: false,
};
let toastTimer = null;

/* ------------------------------------------------------------------ */
/* 工具 */
/* ------------------------------------------------------------------ */
function toast(message, kind = "") {
  const el = $("toast");
  el.textContent = message;
  el.className = "toast " + kind;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 4200);
}

async function api(path, options = {}) {
  const opts = { headers: {}, ...options };
  if (opts.body && !(opts.body instanceof FormData)) {
    opts.headers["Content-Type"] = "application/json";
    if (typeof opts.body !== "string") opts.body = JSON.stringify(opts.body);
  }
  const response = await fetch("/api/v1" + path, opts);
  let data = null;
  try { data = await response.json(); } catch (_) { /* 可能没有 body */ }
  if (!response.ok) {
    const error = new Error((data && (data.error || data.detail)) || `HTTP ${response.status}`);
    error.status = response.status;
    error.data = data;
    throw error;
  }
  return data;
}

function escapeHtml(text) {
  return String(text == null ? "" : text)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function showError(error) {
  if (error.status === 428) {
    const missing = (error.data && error.data.missing_fields) || [];
    toast("还差：" + (missing.length ? missing.join("、") : error.message), "err");
  } else if (error.status === 502) {
    toast("大模型调用失败：" + error.message, "err");
  } else {
    toast(error.message, "err");
  }
  console.error(error);
}

/* ------------------------------------------------------------------ */
/* 步骤与门禁 */
/* ------------------------------------------------------------------ */
const STEPS = ["profile", "market", "kb", "interview", "optimize"];

function currentStep() {
  const r = state.readiness;
  if (!r) return "profile";
  if (!r.llm_ready || !r.profile_ready) return "profile";
  if (!r.market_ready) return "market";
  if (!r.personal_kb_ready) return "kb";
  return "interview";
}

function stepAvailable(step, r) {
  if (step === "profile") return true;
  if (step === "market") return !!r.profile_ready && !!r.llm_ready;
  if (step === "kb") return stepAvailable("market", r) && !!r.market_ready;
  if (step === "interview" || step === "optimize") return stepAvailable("kb", r) && !!r.personal_kb_ready;
  return false;
}

function goto(step, { silent = false } = {}) {
  const r = state.readiness || {};
  if (!stepAvailable(step, r)) {
    if (!silent) {
      const missing = (r.missing_fields || []).join("、");
      toast(missing ? "请先完成前置步骤（还差：" + missing + "）" : "请先完成前置步骤", "err");
    }
    return;
  }
  STEPS.forEach((name) => {
    const section = $("sec-" + name);
    if (section) section.classList.toggle("hidden", name !== step);
  });
  document.querySelectorAll(".step").forEach((button) => {
    const name = button.dataset.step;
    button.classList.toggle("active", name === step);
    button.classList.toggle("locked", !stepAvailable(name, r));
    button.classList.toggle("done", name !== step && stepAvailable(name, r) && STEPS.indexOf(name) < STEPS.indexOf(step));
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderBanner() {
  const r = state.readiness || {};
  const banner = $("banner");
  const blockers = r.blockers || [];
  if (!blockers.length) {
    banner.classList.add("hidden");
    return;
  }
  banner.classList.remove("hidden");
  banner.innerHTML =
    "<b>还有 " + blockers.length + " 项待完成：</b><ul>" +
    blockers.map((item) => "<li>" + escapeHtml(item) + "</li>").join("") +
    "</ul>";
}

function renderLlmChip() {
  const r = state.readiness || {};
  const chip = $("llmChip");
  if (r.llm_ready) {
    chip.className = "chip ok";
    chip.textContent = "✅ 模型已就绪";
  } else {
    chip.className = "chip";
    chip.textContent = "⚠️ 模型未配置，点此设置";
  }
}

/* ------------------------------------------------------------------ */
/* 初始化 */
/* ------------------------------------------------------------------ */
async function refresh() {
  state.readiness = await api("/profile");
  const profile = state.readiness.profile || {};
  fillProfileForm(profile);
  renderBanner();
  renderLlmChip();
  renderTargetJob(profile);
  const step = currentStep();
  goto(step, { silent: true });
  document.querySelectorAll(".step").forEach((button) => {
    const name = button.dataset.step;
    button.classList.toggle("locked", !stepAvailable(name, state.readiness));
  });
}

function fillProfileForm(profile) {
  const map = {
    p_role: "target_role", p_city: "expect_city",
    p_salary_min: "expect_salary_min", p_salary_max: "expect_salary_max",
    p_years: "experience_years", p_education: "education", p_notes: "extra_notes",
  };
  Object.entries(map).forEach(([id, key]) => {
    const value = profile[key];
    if (value === undefined || value === null) return;
    $(id).value = ["expect_salary_min", "expect_salary_max"].includes(key) && !value ? "" : value;
  });
  const skills = profile.skills || [];
  if (skills.length) $("p_skills").value = skills.join(", ");

  if (profile.resume_id) {
    $("resumeStatus").className = "hint ok";
    $("resumeStatus").textContent = "✅ 已上传简历（" + (profile.resume_id || "").slice(0, 8) + "…）";
    state.resumeUploaded = true;
  }
  const missing = state.readiness.missing_fields || [];
  $("profileMissing").textContent = missing.length ? "还差：" + missing.join("、") : "✅ 已填写完整";
}

function renderTargetJob(profile) {
  const box = $("targetJobBox");
  const job = profile.target_job || {};
  if (job.title) {
    box.className = "targetjob";
    box.innerHTML =
      "🎯 <b>目标岗位：</b>" + escapeHtml(job.title) +
      (job.company ? " @ " + escapeHtml(job.company) : "") +
      (job.city ? "　📍" + escapeHtml(job.city) : "") +
      (job.salary_text ? "　💰" + escapeHtml(job.salary_text) : "") +
      '<div class="muted small" style="margin-top:6px">' +
      "知识库与面试题都会围绕这个岗位生成。</div>";
  } else {
    box.className = "targetjob empty";
    box.innerHTML = "还没有选定目标岗位。请先到「抓岗与薪资」抓取，再从结果里选一个。";
  }
  const personal = $("kbStatus");
  if (personal && state.readiness) {
    personal.textContent = state.readiness.personal_kb_ready
      ? `知识库已就绪（${state.readiness.personal_kb_chunks} 块）`
      : "知识库尚未生成";
  }
}

/* ------------------------------------------------------------------ */
/* 档案 */
/* ------------------------------------------------------------------ */
async function saveProfile() {
  const skills = $("p_skills").value.split(/[,，、]/).map((s) => s.trim()).filter(Boolean);
  const payload = {
    target_role: $("p_role").value.trim(),
    expect_city: $("p_city").value.trim(),
    expect_salary_min: parseFloat($("p_salary_min").value) || 0,
    expect_salary_max: parseFloat($("p_salary_max").value) || 0,
    experience_years: parseFloat($("p_years").value) || 0,
    education: $("p_education").value.trim(),
    extra_notes: $("p_notes").value.trim(),
    skills: skills,
  };
  if (!payload.target_role || !payload.expect_city || !payload.expect_salary_min) {
    toast("求职方向、期望城市、期望月薪下限是必填项", "err");
    return;
  }
  try {
    state.readiness = await api("/profile", { method: "PUT", body: payload });
    fillProfileForm(state.readiness.profile);
    renderBanner();
    renderTargetJob(state.readiness.profile);
    toast("档案已保存", "ok");
    if (state.readiness.profile_ready && state.readiness.llm_ready) goto("market");
  } catch (error) { showError(error); }
}

async function uploadResume() {
  const input = $("resumeFile");
  if (!input.files || !input.files.length) { toast("请先选择简历文件", "err"); return; }
  const form = new FormData();
  form.append("file", input.files[0]);
  $("resumeStatus").textContent = "正在上传解析…";
  try {
    const result = await api("/resume/upload", { method: "POST", body: form });
    $("resumeStatus").className = "hint ok";
    $("resumeStatus").textContent = `✅ 已上传「${result.filename}」，解析出 ${result.chars} 字`;
    $("resumePreviewBox").classList.remove("hidden");
    $("resumePreview").textContent = result.preview || "";
    state.readiness = result.state;
    renderBanner(); fillProfileForm(state.readiness.profile); renderLlmChip();
    toast("简历上传成功", "ok");
  } catch (error) {
    $("resumeStatus").className = "hint";
    $("resumeStatus").textContent = "上传失败";
    showError(error);
  }
}

/* ------------------------------------------------------------------ */
/* 抓岗与薪资 */
/* ------------------------------------------------------------------ */
async function crawl() {
  const button = $("btnCrawl");
  button.disabled = true;
  $("crawlProgress").classList.remove("hidden");
  $("crawlStatus").textContent = "正在抓取，可能需要 20~60 秒…";
  try {
    const report = await api("/market/crawl", { method: "POST", body: {} });
    renderSalary(report);
    renderJobs(report.jobs || []);
    state.readiness = await api("/profile");
    renderBanner();
    const total = report.total_jobs || 0;
    $("crawlStatus").textContent =
      `✅ 抓取完成：${total} 个岗位（${report.with_salary} 个有薪资），用时 ${(report.elapsed_ms / 1000).toFixed(1)}s`;
    toast(`抓到 ${total} 个岗位`, "ok");
  } catch (error) {
    $("crawlStatus").textContent = "抓取失败";
    showError(error);
  } finally {
    button.disabled = false;
    $("crawlProgress").classList.add("hidden");
  }
}

function renderSalary(report) {
  const box = $("salaryCards");
  const salary = report.salary || {};
  if (!salary.sample_size) {
    box.innerHTML =
      '<div class="card"><div class="label">薪资样本</div><div class="value">0<small>条</small></div></div>' +
      '<div class="card"><div class="label">说明</div>' +
      '<div class="small muted" style="margin-top:6px">抓到的岗位没有公开薪资，无法统计</div></div>';
  } else {
    const m = salary.monthly;
    box.innerHTML = [
      ["月薪中位", m.median, "K"],
      ["月薪平均", m.mean, "K"],
      ["P25", m.p25, "K"],
      ["P75", m.p75, "K"],
      ["年薪中位", salary.annual.median, "K"],
      ["样本量", salary.sample_size, "条"],
    ].map(([label, value, unit], index) =>
      `<div class="card${index === 0 ? " hl" : ""}"><div class="label">${label}</div>` +
      `<div class="value">${value}<small>${unit}</small></div></div>`
    ).join("");
  }

  const advice = $("salaryAdvice");
  const expectation = salary.expectation;
  if (expectation) {
    advice.classList.remove("hidden");
    advice.innerHTML =
      `<b>期望对比：${escapeHtml(expectation.verdict)}</b>　` +
      `你的期望中位 ${expectation.expected_mid_k}K　|　市场 ${expectation.market_median_k}K　` +
      `（${expectation.gap_k > 0 ? "+" : ""}${expectation.gap_k}K / ${expectation.gap_ratio}%）` +
      `<div class="muted small" style="margin-top:6px">${escapeHtml(expectation.advice)}</div>`;
  } else {
    advice.classList.add("hidden");
  }

  const warnings = $("crawlWarnings");
  if ((report.warnings || []).length) {
    warnings.classList.remove("hidden");
    warnings.innerHTML = "<b>注意：</b><ul>" +
      report.warnings.map((w) => "<li>" + escapeHtml(w) + "</li>").join("") + "</ul>";
  } else {
    warnings.classList.add("hidden");
  }

  const sourceBox = $("crawlWarnings");
  const sources = report.sources || [];
  if (sources.length) {
    sourceBox.classList.remove("hidden");
    const items = sources.map((s) => {
      const mark = s.ok ? "✅" : "❌";
      const detail = s.ok ? `抓取 ${s.fetched} 条` : escapeHtml(s.error || "失败");
      return `<li>${mark} <b>${escapeHtml(s.label)}</b>：${detail}（${s.elapsed_ms}ms）</li>`;
    }).join("");
    sourceBox.innerHTML = (sourceBox.innerHTML || "") +
      "<b>数据来源：</b><ul>" + items + "</ul>";
  }
}

function renderJobs(jobs) {
  state.jobs = jobs;
  const box = $("jobList");
  $("jobCount").textContent = jobs.length ? `共 ${jobs.length} 条（按相关度排序）` : "";
  if (!jobs.length) {
    box.innerHTML = '<div class="muted">没有职位名匹配到你的求职方向。可以换个更通用的方向词再试。</div>';
    return;
  }
  box.innerHTML = jobs.map((job, index) => `
    <div class="job">
      <div class="jt">${escapeHtml(job.title)}</div>
      <div class="jm">
        ${job.company ? "<span>🏢 " + escapeHtml(job.company) + "</span>" : ""}
        ${job.city ? "<span>📍 " + escapeHtml(job.city) + "</span>" : ""}
        <span class="salary">💰 ${escapeHtml(job.salary_text || "未公开薪资")}</span>
        <span>${escapeHtml(job.source || "")}</span>
        ${job.published_at ? "<span>🕒 " + escapeHtml(job.published_at) + "</span>" : ""}
      </div>
      ${(job.tags || []).length ? '<div class="tags">' +
        job.tags.slice(0, 12).map((t) => '<span class="tag">' + escapeHtml(t) + "</span>").join("") +
        "</div>" : ""}
      ${(job.matched_keywords || []).length ? '<div class="tags">' +
        job.matched_keywords.map((t) => '<span class="tag hit">命中 ' + escapeHtml(t) + "</span>").join("") +
        "</div>" : ""}
      <div class="acts">
        <button class="btn sm primary" data-target="${index}">🎯 设为目标岗位</button>
        ${job.url ? `<a class="btn sm ghost" href="${escapeHtml(job.url)}" target="_blank" rel="noopener">🔗 原文</a>` : ""}
        ${job.jd_text ? `<button class="btn sm ghost" data-jd="${index}">📄 查看 JD</button>` : ""}
      </div>
      <div class="jd hidden" id="jd-${index}"><pre>${escapeHtml(job.jd_text || "")}</pre></div>
    </div>`).join("");

  box.querySelectorAll("[data-target]").forEach((button) => {
    button.addEventListener("click", () => chooseTarget(jobs[parseInt(button.dataset.target, 10)]));
  });
  box.querySelectorAll("[data-jd]").forEach((button) => {
    button.addEventListener("click", () => $("jd-" + button.dataset.jd).classList.toggle("hidden"));
  });
}

async function chooseTarget(job) {
  try {
    const result = await api("/market/target", { method: "POST", body: { job_key: job.job_key } });
    state.readiness = result.state;
    renderTargetJob(state.readiness.profile);
    renderBanner();
    toast("已设为目标岗位：" + job.title, "ok");
    goto("kb");
  } catch (error) { showError(error); }
}

async function loadExistingJobs() {
  try {
    const data = await api("/market/jobs?limit=100");
    if (data.jobs && data.jobs.length) {
      renderJobs(data.jobs);
      if (data.market && data.market.sample_size) {
        renderSalary({ salary: data.market, sources: [], warnings: [] });
      }
    }
  } catch (error) { /* 首次运行时没有数据是正常的 */ }
}

/* ------------------------------------------------------------------ */
/* 知识库 */
/* ------------------------------------------------------------------ */
async function buildKb() {
  const button = $("btnBuildKb");
  button.disabled = true;
  $("kbStatus").textContent = "正在生成…";
  try {
    const report = await api("/knowledge/personal/build", { method: "POST", body: {} });
    state.readiness = report.state;
    $("kbReport").classList.remove("hidden");
    $("kbReport").innerHTML =
      `<b>✅ 知识库已生成：${report.chunks} 个文本块</b><ul>` +
      `<li>简历：${report.resume_chunks} 份</li>` +
      `<li>目标岗位 JD：${report.target_job_chunks} 份</li>` +
      `<li>同类岗位 JD：${report.similar_job_chunks} 份</li>` +
      `<li>相关面试题：${report.question_chunks} 道</li>` +
      `<li>耗时：${report.seconds}s</li></ul>` +
      ((report.notes || []).length ? "<div class='muted small' style='margin-top:8px'>" +
        report.notes.map(escapeHtml).join("<br>") + "</div>" : "");
    renderBanner();
    renderTargetJob(state.readiness.profile);
    toast("专属知识库已生成", "ok");
  } catch (error) {
    $("kbStatus").textContent = "生成失败";
    showError(error);
  } finally {
    button.disabled = false;
  }
}

/* ------------------------------------------------------------------ */
/* 模拟面试 */
/* ------------------------------------------------------------------ */
function pushMessage(role, text) {
  state.interview.history.push({ role, text });
  const chat = $("ivChat");
  const div = document.createElement("div");
  div.className = "msg " + (role === "assistant" ? "interviewer" : "me");
  div.innerHTML =
    '<div class="who">' + (role === "assistant" ? "【面试官】" : "【你】") + "</div>" +
    '<div class="body">' + escapeHtml(text) + "</div>";
  chat.appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function startInterview() {
  const button = $("btnStartInterview");
  button.disabled = true;
  $("ivStatus").textContent = "正在根据你的知识库出题…";
  $("ivChat").innerHTML = "";
  $("ivReport").classList.add("hidden");
  state.interview.history = [];
  try {
    const rounds = parseInt($("iv_rounds").value, 10) || 5;
    const result = await api("/interview/start", { method: "POST", body: { rounds } });
    state.interview.rounds = rounds;
    state.interview.index = 0;
    pushMessage("assistant", result.questions);
    $("ivAnswerBox").classList.remove("hidden");
    $("ivStatus").textContent = "面试进行中";
    updateProgress();
  } catch (error) {
    $("ivStatus").textContent = "";
    showError(error);
  } finally {
    button.disabled = false;
  }
}

function updateProgress() {
  $("ivProgress").textContent =
    `已提交 ${state.interview.index} / ${state.interview.rounds} 题`;
}

async function submitAnswer() {
  const box = $("iv_answer");
  const answer = box.value.trim();
  if (!answer) { toast("请先输入你的回答", "err"); return; }

  pushMessage("user", answer);
  box.value = "";            // 提交后立刻清空，避免残留上一轮答案
  const button = $("btnSubmitAnswer");
  button.disabled = true;
  try {
    const history = state.interview.history.map((m) => ({ role: m.role, content: m.text }));
    const result = await api("/interview/answer", {
      method: "POST",
      body: { history, rounds: state.interview.rounds },
    });
    pushMessage("assistant", result.reply);
    state.interview.index += 1;
    updateProgress();
    if (result.memory && result.memory.summary_chars) {
      $("ivStatus").textContent =
        `长上下文：窗口 ${result.memory.window_messages} 条原文 + 摘要 ${result.memory.summary_chars} 字`;
    }
  } catch (error) {
    box.value = answer;      // 失败时把内容还给用户，别让他白打
    showError(error);
  } finally {
    button.disabled = false;
  }
}

async function generateReport() {
  if (state.interview.history.length < 2) { toast("至少回答一题再生成报告", "err"); return; }
  $("ivStatus").textContent = "正在生成报告…";
  try {
    const history = state.interview.history.map((m) => ({ role: m.role, content: m.text }));
    const result = await api("/interview/report", { method: "POST", body: { history } });
    $("ivReport").classList.remove("hidden");
    $("ivReport").textContent = result.content;
    $("ivStatus").textContent = "";
  } catch (error) { showError(error); }
}

function clearInterview() {
  state.interview = { session: null, history: [], index: 0, rounds: 5, busy: false };
  $("ivChat").innerHTML = "";
  $("ivReport").classList.add("hidden");
  $("ivAnswerBox").classList.add("hidden");
  $("ivStatus").textContent = "";
}

/* ------------------------------------------------------------------ */
/* 简历优化 */
/* ------------------------------------------------------------------ */
async function optimize() {
  const button = $("btnOptimize");
  button.disabled = true;
  $("optStatus").textContent = "AI 正在对照目标岗位优化…";
  try {
    const result = await api("/resume/optimize", { method: "POST", body: {} });
    $("optResult").classList.remove("hidden");
    $("optResult").textContent = result.content;
    $("optStatus").textContent =
      `完成（可信度 ${(result.confidence * 100).toFixed(0)}%）`;
    if (result.sources && result.sources.length) {
      $("optResult").textContent += "\n\n=== 引用来源 ===\n" +
        result.sources.map((s) => `[${s.index}] ${s.label}`).join("\n");
    }
  } catch (error) {
    $("optStatus").textContent = "";
    showError(error);
  } finally {
    button.disabled = false;
  }
}

/* ------------------------------------------------------------------ */
/* 模型设置 */
/* ------------------------------------------------------------------ */
async function openSettings() {
  try {
    const data = await api("/settings/llm");
    state.providers = data.providers || [];
    const config = data.config || {};
    $("s_provider").innerHTML = state.providers
      .map((p) => `<option value="${p.key}">${escapeHtml(p.label)}</option>`).join("");
    $("s_provider").value = config.provider || "deepseek";
    $("s_base_url").value = config.base_url || "";
    $("s_model").value = config.model || "";
    $("s_temperature").value = config.temperature ?? 0.2;
    $("s_max_tokens").value = config.max_tokens ?? 2048;
    $("keyHint").textContent = config.key_masked
      ? "当前已保存：" + config.key_masked + "（留空表示不修改）"
      : "还没有配置 Key";
    applyProvider();
    $("verifyResult").classList.add("hidden");
    $("settingsModal").classList.remove("hidden");
  } catch (error) { showError(error); }
}

function applyProvider() {
  const provider = state.providers.find((p) => p.key === $("s_provider").value);
  if (!provider) return;
  if (provider.base_url) $("s_base_url").value = provider.base_url;
  $("s_model").value = provider.default_model || "";
  $("modelList").innerHTML = (provider.models || [])
    .map((m) => `<option value="${escapeHtml(m)}">`).join("");
  $("providerNote").textContent = provider.note || "";
  $("s_api_key").placeholder = provider.api_key_hint || "sk-...";
}

async function saveLlm(verify = true) {
  const payload = {
    provider: $("s_provider").value,
    base_url: $("s_base_url").value.trim(),
    model: $("s_model").value.trim(),
    temperature: parseFloat($("s_temperature").value),
    max_tokens: parseInt($("s_max_tokens").value, 10),
    persist: true,
    verify: verify,
  };
  const key = $("s_api_key").value.trim();
  if (key) payload.api_key = key;
  try {
    const result = await api("/settings/llm", { method: "PUT", body: payload });
    renderVerify(result.verify);
    if (result.verify && result.verify.ok) {
      state.readiness = await api("/profile");
      renderLlmChip(); renderBanner();
      toast("模型已配置并验证通过", "ok");
    }
  } catch (error) { showError(error); }
}

async function verifyLlm() {
  try {
    const result = await api("/settings/llm/verify", { method: "POST", body: {} });
    renderVerify(result);
  } catch (error) { showError(error); }
}

function renderVerify(result) {
  const box = $("verifyResult");
  if (!result) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  if (result.ok) {
    box.className = "ok";
    box.textContent = `✅ 验证通过，模型回复：${result.reply_preview || "(空)"}`;
  } else {
    box.className = "err";
    box.textContent = "❌ 验证失败：" + (result.error || "未知错误");
  }
}

async function clearLlm() {
  try {
    await api("/settings/llm/key", { method: "DELETE" });
    $("s_api_key").value = "";
    $("keyHint").textContent = "已清除";
    state.readiness = await api("/profile");
    renderLlmChip(); renderBanner();
    toast("已清除 API Key", "ok");
  } catch (error) { showError(error); }
}

/* ------------------------------------------------------------------ */
/* 岗位来源 */
/* ------------------------------------------------------------------ */
async function openSources() {
  try {
    const data = await api("/sources");
    renderSources(data.sources || []);
    $("sourcesModal").classList.remove("hidden");
  } catch (error) { showError(error); }
}

function renderSources(sources) {
  $("sourcesList").innerHTML = sources.map((source, index) => `
    <div class="source" data-index="${index}">
      <div class="head">
        <input type="checkbox" class="s-enabled" ${source.enabled !== false ? "checked" : ""}>
        <input type="text" class="s-label" value="${escapeHtml(source.label || "")}" style="flex:1">
        <span class="tag">${escapeHtml(source.kind || "")}</span>
      </div>
      <input type="text" class="s-url" value="${escapeHtml(
        (source.config && source.config.list_url) || source.careers_url || "")}"
        placeholder="职位列表页 URL">
      <div class="sub">
        每行一个来源：勾选启用；自定义官网请填列表页地址，
        程序会用系统自带的 Edge 自动渲染并识别岗位卡片。
      </div>
    </div>`).join("");
}

async function saveSources() {
  try {
    const data = await api("/sources");
    const sources = data.sources || [];
    document.querySelectorAll("#sourcesList .source").forEach((node) => {
      const index = parseInt(node.dataset.index, 10);
      const source = sources[index];
      if (!source) return;
      source.enabled = node.querySelector(".s-enabled").checked;
      source.label = node.querySelector(".s-label").value.trim() || source.label;
      const url = node.querySelector(".s-url").value.trim();
      if (source.kind === "generic_render") {
        source.config = source.config || {};
        source.config.list_url = url;
        source.careers_url = url;
      } else {
        source.careers_url = url;
      }
    });
    await api("/sources", { method: "PUT", body: { sources } });
    toast("来源已保存", "ok");
    $("sourcesModal").classList.add("hidden");
  } catch (error) { showError(error); }
}

function addSource() {
  const container = $("sourcesList");
  const index = container.querySelectorAll(".source").length;
  const div = document.createElement("div");
  div.className = "source";
  div.dataset.index = String(1000 + index);
  div.innerHTML = `
    <div class="head">
      <input type="checkbox" class="s-enabled" checked>
      <input type="text" class="s-label" value="自定义官网" style="flex:1">
      <span class="tag">generic_render</span>
    </div>
    <input type="text" class="s-url" placeholder="https://company.com/careers">
    <div class="sub">保存后会自动分配 key 并启用。</div>`;
  container.appendChild(div);
}

/* ------------------------------------------------------------------ */
/* 事件绑定 */
/* ------------------------------------------------------------------ */
function bind() {
  document.querySelectorAll(".step").forEach((button) => {
    button.addEventListener("click", () => goto(button.dataset.step));
  });
  $("btnSaveProfile").addEventListener("click", saveProfile);
  $("btnUploadResume").addEventListener("click", uploadResume);
  $("btnCrawl").addEventListener("click", crawl);
  $("btnSources").addEventListener("click", openSources);
  $("btnBuildKb").addEventListener("click", buildKb);
  $("btnKbStats").addEventListener("click", async () => {
    const stats = await api("/knowledge/personal");
    toast(stats.ready ? `知识库有 ${stats.chunks} 个文本块` : "知识库尚未生成");
  });
  $("btnStartInterview").addEventListener("click", startInterview);
  $("btnSubmitAnswer").addEventListener("click", submitAnswer);
  $("btnReport").addEventListener("click", generateReport);
  $("btnClearInterview").addEventListener("click", clearInterview);
  $("btnOptimize").addEventListener("click", optimize);
  $("btnCopyOptimize").addEventListener("click", async () => {
    const text = $("optResult").textContent;
    if (!text) { toast("还没有优化结果", "err"); return; }
    await navigator.clipboard.writeText(text);
    toast("已复制到剪贴板", "ok");
  });
  $("llmChip").addEventListener("click", openSettings);
  $("btnSettings").addEventListener("click", openSettings);
  $("btnCloseSettings").addEventListener("click", () => $("settingsModal").classList.add("hidden"));
  $("s_provider").addEventListener("change", applyProvider);
  $("btnSaveLlm").addEventListener("click", () => saveLlm(true));
  $("btnVerifyLlm").addEventListener("click", verifyLlm);
  $("btnClearLlm").addEventListener("click", clearLlm);
  $("btnCloseSources").addEventListener("click", () => $("sourcesModal").classList.add("hidden"));
  $("btnSaveSources").addEventListener("click", saveSources);
  $("btnAddSource").addEventListener("click", addSource);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") document.querySelectorAll(".modal").forEach((m) => m.classList.add("hidden"));
  });
}

async function boot() {
  bind();
  try {
    const system = await api("/system");
    $("appVersion").textContent = "v" + system.version;
    $("cityList").innerHTML = [
      "北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京", "西安",
      "苏州", "天津", "重庆", "长沙", "郑州", "青岛", "合肥", "厦门", "远程",
    ].map((c) => `<option value="${c}">`).join("");
  } catch (error) { console.warn(error); }

  try {
    await refresh();
    await loadExistingJobs();
  } catch (error) { showError(error); }

  if (!state.readiness || !state.readiness.llm_ready) {
    setTimeout(openSettings, 400);
  }
}

boot();
