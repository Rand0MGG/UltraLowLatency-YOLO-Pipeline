const state = {
  config: {},
  defaults: {},
  models: [],
  fieldGroups: [],
  polling: null,
  language: "zh",
  configPath: "",
  lastStatus: null,
  logLines: [],
  perfHistory: [],
  chartsDirty: false,
  chartLoopStarted: false,
};

const MAX_LOG_LINES = 220;
const MAX_PERF_SAMPLES = 90;
const throughputSeries = [
  { key: "grab", color: "#c96442", label: { en: "Grab FPS", zh: "采集 FPS" } },
  { key: "proc", color: "#1f6f78", label: { en: "Proc FPS", zh: "推理 FPS" } },
  { key: "cmd", color: "#5e5d59", label: { en: "Cmd Hz", zh: "目标 Hz" } },
];
const dropSeries = [
  { key: "drop", color: "#d97757", label: { en: "Drop FPS", zh: "掉帧 FPS" } },
];
const latencySeries = [
  { key: "wait", color: "#6c5ce7", label: { en: "Wait", zh: "等待" } },
  { key: "total", color: "#2d3436", label: { en: "Total", zh: "总时延" } },
  { key: "pre", color: "#00b894", label: { en: "Pre", zh: "预处理" } },
  { key: "gpu", color: "#0984e3", label: { en: "GPU", zh: "推理" } },
  { key: "post", color: "#fdcb6e", label: { en: "Post", zh: "后处理" } },
  { key: "ovhd", color: "#e17055", label: { en: "Ovhd", zh: "额外开销" } },
];

const translations = {
  en: {
    eyebrow: "Local Control Panel",
    headerDescription: "Tune runtime parameters, persist them to a config file, then start or stop the detection pipeline from one place.",
    configPath: "Config file",
    saveAndStart: "Save and Start",
    saveConfig: "Save Config",
    stop: "Stop",
    resetDefaults: "Reset Defaults",
    tabConfig: "Config",
    tabStatus: "Status",
    tabLogs: "Logs",
    configCardLabel: "Configuration",
    configTitle: "Parameters",
    configDescription: "Edit the saved config and launch behavior here. Hover a field label to see what that parameter does.",
    statusCardLabel: "Runtime Status",
    statusTitle: "Session",
    statusDescription: "Check whether the pipeline is running and which config file it was started with.",
    logsCardLabel: "Logs",
    logsTitle: "Output",
    logsDescription: "Performance FLOW lines are rendered as charts below. The text log keeps only non-FLOW messages.",
    chartThroughputTitle: "Throughput",
    chartLatencyTitle: "Latency",
    chartDropTitle: "Drop FPS",
    statusPid: "PID",
    statusStarted: "Started",
    statusExit: "Exit Code",
    statusConfig: "Config File",
    cliPreview: "CLI Preview",
    idle: "Idle",
    running: "Running",
    tooltipAria: "Parameter help",
    chartWaiting: "Waiting for performance logs...",
    warning(limit) {
      return `This config will auto-stop after ${limit} second${limit === 1 ? "" : "s"}. Set Max Run Seconds to 0 for continuous running.`;
    },
    previewUnavailable(error) {
      return `Preview unavailable: ${error}`;
    },
    fetchStatusFailed(error) {
      return `Failed to fetch status: ${error}`;
    },
    configSaved: "Config saved.",
    pipelineStarted: "Pipeline started.",
    stopSent: "Stop signal sent.",
    defaultsRestored: "Defaults restored.",
    bootstrapFailed(error) {
      return `Bootstrap failed: ${error}`;
    },
    groupLabels: {
      Runtime: "Runtime",
      Detection: "Detection",
      ROI: "ROI",
      "Auto Capture": "Auto Capture",
      "Output & Smoothing": "Output & Smoothing",
    },
    groupDescriptions: {
      "Basic capture and execution settings.": "Basic capture and execution settings.",
      "Model inference parameters and class filters.": "Model inference parameters and class filters.",
      "Region-of-interest cropping controls.": "Region-of-interest cropping controls.",
      "PID-based capture-rate adaptation.": "PID-based capture-rate adaptation.",
      "Saving, stats, and box smoothing controls.": "Saving, stats, and box smoothing controls.",
    },
  },
  zh: {
    eyebrow: "本地控制台",
    headerDescription: "在这里调整运行参数、保存到配置文件，并直接启动或停止检测流程。",
    configPath: "配置文件",
    saveAndStart: "保存并启动",
    saveConfig: "保存配置",
    stop: "停止运行",
    resetDefaults: "恢复默认",
    tabConfig: "配置",
    tabStatus: "状态",
    tabLogs: "日志",
    configCardLabel: "配置",
    configTitle: "参数设置",
    configDescription: "在这里编辑持久化配置和启动参数。把鼠标悬停在字段旁边的问号上可以查看说明。",
    statusCardLabel: "运行状态",
    statusTitle: "当前会话",
    statusDescription: "查看程序是否正在运行，以及本次启动使用的是哪个配置文件。",
    logsCardLabel: "日志",
    logsTitle: "输出日志",
    logsDescription: "FLOW 性能日志会显示为下方折线图，文本区域仅保留其他运行日志。",
    chartThroughputTitle: "吞吐趋势",
    chartLatencyTitle: "时延趋势",
    chartDropTitle: "掉帧趋势",
    statusPid: "进程号",
    statusStarted: "启动时间",
    statusExit: "退出码",
    statusConfig: "配置文件",
    cliPreview: "命令预览",
    idle: "空闲",
    running: "运行中",
    tooltipAria: "参数说明",
    chartWaiting: "等待性能日志...",
    warning(limit) {
      return `当前配置会在 ${limit} 秒后自动停止。将 Max Run Seconds 设为 0 可持续运行。`;
    },
    previewUnavailable(error) {
      return `命令预览不可用：${error}`;
    },
    fetchStatusFailed(error) {
      return `获取状态失败：${error}`;
    },
    configSaved: "配置已保存。",
    pipelineStarted: "检测流程已启动。",
    stopSent: "已发送停止信号。",
    defaultsRestored: "已恢复默认配置。",
    bootstrapFailed(error) {
      return `初始化失败：${error}`;
    },
    groupLabels: {
      Runtime: "运行",
      Detection: "检测",
      ROI: "区域裁剪",
      "Auto Capture": "自动调速",
      "Output & Smoothing": "输出与平滑",
    },
    groupDescriptions: {
      "Basic capture and execution settings.": "基础采集与运行设置。",
      "Model inference parameters and class filters.": "模型推理参数与类别过滤设置。",
      "Region-of-interest cropping controls.": "感兴趣区域裁剪设置。",
      "PID-based capture-rate adaptation.": "基于 PID 的采集频率自适应调节。",
      "Saving, stats, and box smoothing controls.": "截图保存、统计输出和框体平滑设置。",
    },
  },
};

const fieldTranslations = {
  zh: {
    monitor_index: { label: "显示器编号", help: "选择要抓取的显示器。大多数情况下主屏为 1。" },
    capture_hz: { label: "采集频率", help: "目标抓屏频率。越高越流畅，但 CPU 和 GPU 负载也会更高。" },
    model: { label: "模型文件", help: "用于 YOLO26 姿态推理的权重或 TensorRT engine 文件。" },
    device: { label: "推理设备", help: "指定推理设备，例如 cuda:0 表示第一张显卡，cpu 表示使用处理器。" },
    visualize: { label: "显示悬浮层", help: "在屏幕上显示透明叠加层，把检测框和关键点实时绘制出来。" },
    half: { label: "半精度 FP16", help: "在支持的 CUDA 设备上使用半精度推理，通常更省显存也更快。" },
    end2end: { label: "端到端推理", help: "使用 YOLO26 end-to-end / NMS-free 推理。关闭后可测试传统 NMS 路径。" },
    max_run_seconds: { label: "最长运行秒数", help: "超过这个时间后自动停止。设为 0 表示持续运行直到手动停止。" },
    conf: { label: "置信度阈值", help: "最低检测置信度。数值越高越严格，但可能漏检。" },
    iou: { label: "IoU 阈值", help: "NMS 合并重叠检测框时使用的 IoU 阈值。" },
    imgsz: { label: "输入尺寸", help: "推理图像尺寸。越大通常越准，但延迟也更高。" },
    yolo_classes: { label: "类别过滤", help: "按类别过滤结果。留空表示不过滤，默认 0 表示只识别人。" },
    yolo_max_det: { label: "最大检测数", help: "每一帧最多保留多少个检测结果。" },
    roi_square: { label: "启用中心方形 ROI", help: "只抓取屏幕中央的正方形区域，减少计算量。" },
    roi_radius_px: { label: "ROI 半径", help: "中心 ROI 的半径像素值。越小表示裁剪区域越紧。" },
    auto_capture: { label: "启用自动调速", help: "根据运行负载自动调整抓屏频率，减少堆积和延迟。" },
    auto_capture_warmup_s: { label: "预热时间", help: "启动后等待多久再开始自动调速。" },
    cap_min_hz: { label: "最低抓屏频率", help: "自动调速允许降低到的最小抓屏频率。" },
    cap_max_hz: { label: "最高抓屏频率", help: "自动调速允许提升到的最大抓屏频率。" },
    target_drop_fps: { label: "目标掉帧", help: "控制器希望维持的掉帧水平。" },
    deadband: { label: "死区", help: "误差落在这个范围内时不做调整，避免来回抖动。" },
    kp: { label: "Kp 比例项", help: "PID 比例系数，决定对当前误差的响应强度。" },
    ki: { label: "Ki 积分项", help: "PID 积分系数，决定对累计误差的响应强度。" },
    integral_limit: { label: "积分上限", help: "限制积分项大小，避免自动调速过冲。" },
    min_apply_delta_hz: { label: "最小调整步长", help: "频率变化至少达到这个值才会真正应用。" },
    save_every: { label: "保存间隔（秒）", help: "按固定时间间隔自动保存带结果的截图。设为 0 表示不自动保存。" },
    save_queue: { label: "保存队列长度", help: "异步保存截图的队列长度。" },
    stats_interval: { label: "统计输出间隔", help: "多久向日志输出一次性能统计。" },
    smooth: { label: "启用平滑", help: "平滑检测框和关键点，减少视觉抖动。" },
    smooth_alpha: { label: "平滑强度", help: "平滑程度。越高越稳定，但响应会更慢。" },
    jsonl_log: { label: "JSONL 逐帧日志", help: "把每帧推理结果写入 debug/yolo_log.jsonl，主要用于深度调试。" },
  },
};

function t() {
  return translations[state.language] || translations.en;
}

function getFieldPresentation(field) {
  const translated = fieldTranslations[state.language]?.[field.name];
  return {
    label: translated?.label || field.label,
    help: translated?.help || field.help,
  };
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast._timer);
  showToast._timer = window.setTimeout(() => toast.classList.remove("show"), 2400);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderFieldHelp(help) {
  if (!help) {
    return "";
  }
  return `<span class="help-dot" data-help="${escapeHtml(help)}" aria-label="${escapeHtml(t().tooltipAria)}">?</span>`;
}

function renderGroups() {
  const root = document.getElementById("form-groups");
  root.innerHTML = "";

  state.fieldGroups.forEach((group) => {
    const card = document.createElement("section");
    card.className = "group-card";

    const groupLabel = t().groupLabels[group.label] || group.label;
    const groupDescription = t().groupDescriptions[group.description] || group.description;

    const header = document.createElement("div");
    header.className = "group-header";
    header.innerHTML = `
      <div class="card-label">${groupLabel}</div>
      <h3>${groupLabel}</h3>
      <p>${groupDescription}</p>
    `;

    const grid = document.createElement("div");
    grid.className = "fields-grid";

    group.fields.forEach((field) => {
      const wrapper = document.createElement("div");
      wrapper.className = `field ${field.type === "checkbox" ? "checkbox-field" : ""}`;
      const id = `field-${field.name}`;
      const value = state.config[field.name];
      const presentation = getFieldPresentation(field);
      const help = renderFieldHelp(presentation.help);

      if (field.type === "checkbox") {
        wrapper.innerHTML = `
          <input id="${id}" type="checkbox" ${value ? "checked" : ""}>
          <div>
            <div class="field-label-row">
              <label for="${id}">${presentation.label}</label>
              ${help}
            </div>
          </div>
        `;
      } else if (field.type === "select") {
        const options = state.models
          .map((model) => `<option value="${escapeHtml(model)}" ${value === model ? "selected" : ""}>${escapeHtml(model)}</option>`)
          .join("");
        wrapper.innerHTML = `
          <div class="field-label-row">
            <label for="${id}">${presentation.label}</label>
            ${help}
          </div>
          <select id="${id}">
            ${options}
          </select>
        `;
      } else {
        const min = field.min !== undefined ? `min="${field.min}"` : "";
        const max = field.max !== undefined ? `max="${field.max}"` : "";
        const step = field.step !== undefined ? `step="${field.step}"` : "";
        const placeholder = field.placeholder ? `placeholder="${escapeHtml(field.placeholder)}"` : "";
        wrapper.innerHTML = `
          <div class="field-label-row">
            <label for="${id}">${presentation.label}</label>
            ${help}
          </div>
          <input
            id="${id}"
            type="${field.type}"
            value="${escapeHtml(value ?? "")}" 
            ${min}
            ${max}
            ${step}
            ${placeholder}
          >
        `;
      }

      grid.appendChild(wrapper);
    });

    card.appendChild(header);
    card.appendChild(grid);
    root.appendChild(card);
  });
}

function collectConfig() {
  const next = { ...state.config };
  state.fieldGroups.forEach((group) => {
    group.fields.forEach((field) => {
      const el = document.getElementById(`field-${field.name}`);
      if (!el) return;
      if (field.type === "checkbox") {
        next[field.name] = el.checked;
      } else {
        next[field.name] = el.value;
      }
    });
  });
  return next;
}

function updateRuntimeWarning(config) {
  const warning = document.getElementById("runtime-warning");
  const limit = Number(config.max_run_seconds || 0);
  if (limit > 0) {
    warning.hidden = false;
    warning.textContent = t().warning(limit);
  } else {
    warning.hidden = true;
    warning.textContent = "";
  }
}

async function updateCliPreview() {
  try {
    const payload = collectConfig();
    updateRuntimeWarning(payload);
    const data = await request("/api/preview-args", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    document.getElementById("cli-preview").textContent = `python -m cvcap ${data.cli.join(" ")}`;
  } catch (error) {
    document.getElementById("cli-preview").textContent = t().previewUnavailable(error.message);
  }
}

function renderLegends() {
  document.getElementById("throughput-legend").innerHTML = throughputSeries
    .map((series) => `<span class="legend-item"><span class="legend-swatch" style="background:${series.color}"></span>${series.label[state.language]}</span>`)
    .join("");
  document.getElementById("latency-legend").innerHTML = latencySeries
    .map((series) => `<span class="legend-item"><span class="legend-swatch" style="background:${series.color}"></span>${series.label[state.language]}</span>`)
    .join("");
  document.getElementById("drop-legend").innerHTML = dropSeries
    .map((series) => `<span class="legend-item"><span class="legend-swatch" style="background:${series.color}"></span>${series.label[state.language]}</span>`)
    .join("");
}

function markChartsDirty() {
  state.chartsDirty = true;
}

function computeDomain(seriesList) {
  if (!state.perfHistory.length) {
    return { min: 0, max: 1 };
  }
  let minValue = Infinity;
  let maxValue = -Infinity;
  for (const point of state.perfHistory) {
    for (const series of seriesList) {
      const value = Number(point[series.key]) || 0;
      minValue = Math.min(minValue, value);
      maxValue = Math.max(maxValue, value);
    }
  }
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return { min: 0, max: 1 };
  }
  let span = maxValue - minValue;
  if (span < 0.5) {
    span = Math.max(0.5, maxValue === 0 ? 1 : Math.abs(maxValue) * 0.1);
  }
  const padding = Math.max(span * 0.18, 0.2);
  return {
    min: Math.max(0, minValue - padding),
    max: maxValue + padding,
  };
}

function formatTick(value, unit) {
  return `${value.toFixed(unit === "fps" ? 1 : 2)} ${unit}`;
}

function drawChart(canvas, seriesList, unit) {
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 900;
  const cssHeight = canvas.clientHeight || 220;
  const width = Math.floor(cssWidth * ratio);
  const height = Math.floor(cssHeight * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const padding = { left: 64, right: 16, top: 10, bottom: 24 };
  const chartWidth = cssWidth - padding.left - padding.right;
  const chartHeight = cssHeight - padding.top - padding.bottom;

  ctx.strokeStyle = "#ece8de";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (chartHeight / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(padding.left + chartWidth, y);
    ctx.stroke();
  }

  if (!state.perfHistory.length) {
    ctx.fillStyle = "#87867f";
    ctx.font = "12px Arial";
    ctx.fillText(t().chartWaiting, padding.left, padding.top + 18);
    return;
  }

  const domain = computeDomain(seriesList);
  const span = Math.max(domain.max - domain.min, 0.001);

  ctx.fillStyle = "#87867f";
  ctx.font = "11px Arial";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i += 1) {
    const value = domain.max - (span / 4) * i;
    const y = padding.top + (chartHeight / 4) * i + 4;
    ctx.fillText(formatTick(value, unit), padding.left - 6, y);
  }

  for (const series of seriesList) {
    ctx.strokeStyle = series.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    state.perfHistory.forEach((point, index) => {
      const x = padding.left + (chartWidth * index) / Math.max(1, MAX_PERF_SAMPLES - 1);
      const value = Number(point[series.key]) || 0;
      const y = padding.top + chartHeight - ((value - domain.min) / span) * chartHeight;
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  }
}

function drawCharts() {
  drawChart(document.getElementById("latency-chart"), latencySeries, "ms");
  drawChart(document.getElementById("throughput-chart"), throughputSeries, "fps");
  drawChart(document.getElementById("drop-chart"), dropSeries, "fps");
}

function startChartLoop() {
  if (state.chartLoopStarted) {
    return;
  }
  state.chartLoopStarted = true;
  function frame() {
    if (state.chartsDirty) {
      drawCharts();
      state.chartsDirty = false;
    }
    window.requestAnimationFrame(frame);
  }
  window.requestAnimationFrame(frame);
}

function updateStatus(status) {
  state.lastStatus = status;
  state.logLines = (status.logs || []).slice(-MAX_LOG_LINES);
  state.perfHistory = (status.metrics || []).slice(-MAX_PERF_SAMPLES).map((point) => ({
    grab: Number(point.grab) || 0,
    proc: Number(point.proc) || 0,
    drop: Number(point.drop) || 0,
    cmd: Number(point.cmd) || 0,
    wait: Number(point.wait) || 0,
    total: Number(point.total) || 0,
    pre: Number(point.pre) || 0,
    gpu: Number(point.gpu) || 0,
    post: Number(point.post) || 0,
    ovhd: Number(point.ovhd) || 0,
  }));
  const pill = document.getElementById("status-pill");
  pill.textContent = status.running ? t().running : t().idle;
  pill.classList.toggle("running", Boolean(status.running));
  document.getElementById("status-pid").textContent = status.pid ?? "-";
  document.getElementById("status-started").textContent = status.started_at ? new Date(status.started_at * 1000).toLocaleString() : "-";
  document.getElementById("status-exit").textContent = status.last_exit_code ?? "-";
  document.getElementById("status-config").textContent = status.config_path ?? "-";
  document.getElementById("log-output").textContent = state.logLines.join("\n");
  markChartsDirty();
}

async function pollStatus() {
  try {
    const status = await request("/api/status");
    updateStatus(status);
  } catch (error) {
    showToast(t().fetchStatusFailed(error.message));
  }
}

async function saveConfig() {
  const payload = collectConfig();
  const data = await request("/api/config", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  state.config = data.config;
  renderGroups();
  await updateCliPreview();
  showToast(t().configSaved);
}

async function runPipeline() {
  const payload = collectConfig();
  const data = await request("/api/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  state.config = data.config;
  renderGroups();
  updateStatus(data.status);
  await updateCliPreview();
  showToast(t().pipelineStarted);
}

async function stopPipeline() {
  const data = await request("/api/stop", { method: "POST" });
  updateStatus(data.status);
  showToast(t().stopSent);
}

function activateTab(tabId) {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tabId}`);
  });
  markChartsDirty();
}

function wireTabs() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.tab));
  });
}

function applyStaticTranslations() {
  const copy = t();
  document.getElementById("header-eyebrow").textContent = copy.eyebrow;
  document.getElementById("header-description").textContent = copy.headerDescription;
  document.getElementById("hero-start").textContent = copy.saveAndStart;
  document.getElementById("hero-save").textContent = copy.saveConfig;
  document.getElementById("hero-stop").textContent = copy.stop;
  document.getElementById("hero-reset").textContent = copy.resetDefaults;
  document.getElementById("tab-button-config").textContent = copy.tabConfig;
  document.getElementById("tab-button-status").textContent = copy.tabStatus;
  document.getElementById("tab-button-logs").textContent = copy.tabLogs;
  document.getElementById("config-card-label").textContent = copy.configCardLabel;
  document.getElementById("config-title").textContent = copy.configTitle;
  document.getElementById("config-description").textContent = copy.configDescription;
  document.getElementById("status-card-label").textContent = copy.statusCardLabel;
  document.getElementById("status-title").textContent = copy.statusTitle;
  document.getElementById("status-description").textContent = copy.statusDescription;
  document.getElementById("logs-card-label").textContent = copy.logsCardLabel;
  document.getElementById("logs-title").textContent = copy.logsTitle;
  document.getElementById("logs-description").textContent = copy.logsDescription;
  document.getElementById("chart-throughput-title").textContent = copy.chartThroughputTitle;
  document.getElementById("chart-latency-title").textContent = copy.chartLatencyTitle;
  document.getElementById("chart-drop-title").textContent = copy.chartDropTitle;
  document.getElementById("status-label-pid").textContent = copy.statusPid;
  document.getElementById("status-label-started").textContent = copy.statusStarted;
  document.getElementById("status-label-exit").textContent = copy.statusExit;
  document.getElementById("status-label-config").textContent = copy.statusConfig;
  document.querySelector(".cli-preview-card .card-label").textContent = copy.cliPreview;
  document.getElementById("config-path").textContent = `${copy.configPath}: ${state.configPath || "-"}`;
  document.getElementById("lang-en").classList.toggle("active", state.language === "en");
  document.getElementById("lang-zh").classList.toggle("active", state.language === "zh");
  renderLegends();
}

function setLanguage(language) {
  state.language = language;
  applyStaticTranslations();
  renderGroups();
  if (state.lastStatus) {
    updateStatus(state.lastStatus);
  }
  updateRuntimeWarning(collectConfig());
  markChartsDirty();
}

function wireTooltips() {
  const tooltip = document.getElementById("tooltip");

  function moveTooltip(event) {
    tooltip.style.left = `${event.clientX + 14}px`;
    tooltip.style.top = `${event.clientY + 14}px`;
  }

  document.addEventListener("mouseover", (event) => {
    const target = event.target.closest("[data-help]");
    if (!target) {
      return;
    }
    tooltip.textContent = target.dataset.help;
    tooltip.hidden = false;
    moveTooltip(event);
  });

  document.addEventListener("mousemove", (event) => {
    if (tooltip.hidden) {
      return;
    }
    moveTooltip(event);
  });

  document.addEventListener("mouseout", (event) => {
    if (!event.target.closest("[data-help]")) {
      return;
    }
    tooltip.hidden = true;
    tooltip.textContent = "";
  });
}

function resetDefaults() {
  state.config = { ...state.defaults };
  renderGroups();
  updateCliPreview().catch(() => {});
  showToast(t().defaultsRestored);
}

function wireEvents() {
  document.getElementById("hero-save").addEventListener("click", () => saveConfig().catch((error) => showToast(error.message)));
  document.getElementById("hero-start").addEventListener("click", () => runPipeline().catch((error) => showToast(error.message)));
  document.getElementById("hero-stop").addEventListener("click", () => stopPipeline().catch((error) => showToast(error.message)));
  document.getElementById("hero-reset").addEventListener("click", resetDefaults);
  document.getElementById("lang-en").addEventListener("click", () => setLanguage("en"));
  document.getElementById("lang-zh").addEventListener("click", () => setLanguage("zh"));

  document.getElementById("form-groups").addEventListener("input", () => {
    window.clearTimeout(wireEvents._previewTimer);
    wireEvents._previewTimer = window.setTimeout(() => {
      updateCliPreview().catch(() => {});
    }, 120);
  });

  wireTabs();
  wireTooltips();
  renderLegends();
}

async function bootstrap() {
  const data = await request("/api/bootstrap");
  state.config = data.config;
  state.defaults = data.defaults;
  state.models = data.models;
  state.fieldGroups = data.field_groups;
  state.configPath = data.config_path;
  state.lastStatus = data.status;

  applyStaticTranslations();
  renderGroups();
  updateStatus(data.status);
  updateRuntimeWarning(data.config);
  await updateCliPreview();
  wireEvents();
  startChartLoop();
  window.addEventListener("resize", markChartsDirty);
  state.polling = window.setInterval(pollStatus, 1500);
}

bootstrap().catch((error) => showToast(t().bootstrapFailed(error.message)));
