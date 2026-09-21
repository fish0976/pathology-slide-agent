<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";

const jobs = ref([]),
  current = ref(null),
  loading = ref(false),
  error = ref(""),
  question = ref("");
const messages = ref([]),
  asking = ref(false),
  heatmap = ref(true),
  opacity = ref(60),
  zoom = ref(1);
const activeRegion = ref(null),
  fileInput = ref(null),
  health = ref(null);
const training = ref(null);
const options = ref({
  tile_size: 256,
  max_tiles: 128,
  level: 0,
  normalize: true,
});
let timer,
  selectionVersion = 0;
const report = computed(() => current.value?.report);
const modelMode = computed(
  () => report.value?.mode || health.value?.model_backend || "demo",
);
const processing = computed(() =>
  ["queued", "running"].includes(current.value?.status),
);
const statusNames = {
  uploaded: "待分析",
  queued: "排队中",
  running: "分析中",
  completed: "已完成",
  failed: "失败",
};
const stageNames = {
  read_slide: "读取切片",
  quality_control: "切片质控",
  tile_inference: "组织切块与推理",
  render_heatmap: "生成热力图",
  generate_report: "整理报告",
  completed: "分析完成",
  uploaded: "等待分析",
};
const pct = (x) => `${((x || 0) * 100).toFixed(1)}%`;
const asset = (name) =>
  `/api/jobs/${current.value?.id}/assets/${name}?v=${current.value?.updated_at}`;

async function api(path, init) {
  const response = await fetch(`/api${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : "请求失败，请检查参数或服务状态",
    );
  }
  return response.json();
}
const post = (path, data) =>
  api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
async function refresh() {
  jobs.value = await api("/jobs");
}
async function select(id) {
  const version = ++selectionVersion;
  error.value = "";
  messages.value = [];
  activeRegion.value = null;
  zoom.value = 1;
  try {
    const job = await api(`/jobs/${id}`);
    if (version === selectionVersion) current.value = job;
  } catch (e) {
    error.value = e.message;
  }
}
async function demo() {
  loading.value = true;
  error.value = "";
  try {
    const job = await post("/demo", {});
    await refresh();
    await select(job.id);
    await analyze();
  } catch (e) {
    error.value = e.message;
  } finally {
    loading.value = false;
  }
}
async function upload(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  loading.value = true;
  error.value = "";
  try {
    const body = new FormData();
    body.append("file", file);
    const job = await api("/slides", { method: "POST", body });
    await refresh();
    await select(job.id);
  } catch (e) {
    error.value = e.message;
  } finally {
    loading.value = false;
    event.target.value = "";
  }
}
async function analyze() {
  if (!current.value) return;
  error.value = "";
  activeRegion.value = null;
  messages.value = [];
  const id = current.value.id;
  try {
    const job = await post(`/jobs/${id}/analyze`, options.value);
    if (current.value?.id === id) current.value = job;
    await refresh();
  } catch (e) {
    error.value = e.message;
  }
}
async function poll() {
  try {
    if (
      health.value?.model_backend === "foundation" &&
      training.value?.state !== "completed"
    ) {
      training.value = await api("/training");
    }
    if (processing.value) {
      const id = current.value.id,
        version = selectionVersion;
      const job = await api(`/jobs/${id}`);
      if (version === selectionVersion && id === current.value?.id) {
        current.value = job;
        if (job.status === "failed") error.value = job.error;
      }
      if (!["queued", "running"].includes(job.status)) await refresh();
    }
  } catch (e) {
    error.value = e.message;
  } finally {
    timer = setTimeout(poll, 900);
  }
}
async function ask(text = question.value) {
  if (!text.trim() || !report.value || asking.value) return;
  const id = current.value.id,
    version = selectionVersion;
  messages.value.push({ role: "user", text });
  question.value = "";
  asking.value = true;
  try {
    const answer = await post(`/jobs/${id}/chat`, { question: text });
    if (version === selectionVersion)
      messages.value.push({
        role: "assistant",
        provider:
          answer.source === "llm"
            ? `${answer.provider} · ${answer.model}`
            : "本地证据助手",
        text: answer.answer,
        meta: answer.tools.join(" → "),
        warning: answer.warning,
      });
  } catch (e) {
    error.value = e.message;
  } finally {
    asking.value = false;
  }
}
function regionStyle(region) {
  const s = report.value.slide;
  return {
    left: pct(region.x / s.width),
    top: pct(region.y / s.height),
    width: pct(region.width / s.width),
    height: pct(region.height / s.height),
  };
}
onMounted(async () => {
  try {
    health.value = await api("/health");
    await refresh();
    if (jobs.value.length) await select(jobs.value[0].id);
  } catch (e) {
    error.value = e.message;
  }
  poll();
});
onUnmounted(() => clearTimeout(timer));
</script>

<template>
  <div class="workspace">
    <aside class="sidebar">
      <a class="brand" href="/" aria-label="PathoScope 首页"
        ><span class="brand-icon">✳</span>
        <div>PathoScope<small>PATHOLOGY RESEARCH</small></div></a
      >
      <div class="space-label">工作空间 <span>01</span></div>
      <div class="nav-item">
        <span>◫</span> 切片分析 <span class="nav-dot"></span>
      </div>
      <div class="section-label">
        切片记录 <span>{{ jobs.length }}</span>
      </div>
      <div class="job-list">
        <button
          v-for="job in jobs"
          :key="job.id"
          class="job"
          :class="{ selected: job.id === current?.id }"
          @click="select(job.id)"
        >
          <span class="file-icon">▧</span
          ><span class="job-info"
            ><strong>{{ job.filename }}</strong
            ><small
              >{{ statusNames[job.status] }} · {{ job.id.slice(0, 6) }}</small
            ></span
          >
        </button>
        <p v-if="!jobs.length" class="muted empty-history">
          还没有切片。<br />从合成样本开始探索。
        </p>
      </div>
      <div class="sidebar-bottom">
        <span class="online"></span
        >{{ health ? "本地研究工作台" : "正在连接服务"
        }}<small>v0.1 · 可追溯分析流程</small>
      </div>
    </aside>
    <main>
      <header class="topbar">
        <div>工作空间 <span>/</span> 切片分析</div>
        <span class="research-pill">RESEARCH USE ONLY</span>
      </header>
      <section class="page-title">
        <div class="eyebrow">DIGITAL PATHOLOGY · AGENT WORKSPACE</div>
        <h1>让每一次观察，都有迹可循。</h1>
        <p>从数字切片到区域证据，连接质控、分析与研究报告。</p>
        <div class="title-actions">
          <button class="secondary" :disabled="loading" @click="demo">
            {{ loading ? "准备样本…" : "↗ 体验合成样本" }}</button
          ><button
            class="primary"
            :disabled="loading"
            @click="fileInput.click()"
          >
            ＋ 上传切片</button
          ><input
            ref="fileInput"
            hidden
            type="file"
            accept=".png,.jpg,.jpeg,.svs,.ndpi,.tif,.tiff"
            @change="upload"
          />
        </div>
      </section>
      <div class="notice">
        <span>ⓘ</span>
        {{
          modelMode === "foundation"
            ? "视觉基础模型 · LoRA 微调"
            : modelMode === "torch"
              ? "本地研究模型"
              : "演示模式"
        }}
        ·
        {{
          modelMode !== "demo"
            ? "模型分数尚未经过临床验证。"
            : "颜色纹理分数仅用于演示分析流程，不代表肿瘤概率。"
        }}
        所有结论均需专业人员复核。
      </div>
      <div v-if="error" class="error" role="alert">
        {{ error
        }}<button @click="error = ''" aria-label="关闭错误提示">×</button>
      </div>
      <div v-if="training?.base_model" class="notice">
        <span>◉</span>
        DINOv2 病理微调 ·
        <template v-if="training.state === 'completed'">
          训练完成 · {{ training.samples?.train }} 张训练图块 · 测试敏感度
          {{ pct(training.test?.sensitivity) }} · 测试特异度
          {{ pct(training.test?.specificity) }}。
        </template>
        <template v-else-if="training.state === 'failed'"
          >训练中断，请查看本机训练日志。</template
        >
        <template v-else>
          训练中 · 第 {{ training.epoch || 1 }}/{{ training.epochs }} 轮
          <template v-if="training.steps">
            · {{ training.step }}/{{ training.steps }} 步</template
          >。
        </template>
        公开数据小规模实验，不代表医院数据或临床验证结果。
      </div>
      <section class="stats">
        <div>
          <small>当前切片</small
          ><strong>{{
            report
              ? `${report.slide.width.toLocaleString()} × ${report.slide.height.toLocaleString()}`
              : "—"
          }}</strong
          ><span>LEVEL 0 · PIXELS</span>
        </div>
        <div>
          <small>组织占比</small
          ><strong>{{
            report ? pct(report.quality.tissue_fraction) : "—"
          }}</strong
          ><span>缩略图启发式分割</span>
        </div>
        <div>
          <small>已分析图块</small
          ><strong
            >{{ report?.sampling.analyzed_tiles ?? "—"
            }}<em v-if="report">
              / {{ report.sampling.sampled_tiles }}</em
            ></strong
          ><span>背景图块自动过滤</span>
        </div>
        <div>
          <small>采样网格覆盖率</small
          ><strong>{{
            report ? pct(report.sampling.grid_coverage) : "—"
          }}</strong
          ><span>{{
            report?.sampling.is_full_grid
              ? "当前层级完整网格"
              : "未采样区域不作判断"
          }}</span>
        </div>
      </section>
      <div class="analysis-grid">
        <section class="panel viewer-panel">
          <div class="panel-heading">
            <h2>切片观察窗 <span>SLIDE VIEWER</span></h2>
            <span class="status">{{
              statusNames[current?.status] || "等待切片"
            }}</span>
          </div>
          <div class="viewer-toolbar">
            <span class="filename">{{
              current?.filename || "尚未载入切片"
            }}</span
            ><label><input v-model="heatmap" type="checkbox" /> 热力图</label
            ><input
              v-model="opacity"
              aria-label="热力图透明度"
              type="range"
              min="0"
              max="100"
            />
            <div class="zoom">
              <button
                @click="zoom = Math.max(1, zoom - 0.25)"
                aria-label="缩小"
              >
                −</button
              ><span>{{ Math.round(zoom * 100) }}%</span
              ><button
                @click="zoom = Math.min(3, zoom + 0.25)"
                aria-label="放大"
              >
                ＋
              </button>
            </div>
          </div>
          <div class="viewer-canvas">
            <div
              v-if="current"
              class="slide-image"
              :style="{ width: `${zoom * 100}%` }"
            >
              <img :src="asset('thumbnail.png')" alt="病理切片缩略图" /><img
                v-if="report && heatmap"
                class="overlay"
                :src="asset('heatmap.png')"
                :style="{ opacity: opacity / 100 }"
                alt="已采样图块分数叠加层"
              />
              <div
                v-if="activeRegion"
                class="region-box"
                :style="regionStyle(activeRegion)"
              ></div>
            </div>
            <div v-else class="empty-view">
              <span>◉</span>
              <h3>从一张切片开始</h3>
              <p>
                上传数字切片，或使用无需密钥的合成样本<br />体验完整的分析流程。
              </p>
              <button class="primary" :disabled="loading" @click="demo">
                载入合成样本 →</button
              ><small>SVS / NDPI / TIFF / PNG / JPG</small>
            </div>
          </div>
          <div class="viewer-footer">
            <span>拖动滚动条查看放大后的缩略图 · 坐标基于 Level 0</span
            ><span class="legend">低分 <i></i> 高分</span>
          </div>
          <div v-if="processing" class="progress">
            <div :style="{ width: current.progress + '%' }"></div>
            <span
              >{{ stageNames[current.stage] || "等待工作线程" }} ·
              {{ current.progress }}%</span
            >
          </div>
          <div class="configuration">
            <label
              >图块大小<select v-model.number="options.tile_size">
                <option :value="128">128 px</option>
                <option :value="256">256 px</option>
                <option :value="512">512 px</option>
              </select></label
            ><label
              >采样上限<select v-model.number="options.max_tiles">
                <option :value="48">48 块</option>
                <option :value="128">128 块</option>
                <option :value="512">512 块</option>
              </select></label
            ><label
              >金字塔层级<input
                v-model.number="options.level"
                type="number"
                min="0"
                max="20" /></label
            ><label class="check"
              ><input v-model="options.normalize" type="checkbox" />
              染色标准化</label
            ><button
              class="primary"
              :disabled="!current || processing"
              @click="analyze"
            >
              {{ processing ? "分析中…" : report ? "重新分析" : "开始分析 →" }}
            </button>
          </div>
        </section>
        <section class="panel assistant-panel">
          <div class="panel-heading">
            <h2><span class="spark">✳</span> 研究助手</h2>
            <span class="assistant-type">{{
              health?.assistant === "llm"
                ? `${health.assistant_provider} 已配置`
                : "本地证据模式"
            }}</span>
          </div>
          <div class="assistant-intro">
            <h3>让结果可解释</h3>
            <p>围绕当前切片追问。回答关联质控指标、区域坐标与采样范围。</p>
          </div>
          <div class="suggestions">
            <button :disabled="!report" @click="ask('这张切片的质量如何？')">
              这张切片的质量如何？ ↗</button
            ><button :disabled="!report" @click="ask('高分区域在哪里？')">
              高分区域在哪里？ ↗</button
            ><button :disabled="!report" @click="ask('总结分析报告')">
              总结分析报告 ↗
            </button>
          </div>
          <div class="messages" aria-live="polite">
            <article
              v-for="(message, i) in messages"
              :key="i"
              :class="message.role"
            >
              <small>{{
                message.role === "user" ? "你" : message.provider
              }}</small>
              <p>{{ message.text }}</p>
              <code v-if="message.meta">{{ message.meta }}</code>
              <p v-if="message.warning" class="warning">
                {{ message.warning }}
              </p>
            </article>
            <p v-if="asking" class="muted">正在检索当前切片证据…</p>
            <div v-if="!messages.length" class="assistant-empty">
              分析完成后，即可开始追问。<br />当前会话只使用这张切片的分析结果。
            </div>
          </div>
          <form class="chat-form" @submit.prevent="ask()">
            <input
              v-model="question"
              :disabled="!report || asking"
              maxlength="2000"
              placeholder="询问当前切片…"
              aria-label="向研究助手提问"
            /><button
              class="primary"
              :disabled="!report || asking || !question.trim()"
              aria-label="发送问题"
            >
              ↑
            </button>
          </form>
        </section>
      </div>
      <div class="results-grid">
        <section class="panel">
          <div class="panel-heading">
            <h2>区域证据 <span>REGION EVIDENCE</span></h2>
            <small>点击坐标定位</small>
          </div>
          <div v-if="!report" class="empty-result">
            分析完成后展示已采样图块中的高分区域。
          </div>
          <table v-else>
            <thead>
              <tr>
                <th>区域</th>
                <th>Level 0 坐标</th>
                <th>研究分数</th>
                <th>质控提示</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(r, i) in report.regions.slice(0, 5)" :key="r.id">
                <td>
                  <span class="region-number">{{
                    String(i + 1).padStart(2, "0")
                  }}</span>
                </td>
                <td>
                  <button
                    class="coordinate"
                    @click="
                      activeRegion = r;
                      zoom = 1;
                    "
                  >
                    {{ r.x }}, {{ r.y }} ↗
                  </button>
                </td>
                <td>
                  <div class="score-bar">
                    <i :style="{ width: `${r.score * 100}%` }"></i>
                  </div>
                  {{ r.score.toFixed(3) }}
                </td>
                <td>
                  {{
                    r.qc.blur_warning
                      ? "复核清晰度"
                      : r.qc.dark_warning
                        ? "复核深色区域"
                        : "未触发阈值"
                  }}
                </td>
              </tr>
            </tbody>
          </table>
        </section>
        <section class="panel report-panel">
          <div class="panel-heading">
            <h2>结构化报告</h2>
            <span>↗</span>
          </div>
          <template v-if="report"
            ><p>{{ report.summary }}</p>
            <div class="trace">
              <div v-for="step in report.trace" :key="step.tool">
                <span>✓</span>{{ step.message }}
              </div>
            </div>
            <div class="download-links">
              <a :href="`/api/jobs/${current.id}/report?format=md`"
                >↓ Markdown 报告</a
              ><a :href="`/api/jobs/${current.id}/report?format=json`"
                >↓ JSON 数据</a
              >
            </div></template
          >
          <div v-else class="empty-result">
            每一步工具调用都将记录到报告中。<br />支持 Markdown 与 JSON 导出。
          </div>
        </section>
      </div>
      <footer>
        PathoScope · 肿瘤病理切片分析智能体<span
          >合成样本演示 / 本地处理 / 研究用途</span
        >
      </footer>
    </main>
  </div>
</template>
