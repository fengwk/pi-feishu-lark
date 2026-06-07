import { existsSync, readFileSync, realpathSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { basename, isAbsolute, join, resolve } from "node:path";
import type { AgentSession, SessionInfo } from "@earendil-works/pi-coding-agent";
import {
  AuthStorage,
  createAgentSession,
  DefaultResourceLoader,
  getAgentDir,
  ModelRegistry,
  SessionManager,
} from "@earendil-works/pi-coding-agent";
import type { FeishuBridgeRuntime } from "./bridge-runtime.js";
import { CHILD_SESSION_ENV, ensureRoot, readJson, STATE_PATH, writeJson } from "./config.js";
import { debugLog } from "./debug.js";
import type { ResumeScope, ResumeSessionPage } from "./cards.js";
import type { TaskStatusSink } from "./task-status-card.js";
import type { FeishuState } from "./types.js";

type ReplyTarget = (text: string) => Promise<void>;
type ActiveRun = {
  session: AgentSession;
  runId?: string;
  stopped: boolean;
  status?: TaskStatusSink;
  fallbackReply: ReplyTarget;
  replyTargets: ReplyTarget[];
  currentTurnReplyTargets: ReplyTarget[];
  assistantReplyCount: number;
  assistantReplies: Promise<void>[];
};

export type StopConversationResult =
  | { status: "stopped"; message: string }
  | { status: "not_running"; message: string }
  | { status: "stale"; message: string }
  | { status: "failed"; message: string };

const RESUME_PAGE_SIZE = 10;
const PROMPT_TIMEOUT_MS = 60 * 60 * 1000;

export class ConversationManager {
  private readonly sessions = new Map<string, Promise<AgentSession>>();
  private readonly queues = new Map<string, Promise<void>>();
  private readonly startingRuns = new Map<string, Promise<void>>();
  private readonly activeRuns = new Map<string, ActiveRun>();
  private readonly authStorage = AuthStorage.create();
  private readonly modelRegistry = ModelRegistry.create(this.authStorage);
  private defaultProvider: string | undefined;
  private defaultModelId: string | undefined;
  private state: FeishuState;

  constructor(
    private readonly cwd: string,
    private readonly bridge?: FeishuBridgeRuntime,
  ) {
    ensureRoot();
    this.state = readJson<FeishuState>(STATE_PATH, { sessions: {} });
    this.state.sessions ||= {};
    this.state.models ||= {};
    this.state.workspaces ||= {};
    this.loadSettingsDefault();
  }

  /** Read global settings default model for fallback in getSelectedModel. */
  private loadSettingsDefault() {
    try {
      const settingsPath = join(getAgentDir(), "settings.json");
      const raw = readFileSync(settingsPath, "utf-8");
      const settings = JSON.parse(raw);
      if (settings.defaultProvider && settings.defaultModel) {
        this.defaultProvider = settings.defaultProvider;
        this.defaultModelId = settings.defaultModel;
      }
    } catch {}
  }

  async prompt(key: string, userText: string, onReply: (text: string) => Promise<void>) {
    return this.promptWithImages(key, userText, [], onReply);
  }

  async promptWithImages(
    key: string,
    userText: string,
    images: Array<{ type: "image"; data: string; mimeType: string }>,
    onReply: (text: string) => Promise<void>,
    status?: TaskStatusSink,
  ) {
    const active = this.activeRuns.get(key);
    if (active?.session.isStreaming) {
      await this.queueSteering(key, active, userText, images, onReply);
      return;
    }

    const starting = this.startingRuns.get(key);
    if (starting) {
      await starting.catch(() => undefined);
      const startedActive = this.activeRuns.get(key);
      if (startedActive?.session.isStreaming) {
        await this.queueSteering(key, startedActive, userText, images, onReply);
        return;
      }
    }

    const previous = this.queues.get(key);
    if (previous) {
      await previous.catch((error) => {
        debugLog("feishu.queue.previous_error", {
          key,
          error: error instanceof Error ? error.message : String(error),
        });
      });
      const queuedActive = this.activeRuns.get(key);
      if (queuedActive?.session.isStreaming) {
        await this.queueSteering(key, queuedActive, userText, images, onReply);
        return;
      }
    }

    let resolveStarted: () => void = () => undefined;
    let startedResolved = false;
    const started = new Promise<void>((resolve) => {
      resolveStarted = resolve;
    });
    const finishStarting = () => {
      if (!startedResolved) {
        startedResolved = true;
        resolveStarted();
      }
      if (this.startingRuns.get(key) === started) this.startingRuns.delete(key);
    };
    this.startingRuns.set(key, started);

    const next = (async () => {
      let run: ActiveRun | undefined;
      try {
        debugLog("feishu.prompt.start", { key, textLength: userText.length, imageCount: images.length });
        const session = await this.getFreshSessionForPrompt(key);
        run = {
          session,
          runId: status?.runId,
          stopped: false,
          status,
          fallbackReply: onReply,
          replyTargets: [onReply],
          currentTurnReplyTargets: [],
          assistantReplyCount: 0,
          assistantReplies: [],
        };
        this.activeRuns.set(key, run);
        finishStarting();
        await status?.start();
        if (run.stopped) return;
        this.bridge?.beginFeishuInput(session.sessionId);
        try {
          try {
            await withTimeout(
              session.prompt(userText, images.length ? { images } : undefined),
              PROMPT_TIMEOUT_MS,
              "Pi 模型处理超时，请稍后重试；如果是图片消息，可以先切换到明确支持图片的模型。",
            );
          } catch (error) {
            if (run.stopped) {
              debugLog("feishu.prompt.stopped", { key });
              return;
            }
            throw error;
          }
        } finally {
          if (this.activeRuns.get(key) === run) this.activeRuns.delete(key);
          this.bridge?.endFeishuInput(session.sessionId);
        }
        if (run.stopped) return;
        if (run.assistantReplyCount === 0) {
          const answer = extractLastAssistantText(session);
          debugLog("feishu.prompt.done", { key, answerLength: answer.length, source: "fallback" });
          await run.fallbackReply(answer || "No response.");
        } else {
          debugLog("feishu.prompt.done", { key, replyCount: run.assistantReplyCount, source: "turn_end" });
        }
        if (run.assistantReplies.length) await Promise.allSettled(run.assistantReplies);
        await status?.finish("done");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        debugLog("feishu.prompt.error", { key, error: message });
        await status?.finish("failed", message);
        await onReply(`Pi error: ${message}`);
      } finally {
        finishStarting();
        if (run && this.activeRuns.get(key) === run) this.activeRuns.delete(key);
      }
    })();
    this.queues.set(key, next);
    await next;
  }

  async stopConversation(key: string, onReply: (text: string) => Promise<void>, runId?: string): Promise<StopConversationResult> {
    const active = this.activeRuns.get(key);
    if (!active) {
      const message = "当前没有进行中的处理。";
      await onReply(message);
      return { status: "not_running", message };
    }
    if (runId && active.runId && active.runId !== runId) {
      const message = "这张任务卡片已不是当前进行中的任务。";
      await onReply(message);
      debugLog("feishu.prompt.stop_stale", { key, runId, activeRunId: active.runId });
      return { status: "stale", message };
    }

    active.stopped = true;
    await active.status?.stopImmediately("用户已停止任务");
    try {
      await active.session.abort();
      debugLog("feishu.prompt.abort", { key });
      const message = "已停止当前处理。";
      await onReply(message);
      return { status: "stopped", message };
    } catch (error) {
      active.stopped = false;
      debugLog("feishu.prompt.abort_error", { key, error: error instanceof Error ? error.message : String(error) });
      const message = "停止失败，请重试。";
      await onReply(message);
      return { status: "failed", message };
    }
  }

  async newConversation(key: string, onReply: (text: string) => Promise<void>) {
    const previous = this.previousTurn(key);
    const next = previous.then(async () => {
      const cached = this.sessions.get(key);
      if (cached) {
        try { (await cached).dispose(); } catch {}
      }
      this.sessions.delete(key);
      delete this.state.sessions[key];
      writeJson(STATE_PATH, this.state);
      await onReply("已创建新会话。旧会话历史已保留，下一条消息会从新上下文开始。");
    }).catch(async (error) => {
      await onReply(`Pi error: ${error instanceof Error ? error.message : String(error)}`);
    });
    this.queues.set(key, next);
    await next;
  }

  async listResumeSessions(key: string, scope: ResumeScope, page: number): Promise<ResumeSessionPage> {
    const sessions = await this.getResumeSessions(key, scope);
    const normalizedPage = Math.max(0, Math.floor(page));
    const total = sessions.length;
    const totalPages = Math.max(1, Math.ceil(total / RESUME_PAGE_SIZE));
    const clampedPage = Math.min(normalizedPage, totalPages - 1);
    const currentSessionPath = this.normalizeSessionPath(this.state.sessions[key]);
    const start = clampedPage * RESUME_PAGE_SIZE;
    const items = sessions.slice(start, start + RESUME_PAGE_SIZE).map((session) => {
      const sessionPath = this.normalizeSessionPath(session.path) || session.path;
      return {
        path: session.path,
        title: session.name?.trim() || summarizeFirstMessage(session.firstMessage),
        subtitle: session.name?.trim()
          ? summarizeFirstMessage(session.firstMessage)
          : `消息数：${session.messageCount}`,
        modifiedLabel: formatModifiedLabel(session.modified),
        workspaceLabel: scope === "all" ? formatWorkspaceLabel(session.cwd) : undefined,
        isCurrent: Boolean(currentSessionPath && sessionPath && currentSessionPath === sessionPath),
      };
    });

    return {
      key,
      scope,
      page: clampedPage,
      total,
      totalPages,
      items,
    };
  }

  async resumeConversation(key: string, sessionPathInput: string, onReply: (text: string) => Promise<void>) {
    if (this.activeRuns.has(key)) {
      await onReply("当前还有进行中的处理，请先发送 /stop，再切换历史会话。");
      return;
    }

    const previous = this.previousTurn(key);
    const next = previous.then(async () => {
      const sessionPath = this.normalizeExistingSessionPath(sessionPathInput);
      const sessionInfo = await this.findSessionInfo(sessionPath);
      if (!sessionInfo) {
        await onReply("这条历史会话不存在，可能已经被删除。请重新打开 /resume 选择。");
        return;
      }

      const currentPath = this.normalizeSessionPath(this.state.sessions[key]);
      if (currentPath === sessionPath) {
        this.state.workspaces![key] = sessionInfo.cwd || this.getWorkspace(key);
        writeJson(STATE_PATH, this.state);
        await onReply(`你已经在这个历史会话里了。\n当前工作区：${this.state.workspaces![key]}`);
        return;
      }

      const cached = this.sessions.get(key);
      if (cached) {
        try { (await cached).dispose(); } catch {}
      }

      this.sessions.delete(key);
      this.state.sessions[key] = sessionPath;
      this.state.workspaces![key] = sessionInfo.cwd || this.cwd;
      writeJson(STATE_PATH, this.state);
      await onReply([
        `已切换到历史会话：${sessionInfo.name?.trim() || summarizeFirstMessage(sessionInfo.firstMessage)}`,
        `工作区：${this.state.workspaces![key]}`,
        "下一条消息会继续接着这个会话往下聊。",
      ].join("\n"));
    }).catch(async (error) => {
      await onReply(`Pi error: ${error instanceof Error ? error.message : String(error)}`);
    });
    this.queues.set(key, next);
    await next;
  }

  async selectModel(key: string, provider: string, modelId: string, onReply: (text: string) => Promise<void>) {
    const previous = this.previousTurn(key);
    const next = previous.then(async () => {
      const model = this.modelRegistry.find(provider, modelId);
      if (!model || !this.modelRegistry.hasConfiguredAuth(model)) {
        await onReply(`这个模型当前不可用：${provider}/${modelId}。请发送 /model 重新选择。`);
        return;
      }

      this.state.models![key] = { provider, id: modelId };
      writeJson(STATE_PATH, this.state);

      const cached = this.sessions.get(key);
      if (cached) {
        try { (await cached).dispose(); } catch {}
      }
      this.sessions.delete(key);
      await onReply(`已切换到 ${provider}/${modelId}。当前飞书会话后续都会使用这个模型。`);
    }).catch(async (error) => {
      await onReply(`Pi error: ${error instanceof Error ? error.message : String(error)}`);
    });
    this.queues.set(key, next);
    await next;
  }

  getWorkspace(key: string) {
    return this.state.workspaces?.[key] || this.cwd;
  }

  async switchWorkspace(key: string, workspaceInput: string | undefined, onReply: (text: string) => Promise<void>) {
    if (!workspaceInput) {
      const current = this.getWorkspace(key);
      await onReply([
        `当前工作区：${current}`,
        "用法：/workspace /绝对路径",
        "也支持：/workspace ~/your/project",
      ].join("\n"));
      return;
    }

    const previous = this.previousTurn(key);
    const next = previous.then(async () => {
      const workspace = resolveWorkspacePath(workspaceInput);
      const cached = this.sessions.get(key);
      if (cached) {
        try { (await cached).dispose(); } catch {}
      }
      this.sessions.delete(key);
      delete this.state.sessions[key];
      this.state.workspaces![key] = workspace;
      writeJson(STATE_PATH, this.state);
      await onReply(`已切换到工作区：${workspace}\n下一条消息会在这个目录里创建新的 Pi 会话。`);
    }).catch(async (error) => {
      await onReply(error instanceof Error ? error.message : `Pi error: ${String(error)}`);
    });
    this.queues.set(key, next);
    await next;
  }

  getAvailableModels() {
    return this.modelRegistry.getAvailable().sort((a, b) => {
      const providerCmp = a.provider.localeCompare(b.provider);
      if (providerCmp !== 0) return providerCmp;
      return a.id.localeCompare(b.id);
    });
  }

  getSelectedModel(key: string) {
    const selected = this.state.models?.[key];
    if (selected) {
      const model = this.modelRegistry.find(selected.provider, selected.id);
      if (model && this.modelRegistry.hasConfiguredAuth(model)) return model;
    }
    const cached = this.sessions.get(key);
    if (cached) {
      return cached.then((session) => session.model);
    }
    // Check settings default model before falling back to first available
    if (this.defaultProvider && this.defaultModelId) {
      const defaultModel = this.modelRegistry.find(this.defaultProvider, this.defaultModelId);
      if (defaultModel && this.modelRegistry.hasConfiguredAuth(defaultModel)) {
        return defaultModel;
      }
    }
    const available = this.getAvailableModels();
    return available[0];
  }

  resetMemory() {
    for (const session of this.sessions.values()) {
      void session.then((s) => s.dispose()).catch(() => undefined);
    }
    this.sessions.clear();
    this.queues.clear();
    this.startingRuns.clear();
    this.state = { sessions: {}, models: {}, workspaces: {} };
  }

  private getSession(key: string): Promise<AgentSession> {
    const cached = this.sessions.get(key);
    if (cached) return cached;
    const created = this.createSession(key);
    this.sessions.set(key, created);
    return created;
  }

  private async getFreshSessionForPrompt(key: string): Promise<AgentSession> {
    const cached = this.sessions.get(key);
    if (cached) {
      try { (await cached).dispose(); } catch {}
      this.sessions.delete(key);
    }
    return this.getSession(key);
  }

  private async queueSteering(
    key: string,
    active: ActiveRun,
    userText: string,
    images: Array<{ type: "image"; data: string; mimeType: string }>,
    onReply: (text: string) => Promise<void>,
  ) {
    if (active.stopped) {
      await onReply("当前任务正在停止，请稍后重试。");
      return;
    }

    const targetIndex = active.replyTargets.length;
    active.replyTargets.push(onReply);
    try {
      await active.session.steer(userText, images.length ? images : undefined);
    } catch (error) {
      active.replyTargets.splice(targetIndex, 1);
      const message = error instanceof Error ? error.message : String(error);
      debugLog("feishu.prompt.steering_error", { key, error: message });
      await onReply(`Pi error: ${message}`);
      return;
    }
    debugLog("feishu.prompt.steering_queued", { key, textLength: userText.length, imageCount: images.length });
    await onReply("已追加到当前任务队列");
  }

  private captureUserTurnReplyTarget(key: string) {
    const active = this.activeRuns.get(key);
    if (!active || active.stopped) return;
    const target = active.replyTargets.shift();
    if (target) active.currentTurnReplyTargets.push(target);
  }

  private deliverAssistantTurnResult(key: string, message: any, _toolResults: unknown) {
    const active = this.activeRuns.get(key);
    if (!active || active.stopped) return;
    if (!isAssistantAnswerMessage(message)) return;

    const answer = extractAssistantText(message);
    if (!answer) return;

    const target = active.currentTurnReplyTargets[0] || active.replyTargets.shift() || active.fallbackReply;
    active.currentTurnReplyTargets = [];
    active.assistantReplyCount += 1;
    const sequence = active.assistantReplyCount;
    debugLog("feishu.prompt.turn_result", { key, sequence, answerLength: answer.length });
    const reply = target(answer).catch((error) => {
      debugLog("feishu.prompt.turn_reply_error", {
        key,
        sequence,
        error: error instanceof Error ? error.message : String(error),
      });
    });
    active.assistantReplies.push(reply);
  }

  private previousTurn(key: string) {
    const previous = this.queues.get(key) || Promise.resolve();
    return previous.catch((error) => {
      debugLog("feishu.queue.previous_error", {
        key,
        error: error instanceof Error ? error.message : String(error),
      });
    });
  }

  private async createSession(key: string): Promise<AgentSession> {
    const workspaceCwd = this.getWorkspace(key);
    ensureWorkspaceExists(workspaceCwd);
    const existingFile = this.state.sessions[key];
    const selected = this.state.models?.[key];
    const model = selected ? this.modelRegistry.find(selected.provider, selected.id) : undefined;
    const sessionManager = existingFile && existsSync(existingFile)
      ? SessionManager.open(existingFile, undefined, workspaceCwd)
      : SessionManager.create(workspaceCwd);

    const loader = new DefaultResourceLoader({
      cwd: workspaceCwd,
      agentDir: getAgentDir(),
      systemPromptOverride: (base) => {
        const extra = "You are replying through Feishu/Lark. Keep answers concise and readable in chat. Do not use markdown tables.";
        return base?.trim() ? `${base}\n\n${extra}` : extra;
      },
    });

    const previousChildEnv = process.env[CHILD_SESSION_ENV];
    process.env[CHILD_SESSION_ENV] = "1";
    try {
      await loader.reload();
    } finally {
      if (previousChildEnv === undefined) delete process.env[CHILD_SESSION_ENV];
      else process.env[CHILD_SESSION_ENV] = previousChildEnv;
    }

    const { session } = await createAgentSession({
      cwd: workspaceCwd,
      agentDir: getAgentDir(),
      authStorage: this.authStorage,
      modelRegistry: this.modelRegistry,
      model,
      sessionManager,
      resourceLoader: loader,
    });

    await session.bindExtensions({});
    this.bridge?.attachSession(key, session.sessionId);
    session.subscribe((event) => {
      this.activeRuns.get(key)?.status?.updateFromEvent(event);
      if (event.type === "message_end") {
        this.bridge?.handleMessageEnd(session.sessionId, key, event.message);
        if ((event.message as any)?.role === "user") this.captureUserTurnReplyTarget(key);
      }
      if (event.type === "turn_end") {
        this.deliverAssistantTurnResult(key, (event as any).message, (event as any).toolResults);
      }
    });

    if (session.sessionFile && this.state.sessions[key] !== session.sessionFile) {
      this.state.sessions[key] = session.sessionFile;
      writeJson(STATE_PATH, this.state);
    }
    return session;
  }

  private async getResumeSessions(key: string, scope: ResumeScope) {
    const base = scope === "all"
      ? await SessionManager.listAll()
      : await SessionManager.list(this.getWorkspace(key));
    return [...base].sort((a, b) => toTimeMs(b.modified) - toTimeMs(a.modified));
  }

  private async findSessionInfo(sessionPath: string): Promise<SessionInfo | undefined> {
    const currentWorkspace = this.getWorkspaceFromSessionFile(sessionPath);
    const localSessions = currentWorkspace ? await SessionManager.list(currentWorkspace) : [];
    const normalizedTarget = this.normalizeSessionPath(sessionPath);
    const fromLocal = localSessions.find((item) => this.normalizeSessionPath(item.path) === normalizedTarget);
    if (fromLocal) return fromLocal;
    const allSessions = await SessionManager.listAll();
    return allSessions.find((item) => this.normalizeSessionPath(item.path) === normalizedTarget);
  }

  private getWorkspaceFromSessionFile(sessionPath: string) {
    try {
      return SessionManager.open(sessionPath).getCwd();
    } catch {
      return undefined;
    }
  }

  private normalizeExistingSessionPath(path: string) {
    if (!path || !existsSync(path)) {
      throw new Error("历史会话不存在，可能已经被删除。");
    }
    return realpathSync(path);
  }

  private normalizeSessionPath(path: string | undefined) {
    if (!path) return undefined;
    try {
      return existsSync(path) ? realpathSync(path) : path;
    } catch {
      return path;
    }
  }
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number, timeoutMessage: string): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timer = setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

function extractLastAssistantText(session: AgentSession): string {
  const messages = [...(session.messages || [])].reverse();
  for (const msg of messages as any[]) {
    if (!isAssistantAnswerMessage(msg)) continue;
    const text = extractAssistantText(msg);
    if (text) return text;
  }
  return "";
}

function isAssistantAnswerMessage(msg: any): boolean {
  if (msg?.role !== "assistant") return false;
  if (msg.stopReason === "error" || msg.stopReason === "aborted" || msg.stopReason === "toolUse") return false;
  if (!Array.isArray(msg.content)) return typeof msg.content === "string";
  return !msg.content.some((p) => p?.type === "toolCall");
}
function extractAssistantText(msg: any): string {
  if (msg?.role !== "assistant") return "";
  if (msg.stopReason === "error" || msg.stopReason === "aborted") return "";
  const content = msg.content;
  if (typeof content === "string") return content.trim();
  if (Array.isArray(content)) {
    return content
      .map((p) => p?.type === "text" ? p.text : "")
      .join("")
      .trim();
  }
  return "";
}

function resolveWorkspacePath(input: string) {
  const trimmed = input.trim();
  if (!trimmed) {
    throw new Error("请在 /workspace 后面带上目录路径，例如：/workspace /Users/ax/project");
  }

  const expanded = trimmed === "~" || trimmed.startsWith("~/")
    ? join(homedir(), trimmed.slice(2))
    : trimmed;

  if (!isAbsolute(expanded)) {
    throw new Error("当前只支持绝对路径或 ~/ 开头的路径。");
  }

  const resolved = resolve(expanded);
  ensureWorkspaceExists(resolved);
  return realpathSync(resolved);
}

function ensureWorkspaceExists(path: string) {
  if (!existsSync(path)) {
    throw new Error(`工作区不存在：${path}`);
  }

  let stat;
  try {
    stat = statSync(path);
  } catch {
    throw new Error(`无法访问工作区：${path}`);
  }

  if (!stat.isDirectory()) {
    throw new Error(`工作区不是目录：${path}`);
  }
}

function summarizeFirstMessage(text: string) {
  const normalized = (text || "").replace(/\s+/g, " ").trim();
  if (!normalized) return "未命名会话";
  return normalized.length > 36 ? `${normalized.slice(0, 35)}...` : normalized;
}

function formatModifiedLabel(value: Date | string) {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "未知";
  const yyyy = date.getFullYear();
  const mm = String(date.getMonth() + 1).padStart(2, "0");
  const dd = String(date.getDate()).padStart(2, "0");
  const hh = String(date.getHours()).padStart(2, "0");
  const min = String(date.getMinutes()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd} ${hh}:${min}`;
}

function formatWorkspaceLabel(cwd: string) {
  if (!cwd) return "(unknown)";
  return `${basename(cwd)} · ${cwd}`;
}

function toTimeMs(value: Date | string) {
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? 0 : date.getTime();
}
