import { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Send,
  MessageSquare,
  User,
  Bot,
  Clock,
  Sparkles,
  Trash2,
  ChevronDown,
} from "lucide-react";

/* ================================================================
   Markdown component overrides (compact version for chat bubbles)
   ================================================================ */
const chatMarkdownComponents = {
  /* Keep paragraphs tight */
  p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,

  /* Headings — scaled down for chat context */
  h1: ({ children }) => <h4 className="text-sm font-bold text-bright mt-3 mb-1">{children}</h4>,
  h2: ({ children }) => <h4 className="text-sm font-bold text-bright mt-2 mb-1">{children}</h4>,
  h3: ({ children }) => <h5 className="text-xs font-semibold text-bright mt-2 mb-1">{children}</h5>,

  /* Lists */
  ul: ({ children }) => <ul className="list-disc pl-4 mb-2 space-y-0.5">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal pl-4 mb-2 space-y-0.5">{children}</ol>,
  li: ({ children }) => <li className="text-[13px]">{children}</li>,

  /* Code blocks — compact dark style */
  pre: ({ children }) => (
    <pre className="my-2 p-3 bg-void rounded-md border border-border-subtle overflow-x-auto text-[12px] leading-relaxed">
      {children}
    </pre>
  ),
  code: ({ className, children, ...props }) => {
    const isBlock = className?.startsWith("language-");
    if (isBlock) {
      return <code className={`${className} text-text`} {...props}>{children}</code>;
    }
    return (
      <code
        className="text-neon-dim bg-surface-alt px-1 py-0.5 rounded text-[12px] border border-border-subtle"
        {...props}
      >
        {children}
      </code>
    );
  },

  /* Links */
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-neon-dim underline underline-offset-2 hover:text-neon transition-colors"
    >
      {children}
    </a>
  ),

  /* Blockquotes */
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-neon-dim/40 pl-3 my-2 text-dim italic text-[13px]">
      {children}
    </blockquote>
  ),

  /* Tables — compact for chat */
  table: ({ children }) => (
    <div className="overflow-x-auto my-2 rounded border border-border">
      <table className="w-full text-[12px]">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="text-left px-2 py-1 bg-surface text-bright font-semibold border-b border-border text-[11px]">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="px-2 py-1 border-b border-border-subtle text-[12px]">{children}</td>
  ),

  /* Strong & emphasis */
  strong: ({ children }) => <strong className="font-semibold text-bright">{children}</strong>,
  em: ({ children }) => <em className="italic text-dim">{children}</em>,

  /* Horizontal rules */
  hr: () => <hr className="border-border my-2" />,
};

/* ================================================================
   Format seconds → "2:34" or "1:05:12"
   ================================================================ */
function formatTimestamp(seconds) {
  if (!seconds || seconds <= 0) return "0:00";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

/* ================================================================
   SOURCE CHIPS — collapsible section showing retrieved chunks
   ================================================================ */
function SourceChips({ sources }) {
  const [expanded, setExpanded] = useState(false);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-2 pt-2 border-t border-border-subtle/50">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-[11px] font-mono text-muted hover:text-neon-dim
                   transition-colors cursor-pointer"
      >
        <Sparkles size={10} />
        {sources.length} source{sources.length > 1 ? "s" : ""} referenced
        <ChevronDown
          size={10}
          className={`transition-transform duration-200 ${expanded ? "rotate-180" : ""}`}
        />
      </button>

      {expanded && (
        <div className="flex flex-wrap gap-1.5 mt-2 animate-fade-in-up">
          {sources.map((src, i) => (
            <span
              key={i}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full
                         bg-neon/[0.06] border border-neon/15 text-[10px] font-mono text-neon-dim"
              title={src.text?.slice(0, 120) || ""}
            >
              <Clock size={8} />
              {formatTimestamp(src.start_time)} – {formatTimestamp(src.end_time)}
              {src.chapter_title && (
                <span className="text-muted ml-0.5">· {src.chapter_title}</span>
              )}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/* ================================================================
   TYPING INDICATOR — three pulsing dots
   ================================================================ */
function TypingIndicator() {
  return (
    <div className="chat-bubble chat-bubble--ai animate-fade-in-up">
      <div className="flex items-center gap-2">
        <div className="flex gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-neon/70 animate-bounce [animation-delay:0ms] [animation-duration:1s]" />
          <span className="w-1.5 h-1.5 rounded-full bg-neon/70 animate-bounce [animation-delay:150ms] [animation-duration:1s]" />
          <span className="w-1.5 h-1.5 rounded-full bg-neon/70 animate-bounce [animation-delay:300ms] [animation-duration:1s]" />
        </div>
        <span className="text-[11px] text-muted font-mono">Thinking...</span>
      </div>
    </div>
  );
}

/* ================================================================
   CHAT PANEL — main component
   ================================================================ */

/**
 * Chat side-panel for conversing with the AI about the video.
 *
 * Sends multi-turn chat_history to the backend so follow-up
 * questions have full conversational context.
 *
 * Props:
 *  - videoId: string — YouTube video ID for scoping RAG retrieval
 */
export default function ChatPanel({ videoId }) {
  /* ── State ──────────────────────────────────────────────── */
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "I've analyzed the video and indexed its content. Ask me anything — concepts, code explanations, deeper dives, or quick summaries.",
      sources: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showScrollDown, setShowScrollDown] = useState(false);

  /* ── Refs ────────────────────────────────────────────────── */
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const inputRef = useRef(null);
  const textareaRef = useRef(null);

  /* ── Auto-scroll to newest message ──────────────────────── */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  /* ── Track scroll position for "scroll down" indicator ──── */
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      setShowScrollDown(scrollHeight - scrollTop - clientHeight > 100);
    };

    container.addEventListener("scroll", handleScroll, { passive: true });
    return () => container.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  /* ── Build chat_history array for the backend ───────────── */
  const buildChatHistory = useCallback(() => {
    // Exclude the initial welcome message and only send user + assistant turns
    return messages
      .filter((m) => m.role === "user" || (m.role === "assistant" && messages.indexOf(m) > 0))
      .slice(-10) // Last 10 turns to stay within token limits
      .map((m) => ({ role: m.role, content: m.content }));
  }, [messages]);

  /* ── Send message ───────────────────────────────────────── */
  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    // 1) Immediately append user message
    const userMessage = { role: "user", content: text, sources: [] };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    try {
      // 2) Build the request with full chat history
      const chatHistory = buildChatHistory();

      const response = await fetch("http://localhost:8000/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: videoId,
          message: text,
          chat_history: chatHistory,
          top_k: 5,
        }),
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => null);
        throw new Error(errBody?.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();

      // 3) Append assistant response with sources
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer || "I couldn't generate a response.",
          sources: data.sources || [],
          tokensUsed: data.tokens_used,
          processingTime: data.processing_time_ms,
        },
      ]);
    } catch (err) {
      console.error("Chat error:", err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `⚠ **Error**: ${err.message}\n\nPlease try again or rephrase your question.`,
          sources: [],
          isError: true,
        },
      ]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }, [input, isLoading, videoId, buildChatHistory]);

  /* ── Keyboard handler (Enter to send, Shift+Enter for newline) */
  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  /* ── Auto-resize textarea ───────────────────────────────── */
  const handleInputChange = useCallback((e) => {
    setInput(e.target.value);
    const textarea = e.target;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
  }, []);

  /* ── Clear conversation ─────────────────────────────────── */
  const handleClear = useCallback(() => {
    setMessages([
      {
        role: "assistant",
        content: "Conversation cleared. Ask me a new question about the video.",
        sources: [],
      },
    ]);
  }, []);

  /* ── Message count (excluding welcome) ──────────────────── */
  const messageCount = messages.length - 1;

  /* ================================================================
     RENDER
     ================================================================ */
  return (
    <div className="chat-panel">
      {/* ── Header ──────────────────────────────────────────── */}
      <div className="chat-header">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md bg-neon/10 border border-neon/20 flex items-center justify-center">
              <MessageSquare size={12} className="text-neon" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-bright font-mono tracking-wide leading-none">
                CHAT
              </h2>
              <p className="text-[10px] text-muted mt-0.5 font-mono">
                RAG-powered Q&A
              </p>
            </div>
          </div>

          {/* Clear + message count */}
          <div className="flex items-center gap-2">
            {messageCount > 0 && (
              <span className="text-[10px] font-mono text-muted">
                {messageCount} msg{messageCount !== 1 ? "s" : ""}
              </span>
            )}
            <button
              id="btn-clear-chat"
              onClick={handleClear}
              className="p-1.5 rounded-md text-muted hover:text-error hover:bg-error/10
                         border border-transparent hover:border-error/20
                         transition-all duration-150 cursor-pointer"
              title="Clear conversation"
            >
              <Trash2 size={12} />
            </button>
          </div>
        </div>
      </div>

      {/* ── Messages ────────────────────────────────────────── */}
      <div
        ref={messagesContainerRef}
        className="chat-messages relative"
      >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex gap-2.5 mb-4 animate-fade-in-up ${
              msg.role === "user" ? "flex-row-reverse" : "flex-row"
            }`}
          >
            {/* Avatar */}
            <div
              className={`shrink-0 w-6 h-6 rounded-full flex items-center justify-center mt-0.5 ${
                msg.role === "user"
                  ? "bg-neon/15 border border-neon/25"
                  : "bg-surface border border-border"
              }`}
            >
              {msg.role === "user" ? (
                <User size={11} className="text-neon" />
              ) : (
                <Bot size={11} className="text-dim" />
              )}
            </div>

            {/* Bubble */}
            <div
              className={`${
                msg.role === "user" ? "chat-bubble--user" : "chat-bubble--ai"
              } chat-bubble ${msg.isError ? "!border-error/30 !bg-error/5" : ""}`}
            >
              {msg.role === "user" ? (
                /* User messages: plain text */
                <div className="text-[13px] leading-relaxed whitespace-pre-wrap">
                  {msg.content}
                </div>
              ) : (
                /* Assistant messages: rendered Markdown */
                <div className="text-[13px] leading-relaxed chat-md">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={chatMarkdownComponents}
                  >
                    {msg.content}
                  </ReactMarkdown>
                </div>
              )}

              {/* Source chips (assistant only) */}
              {msg.role === "assistant" && msg.sources?.length > 0 && (
                <SourceChips sources={msg.sources} />
              )}

              {/* Processing metadata (assistant only) */}
              {msg.role === "assistant" && msg.processingTime > 0 && (
                <div className="flex items-center gap-2 mt-2 pt-1.5 border-t border-border-subtle/30">
                  <span className="text-[9px] font-mono text-muted/60">
                    {msg.processingTime}ms
                  </span>
                  {msg.tokensUsed > 0 && (
                    <span className="text-[9px] font-mono text-muted/60">
                      · {msg.tokensUsed.toLocaleString()} tokens
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {isLoading && (
          <div className="flex gap-2.5 mb-4">
            <div className="shrink-0 w-6 h-6 rounded-full flex items-center justify-center mt-0.5 bg-surface border border-border">
              <Bot size={11} className="text-dim" />
            </div>
            <TypingIndicator />
          </div>
        )}

        <div ref={messagesEndRef} />

        {/* Scroll to bottom indicator */}
        {showScrollDown && (
          <button
            onClick={scrollToBottom}
            className="sticky bottom-2 left-1/2 -translate-x-1/2 z-10
                       flex items-center gap-1 px-3 py-1 rounded-full
                       bg-surface/90 border border-border backdrop-blur-sm
                       text-[11px] font-mono text-muted hover:text-neon-dim
                       hover:border-neon-dim/30 transition-all duration-200
                       shadow-lg cursor-pointer"
          >
            <ChevronDown size={10} />
            New messages
          </button>
        )}
      </div>

      {/* ── Input area ──────────────────────────────────────── */}
      <div className="chat-input-area">
        {/* Suggestion chips (only when no conversation yet) */}
        {messages.length <= 1 && !isLoading && (
          <div className="flex flex-wrap gap-1.5 mb-2.5">
            {[
              "Summarize the key points",
              "Explain the main concept",
              "What are the takeaways?",
            ].map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => {
                  setInput(suggestion);
                  inputRef.current?.focus();
                }}
                className="px-2.5 py-1 rounded-full text-[11px] font-mono
                           text-muted bg-surface border border-border
                           hover:text-neon-dim hover:border-neon-dim/30 hover:bg-neon/[0.04]
                           transition-all duration-150 cursor-pointer"
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}

        <div className="flex items-end gap-2">
          <textarea
            ref={(el) => {
              textareaRef.current = el;
              inputRef.current = el;
            }}
            id="chat-input"
            className="flex-1 bg-transparent border border-border rounded-lg px-3 py-2
                       text-[13px] text-text placeholder-muted font-sans
                       focus:outline-none focus:border-neon-dim/40 focus:ring-1 focus:ring-neon/15
                       transition-all duration-200 resize-none leading-relaxed
                       min-h-[38px] max-h-[120px]"
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the video..."
            disabled={isLoading}
            rows={1}
          />
          <button
            id="btn-send-chat"
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="shrink-0 p-2 rounded-lg bg-neon/10 text-neon border border-neon/20
                       hover:bg-neon/20 hover:border-neon/40
                       disabled:opacity-25 disabled:cursor-not-allowed
                       transition-all duration-200 cursor-pointer
                       active:scale-95"
            aria-label="Send message"
          >
            <Send size={14} />
          </button>
        </div>

        <p className="text-[9px] text-muted/50 mt-1.5 font-mono text-center">
          Shift+Enter for new line · Responses may include timestamps
        </p>
      </div>
    </div>
  );
}
