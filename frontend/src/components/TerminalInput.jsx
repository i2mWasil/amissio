import { useState, useRef, useEffect, useCallback } from "react";
import { Loader2 } from "lucide-react";

/* ================================================================
   Fake processing logs — displayed one-by-one with staggered delays
   while the real API request is in-flight.
   ================================================================ */
const PROCESSING_LOGS = [
  { text: "[SYS]  Validating YouTube URL format...",               type: "info",    delay: 300  },
  { text: "[OK]   URL accepted — resolving video metadata",        type: "success", delay: 500  },
  { text: "[+]    Connecting to download service...",              type: "info",    delay: 700  },
  { text: "[+]    Downloading audio stream (opus@128k)...",        type: "info",    delay: 1400 },
  { text: "[+]    Downloading video stream (mp4@720p)...",         type: "info",    delay: 1800 },
  { text: "[+]    Fetching YouTube captions...",                   type: "info",    delay: 800  },
  { text: "[OK]   Media download complete",                        type: "success", delay: 400  },
  { text: "[+]    Extracting audio for Whisper transcription...",  type: "info",    delay: 1200 },
  { text: "[+]    Running speech-to-text (Whisper)...",            type: "info",    delay: 2200 },
  { text: "[OK]   Transcript generated — segmenting...",           type: "success", delay: 600  },
  { text: "[+]    Extracting keyframes from video...",             type: "info",    delay: 1600 },
  { text: "[+]    Deduplicating frames (perceptual hash)...",      type: "info",    delay: 1000 },
  { text: "[+]    Running OCR on unique keyframes...",             type: "info",    delay: 1400 },
  { text: "[+]    Gemini Vision — analysing visual content...",    type: "info",    delay: 2000 },
  { text: "[OK]   Visual analysis complete",                       type: "success", delay: 500  },
  { text: "[+]    Building multimodal timeline...",                type: "info",    delay: 900  },
  { text: "[+]    Semantic chunking (topic + slide boundaries)...",type: "info",    delay: 1200 },
  { text: "[+]    Generating vector embeddings...",                type: "info",    delay: 1400 },
  { text: "[+]    Storing embeddings in Qdrant...",                type: "info",    delay: 800  },
  { text: "[OK]   Vector store populated",                         type: "success", delay: 400  },
  { text: "[+]    Map-Reduce note generation (Gemini 2.5)...",     type: "info",    delay: 2400 },
  { text: "[+]    Merging segment notes into final document...",   type: "info",    delay: 1600 },
  { text: "[+]    Generating glossary + quiz questions...",        type: "info",    delay: 1200 },
  { text: "[+]    Formatting output as Markdown...",               type: "info",    delay: 800  },
];

/* After all fake logs play, these keep cycling until the API returns */
const WAITING_MESSAGES = [
  "[…]    Pipeline in progress — finalising notes",
  "[…]    Still processing — almost there",
  "[…]    Waiting for Gemini response",
  "[…]    Optimising note structure",
];

/**
 * TerminalInput — hacker-style terminal for YouTube URL submission.
 *
 * Responsibilities:
 *  1. Accept a YouTube URL from the user.
 *  2. POST it to the backend at /generate-notes.
 *  3. Stream fake processing logs while waiting.
 *  4. On success, pass { notes, videoId, title, summary } to the parent.
 *  5. On error, display the error in the terminal.
 *
 * Props:
 *  - onProcessingStart()           — signals the parent that processing began
 *  - onProcessingComplete(data)    — returns the API response payload
 *  - onProcessingError(errorMsg)   — returns the error message string
 */
export default function TerminalInput({
  onProcessingStart,
  onProcessingComplete,
  onProcessingError,
}) {
  /* ── Local state ────────────────────────────────────────── */
  const [inputValue, setInputValue] = useState("");
  const [logLines, setLogLines] = useState([]);        // { text, type }
  const [isProcessing, setIsProcessing] = useState(false);
  const [hasError, setHasError] = useState(false);

  /* ── Refs ────────────────────────────────────────────────── */
  const inputRef = useRef(null);
  const bodyRef = useRef(null);
  const timersRef = useRef([]);        // setTimeout IDs for cleanup
  const waitingIntervalRef = useRef(null);
  const apiDoneRef = useRef(false);    // flag: API finished before logs end

  /* ── Focus input on mount ───────────────────────────────── */
  useEffect(() => {
    if (!isProcessing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isProcessing]);

  /* ── Auto-scroll the terminal body ──────────────────────── */
  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [logLines]);

  /* ── Cleanup all timers on unmount ──────────────────────── */
  useEffect(() => {
    return () => {
      timersRef.current.forEach(clearTimeout);
      if (waitingIntervalRef.current) clearInterval(waitingIntervalRef.current);
    };
  }, []);

  /* ── Append a single log line ───────────────────────────── */
  const appendLog = useCallback((text, type = "info") => {
    setLogLines((prev) => [...prev, { text, type }]);
  }, []);

  /* ── Stream fake logs with staggered delays ─────────────── */
  const startFakeLogs = useCallback(() => {
    let cumulativeDelay = 0;

    PROCESSING_LOGS.forEach((log) => {
      cumulativeDelay += log.delay;
      const timer = setTimeout(() => {
        // Don't add more fake logs if the API already returned an error
        if (apiDoneRef.current) return;
        appendLog(log.text, log.type);
      }, cumulativeDelay);
      timersRef.current.push(timer);
    });

    // After all scripted logs, start a "waiting" cycle
    const waitStart = cumulativeDelay + 800;
    const waitTimer = setTimeout(() => {
      if (apiDoneRef.current) return;
      let idx = 0;
      waitingIntervalRef.current = setInterval(() => {
        if (apiDoneRef.current) {
          clearInterval(waitingIntervalRef.current);
          return;
        }
        appendLog(WAITING_MESSAGES[idx % WAITING_MESSAGES.length], "info");
        idx++;
      }, 3000);
    }, waitStart);
    timersRef.current.push(waitTimer);
  }, [appendLog]);

  /* ── Stop all fake-log timers ───────────────────────────── */
  const stopFakeLogs = useCallback(() => {
    apiDoneRef.current = true;
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
    if (waitingIntervalRef.current) {
      clearInterval(waitingIntervalRef.current);
      waitingIntervalRef.current = null;
    }
  }, []);

  /* ── Submit URL → run pipeline ──────────────────────────── */
  const handleSubmit = useCallback(
    async (url) => {
      // Reset state
      setHasError(false);
      apiDoneRef.current = false;
      setIsProcessing(true);

      // Echo the command to the terminal
      setLogLines([
        { text: `$ amissio generate-notes "${url}"`, type: "command" },
        { text: "", type: "info" }, // blank line spacer
      ]);

      // Notify parent
      onProcessingStart?.();

      // Start fake logs in parallel with the real API call
      startFakeLogs();

      try {
        const response = await fetch("http://localhost:8000/api/v1/generate-notes", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url }),
        });

        // Stop fake logs immediately
        stopFakeLogs();

        if (!response.ok) {
          const errorBody = await response.json().catch(() => null);
          const detail =
            errorBody?.detail || errorBody?.error || `HTTP ${response.status}`;
          throw new Error(detail);
        }

        const data = await response.json();

        // Success logs
        appendLog("", "info");
        appendLog("[OK]   Notes generated successfully!", "success");
        appendLog(
          `[OK]   ${data.title || "Video"} (${formatDuration(data.duration)})`,
          "success"
        );
        appendLog(
          `[OK]   ${(data.notes || "").length.toLocaleString()} characters of Markdown produced`,
          "success"
        );
        appendLog("", "info");
        appendLog("[SYS]  Opening notes viewer...", "info");

        // Small delay so the user can see the success logs
        setTimeout(() => {
          onProcessingComplete?.({
            videoId: data.video_id,
            title: data.title,
            notes: data.notes,
            summary: data.summary,
            duration: data.duration,
            channel: data.channel,
            thumbnailUrl: data.thumbnail_url,
          });
        }, 1200);
      } catch (err) {
        stopFakeLogs();
        setHasError(true);

        appendLog("", "info");
        appendLog(`[ERR]  Pipeline failed: ${err.message}`, "error");
        appendLog("[SYS]  Type a new URL to retry.", "info");

        setIsProcessing(false);
        onProcessingError?.(err.message);
      }
    },
    [onProcessingStart, onProcessingComplete, onProcessingError, startFakeLogs, stopFakeLogs, appendLog]
  );

  /* ── Keyboard handler ───────────────────────────────────── */
  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === "Enter" && inputValue.trim() && !isProcessing) {
        e.preventDefault();
        const url = inputValue.trim();
        setInputValue("");
        handleSubmit(url);
      }
    },
    [inputValue, isProcessing, handleSubmit]
  );

  /* ── Map log type → CSS class ───────────────────────────── */
  const logTypeClass = (type) => {
    switch (type) {
      case "success": return "terminal-log-line--success";
      case "error":   return "terminal-log-line--error";
      case "warn":    return "terminal-log-line--warn";
      case "command": return "text-bright font-bold";
      default:        return "terminal-log-line";
    }
  };

  /* ================================================================
     RENDER
     ================================================================ */
  return (
    <div className={`terminal-container scanline-effect ${isProcessing ? "animate-pulse-glow" : ""}`}>
      {/* ── macOS title bar ─────────────────────────────────── */}
      <div className="terminal-header">
        <div className="flex items-center gap-2">
          <span className="terminal-dot terminal-dot--red" />
          <span className="terminal-dot terminal-dot--yellow" />
          <span className="terminal-dot terminal-dot--green" />
        </div>
        <span className="terminal-title">amissio — zsh</span>
        {isProcessing && (
          <div className="ml-auto flex items-center gap-1.5">
            <Loader2 size={12} className="animate-spin text-neon-dim" />
            <span className="text-[11px] text-neon-dim font-mono">processing</span>
          </div>
        )}
      </div>

      {/* ── Terminal body ───────────────────────────────────── */}
      <div
        ref={bodyRef}
        className="terminal-body relative min-h-[280px] max-h-[460px]"
      >
        {/* Welcome message (only when no logs yet) */}
        {logLines.length === 0 && (
          <div className="mb-4 text-dim leading-relaxed">
            <div>Welcome to <span className="text-neon">Amissio</span> Multimodal RAG System</div>
            <div className="text-muted text-xs mt-1">
              Paste a YouTube URL below to generate structured study notes.
            </div>
            <div className="border-t border-border mt-3 mb-3" />
          </div>
        )}

        {/* Log lines */}
        {logLines.map((line, idx) => (
          <div
            key={idx}
            className={`${logTypeClass(line.type)} animate-fade-in-up`}
            style={{ animationDelay: `${Math.min(idx * 20, 200)}ms` }}
          >
            {line.text || "\u00A0" /* non-breaking space for blank lines */}
          </div>
        ))}

        {/* ── Prompt + input (visible when NOT processing) ── */}
        {!isProcessing && (
          <div className="flex items-center gap-1 mt-2">
            <span className="terminal-prompt shrink-0">
              <span className="text-neon">root</span>
              <span className="text-dim">@</span>
              <span className="text-neon-dim">rag-system</span>
              <span className="text-dim">:</span>
              <span className="text-neon-dim">~</span>
              <span className="text-dim">#</span>
            </span>

            <input
              ref={inputRef}
              id="terminal-url-input"
              type="text"
              className="terminal-input"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="paste youtube url here..."
              spellCheck={false}
              autoComplete="off"
              autoFocus
            />

            {/* Blinking cursor when input is empty */}
            {!inputValue && <span className="terminal-cursor" />}
          </div>
        )}

        {/* ── Processing cursor (visible when processing) ── */}
        {isProcessing && (
          <div className="flex items-center gap-2 mt-3">
            <span className="terminal-prompt shrink-0">
              <span className="text-neon">root</span>
              <span className="text-dim">@</span>
              <span className="text-neon-dim">rag-system</span>
              <span className="text-dim">:</span>
              <span className="text-neon-dim">~</span>
              <span className="text-dim">#</span>
            </span>
            <span className="terminal-cursor" />
          </div>
        )}
      </div>

      {/* ── Bottom status bar ───────────────────────────────── */}
      <div className="flex items-center justify-between px-4 py-2 border-t border-border bg-surface/50 rounded-b-lg">
        <span className="text-[11px] font-mono text-muted">
          {isProcessing
            ? `${logLines.length} operations logged`
            : hasError
              ? "Ready — enter a new URL"
              : "Ready"}
        </span>
        <span className="text-[11px] font-mono text-muted">
          POST /generate-notes
        </span>
      </div>
    </div>
  );
}

/* ── Helper: format seconds → "4m 32s" ────────────────────── */
function formatDuration(seconds) {
  if (!seconds || seconds <= 0) return "0s";
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}
