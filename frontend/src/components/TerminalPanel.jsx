import { useState, useRef, useEffect, useCallback } from "react";

const FAKE_LOGS = [
  { text: "[+] Validating YouTube URL...", delay: 400 },
  { text: "[+] Connecting to backend service...", delay: 600 },
  { text: "[+] Downloading audio stream...", delay: 1200 },
  { text: "[+] Extracting keyframes from video...", delay: 1800 },
  { text: "[+] Running speech-to-text transcription...", delay: 2200 },
  { text: "[+] Chunking transcript into segments...", delay: 1000 },
  { text: "[+] Generating vector embeddings...", delay: 1400 },
  { text: "[+] Storing embeddings in Qdrant...", delay: 800 },
  { text: "[+] Analyzing visual content from keyframes...", delay: 1600 },
  { text: "[+] Building multimodal context graph...", delay: 1200 },
  { text: "[+] Generating structured study notes...", delay: 2000 },
  { text: "[+] Formatting output with Markdown...", delay: 800 },
];

/**
 * Terminal-style input panel.
 *
 * Props:
 *  - videoStatus: "idle" | "processing"
 *  - onSubmit(url: string): void
 */
export default function TerminalPanel({ videoStatus, onSubmit }) {
  const [inputValue, setInputValue] = useState("");
  const [logLines, setLogLines] = useState([]);
  const inputRef = useRef(null);
  const bodyRef = useRef(null);

  /* Focus the input on mount */
  useEffect(() => {
    if (videoStatus === "idle" && inputRef.current) {
      inputRef.current.focus();
    }
  }, [videoStatus]);

  /* Auto-scroll log to bottom */
  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [logLines]);

  /* Stream fake logs when processing */
  useEffect(() => {
    if (videoStatus !== "processing") return;

    let i = 0;
    let totalDelay = 0;
    const timers = [];

    FAKE_LOGS.forEach((log) => {
      totalDelay += log.delay;
      const timer = setTimeout(() => {
        setLogLines((prev) => [...prev, log.text]);
      }, totalDelay);
      timers.push(timer);
    });

    return () => timers.forEach(clearTimeout);
  }, [videoStatus]);

  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === "Enter" && inputValue.trim()) {
        e.preventDefault();
        setLogLines([`$ process "${inputValue.trim()}"`]);
        onSubmit(inputValue.trim());
        setInputValue("");
      }
    },
    [inputValue, onSubmit]
  );

  return (
    <div className="terminal-container scanline-effect">
      {/* ── Title bar ───────────────────────────────────────── */}
      <div className="terminal-header">
        <span className="terminal-dot terminal-dot--red" />
        <span className="terminal-dot terminal-dot--yellow" />
        <span className="terminal-dot terminal-dot--green" />
        <span className="terminal-title">amissio — zsh</span>
      </div>

      {/* ── Body ────────────────────────────────────────────── */}
      <div ref={bodyRef} className="terminal-body relative min-h-[260px] max-h-[420px]">
        {/* Previous log lines */}
        {logLines.map((line, idx) => (
          <div key={idx} className="terminal-log-line animate-fade-in-up">
            {line}
          </div>
        ))}

        {/* Input row (only when idle) */}
        {videoStatus === "idle" && (
          <div className="flex items-center gap-1 mt-1">
            <span className="terminal-prompt shrink-0">
              root@rag-system<span className="text-dim">:</span>
              <span className="text-neon-dim">~</span>
              <span className="text-dim">#</span>
            </span>
            <input
              ref={inputRef}
              id="terminal-input"
              type="text"
              className="terminal-input"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="paste youtube url here..."
              spellCheck={false}
              autoComplete="off"
            />
            {!inputValue && <span className="terminal-cursor" />}
          </div>
        )}

        {/* Processing indicator */}
        {videoStatus === "processing" && (
          <div className="flex items-center gap-2 mt-3">
            <span className="terminal-prompt">
              root@rag-system<span className="text-dim">:</span>
              <span className="text-neon-dim">~</span>
              <span className="text-dim">#</span>
            </span>
            <span className="terminal-cursor" />
          </div>
        )}
      </div>
    </div>
  );
}
