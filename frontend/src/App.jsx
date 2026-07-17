import { useState, useCallback } from "react";
import TerminalInput from "./components/TerminalInput";
import NotesViewer from "./components/NotesViewer";
import ChatPanel from "./components/ChatPanel";

/**
 * Main application shell.
 *
 * Manages three global states:
 *  - videoStatus: "idle" | "processing" | "completed"
 *  - notesMarkdown: raw Markdown string of the generated study notes
 *  - videoId: the YouTube video ID extracted from the submitted URL
 *
 * Layout behaviour:
 *  idle / processing → full-screen centred terminal
 *  completed         → 70 / 30 split  (Notes | Chat)
 */
export default function App() {
  const [videoStatus, setVideoStatus] = useState("idle"); // idle | processing | completed
  const [notesMarkdown, setNotesMarkdown] = useState("");
  const [videoId, setVideoId] = useState("");

  /* ── TerminalInput callbacks ────────────────────────────── */

  const handleProcessingStart = useCallback(() => {
    setVideoStatus("processing");
  }, []);

  const handleProcessingComplete = useCallback((data) => {
    setVideoId(data.videoId || "");
    setNotesMarkdown(data.notes || "");
    setVideoStatus("completed");
  }, []);

  const handleProcessingError = useCallback((errorMsg) => {
    console.error("Pipeline error:", errorMsg);
    setVideoStatus("idle");
  }, []);

  /* ── Called when user clicks "New Video" to reset ───────── */
  const handleReset = useCallback(() => {
    setVideoStatus("idle");
    setNotesMarkdown("");
    setVideoId("");
  }, []);

  /* ================================================================
     RENDER
     ================================================================ */
  return (
    <div className="relative min-h-screen bg-void overflow-hidden">
      {/* ── Ambient background glow ─────────────────────────── */}
      <div
        className="pointer-events-none fixed inset-0 z-0"
        aria-hidden="true"
      >
        <div className="absolute top-[-20%] left-[10%] w-[500px] h-[500px] rounded-full bg-neon/[0.02] blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[5%] w-[400px] h-[400px] rounded-full bg-neon/[0.015] blur-[100px]" />
      </div>

      {/* ── Idle / Processing: Full-screen terminal ─────────── */}
      {videoStatus !== "completed" && (
        <div className="relative z-10 flex items-center justify-center min-h-screen px-4 py-8">
          <div className="w-full max-w-3xl animate-fade-in-up">
            {/* Logo / Title */}
            <div className="text-center mb-8">
              <h1 className="font-mono text-4xl font-bold tracking-tight text-neon glow-text mb-2">
                AMISSIO
              </h1>
              <p className="font-mono text-sm text-muted tracking-widest uppercase">
                Multimodal RAG System v1.0
              </p>
            </div>

            <TerminalInput
              onProcessingStart={handleProcessingStart}
              onProcessingComplete={handleProcessingComplete}
              onProcessingError={handleProcessingError}
            />

            {/* Footer hint (only in idle state) */}
            {videoStatus === "idle" && (
              <p className="text-center text-muted text-xs mt-6 font-mono">
                Paste a YouTube URL and press{" "}
                <kbd className="px-1.5 py-0.5 rounded bg-surface border border-border text-dim text-[11px]">
                  Enter
                </kbd>{" "}
                to begin
              </p>
            )}
          </div>
        </div>
      )}

      {/* ── Completed: Split layout (Notes 70% + Chat 30%) ── */}
      {videoStatus === "completed" && (
        <div className="relative z-10 flex h-screen">
          {/* Notes viewer */}
          <main className="w-[70%] h-full overflow-y-auto">
            <NotesViewer
              markdown={notesMarkdown}
              videoId={videoId}
              onReset={handleReset}
            />
          </main>

          {/* Chat panel */}
          <aside className="w-[30%] h-full border-l border-border">
            <ChatPanel videoId={videoId} />
          </aside>
        </div>
      )}
    </div>
  );
}

