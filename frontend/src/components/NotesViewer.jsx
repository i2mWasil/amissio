import { useState, useEffect, useCallback, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowLeft,
  ExternalLink,
  ChevronUp,
  Copy,
  Check,
  Clock,
  Hash,
  FileText,
  BookOpen,
} from "lucide-react";

/* ================================================================
   TIMESTAMP REGEX
   Matches patterns like [12:34], [1:23:45], (05:12), 12:34, etc.
   ================================================================ */
const TIMESTAMP_RE = /\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?/g;

/**
 * Parse a timestamp string like "12:34" or "1:23:45" into total seconds.
 */
function parseTimestamp(ts) {
  const parts = ts.split(":").map(Number);
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  return parts[0] * 60 + parts[1];
}

/**
 * Splits a text string so that timestamp patterns become interactive pills
 * while preserving the surrounding text as plain React nodes.
 */
function renderTimestamps(text) {
  if (typeof text !== "string") return text;

  const parts = [];
  let lastIndex = 0;
  let match;

  TIMESTAMP_RE.lastIndex = 0; // reset regex state
  while ((match = TIMESTAMP_RE.exec(text)) !== null) {
    // Push text before the match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    const raw = match[1]; // e.g. "12:34"
    const seconds = parseTimestamp(raw);

    parts.push(
      <button
        key={`ts-${match.index}`}
        className="timestamp-pill"
        onClick={() => {
          console.log(`⏱ Timestamp clicked: ${raw} (${seconds}s)`);
        }}
        title={`Jump to ${raw}`}
      >
        <Clock size={10} />
        {raw}
      </button>
    );

    lastIndex = match.index + match[0].length;
  }

  // Push remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length > 0 ? parts : text;
}

/* ================================================================
   CODE BLOCK with language header + copy button
   ================================================================ */
function CodeBlock({ className, children }) {
  const [copied, setCopied] = useState(false);
  const language = className?.replace("language-", "") || "";
  const code = String(children).replace(/\n$/, "");

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [code]);

  return (
    <div className="code-block-wrapper">
      <div className="code-block-header">
        <span className="lang-label">
          <span className="lang-dot" />
          {language || "plain text"}
        </span>
        <button className={`code-copy-btn ${copied ? "copied" : ""}`} onClick={handleCopy}>
          {copied ? <Check size={10} /> : <Copy size={10} />}
          {copied ? "copied" : "copy"}
        </button>
      </div>
      <pre>
        <code className={className}>{code}</code>
      </pre>
    </div>
  );
}

/* ================================================================
   HEADING with anchor link
   ================================================================ */
function HeadingWithAnchor({ level, children }) {
  const Tag = `h${level}`;
  const text = extractText(children);
  const id = text
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .trim();

  return (
    <Tag id={id} className="heading-anchor">
      <a href={`#${id}`} className="anchor-link" aria-hidden="true">
        <Hash size={level <= 2 ? 16 : 14} />
      </a>
      {children}
    </Tag>
  );
}

/** Recursively extract plain text from React children */
function extractText(children) {
  if (typeof children === "string") return children;
  if (Array.isArray(children)) return children.map(extractText).join("");
  if (children?.props?.children) return extractText(children.props.children);
  return "";
}

/* ================================================================
   CUSTOM PARAGRAPH — injects timestamp pills into text nodes
   ================================================================ */
function ParagraphWithTimestamps({ children }) {
  const processed = processChildren(children);
  return <p>{processed}</p>;
}

function ListItemWithTimestamps({ children }) {
  const processed = processChildren(children);
  return <li>{processed}</li>;
}

/**
 * Walk the children tree, replacing string nodes that contain
 * timestamp patterns with interactive pills.
 */
function processChildren(children) {
  if (typeof children === "string") return renderTimestamps(children);
  if (Array.isArray(children)) return children.map((child, i) => {
    if (typeof child === "string") return <span key={i}>{renderTimestamps(child)}</span>;
    return child;
  });
  return children;
}

/* ================================================================
   REACT-MARKDOWN COMPONENT OVERRIDES
   ================================================================ */
const markdownComponents = {
  /* Headings */
  h1: ({ children }) => <HeadingWithAnchor level={1}>{children}</HeadingWithAnchor>,
  h2: ({ children }) => <HeadingWithAnchor level={2}>{children}</HeadingWithAnchor>,
  h3: ({ children }) => <HeadingWithAnchor level={3}>{children}</HeadingWithAnchor>,
  h4: ({ children }) => <HeadingWithAnchor level={4}>{children}</HeadingWithAnchor>,
  h5: ({ children }) => <HeadingWithAnchor level={5}>{children}</HeadingWithAnchor>,
  h6: ({ children }) => <HeadingWithAnchor level={6}>{children}</HeadingWithAnchor>,

  /* Paragraphs — inject timestamp pills */
  p: ParagraphWithTimestamps,

  /* List items — inject timestamp pills */
  li: ListItemWithTimestamps,

  /* Code — distinguish inline vs. block */
  code: ({ className, children, ...props }) => {
    const isBlock = className?.startsWith("language-") || String(children).includes("\n");
    if (isBlock) {
      return <CodeBlock className={className}>{children}</CodeBlock>;
    }
    // Inline code — just use the styled <code> from CSS
    return <code className={className} {...props}>{children}</code>;
  },

  /* Pre — pass through (CodeBlock handles its own <pre>) */
  pre: ({ children }) => {
    // If the child is already a CodeBlock wrapper, don't double-wrap
    if (children?.type === CodeBlock) return children;
    // For code blocks, react-markdown wraps <code> inside <pre>
    // Extract the code child and render as CodeBlock
    if (children?.props?.className?.startsWith("language-") || children?.props?.children) {
      return (
        <CodeBlock className={children.props.className}>
          {children.props.children}
        </CodeBlock>
      );
    }
    return <pre>{children}</pre>;
  },

  /* Tables — add wrapper for horizontal scroll on narrow viewports */
  table: ({ children }) => (
    <div className="overflow-x-auto rounded-lg">
      <table>{children}</table>
    </div>
  ),

  /* Links — external links get an icon */
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1">
      {children}
      <ExternalLink size={11} className="opacity-50" />
    </a>
  ),

  /* Blockquotes — style with a neon accent */
  blockquote: ({ children }) => (
    <blockquote className="flex gap-2">
      <div className="shrink-0 mt-1">
        <BookOpen size={14} className="text-neon-dim opacity-60" />
      </div>
      <div>{children}</div>
    </blockquote>
  ),
};

/* ================================================================
   NOTES VIEWER — main component
   ================================================================ */

/**
 * Renders the generated study notes as beautifully styled Markdown.
 *
 * Props:
 *  - markdown: string        — raw Markdown from the backend
 *  - videoId: string          — YouTube video ID
 *  - onReset: () => void     — resets the app to the terminal view
 */
export default function NotesViewer({ markdown, videoId, onReset }) {
  const [showBackToTop, setShowBackToTop] = useState(false);

  /* ── Scroll listener for back-to-top button ─────────────── */
  useEffect(() => {
    const container = document.querySelector("main");
    if (!container) return;

    const handleScroll = () => {
      setShowBackToTop(container.scrollTop > 400);
    };

    container.addEventListener("scroll", handleScroll, { passive: true });
    return () => container.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToTop = useCallback(() => {
    const container = document.querySelector("main");
    container?.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  /* ── Estimate reading time ──────────────────────────────── */
  const readingTime = useMemo(() => {
    if (!markdown) return 0;
    const words = markdown.split(/\s+/).length;
    return Math.max(1, Math.ceil(words / 200));
  }, [markdown]);

  return (
    <div className="min-h-screen bg-void">
      {/* ── Sticky top bar ──────────────────────────────────── */}
      <header
        className="sticky top-0 z-20 flex items-center justify-between px-8 py-3
                   bg-void/80 backdrop-blur-md border-b border-border"
      >
        <div className="flex items-center gap-4">
          <button
            id="btn-new-video"
            onClick={onReset}
            className="flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-mono text-dim
                       hover:text-neon hover:bg-neon/[0.06] border border-border hover:border-neon/30
                       transition-all duration-200 cursor-pointer"
          >
            <ArrowLeft size={14} />
            New Video
          </button>

          {/* Reading time badge */}
          <span className="flex items-center gap-1.5 text-xs font-mono text-muted">
            <FileText size={12} />
            {readingTime} min read
          </span>
        </div>

        {videoId && (
          <a
            href={`https://www.youtube.com/watch?v=${videoId}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs font-mono text-muted
                       hover:text-neon-dim transition-colors"
          >
            <ExternalLink size={12} />
            {videoId}
          </a>
        )}
      </header>

      {/* ── Notes content ───────────────────────────────────── */}
      <article className="prose-notes px-8 py-8 max-w-none animate-fade-in-up">
        {markdown ? (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={markdownComponents}
          >
            {markdown}
          </ReactMarkdown>
        ) : (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <FileText size={48} className="text-border mb-4" />
            <p className="text-muted font-mono text-sm">No notes generated.</p>
            <p className="text-muted/60 font-mono text-xs mt-1">
              Process a video to see your study notes here.
            </p>
          </div>
        )}
      </article>

      {/* ── Back to top ─────────────────────────────────────── */}
      {showBackToTop && (
        <button
          id="btn-back-to-top"
          onClick={scrollToTop}
          className="back-to-top animate-fade-in-up"
          aria-label="Back to top"
        >
          <ChevronUp size={18} />
        </button>
      )}
    </div>
  );
}
