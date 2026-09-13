/**
 * AI Quick Answer streaming client.
 * Triggers on search results load, streams answer in a side panel,
 * and handles clickable citations that open the reference material in the slide-over drawer.
 */
(function () {
  let activeAbortController = null;
  let accumulatedText = "";
  let isCheckingRelevance = true;
  let currentResults = [];

  function getElements() {
    return {
      card: document.getElementById("quick-answer-card"),
      loading: document.getElementById("quick-answer-loading"),
      body: document.getElementById("quick-answer-body"),
      content: document.getElementById("quick-answer-content"),
      pulse: document.getElementById("quick-answer-pulse"),
      footer: document.getElementById("quick-answer-footer"),
      closeBtn: document.getElementById("quick-answer-close-btn"),
    };
  }

  function hideQuickAnswer() {
    if (activeAbortController) {
      activeAbortController.abort();
      activeAbortController = null;
    }
    const els = getElements();
    if (els.card) {
      els.card.classList.add("hidden");
    }
    if (els.content) {
      els.content.innerHTML = "";
    }
    currentResults = [];
    accumulatedText = "";
  }

  function escapeHtmlText(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  /**
   * Parses Markdown and transforms numeric citations like [1], [2], [1, 2] into interactive badges.
   */
  function renderMarkdownWithCitations(markdown) {
    if (!markdown) return "";

    let html = "";
    if (typeof marked !== "undefined" && typeof marked.parse === "function") {
      try {
        html = marked.parse(markdown, { breaks: true, gfm: true });
      } catch (e) {
        html = escapeHtmlText(markdown).replace(/\n\n+/g, '<p class="my-1.5"></p>').replace(/\n/g, "<br>");
      }
    } else {
      html = escapeHtmlText(markdown)
        .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-zinc-900 dark:text-zinc-100">$1</strong>')
        .replace(/(^|[^\*])\*(?!\*)([^\*]+)\*(?!\*)/g, '$1<em>$2</em>')
        .replace(/`([^`]+)`/g, '<code class="rounded bg-zinc-100 px-1 py-0.5 text-[11px] font-mono text-zinc-800 dark:bg-zinc-800 dark:text-zinc-200">$1</code>')
        .replace(/\n\n+/g, '<p class="my-1.5"></p>')
        .replace(/\n/g, "<br>");
    }

    // Replace citations like [1], [2], [1, 2] with interactive badges
    html = html.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, function (match, group) {
      const nums = group.split(",").map(function (s) { return s.trim(); }).filter(Boolean);
      return nums.map(function (numStr) {
        const idx = parseInt(numStr, 10) - 1;
        if (currentResults && currentResults[idx]) {
          const item = currentResults[idx];
          const title = item.title || item.path || `Source [${numStr}]`;
          return `<button type="button" data-citation-idx="${idx}" class="inline-flex items-center justify-center font-mono font-semibold text-[10px] text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-950/70 hover:bg-indigo-100 dark:hover:bg-indigo-900/80 border border-indigo-200 dark:border-indigo-800/80 rounded px-1.5 py-0.2 mx-0.5 align-baseline cursor-pointer transition-colors" title="${escapeHtmlText(title)}">[${numStr}]</button>`;
        }
        return `<span class="font-mono text-[10px] text-zinc-400 dark:text-zinc-500 font-semibold">[${numStr}]</span>`;
      }).join("");
    });

    return html;
  }

  /**
   * Opens the slide-over document drawer for a given citation index.
   */
  window.quickAnswerOpenCitation = function (event, idx) {
    if (event) {
      event.preventDefault();
      event.stopPropagation();
    }
    if (!currentResults || !currentResults[idx]) return;

    const item = currentResults[idx];
    const collection = item.collection || "";
    const path = item.path || "";
    const targetText = (item.chunks && item.chunks.length > 0)
      ? (item.chunks[0].text || "")
      : (item.text || (item.snippets ? item.snippets[0] : ""));

    if (typeof window.openDocument === "function") {
      window.openDocument(collection, path, targetText);
    } else {
      console.warn("openDocument function not found on window object.");
    }
  };

  /**
   * Main entry point: begins streaming the Quick Answer for a query using search XML.
   */
  async function triggerQuickAnswer(query, xmlContext, results = []) {
    if (!query || !xmlContext || xmlContext.trim().length === 0) {
      hideQuickAnswer();
      return;
    }

    currentResults = Array.isArray(results) ? results : [];

    // Cancel any ongoing stream
    if (activeAbortController) {
      activeAbortController.abort();
    }
    activeAbortController = new AbortController();

    const els = getElements();
    if (!els.card) return;

    // Initialize UI state
    accumulatedText = "";
    isCheckingRelevance = true;
    els.card.classList.remove("hidden");
    els.loading.classList.remove("hidden");
    els.body.classList.add("hidden");
    els.content.innerHTML = "";
    if (els.pulse) els.pulse.classList.remove("hidden");
    if (els.footer) els.footer.classList.add("hidden");

    try {
      const response = await fetch("/api/quick_answer", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          xml: xmlContext,
        }),
        signal: activeAbortController.signal,
      });

      if (!response.ok || !response.body) {
        hideQuickAnswer();
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // Keep incomplete line

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;

          const dataStr = trimmed.slice(5).trim();
          if (dataStr === "[DONE]") {
            finishStream(els);
            return;
          }

          try {
            const data = JSON.parse(dataStr);
            if (data.error) {
              hideQuickAnswer();
              return;
            }
            if (data.delta) {
              accumulatedText += data.delta;

              // Relevance Check: If LLM answers NOT RELEVANT, immediately dismiss
              if (isCheckingRelevance) {
                const normalized = accumulatedText.trim().toUpperCase();
                if (normalized.startsWith("NOT RELEVANT")) {
                  hideQuickAnswer();
                  return;
                }
                // If it starts producing content other than NOT RELEVANT, reveal the body
                if (normalized.length > 12 || (!normalized.startsWith("NOT") && normalized.length > 3)) {
                  isCheckingRelevance = false;
                  els.loading.classList.add("hidden");
                  els.body.classList.remove("hidden");
                }
              }

              if (!isCheckingRelevance) {
                els.content.innerHTML = renderMarkdownWithCitations(accumulatedText);
              }
            }
          } catch (e) {
            // Ignore malformed chunk
          }
        }
      }

      finishStream(els);
    } catch (err) {
      if (err.name !== "AbortError") {
        console.warn("Quick answer stream aborted or failed:", err);
      }
      hideQuickAnswer();
    }
  }

  function finishStream(els) {
    if (els.pulse) els.pulse.classList.add("hidden");
    if (els.footer && !els.body.classList.contains("hidden")) {
      els.footer.classList.remove("hidden");
    }
    activeAbortController = null;
  }

  // Setup UI event handlers
  document.addEventListener("DOMContentLoaded", function () {
    const els = getElements();
    if (els.closeBtn) {
      els.closeBtn.addEventListener("click", hideQuickAnswer);
    }

    // Event delegation for citation clicks and qmd:// links
    if (els.content) {
      els.content.addEventListener("click", function (e) {
        const btn = e.target.closest("[data-citation-idx]");
        if (btn) {
          e.preventDefault();
          e.stopPropagation();
          const idx = parseInt(btn.getAttribute("data-citation-idx"), 10);
          window.quickAnswerOpenCitation(e, idx);
          return;
        }

        const qmdLink = e.target.closest('a[href^="qmd://"]');
        if (qmdLink) {
          e.preventDefault();
          e.stopPropagation();
          try {
            const href = qmdLink.getAttribute("href");
            const parsedUrl = new URL(href.replace("qmd://", "http://qmd.local/"));
            const collection = parsedUrl.searchParams.get("collection") || "";
            const path = parsedUrl.searchParams.get("path") || "";
            const text = parsedUrl.searchParams.get("text") || "";
            if (typeof window.openDocument === "function") {
              window.openDocument(collection, path, text);
            }
          } catch (err) {
            console.warn("Could not parse qmd link:", err);
          }
        }
      });
    }
  });

  // Expose public API on window for search integration
  window.QuickAnswer = {
    trigger: triggerQuickAnswer,
    hide: hideQuickAnswer,
  };
})();