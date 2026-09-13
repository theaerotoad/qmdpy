/**
 * AI Quick Answer streaming client.
 * Triggers on search results load, streams answer in a side panel,
 * and handles clickable citations that open the reference material in the slide-over drawer.
 */
(function () {
  let activeAbortController = null;
  let accumulatedText = "";
  let isCheckingRelevance = true;

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
    accumulatedText = "";
  }

  /**
   * Parses simple Markdown and transforms qmd:// citations into interactive slide-over triggers.
   */
  function renderMarkdownWithCitations(markdown) {
    if (!markdown) return "";

    let html = escapeHtmlText(markdown);

    // Bold: **text**
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-zinc-900 dark:text-zinc-100">$1</strong>');
    
    // Italic: *text*
    html = html.replace(/(^|[^\*])\*(?!\*)([^\*]+)\*(?!\*)/g, '$1<em>$2</em>');

    // Inline code: `code`
    html = html.replace(/`([^`]+)`/g, '<code class="rounded bg-zinc-100 px-1 py-0.5 text-[11px] font-mono text-zinc-800 dark:bg-zinc-800 dark:text-zinc-200">$1</code>');

    // Citations: [Title](qmd://open?collection=...&path=...&text=...)
    html = html.replace(
      /\[(.*?)\]\((qmd:\/\/open\?[^\)]+)\)/g,
      function (match, label, uri) {
        return createCitationLink(label, uri);
      }
    );

    // Newlines to line breaks
    html = html.replace(/\n\n+/g, '<p class="my-1.5"></p>');
    html = html.replace(/\n/g, "<br>");

    return html;
  }

  function escapeHtmlText(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function createCitationLink(label, uri) {
    try {
      const parsedUrl = new URL(uri.replace("qmd://", "http://qmd.local/"));
      const collection = encodeURIComponent(parsedUrl.searchParams.get("collection") || "");
      const path = encodeURIComponent(parsedUrl.searchParams.get("path") || "");
      const text = encodeURIComponent(parsedUrl.searchParams.get("text") || "");

      return `<a href="#" 
                 onclick="window.quickAnswerOpenRef(event, '${collection}', '${path}', '${text}')" 
                 class="inline-flex items-center gap-0.5 rounded bg-indigo-50 px-1.5 py-0.5 text-[11px] font-medium text-indigo-700 hover:bg-indigo-100 hover:underline dark:bg-indigo-950/50 dark:text-indigo-300 dark:hover:bg-indigo-900/60" 
                 title="Open in document viewer">
                 <svg class="h-2.5 w-2.5 opacity-70" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor">
                   <path stroke-linecap="round" stroke-linejoin="round" d="M13.5 6H5.25A2.25 2.25 0 003 8.25v10.5A2.25 2.25 0 005.25 21h10.5A2.25 2.25 0 0018 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" />
                 </svg>
                 ${label}
              </a>`;
    } catch (e) {
      return `<span class="underline decoration-dotted">${label}</span>`;
    }
  }

  /**
   * Opens the slide-over document drawer using existing modals.js functions.
   */
  window.quickAnswerOpenRef = function (event, collectionEncoded, pathEncoded, textEncoded) {
    if (event) {
      event.preventDefault();
      event.stopPropagation();
    }
    const collection = decodeURIComponent(collectionEncoded || "");
    const path = decodeURIComponent(pathEncoded || "");
    const text = decodeURIComponent(textEncoded || "");

    if (typeof window.openDocument === "function") {
      window.openDocument(collection, path, text);
    } else {
      console.warn("openDocument function not found on window object.");
    }
  };

  /**
   * Main entry point: begins streaming the Quick Answer for a query using search XML.
   */
  async function triggerQuickAnswer(query, xmlContext) {
    if (!query || !xmlContext || xmlContext.trim().length === 0) {
      hideQuickAnswer();
      return;
    }

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
  });

  // Expose public API on window for search integration
  window.QuickAnswer = {
    trigger: triggerQuickAnswer,
    hide: hideQuickAnswer,
  };
})();