/**
 * AI Quick Answer streaming client.
 * Triggers on search results load, streams answer in a side panel,
 * and handles clickable citations that open the reference material in the slide-over drawer.
 */
(function () {
  let activeAbortController = null;
  let accumulatedText = "";
  let isCheckingRelevance = true;
  let citationSources = {};
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
    citationSources = {};
    accumulatedText = "";
  }

  /**
   * Builds a comprehensive lookup dictionary for all numeric citation keys (e.g. rank, seq, doc index)
   * from both the search results array and the raw search XML.
   */
  function buildCitationSources(results, xmlContext) {
    const map = {};

    // 1. Traverse results object/array
    if (Array.isArray(results)) {
      results.forEach((item, docIdx) => {
        const docNum = (docIdx + 1).toString();
        const docTitle = item.title || item.path || `Document ${docNum}`;
        const docCollection = item.collection || "";
        const docPath = item.path || "";

        // Grouped document view with multiple chunks
        if (Array.isArray(item.chunks) && item.chunks.length > 0) {
          item.chunks.forEach((chunk) => {
            const chunkObj = {
              collection: docCollection,
              path: docPath,
              title: docTitle,
              text: chunk.text || "",
            };

            if (chunk.rank !== undefined && chunk.rank !== null) {
              map[chunk.rank.toString()] = chunkObj;
            }
            if (chunk.seq_id !== undefined && chunk.seq_id !== null) {
              const seqKey = chunk.seq_id.toString();
              if (!map[seqKey]) map[seqKey] = chunkObj;
            }
          });

          if (!map[docNum]) {
            map[docNum] = {
              collection: docCollection,
              path: docPath,
              title: docTitle,
              text: item.chunks[0].text || "",
            };
          }
        } else {
          // Flat chunk or discover mode
          const text = item.text || (item.snippets && item.snippets[0]) || "";
          const itemObj = {
            collection: docCollection,
            path: docPath,
            title: docTitle,
            text: text,
          };

          map[docNum] = itemObj;
          if (item.rank !== undefined && item.rank !== null) {
            map[item.rank.toString()] = itemObj;
          }
        }
      });
    }

    // 2. Parse raw XML context for complete rank/attribute resolution
    if (xmlContext && typeof DOMParser !== "undefined") {
      try {
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(xmlContext, "text/xml");
        const nodes = xmlDoc.querySelectorAll("chunk, result, document, doc, match");
        nodes.forEach((node, nodeIdx) => {
          const rank = node.getAttribute("rank");
          const index = node.getAttribute("index");
          const id = node.getAttribute("id");
          const seq = node.getAttribute("seq") || node.getAttribute("seq_id");

          const parentDoc = node.closest("document, doc");
          const collection = node.getAttribute("collection") || (parentDoc && parentDoc.getAttribute("collection")) || "";
          const path = node.getAttribute("path") || (parentDoc && parentDoc.getAttribute("path")) || "";
          const title = node.getAttribute("title") || (parentDoc && parentDoc.getAttribute("title")) || path;
          const text = node.textContent ? node.textContent.trim() : "";

          const srcObj = { collection, path, title, text };

          if (rank && (!map[rank] || !map[rank].text)) map[rank] = srcObj;
          if (index && (!map[index] || !map[index].text)) map[index] = srcObj;
          if (id && (!map[id] || !map[id].text)) map[id] = srcObj;
          if (seq && !map[seq]) map[seq] = srcObj;

          const fallbackIdx = (nodeIdx + 1).toString();
          if (!map[fallbackIdx]) map[fallbackIdx] = srcObj;
        });
      } catch (e) {
        console.warn("Could not parse XML for citations:", e);
      }
    }

    return map;
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
        const src = citationSources[numStr];
        if (src) {
          const title = src.title || src.path || `Source [${numStr}]`;
          return `<button type="button" data-citation-key="${escapeHtmlText(numStr)}" class="inline-flex items-center justify-center font-mono font-semibold text-[10px] text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-950/70 hover:bg-indigo-100 dark:hover:bg-indigo-900/80 border border-indigo-200 dark:border-indigo-800/80 rounded px-1.5 py-0.2 mx-0.5 align-baseline cursor-pointer transition-colors" title="${escapeHtmlText(title)}">[${numStr}]</button>`;
        }
        return `<span class="font-mono text-[10px] text-zinc-400 dark:text-zinc-500 font-semibold">[${numStr}]</span>`;
      }).join("");
    });

    return html;
  }

  /**
   * Opens the slide-over document drawer for a given citation key.
   */
  window.quickAnswerOpenCitation = function (event, citationKey) {
    if (event) {
      event.preventDefault();
      event.stopPropagation();
    }
    const src = citationSources[citationKey];
    if (!src) return;

    const collection = src.collection || "";
    const path = src.path || "";
    const targetText = src.text || "";

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
    citationSources = buildCitationSources(currentResults, xmlContext);

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
        const btn = e.target.closest("[data-citation-key]");
        if (btn) {
          e.preventDefault();
          e.stopPropagation();
          const key = btn.getAttribute("data-citation-key");
          window.quickAnswerOpenCitation(e, key);
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