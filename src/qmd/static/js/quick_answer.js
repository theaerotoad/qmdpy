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
      collapseContainer: document.getElementById("quick-answer-collapse-container"),
      fade: document.getElementById("quick-answer-fade"),
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

    // Proactively kill upstream LLM request on the backend
    try {
      const sid = (typeof currentSessionId !== "undefined" && currentSessionId) ? currentSessionId : "default";
      fetch("api/quick_answer/abort", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sid }),
        keepalive: true,
      }).catch(function () {});
    } catch (e) {}

    const els = getElements();
    if (els.card) {
      els.card.classList.add("hidden");
    }
    if (els.content) {
      els.content.innerHTML = "";
    }
    if (els.collapseContainer) {
      els.collapseContainer.classList.add("max-h-[140px]", "cursor-pointer");
      els.collapseContainer.classList.remove("max-h-[2000px]");
    }
    if (els.fade) {
      // Clear inline display style so Tailwind's lg:hidden can apply on desktop
      els.fade.style.display = "";
    }
    currentResults = [];
    citationSources = {};
    accumulatedText = "";
  }

  function unescapeXml(str) {
    if (!str) return "";
    return str
      .replace(/&quot;/g, '"')
      .replace(/&apos;/g, "'")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&#39;/g, "'")
      .replace(/&#039;/g, "'")
      .replace(/&#34;/g, '"')
      .replace(/&#034;/g, '"')
      .replace(/&#(\d+);/g, function (_, dec) { return String.fromCharCode(parseInt(dec, 10)); })
      .replace(/&#x([0-9a-fA-F]+);/g, function (_, hex) { return String.fromCharCode(parseInt(hex, 16)); })
      .replace(/&amp;/g, "&");
  }

  function parseXmlAttrs(attrStr) {
    const attrs = {};
    if (!attrStr) return attrs;
    const re = /([a-zA-Z_:][a-zA-Z0-9_:-]*)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;
    let match;
    while ((match = re.exec(attrStr)) !== null) {
      attrs[match[1]] = unescapeXml(match[2] !== undefined ? match[2] : match[3]);
    }
    return attrs;
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

    // 2. Parse raw XML context for complete rank/attribute resolution without
    // triggering browser DOMParser XML parsing error logs in Firefox.
    if (xmlContext && typeof xmlContext === "string") {
      let nodeIdx = 0;

      // 2a. Match <document ...>...</document> structures (grouped doc mode or discover mode)
      const docRegex = /<document\b([^>]*)>([\s\S]*?)<\/document>/gi;
      let docMatch;
      let docCount = 0;

      while ((docMatch = docRegex.exec(xmlContext)) !== null) {
        docCount++;
        const docAttrs = parseXmlAttrs(docMatch[1]);
        const innerContent = docMatch[2];

        const docCollection = docAttrs.collection || "";
        const docPath = docAttrs.path || "";
        const docTitle = docAttrs.title || docPath || `Document ${docCount}`;

        // Look for nested <chunk ...>...</chunk> tags inside this document
        const chunkRegex = /<chunk\b([^>]*)>([\s\S]*?)<\/chunk>/gi;
        let chunkMatch;
        let hasChunks = false;

        while ((chunkMatch = chunkRegex.exec(innerContent)) !== null) {
          hasChunks = true;
          nodeIdx++;
          const chunkAttrs = parseXmlAttrs(chunkMatch[1]);
          const chunkText = unescapeXml(chunkMatch[2].trim());

          const srcObj = {
            collection: chunkAttrs.collection || docCollection,
            path: chunkAttrs.path || docPath,
            title: chunkAttrs.title || docTitle,
            text: chunkText,
          };

          const rank = chunkAttrs.rank;
          const seq = chunkAttrs.seq || chunkAttrs.seq_id;
          const id = chunkAttrs.id;
          const index = chunkAttrs.index;

          if (rank && (!map[rank] || !map[rank].text)) map[rank] = srcObj;
          if (seq && !map[seq]) map[seq] = srcObj;
          if (id && (!map[id] || !map[id].text)) map[id] = srcObj;
          if (index && (!map[index] || !map[index].text)) map[index] = srcObj;

          const fallbackIdx = nodeIdx.toString();
          if (!map[fallbackIdx]) map[fallbackIdx] = srcObj;
        }

        if (!hasChunks) {
          // Discover mode: <document ...>text</document>
          nodeIdx++;
          const docText = unescapeXml(innerContent.trim());
          const srcObj = {
            collection: docCollection,
            path: docPath,
            title: docTitle,
            text: docText,
          };

          const rank = docAttrs.rank;
          const seq = docAttrs.top_chunk_seq || docAttrs.seq || docAttrs.seq_id;
          const id = docAttrs.id;

          if (rank && (!map[rank] || !map[rank].text)) map[rank] = srcObj;
          if (seq && !map[seq]) map[seq] = srcObj;
          if (id && (!map[id] || !map[id].text)) map[id] = srcObj;

          const docNum = docCount.toString();
          if (!map[docNum]) map[docNum] = srcObj;
          const fallbackIdx = nodeIdx.toString();
          if (!map[fallbackIdx]) map[fallbackIdx] = srcObj;
        }
      }

      // 2b. Match flat <result ...>...</result> tags (passages mode)
      const resultRegex = /<result\b([^>]*)>([\s\S]*?)<\/result>/gi;
      let resMatch;
      let resCount = 0;

      while ((resMatch = resultRegex.exec(xmlContext)) !== null) {
        resCount++;
        nodeIdx++;
        const resAttrs = parseXmlAttrs(resMatch[1]);
        const resText = unescapeXml(resMatch[2].trim());

        const coll = resAttrs.collection || "";
        const path = resAttrs.path || resAttrs.document || "";
        const title = resAttrs.title || path || `Result ${resCount}`;

        const srcObj = {
          collection: coll,
          path: path,
          title: title,
          text: resText,
        };

        const rank = resAttrs.rank;
        const seq = resAttrs.seq || resAttrs.seq_id;
        const id = resAttrs.id;
        const index = resAttrs.index;

        if (rank && (!map[rank] || !map[rank].text)) map[rank] = srcObj;
        if (seq && !map[seq]) map[seq] = srcObj;
        if (id && (!map[id] || !map[id].text)) map[id] = srcObj;
        if (index && (!map[index] || !map[index].text)) map[index] = srcObj;

        const resNum = resCount.toString();
        if (!map[resNum]) map[resNum] = srcObj;
        const fallbackIdx = nodeIdx.toString();
        if (!map[fallbackIdx]) map[fallbackIdx] = srcObj;
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
    if (typeof featureStates !== "undefined" && !featureStates.offer_llm) {
      hideQuickAnswer();
      return;
    }

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
      const sid = (typeof currentSessionId !== "undefined" && currentSessionId) ? currentSessionId : "default";
      const response = await fetch("api/quick_answer", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          xml: xmlContext,
          session_id: sid,
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
              isCheckingRelevance = false;
              els.loading.classList.add("hidden");
              els.body.classList.remove("hidden");
              els.content.innerHTML = `<div class="mb-2 text-sm font-semibold text-red-500 dark:text-red-400">LLM Error</div><div class="text-xs text-red-600 dark:text-red-300 break-words">${escapeHtmlText(data.error)}</div>`;
              finishStream(els);
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

  /**
   * Hook global tab switching and query reset functions to immediately abort
   * in-flight Quick Answer requests.
   */
  function setupSmartsHooks() {
    if (typeof window.setSearchTab === "function" && !window.setSearchTab._qaHooked) {
      const origSetSearchTab = window.setSearchTab;
      window.setSearchTab = function (tabName) {
        hideQuickAnswer();
        return origSetSearchTab.apply(this, arguments);
      };
      window.setSearchTab._qaHooked = true;
    }

    if (typeof window.returnToHero === "function" && !window.returnToHero._qaHooked) {
      const origReturnToHero = window.returnToHero;
      window.returnToHero = function () {
        hideQuickAnswer();
        return origReturnToHero.apply(this, arguments);
      };
      window.returnToHero._qaHooked = true;
    }

    if (typeof window.clearQuery === "function" && !window.clearQuery._qaHooked) {
      const origClearQuery = window.clearQuery;
      window.clearQuery = function () {
        hideQuickAnswer();
        return origClearQuery.apply(this, arguments);
      };
      window.clearQuery._qaHooked = true;
    }
  }

  // Attempt hooking immediately if state.js is already loaded
  setupSmartsHooks();

  // Setup UI event handlers
  document.addEventListener("DOMContentLoaded", function () {
    setupSmartsHooks();
    const els = getElements();
    if (els.closeBtn) {
      els.closeBtn.addEventListener("click", hideQuickAnswer);
    }

    // Cancel in-flight quick answer when user clicks any tab button
    document.querySelectorAll("#tab-all, #tab-passages, #tab-documents").forEach(function (tabEl) {
      tabEl.addEventListener("click", function () {
        hideQuickAnswer();
      });
    });

    // Cancel in-flight quick answer when user starts typing a new query
    ["serp-query", "hero-query"].forEach(function (id) {
      const inputEl = document.getElementById(id);
      if (inputEl) {
        inputEl.addEventListener("input", function () {
          if (activeAbortController) {
            hideQuickAnswer();
          }
        });
      }
    });

    const clearBtn = document.getElementById("clear-btn");
    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        hideQuickAnswer();
      });
    }

    window.addEventListener("beforeunload", function () {
      if (activeAbortController) {
        activeAbortController.abort();
        activeAbortController = null;
      }
    });

    // Event delegation for citation clicks, qmd:// links, and mobile expand
    if (els.collapseContainer) {
      els.collapseContainer.addEventListener("click", function (e) {
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
          return;
        }

        // Expand behavior on mobile if not clicking a link/citation
        if (window.innerWidth < 1024 && els.collapseContainer.classList.contains("max-h-[140px]")) {
          els.collapseContainer.classList.remove("max-h-[140px]", "cursor-pointer");
          els.collapseContainer.classList.add("max-h-[2000px]");
          if (els.fade) els.fade.style.display = "none";
        }
      });
    }
  });

  // Expose public API on window for search integration
  window.QuickAnswer = {
    trigger: triggerQuickAnswer,
    hide: hideQuickAnswer,
    abort: hideQuickAnswer,
  };
})();
