// XML Formatter for Results
function formatResultsToXml(results, type, query, sessionId) {
    const sessionAttr = sessionId ? ` session_id="${escapeXmlAttr(sessionId)}"` : '';
    if (type === 'doc') {
        const lines = [`<search_results query="${escapeXmlAttr(query)}"${sessionAttr}>`];
        results.forEach(d => {
            lines.push(`  <document uri="${escapeXmlAttr(d.collection ? `qmd://${d.collection}/${d.path}` : d.path)}" title="${escapeXmlAttr(d.title || '')}">`);
            (d.chunks || []).forEach(c => {
                lines.push(`    <chunk seq="${c.seq_id || 0}">\n${(c.text || '').trim()}\n    </chunk>`);
            });
            lines.push(`  </document>`);
        });
        lines.push('</search_results>');
        return lines.join('\n');
    } else {
        const lines = [`<search_results query="${escapeXmlAttr(query)}"${sessionAttr}>`];
        results.forEach((r, i) => {
            lines.push(`  <result rank="${i + 1}" document="${escapeXmlAttr(r.collection ? `qmd://${r.collection}/${r.path}` : r.path)}" title="${escapeXmlAttr(r.title || '')}">\n${(r.text || '').trim()}\n  </result>`);
        });
        lines.push('</search_results>');
        return lines.join('\n');
    }
}

// Single Result XML Copy Handler
function copySingleXml(index, btnEl = null) {
    if (!lastRawJson || !lastRawJson[index]) return;
    const item = lastRawJson[index];
    const xml = formatResultsToXml([item], lastSearchType, document.getElementById('serp-query').value, currentSessionId);
    navigator.clipboard.writeText(xml).then(() => {
        if (btnEl) {
            const orig = btnEl.innerHTML;
            btnEl.innerHTML = `<span class="text-emerald-600 dark:text-emerald-400 font-semibold">Copied!</span>`;
            setTimeout(() => { btnEl.innerHTML = orig; }, 1800);
        }
    });
}

// Form Submit & Search Execution
async function handleFormSubmit(e, source) {
    e.preventDefault();
    const rawVal = document.getElementById(`${source}-query`).value;
    const parsed = parseQueryDirectives(rawVal);
    if (!parsed.cleanQuery && !rawVal.trim()) return;

    saveQueryToHistory(rawVal);
    document.getElementById('hero-query').value = rawVal;
    document.getElementById('serp-query').value = rawVal;
    setAppState('serp');

    const resultsEl = document.getElementById('results');
    const statsEl = document.getElementById('result-stats');
    const toolbarEl = document.getElementById('results-toolbar');

    resultsEl.innerHTML = `<div class="flex items-center justify-center py-20 text-slate-500 gap-3"><svg class="animate-spin h-5 w-5 text-blue-500" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg><span>Searching knowledge base...</span></div>`;
    toolbarEl.classList.add('hidden');

    const scopedPaths = selectedScopes.filter(s => s.path).map(s => s.path);
    if (parsed.path && !scopedPaths.includes(parsed.path)) scopedPaths.push(parsed.path);

    const collFilter = parsed.collection || document.getElementById(`${source}-collection-filter`).value;
    const limitVal = parsed.limit || parseInt(document.getElementById(`${source}-limit`).value) || 10;

    try {
        if (activeMode === 'grep') {
            const payload = {
                pattern: parsed.cleanQuery || rawVal,
                regex: featureStates.grep_regex,
                case_sensitive: featureStates.grep_case,
                limit: limitVal,
                collection: collFilter || null,
                path: scopedPaths.length === 1 ? scopedPaths[0] : (scopedPaths.length > 1 ? scopedPaths : null),
                paths: scopedPaths.length > 0 ? scopedPaths : undefined
            };
            const res = await fetch('/api/grep', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Grep failed");
            lastRawJson = data.results;
            lastSearchType = 'grep';
            statsEl.innerHTML = `Found <span class="font-semibold text-slate-800 dark:text-slate-200">${data.total_matches}</span> matches (${data.time_taken || 0}s)`;
            toolbarEl.classList.remove('hidden');
            renderResults(data.results, 'grep', parsed.cleanQuery || rawVal);
        } else {
            const isDoc = activeMode === 'doc';
            const payload = {
                query: parsed.cleanQuery || rawVal,
                doc: isDoc,
                rerank: featureStates.rerank,
                exclude_seen: featureStates.exclude_seen,
                redact_pii: featureStates.redact_pii,
                limit: limitVal,
                collection: collFilter || null,
                lex: parsed.lex || null,
                title: parsed.title || null,
                path: scopedPaths.length === 1 ? scopedPaths[0] : (scopedPaths.length > 1 ? scopedPaths : null),
                paths: scopedPaths.length > 0 ? scopedPaths : undefined,
                session_id: currentSessionId
            };
            const res = await fetch('/api/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Search failed");
            lastRawJson = data.results;
            lastSearchType = data.type;
            if (data.session_id) setSessionId(data.session_id, false);
            currentExcludedCount = data.excluded_count || 0;
            updateSessionBadges();
            statsEl.innerHTML = `Found <span class="font-semibold text-slate-800 dark:text-slate-200">${data.results.length}</span> ${data.type === 'doc' ? 'documents' : 'chunks'} (${data.time_taken || 0}s)${currentExcludedCount ? ` • <span class="text-amber-500 font-semibold">-${currentExcludedCount} seen</span>` : ''}`;
            toolbarEl.classList.remove('hidden');
            renderResults(data.results, data.type, parsed.cleanQuery || rawVal);
        }
    } catch(e) {
        resultsEl.innerHTML = `<div class="bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900/60 rounded-xl p-5 text-red-700 dark:text-red-300 text-sm font-semibold">${escapeHtml(e.message)}</div>`;
    }
}

// Render Results Viewport
function renderResults(results, type, query) {
    const container = document.getElementById('results');
    container.innerHTML = '';
    if (!results || results.length === 0) {
        container.innerHTML = `<div class="text-center py-16 text-slate-500"><div class="text-base font-medium mb-1">No matches found</div><div class="text-xs">Try broader search terms or adjusting filters.</div></div>`;
        return;
    }

    if (type === 'grep') {
        const grouped = {};
        results.forEach(m => {
            const key = (m.collection || '') + '/' + m.path;
            if (!grouped[key]) grouped[key] = { collection: m.collection, path: m.path, title: m.title, matches: [] };
            grouped[key].matches.push(m);
        });
        Object.values(grouped).forEach(doc => {
            const card = document.createElement('div');
            card.className = "bg-white dark:bg-slate-900/90 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-3";
            const uri = doc.collection ? `qmd://${doc.collection}/${doc.path}` : doc.path;
            card.innerHTML = `
                <div class="flex items-center justify-between text-xs font-mono text-slate-500 dark:text-slate-400">
                    <span class="text-blue-600 dark:text-blue-400 truncate">${escapeHtml(uri)}</span>
                    <span class="bg-slate-100 dark:bg-slate-950 px-2 py-0.5 rounded font-mono text-[11px]">${doc.matches.length} match(es)</span>
                </div>
                <h3 class="doc-link text-lg font-semibold text-blue-600 dark:text-blue-400 hover:underline cursor-pointer">${escapeHtml(doc.title || doc.path)}</h3>
                <div class="space-y-1 font-mono text-xs bg-slate-50 dark:bg-slate-950/70 p-3 rounded-xl border border-slate-200 dark:border-slate-900">
                    ${doc.matches.map(m => `
                        <div class="grep-line flex items-start gap-3 hover:bg-slate-200/60 dark:hover:bg-slate-800/60 p-1.5 rounded cursor-pointer transition select-none">
                            <span class="text-sky-600 font-semibold w-12 text-right flex-shrink-0">L${m.line_number}:</span>
                            <span class="break-all">${highlightKeywords(escapeHtml(m.line_text), query)}</span>
                        </div>
                    `).join('')}
                </div>
            `;
            card.querySelector('.doc-link').onclick = () => openDocument(doc.collection, doc.path, '');
            card.querySelectorAll('.grep-line').forEach((el, idx) => {
                el.onclick = () => openDocument(doc.collection, doc.path, doc.matches[idx].line_text);
            });
            container.appendChild(card);
        });
        return;
    }

    results.forEach((item, index) => {
        const card = document.createElement('div');
        card.className = "bg-white dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800/90 rounded-2xl p-5 shadow-sm space-y-3";
        const uri = item.collection ? `qmd://${item.collection}/${item.path}` : item.path;

        if (type === 'doc') {
            const snippets = item.snippets || [];
            card.innerHTML = `
                <div class="text-xs text-slate-500 font-mono text-blue-600 dark:text-blue-400 truncate">${escapeHtml(uri)}</div>
                <h3 class="doc-link text-xl font-semibold text-blue-600 dark:text-blue-400 hover:underline cursor-pointer">${escapeHtml(item.title || item.path)}</h3>
                <div class="space-y-2.5 my-2">
                    ${snippets.map(snip => `<div class="snip-box cursor-pointer bg-slate-50 dark:bg-slate-950/60 hover:bg-slate-100 p-3.5 rounded-xl border border-slate-200 dark:border-slate-800/80 prose dark:prose-invert max-w-none text-xs leading-relaxed">${marked.parse(snip)}</div>`).join('')}
                </div>
                <div class="flex items-center justify-between pt-2 border-t border-slate-100 dark:border-slate-800/80 text-xs font-mono text-slate-500">
                    <span>Score: ${Number(item.score || 0).toFixed(4)} • ${snippets.length} snippet(s)</span>
                    <div class="flex items-center gap-1.5 font-sans">
                        <button type="button" class="btn-copy-xml hover:bg-slate-100 dark:hover:bg-slate-800 px-2.5 py-1 rounded text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 transition">XML</button>
                        <button type="button" class="btn-open-doc hover:bg-slate-100 dark:hover:bg-slate-800 px-2.5 py-1 rounded text-blue-600 dark:text-blue-400 transition">Open Doc</button>
                    </div>
                </div>
            `;
            card.querySelector('.doc-link').onclick = () => openDocument(item.collection, item.path, '');
            card.querySelector('.btn-open-doc').onclick = () => openDocument(item.collection, item.path, '');
            card.querySelector('.btn-copy-xml').onclick = (e) => copySingleXml(index, e.currentTarget);
            card.querySelectorAll('.snip-box').forEach((el, sIdx) => {
                el.onclick = () => openDocument(item.collection, item.path, snippets[sIdx]);
            });
        } else {
            card.innerHTML = `
                <div class="text-xs text-slate-500 font-mono text-blue-600 dark:text-blue-400 truncate">${escapeHtml(uri)}${item.headers ? ` › ${escapeHtml(item.headers)}` : ''}</div>
                <h3 class="doc-link text-xl font-semibold text-blue-600 dark:text-blue-400 hover:underline cursor-pointer">${escapeHtml(item.title || item.path)}</h3>
                <div class="chunk-box cursor-pointer bg-slate-50 dark:bg-slate-950/60 hover:bg-slate-100 p-4 rounded-xl border border-slate-200 dark:border-slate-800/80 prose dark:prose-invert max-w-none text-xs leading-relaxed">${marked.parse(item.text || '')}</div>
                <div class="flex items-center justify-between pt-2 border-t border-slate-100 dark:border-slate-800/80 text-xs font-mono text-slate-500">
                    <span>Score: ${Number(item.score || 0).toFixed(4)} ${item.rank ? `• #${item.rank}` : ''}</span>
                    <div class="flex items-center gap-1.5 font-sans">
                        <button type="button" class="btn-copy-xml hover:bg-slate-100 dark:hover:bg-slate-800 px-2.5 py-1 rounded text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 transition">XML</button>
                        <button type="button" class="btn-open-doc hover:bg-slate-100 dark:hover:bg-slate-800 px-2.5 py-1 rounded text-blue-600 dark:text-blue-400 transition">Open Doc</button>
                    </div>
                </div>
            `;
            card.querySelector('.doc-link').onclick = () => openDocument(item.collection, item.path, item.text || '');
            card.querySelector('.chunk-box').onclick = () => openDocument(item.collection, item.path, item.text || '');
            card.querySelector('.btn-copy-xml').onclick = (e) => copySingleXml(index, e.currentTarget);
            card.querySelector('.btn-open-doc').onclick = () => openDocument(item.collection, item.path, item.text || '');
        }
        container.appendChild(card);
    });
}

// Copy Buttons Toolbar
const copyJsonBtn = document.getElementById('copy-json');
if (copyJsonBtn) {
    copyJsonBtn.onclick = (e) => {
        if (!lastRawJson) return;
        const btn = e.currentTarget;
        const orig = btn.innerHTML;
        const xml = formatResultsToXml(lastRawJson, lastSearchType, document.getElementById('serp-query').value, currentSessionId);
        navigator.clipboard.writeText(xml).then(() => {
            btn.innerHTML = `<svg class="w-3.5 h-3.5 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg><span class="text-emerald-600 dark:text-emerald-400 font-semibold">Copied!</span>`;
            setTimeout(() => { btn.innerHTML = orig; }, 1800);
        });
    };
}

const copyPromptBtn = document.getElementById('copy-prompt');
if (copyPromptBtn) {
    copyPromptBtn.onclick = (e) => {
        if (!lastRawJson) return;
        const btn = e.currentTarget;
        const orig = btn.innerHTML;
        const q = document.getElementById('serp-query').value;
        const xml = formatResultsToXml(lastRawJson, lastSearchType, q, currentSessionId);
        navigator.clipboard.writeText(`Task / Question: ${q}\n\nContext / References:\n${xml}`).then(() => {
            btn.innerHTML = `<svg class="w-3.5 h-3.5 text-emerald-200" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg><span class="text-white font-semibold">Copied!</span>`;
            setTimeout(() => { btn.innerHTML = orig; }, 1800);
        });
    };
}

// Tree & Scope Filter Debouncers
const scopeFilterInput = document.getElementById('scope-tree-filter');
if (scopeFilterInput) {
    scopeFilterInput.addEventListener('input', (e) => {
        clearTimeout(scopeFilterTimer);
        scopeFilterTimer = setTimeout(() => loadScopePickerTree(e.target.value.trim()), 250);
    });
}

const treeFilterInput = document.getElementById('tree-filter-input');
if (treeFilterInput) {
    treeFilterInput.addEventListener('input', (e) => {
        clearTimeout(treeFilterTimer);
        treeFilterTimer = setTimeout(() => loadCollectionTree(e.target.value.trim()), 250);
    });
}

// Mode Buttons Event Wiring
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', (e) => setMode(e.target.dataset.mode));
});

// Tree Slide-over Trigger
document.querySelectorAll('.btn-open-tree').forEach(btn => {
    btn.onclick = openTreeDrawer;
});

// Escape Key Listener
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeScopePicker();
        closeSessionModal();
        closeSettings();
        closeDocument();
        closeTreeDrawer();
    }
});

// Dropdown Synchronization
document.querySelectorAll('.limit-select').forEach(sel => {
    sel.addEventListener('change', (e) => {
        const val = e.target.value;
        document.querySelectorAll('.limit-select').forEach(s => s.value = val);
    });
});

document.querySelectorAll('.collection-select').forEach(sel => {
    sel.addEventListener('change', (e) => {
        const val = e.target.value;
        document.querySelectorAll('.collection-select').forEach(s => s.value = val);
    });
});

// Initialize on Startup
initSession();
applyTheme(localStorage.getItem(THEME_STORAGE_KEY) || 'system');
applyDefaultPreferences();
loadSettingsModalValues();
updateQueryHistoryDatalist();
loadCollections();
