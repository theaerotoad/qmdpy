// Single Result XML Copy Handler
function copySingleXml(index, btnEl = null) {
    if (!lastRawJson || !lastRawJson[index]) return;
    const item = lastRawJson[index];
    const xml = item.xml || lastRawXml;
    navigator.clipboard.writeText(xml).then(() => {
        if (btnEl) {
            const orig = btnEl.innerHTML;
            btnEl.innerHTML = `<span class="text-emerald-600 dark:text-emerald-400 font-semibold">Copied!</span>`;
            setTimeout(() => { btnEl.innerHTML = orig; }, 1800);
        }
        showToast("Chunk XML copied to clipboard");
    });
}

function copySingleExcerpt(index, btnEl = null) {
    if (!lastRawJson || !lastRawJson[index]) return;
    const item = lastRawJson[index];
    const text = item.text || (item.snippets ? item.snippets.join('\n\n') : '');
    navigator.clipboard.writeText(text).then(() => {
        if (btnEl) {
            const orig = btnEl.innerHTML;
            btnEl.innerHTML = `<span class="text-emerald-600 dark:text-emerald-400 font-semibold">Copied!</span>`;
            setTimeout(() => { btnEl.innerHTML = orig; }, 1800);
        }
        showToast("Excerpt copied to clipboard");
    });
}

function triggerSearchWithMode(mode, source) {
    activeMode = mode;
    if (mode === 'discover') {
        setSearchTab('documents');
    } else {
        setSearchTab('all');
    }
    const form = document.getElementById(`${source}-search-form`);
    if (form) {
        handleFormSubmit(new Event('submit'), source);
    }
}

// Form Submit & Search Execution
async function handleFormSubmit(e, source) {
    if (e && e.preventDefault) e.preventDefault();
    const rawVal = document.getElementById(`${source}-query`) 
        ? document.getElementById(`${source}-query`).value 
        : (document.getElementById('serp-query').value || document.getElementById('hero-query').value);

    const parsed = parseQueryDirectives(rawVal);
    if (!parsed.cleanQuery && !rawVal.trim()) return;

    saveQueryToHistory(rawVal);
    document.getElementById('hero-query').value = rawVal;
    document.getElementById('serp-query').value = rawVal;
    setAppState('serp');

    const resultsEl = document.getElementById('results');
    const statsEl = document.getElementById('result-stats');
    const progressEl = document.getElementById('search-progress');
    const footerEl = document.getElementById('results-footer');

    if (progressEl) progressEl.classList.remove('hidden');
    resultsEl.innerHTML = `<div class="flex items-center justify-center py-20 text-gray-500 gap-3"><svg class="animate-spin h-5 w-5 text-blue-500" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg><span>Searching knowledge base...</span></div>`;
    statsEl.textContent = "Querying multi-corpus index...";

    const scopedPaths = selectedScopes.filter(s => s.path).map(s => s.path);
    if (parsed.path && !scopedPaths.includes(parsed.path)) scopedPaths.push(parsed.path);

    const collFilter = parsed.collection || null;
    const limitSelectEl = document.getElementById('filter-limit');
    const limitVal = parsed.limit || (limitSelectEl ? parseInt(limitSelectEl.value) : 10) || 10;

    try {
        const isDiscoverMode = activeTab === 'documents' || (activeTab === 'all' && activeMode === 'discover');
        const isPassagesMode = activeTab === 'passages';
        const isLLMContextMode = activeTab === 'llm-context';

        if (isDiscoverMode) {
            const payload = {
                query: parsed.cleanQuery || rawVal,
                rerank: featureStates.rerank,
                exclude_seen: featureStates.exclude_seen,
                redact_pii: featureStates.redact_pii,
                limit: limitVal,
                collection: collFilter,
                lex: parsed.lex || null,
                title: parsed.title || null,
                path: scopedPaths.length === 1 ? scopedPaths[0] : (scopedPaths.length > 1 ? scopedPaths : null),
                paths: scopedPaths.length > 0 ? scopedPaths : undefined,
                session_id: currentSessionId
            };
            const res = await fetch(apiUrl('/api/discover'), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Discovery failed");
            
            lastRawJson = data.results;
            lastRawXml = data.xml || '';
            lastSearchType = 'discover';
            if (data.session_id) setSessionId(data.session_id, false);
            currentExcludedCount = data.excluded_count || 0;
            updateSessionBadges();

            statsEl.innerHTML = `About <strong class="text-gray-700 dark:text-gray-200">${data.results.length} documents</strong> across indexed repositories (<span class="font-mono">${data.time_taken || 0} seconds</span>)${currentExcludedCount ? ` • <span class="text-amber-500 font-semibold">-${currentExcludedCount} seen</span>` : ''}`;
            renderResults(data.results, 'discover', parsed.cleanQuery || rawVal);
        } else {
            const payload = {
                query: parsed.cleanQuery || rawVal,
                doc: !isPassagesMode,
                flat: isPassagesMode,
                rerank: featureStates.rerank,
                exclude_seen: featureStates.exclude_seen,
                redact_pii: featureStates.redact_pii,
                limit: limitVal,
                collection: collFilter,
                lex: parsed.lex || null,
                title: parsed.title || null,
                path: scopedPaths.length === 1 ? scopedPaths[0] : (scopedPaths.length > 1 ? scopedPaths : null),
                paths: scopedPaths.length > 0 ? scopedPaths : undefined,
                session_id: currentSessionId
            };
            const res = await fetch(apiUrl('/api/search'), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Search failed");

            lastRawJson = data.results;
            lastRawXml = data.xml || '';
            lastSearchType = data.type;
            if (data.session_id) setSessionId(data.session_id, false);
            currentExcludedCount = data.excluded_count || 0;
            updateSessionBadges();

            statsEl.innerHTML = `About <strong class="text-gray-700 dark:text-gray-200">${data.results.length} ${data.type === 'doc' ? 'documents' : 'passages'}</strong> across indexed repositories (<span class="font-mono">${data.time_taken || 0} seconds</span>)${currentExcludedCount ? ` • <span class="text-amber-500 font-semibold">-${currentExcludedCount} seen</span>` : ''}`;
            
            if (isLLMContextMode) {
                renderLLMContextView(data.results, data.type, parsed.cleanQuery || rawVal);
            } else {
                renderResults(data.results, data.type, parsed.cleanQuery || rawVal);
            }
        }
        if (footerEl) footerEl.classList.remove('hidden');
    } catch(e) {
        resultsEl.innerHTML = `<div class="bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900/60 rounded-xl p-5 text-red-700 dark:text-red-300 text-sm font-semibold">${escapeHtml(e.message)}</div>`;
        statsEl.textContent = "Search encountered an error";
    } finally {
        if (progressEl) progressEl.classList.add('hidden');
    }
}

// Collapsible Skipped Chunks Handler
function toggleSkipped(btn) {
    const isExpanded = btn.getAttribute('data-expanded') === 'true';
    const arrow = btn.querySelector('svg');
    const textSpan = btn.querySelector('.skipped-text');
    const hiddenChunks = btn.closest('.skipped-wrapper').querySelector('.hidden-chunks');

    if (!isExpanded) {
        btn.setAttribute('data-expanded', 'true');
        if (arrow) arrow.classList.add('rotate-90');
        if (textSpan) textSpan.textContent = "Hide skipped chunk context";
        if (hiddenChunks) hiddenChunks.classList.remove('hidden');
    } else {
        btn.setAttribute('data-expanded', 'false');
        if (arrow) arrow.classList.remove('rotate-90');
        if (textSpan) textSpan.textContent = btn.getAttribute('data-original-text') || "... chunks skipped ...";
        if (hiddenChunks) hiddenChunks.classList.add('hidden');
    }
}

// Deep Search inside Document
function deepSearchSnippet(docTitle, docPath, docCollection) {
    activeMode = 'search';
    setSearchTab('all');
    clearAllScopes();
    toggleScope(docCollection || '', docPath || '', 'file', docTitle || docPath);
    const query = document.getElementById('serp-query').value.trim();
    handleFormSubmit(new Event('submit'), 'serp');
    showToast(`Focused search inside: ${docTitle || docPath}`);
}

// Render Results Viewport in Google Mid-2010s Layout
function renderResults(results, type, query) {
    const container = document.getElementById('results');
    container.innerHTML = '';

    if (!results || results.length === 0) {
        container.innerHTML = `
            <div class="py-14 text-gray-500">
                <p class="text-base font-medium mb-1 text-gray-700 dark:text-gray-300">Your search - <strong class="font-bold text-gray-900 dark:text-white">${escapeHtml(query)}</strong> - did not match any documents.</p>
                <div class="text-xs space-y-1 mt-3">
                    <p>Suggestions:</p>
                    <ul class="list-disc pl-5 space-y-0.5">
                        <li>Make sure that all words are spelled correctly.</li>
                        <li>Try different or broader natural language queries.</li>
                        <li>Try removing directory or collection scope filters.</li>
                    </ul>
                </div>
            </div>
        `;
        return;
    }

    results.forEach((item, index) => {
        const article = document.createElement('article');
        article.className = "space-y-1.5 group";
        const uri = item.collection ? `qmd://${item.collection}/${item.path}` : item.path;
        const matchCount = item.match_count || (item.snippets ? item.snippets.length : 1);
        const matchesBadge = matchCount > 1 
            ? `<span class="px-1.5 py-0.5 rounded text-[10px] font-medium bg-gray-100 dark:bg-[#303134] text-gray-600 dark:text-gray-300 border border-gray-200 dark:border-gray-700 flex-shrink-0">${matchCount} matches in doc</span>` 
            : '';

        let contentHtml = '';
        if (type === 'doc' && item.snippets && item.snippets.length > 0) {
            contentHtml = `
                <div class="border-l-2 border-gray-200 dark:border-gray-700 pl-4 space-y-3 mt-2">
                    ${item.snippets.map((snip, sIdx) => {
                        const skipNotice = sIdx > 0 
                            ? `<div class="py-0.5 skipped-wrapper">
                                 <button type="button" onclick="toggleSkipped(this)" data-original-text="... preceding chunks skipped ..." class="text-xs text-gray-500 hover:text-blue-600 dark:hover:text-blue-400 font-mono flex items-center gap-1.5">
                                   <svg class="w-3 h-3 transform rotate-0 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
                                   <span class="skipped-text">... preceding chunks skipped ...</span>
                                 </button>
                               </div>` 
                            : '';
                        return `
                            ${skipNotice}
                            <div>
                                <div class="g-snippet">${marked.parse(snip)}</div>
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
        } else {
            const rawBody = item.text || (item.snippets ? item.snippets[0] : '');
            contentHtml = `
                <div class="g-snippet space-y-2 mt-1">
                    ${marked.parse(rawBody || '')}
                </div>
            `;
        }

        article.innerHTML = `
            <div class="flex items-center gap-2 text-xs">
                <span class="g-url font-mono truncate max-w-xl">${escapeHtml(uri)}${item.headers ? ` › ${escapeHtml(item.headers)}` : ''}</span>
                ${matchesBadge}
            </div>

            <h2 class="text-xl leading-snug">
                <a href="javascript:void(0)" class="doc-link g-link font-medium">
                    ${escapeHtml(item.title || item.path)}
                </a>
            </h2>

            ${contentHtml}

            <div class="pt-2 flex items-center gap-3 text-xs text-gray-500 dark:text-gray-400 flex-wrap">
                <span class="font-mono text-[11px]">Score: ${Number(item.score || 0).toFixed(4)}</span>
                <span>•</span>
                <span class="font-mono text-[11px]">Top Chunk: #${item.seq_id || item.rank || (index + 1)}</span>
                <span>•</span>
                <button type="button" class="btn-open-doc text-blue-600 dark:text-blue-400 hover:underline">Open Document</button>
                <span>•</span>
                <button type="button" class="btn-deep-search text-blue-600 dark:text-blue-400 hover:underline">Deep Search Inside</button>
                <span>•</span>
                <button type="button" class="btn-copy-chunk text-gray-500 hover:text-gray-700 dark:hover:text-gray-200">Copy chunk</button>
                <span>•</span>
                <button type="button" class="btn-copy-xml text-gray-500 hover:text-gray-700 dark:hover:text-gray-200">XML</button>
            </div>
        `;

        article.querySelector('.doc-link').onclick = () => openDocument(item.collection, item.path, item.text || (item.snippets ? item.snippets[0] : ''));
        article.querySelector('.btn-open-doc').onclick = () => openDocument(item.collection, item.path, item.text || (item.snippets ? item.snippets[0] : ''));
        article.querySelector('.btn-deep-search').onclick = () => deepSearchSnippet(item.title || item.path, item.path, item.collection);
        article.querySelector('.btn-copy-chunk').onclick = (e) => copySingleExcerpt(index, e.currentTarget);
        article.querySelector('.btn-copy-xml').onclick = (e) => copySingleXml(index, e.currentTarget);

        container.appendChild(article);
    });
}

// Render LLM Context Tab View
function renderLLMContextView(results, type, query) {
    const container = document.getElementById('results');
    container.innerHTML = `
        <div class="space-y-4">
            <div class="p-4 bg-purple-50 dark:bg-purple-950/30 border border-purple-200 dark:border-purple-800 rounded-xl text-xs space-y-2">
                <div class="font-semibold text-purple-900 dark:text-purple-300 flex items-center gap-1.5">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path></svg>
                    <span>LLM Context Bundle Ready</span>
                </div>
                <p class="text-purple-800 dark:text-purple-400">Below is the canonical formatted XML context generated for AI agents, including chunk sequence metadata and action attributes.</p>
            </div>
            <pre class="bg-gray-900 text-gray-100 p-4 rounded-xl font-mono text-xs overflow-x-auto max-h-[600px] border border-gray-800 leading-relaxed">${escapeHtml(lastRawXml || '')}</pre>
        </div>
    `;
}

// Copy XML and Prompt Context Event Listeners
const copyJsonBtn = document.getElementById('copy-json');
if (copyJsonBtn) {
    copyJsonBtn.onclick = () => {
        if (!lastRawXml) return;
        navigator.clipboard.writeText(lastRawXml).then(() => {
            showToast("Copied XML for LLM to clipboard ✓");
        });
    };
}

const copyPromptBtn = document.getElementById('copy-prompt');
if (copyPromptBtn) {
    copyPromptBtn.onclick = () => {
        if (!lastRawXml) return;
        const q = document.getElementById('serp-query').value;
        const promptPayload = `Answer the user question based on the following sources:\n\nUser Question: "${q}"\n\nContext & Citations:\n${lastRawXml}`;
        navigator.clipboard.writeText(promptPayload).then(() => {
            showToast("Copied Prompt & References to clipboard ✓");
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

// Synchronize limit dropdowns
document.querySelectorAll('.limit-select').forEach(sel => {
    sel.addEventListener('change', (e) => {
        const val = e.target.value;
        document.querySelectorAll('.limit-select').forEach(s => s.value = val);
    });
});

// Initialize on Startup
initSession();
applyTheme(localStorage.getItem(THEME_STORAGE_KEY) || 'system');
applyDefaultPreferences();
loadSettingsModalValues();
updateQueryHistoryDatalist();
loadCollections();