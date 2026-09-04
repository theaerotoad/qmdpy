// Scope Multi-Select & Indicator Helpers
function getScopeId(collection, path) { return `${collection || ''}::${path || ''}`; }
function isScopeSelected(coll, path) { return selectedScopes.some(s => s.id === getScopeId(coll, path)); }

function formatPathMiddle(str, max = 28) {
    if (!str || str.length <= max) return str;
    const half = Math.floor((max - 3) / 2);
    return str.slice(0, half) + '...' + str.slice(str.length - half);
}

function toggleScope(coll = '', path = '', type = '', label = '') {
    const id = getScopeId(coll, path);
    const idx = selectedScopes.findIndex(s => s.id === id);
    if (idx >= 0) selectedScopes.splice(idx, 1);
    else selectedScopes.push({ id, collection: coll, path, type: type || (path ? (path.endsWith('/') ? 'folder' : 'file') : 'collection'), label: label || path || coll });
    updateScopeUI();
}

function removeScope(id) { selectedScopes = selectedScopes.filter(s => s.id !== id); updateScopeUI(); }
function clearAllScopes() { selectedScopes = []; updateScopeUI(); }

function updateScopeUI() {
    renderScopeChips();
    updateScopePickerIndicators();
}

function renderScopeChips() {
    const containers = [document.getElementById('hero-scope-chips'), document.getElementById('serp-scope-chips')].filter(Boolean);
    containers.forEach(c => { c.innerHTML = ''; });
    if (!selectedScopes.length) { containers.forEach(c => c.classList.add('hidden')); return; }
    containers.forEach(container => {
        container.classList.remove('hidden');
        selectedScopes.forEach(s => {
            const full = s.path || s.collection;
            const chip = document.createElement('span');
            chip.className = "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono border shadow-sm max-w-[260px] bg-blue-50 dark:bg-blue-900/30 border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300";
            chip.innerHTML = `<span class="truncate">${escapeHtml(formatPathMiddle(full, 24))}</span><button type="button" class="hover:text-red-500 transition">✕</button>`;
            chip.querySelector('button').onclick = () => removeScope(s.id);
            container.appendChild(chip);
        });
    });
}

function updateScopePickerIndicators() {
    const cnt = document.getElementById('scope-selected-count');
    if (cnt) {
        cnt.textContent = `${selectedScopes.length} selected`;
        cnt.classList.toggle('hidden', selectedScopes.length === 0);
    }
    document.querySelectorAll('.scope-item-row').forEach(row => {
        const isSel = isScopeSelected(row.dataset.scopeColl || '', row.dataset.scopePath || '');
        const btn = row.querySelector('.scope-toggle-btn');
        if (btn) {
            btn.textContent = isSel ? '✓' : '+';
            btn.title = isSel ? 'Remove from scope (✕)' : 'Add to scope (+)';
            btn.className = `scope-toggle-btn w-6 h-6 flex items-center justify-center rounded-lg text-xs font-bold transition ${isSel ? 'bg-blue-600 text-white shadow-sm' : 'bg-white dark:bg-[#303134] text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 border border-gray-300 dark:border-gray-600'}`;
        }
    });
}

// Scope Picker Modal
async function openScopePicker() {
    const modal = document.getElementById('scope-modal');
    modal.classList.remove('hidden');
    void modal.offsetWidth;
    document.getElementById('scope-backdrop').classList.remove('opacity-0');
    document.getElementById('scope-panel').classList.remove('scale-95', 'opacity-0');
    loadScopePickerTree();
}

function closeScopePicker() {
    document.getElementById('scope-backdrop').classList.add('opacity-0');
    document.getElementById('scope-panel').classList.add('scale-95', 'opacity-0');
    setTimeout(() => document.getElementById('scope-modal').classList.add('hidden'), 150);
}

async function loadScopePickerTree(pattern = '') {
    const container = document.getElementById('scope-picker-tree');
    container.innerHTML = '<div class="text-gray-500 py-6 text-center">Loading directories...</div>';
    try {
        const url = pattern ? apiUrl(`/api/collections/tree?pattern=${encodeURIComponent(pattern)}`) : apiUrl('/api/collections/tree');
        const res = await fetch(url);
        const data = await res.json();
        renderScopePickerTree(data);
    } catch(e) {
        container.innerHTML = `<div class="text-red-500 p-2">Error loading tree</div>`;
    }
}

function renderScopePickerTree(treeData) {
    const container = document.getElementById('scope-picker-tree');
    container.innerHTML = '';
    const colls = Array.isArray(treeData) ? treeData : [treeData];
    colls.forEach(item => {
        const div = document.createElement('div');
        div.className = "bg-gray-50 dark:bg-[#202124] rounded-lg p-3 border border-gray-200 dark:border-gray-700";
        div.innerHTML = `
            <div class="scope-item-row flex items-center justify-between pb-2 border-b border-gray-200 dark:border-gray-700 mb-2 cursor-pointer" data-scope-coll="${escapeXmlAttr(item.collection)}" data-scope-path="">
                <div class="font-bold text-blue-600 dark:text-blue-400">${escapeHtml(item.collection)}/</div>
                <button type="button" class="scope-toggle-btn w-6 h-6 flex items-center justify-center rounded-lg text-xs font-bold transition bg-white dark:bg-[#303134] text-gray-400 border border-gray-300 dark:border-gray-600" title="Add to scope">+</button>
            </div>
            <div class="tree-nodes pl-1 space-y-1"></div>
        `;
        div.querySelector('.scope-item-row').onclick = () => toggleScope(item.collection, '', 'collection', item.collection + '/');

        function renderNodes(node, parentEl, currentPath = '') {
            (node.children || []).forEach(child => {
                const row = document.createElement('div');
                if (child.type === 'directory') {
                    const p = (currentPath ? currentPath + '/' : '') + child.name + '/';
                    row.innerHTML = `
                        <div class="scope-item-row flex items-center justify-between text-amber-600 dark:text-amber-400 font-semibold py-1 px-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-[#303134] cursor-pointer" data-scope-coll="${escapeXmlAttr(item.collection)}" data-scope-path="${escapeXmlAttr(p)}">
                            <span>📁 ${escapeHtml(child.name)}/</span>
                            <button type="button" class="scope-toggle-btn w-6 h-6 flex items-center justify-center rounded-lg text-xs font-bold transition bg-white dark:bg-[#303134] text-gray-400 border border-gray-300 dark:border-gray-600" title="Add to scope">+</button>
                        </div>
                        <div class="child-nodes pl-3 border-l border-gray-200 dark:border-gray-700 ml-2 space-y-1"></div>
                    `;
                    row.querySelector('.scope-item-row').onclick = () => toggleScope(item.collection, p, 'folder', p);
                    renderNodes(child, row.querySelector('.child-nodes'), (currentPath ? currentPath + '/' : '') + child.name);
                } else {
                    row.className = "scope-item-row flex items-center justify-between text-gray-700 dark:text-gray-300 px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-[#303134] cursor-pointer";
                    row.dataset.scopeColl = item.collection;
                    row.dataset.scopePath = child.path;
                    row.innerHTML = `<span class="truncate">📄 ${escapeHtml(child.name)}</span><button type="button" class="scope-toggle-btn w-6 h-6 flex items-center justify-center rounded-lg text-xs font-bold transition bg-white dark:bg-[#303134] text-gray-400 border border-gray-300 dark:border-gray-600" title="Add to scope">+</button>`;
                    row.onclick = () => toggleScope(item.collection, child.path, 'file', child.name);
                }
                parentEl.appendChild(row);
            });
        }
        renderNodes(item.tree || {}, div.querySelector('.tree-nodes'));
        container.appendChild(div);
    });
    updateScopePickerIndicators();
}

// Settings Modal & Tabs
function setSettingsTab(tabName) {
    document.querySelectorAll('.settings-tab-btn').forEach(btn => {
        const isCur = btn.dataset.tab === tabName;
        btn.className = `settings-tab-btn px-3 py-2.5 border-b-2 transition ${isCur ? 'border-blue-600 text-blue-600 dark:text-blue-400 font-semibold' : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'}`;
    });
    document.querySelectorAll('.settings-tab-panel').forEach(p => p.classList.add('hidden'));
    const activePanel = document.getElementById(`tab-content-${tabName}`);
    if (activePanel) activePanel.classList.remove('hidden');
    if (tabName === 'collections') loadCollections(true);
}

function openSettings() {
    loadSettingsModalValues();
    ensureOpenFileSettingInSettings();
    const chk = document.getElementById('setting-allow-open-file');
    if (chk) chk.checked = isOpenFileSettingEnabled();
    const modal = document.getElementById('settings-modal');
    modal.classList.remove('hidden');
    void modal.offsetWidth;
    document.getElementById('settings-backdrop').classList.remove('opacity-0');
    document.getElementById('settings-panel').classList.remove('scale-95', 'opacity-0');
}

function closeSettings() {
    document.getElementById('settings-backdrop').classList.add('opacity-0');
    document.getElementById('settings-panel').classList.add('scale-95', 'opacity-0');
    setTimeout(() => document.getElementById('settings-modal').classList.add('hidden'), 150);
}

// Collections List & Re-indexing
async function loadCollections(populateSettings = false) {
    try {
        const res = await fetch(apiUrl('/api/collections'));
        const data = await res.json();

        const footerDocEl = document.getElementById('footer-doc-count');
        if (footerDocEl && Array.isArray(data)) {
            const totalDocs = data.reduce((acc, c) => acc + (c.doc_count || 0), 0);
            footerDocEl.textContent = `${data.length} Collections (${totalDocs.toLocaleString()} docs)`;
        }

        if (populateSettings) {
            const listEl = document.getElementById('settings-collections-list');
            if (listEl) {
                listEl.innerHTML = '';
                if (!data.length) { listEl.innerHTML = '<div class="text-gray-500 py-3 text-center">No collections configured in qmd.yml</div>'; return; }
                data.forEach(c => {
                    const card = document.createElement('div');
                    card.className = "flex items-center justify-between p-3 bg-gray-50 dark:bg-[#202124] rounded-lg border border-gray-200 dark:border-gray-700";
                    const canReindex = c.can_reindex !== false && !c.is_federated;
                    const actionHtml = canReindex ? `
                        <button type="button" class="btn-reindex bg-white dark:bg-[#303134] border border-gray-300 dark:border-gray-600 hover:bg-gray-50 px-2.5 py-1.5 rounded text-xs font-medium text-gray-700 dark:text-gray-200 transition shadow-sm flex items-center gap-1.5">
                            Re-index
                        </button>
                    ` : `
                        <span class="text-[11px] text-gray-400 bg-gray-100 dark:bg-[#303134] border border-gray-200 dark:border-gray-700 px-2 py-0.5 rounded font-mono">
                            Read-Only
                        </span>
                    `;
                    card.innerHTML = `
                        <div>
                            <div class="font-semibold text-gray-900 dark:text-white flex items-center gap-1.5">
                                <svg class="w-3.5 h-3.5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                                ${escapeHtml(c.name)}
                            </div>
                            <div class="text-[11px] text-gray-500 font-mono">${escapeHtml(c.path || '')} • ${c.doc_count || 0} docs</div>
                        </div>
                        ${actionHtml}
                    `;
                    const reindexBtn = card.querySelector('.btn-reindex');
                    if (reindexBtn) {
                        reindexBtn.onclick = (e) => triggerReindex(c.name, e.currentTarget);
                    }
                    listEl.appendChild(card);
                });
            }
        }
    } catch(e) { console.error("Collections load failed", e); }
}

async function triggerReindex(name, btn) {
    const orig = btn.innerHTML;
    btn.innerHTML = `Indexing...`;
    btn.disabled = true;
    try {
        const res = await fetch(apiUrl('/api/update'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ collection: name, force: false })
        });
        const data = await res.json();
        if (data.status === 'success') {
            btn.innerHTML = `<span class="text-emerald-500 font-semibold">✓ Done</span>`;
            showToast(`Re-indexed collection: ${name}`);
            setTimeout(() => { btn.innerHTML = orig; btn.disabled = false; }, 2000);
        } else {
            btn.innerHTML = `<span class="text-red-500 font-semibold">Failed</span>`;
            setTimeout(() => { btn.innerHTML = orig; btn.disabled = false; }, 2000);
        }
    } catch(e) {
        btn.innerHTML = `<span class="text-red-500 font-semibold">Failed</span>`;
        setTimeout(() => { btn.innerHTML = orig; btn.disabled = false; }, 2000);
    }
}

// Session Management
function generateRandomHex(length = 4) {
    const bytes = new Uint8Array(length);
    crypto.getRandomValues(bytes);
    return Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
}

function initSession() {
    const newId = generateRandomHex(4);
    sessionStorage.setItem(SESSION_STORAGE_KEY, newId);
    setSessionId(newId, false);
}

function setSessionId(newId, fetchStats = true) {
    currentSessionId = newId.trim();
    sessionStorage.setItem(SESSION_STORAGE_KEY, currentSessionId);
    document.querySelectorAll('.session-id-val').forEach(el => el.textContent = currentSessionId);
    const m = document.getElementById('modal-session-id');
    if (m) m.textContent = currentSessionId;
    updateSessionBadges();
    if (fetchStats) refreshSessionStats(currentSessionId);
}

function updateSessionBadges() {
    document.querySelectorAll('.session-excluded-badge').forEach(b => {
        if (currentExcludedCount > 0) {
            b.textContent = `-${currentExcludedCount} seen`;
            b.classList.remove('hidden');
        } else b.classList.add('hidden');
    });
}

async function refreshSessionStats(id) {
    if (!id) return;
    try {
        const res = await fetch(apiUrl(`/api/session/${encodeURIComponent(id)}`));
        if (!res.ok) return;
        const d = await res.json();
        const s = document.getElementById('modal-seen-count'), e = document.getElementById('modal-events-count');
        if (s) s.textContent = d.seen_chunks_count || 0;
        if (e) e.textContent = d.events_count || 0;
    } catch(e) {}
}

function generateNewSession() {
    currentExcludedCount = 0;
    const freshId = generateRandomHex(4);
    setSessionId(freshId, false);
    const s = document.getElementById('modal-seen-count'), e = document.getElementById('modal-events-count');
    if (s) s.textContent = '0';
    if (e) e.textContent = '0';
    showToast(`Started fresh session: ${freshId}`);
}

function applySwitchedSession() {
    const val = document.getElementById('input-switch-session').value.trim();
    if (!val) return;
    setSessionId(val, true);
    document.getElementById('input-switch-session').value = '';
    closeSessionModal();
    showToast(`Switched to session: ${val}`);
}

function copySessionId(type = 'id') {
    const txt = type === 'cli' ? `--session ${currentSessionId}` : currentSessionId;
    const btn = document.getElementById(type === 'cli' ? 'btn-copy-session-cli' : 'btn-copy-session-id');
    navigator.clipboard.writeText(txt).then(() => {
        if (btn) {
            const orig = btn.innerHTML;
            btn.innerHTML = `<span class="text-emerald-600 dark:text-emerald-400 font-semibold">Copied!</span>`;
            setTimeout(() => { btn.innerHTML = orig; }, 1800);
        }
        showToast(`Copied ${type === 'cli' ? 'CLI argument' : 'Session Key'}: ${txt}`);
    });
}

function openSessionModal() {
    const modal = document.getElementById('session-modal');
    modal.classList.remove('hidden');
    void modal.offsetWidth;
    document.getElementById('session-backdrop').classList.remove('opacity-0');
    document.getElementById('session-panel').classList.remove('scale-95', 'opacity-0');
    refreshSessionStats(currentSessionId);
}

function closeSessionModal() {
    document.getElementById('session-backdrop').classList.add('opacity-0');
    document.getElementById('session-panel').classList.add('scale-95', 'opacity-0');
    setTimeout(() => document.getElementById('session-modal').classList.add('hidden'), 150);
}

// Batch Command Modal Logic
let lastBatchBundleXml = '';

function openBatchModal() {
    const modal = document.getElementById('batch-modal');
    modal.classList.remove('hidden');
    modal.classList.add('flex');
    void modal.offsetWidth;
    document.getElementById('batch-backdrop').classList.remove('opacity-0');
    document.getElementById('batch-panel').classList.remove('scale-95', 'opacity-0');
    const input = document.getElementById('batch-input');
    if (input && !input.value.trim()) input.focus();
}

function closeBatchModal() {
    document.getElementById('batch-backdrop').classList.add('opacity-0');
    document.getElementById('batch-panel').classList.add('scale-95', 'opacity-0');
    setTimeout(() => {
        const modal = document.getElementById('batch-modal');
        modal.classList.remove('flex');
        modal.classList.add('hidden');
    }, 150);
}

function showBatchModal() { openBatchModal(); }
function hideBatchModal() { closeBatchModal(); }

function clearBatchInput() {
    const input = document.getElementById('batch-input');
    if (input) input.value = '';
    document.getElementById('batch-progress-wrapper').classList.add('hidden');
    document.getElementById('batch-output-wrapper').classList.add('hidden');
    document.getElementById('btn-copy-batch-output').classList.add('hidden');
}

async function copyBatchGuide() {
    const btn = document.getElementById('btn-copy-batch-guide');
    try {
        const res = await fetch(apiUrl('/api/guide?format=xml'));
        const data = await res.json();
        const guideText = data.guide || '';
        await navigator.clipboard.writeText(guideText);
        showToast("XML LLM Guide copied to clipboard ✓");
    } catch(e) {
        showToast("Error fetching prompt guide");
    }
}

async function runBatchCommands() {
    const rawText = document.getElementById('batch-input').value;
    if (!rawText || !rawText.trim()) return;

    const runBtn = document.getElementById('btn-run-batch');
    const origBtnHtml = runBtn.innerHTML;
    runBtn.disabled = true;
    runBtn.innerHTML = `Running...`;

    const progressWrapper = document.getElementById('batch-progress-wrapper');
    const stepList = document.getElementById('batch-step-list');
    const outputWrapper = document.getElementById('batch-output-wrapper');
    const outputEl = document.getElementById('batch-output-xml');
    const copyOutBtn = document.getElementById('btn-copy-batch-output');

    progressWrapper.classList.remove('hidden');
    outputWrapper.classList.add('hidden');
    stepList.innerHTML = `<div class="text-gray-500 py-3 text-center">Parsing and evaluating commands...</div>`;

    try {
        const res = await fetch(apiUrl('/api/batch'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: rawText, max_commands: 5 })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to execute batch');

        stepList.innerHTML = '';
        (data.results || []).forEach(r => {
            const card = document.createElement('div');
            const isSuccess = r.status === 'success';
            card.className = `p-2.5 rounded-lg border flex items-start justify-between gap-3 text-xs font-mono ${
                isSuccess
                    ? 'bg-gray-50 dark:bg-[#202124] border-gray-200 dark:border-gray-700'
                    : 'bg-red-50/60 dark:bg-red-950/30 border-red-200 dark:border-red-900/50 text-red-700 dark:text-red-300'
            }`;
            card.innerHTML = `
                <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-2 mb-1">
                        <span class="px-1.5 py-0.2 rounded font-semibold text-[10px] ${isSuccess ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300' : 'bg-red-100 text-red-700'}">#${r.index}</span>
                        <span class="font-bold text-gray-800 dark:text-gray-200 truncate">${escapeHtml(r.command)}</span>
                    </div>
                    <div class="text-[11px] text-gray-500 truncate">${escapeHtml(r.output.split('\n')[0] || '')}</div>
                </div>
                <div class="flex items-center gap-2 flex-shrink-0 pt-0.5">
                    <span class="text-[11px] text-gray-400">${r.time_taken}s</span>
                    <span class="font-semibold ${isSuccess ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-500'}">${isSuccess ? '✓' : '✕'}</span>
                </div>
            `;
            stepList.appendChild(card);
        });

        lastBatchBundleXml = data.xml || '';
        outputEl.textContent = lastBatchBundleXml;
        outputWrapper.classList.remove('hidden');
        if (copyOutBtn) copyOutBtn.classList.remove('hidden');
        document.getElementById('batch-progress-title').textContent = `Completed ${data.total_commands} command(s)`;
        document.getElementById('batch-progress-stats').textContent = `Total: ${data.time_taken}s`;
        showToast(`Executed ${data.total_commands} batch operations`);
    } catch(e) {
        stepList.innerHTML = `<div class="p-3 bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 rounded-lg border border-red-200 dark:border-red-900/60 font-semibold">${escapeHtml(e.message)}</div>`;
    } finally {
        runBtn.disabled = false;
        runBtn.innerHTML = origBtnHtml;
    }
}

function copyBatchOutputXml() {
    if (!lastBatchBundleXml) return;
    navigator.clipboard.writeText(lastBatchBundleXml).then(() => {
        showToast("Combined batch XML copied to clipboard ✓");
    });
}

// Document Slide-Over Viewer & Target Scroll Helper
let currentDocCollection = '';
let currentDocPath = '';

function downloadCurrentOriginalFile() {
    if (!currentDocPath) {
        showToast("No document selected");
        return;
    }
    const url = apiUrl(`/api/document/download?collection=${encodeURIComponent(currentDocCollection)}&path=${encodeURIComponent(currentDocPath)}`);
    const a = document.createElement('a');
    a.href = url;
    a.download = currentDocPath.split('/').pop() || 'document';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    showToast("Starting download...");
}

function isOpenFileSettingEnabled() {
    return localStorage.getItem('qmd_allow_open_file') === 'true';
}

function setOpenFileSetting(enabled) {
    localStorage.setItem('qmd_allow_open_file', enabled ? 'true' : 'false');
    if (enabled) {
        window._serverAllowsOpen = true;
    }
    updateOpenFileButtonVisibility();
}

function updateOpenFileButtonVisibility(serverAllows = null) {
    if (serverAllows !== null) window._serverAllowsOpen = serverAllows;
    const btn = document.getElementById('btn-open-original');
    if (!btn) return;
    const isAllowedByClient = isOpenFileSettingEnabled();
    const isForbiddenByServer = window._serverAllowsOpen === false;
    if (isAllowedByClient && !isForbiddenByServer) {
        btn.classList.remove('hidden');
    } else {
        btn.classList.add('hidden');
    }
}

function ensureOpenFileSettingInSettings() {
    const container = document.getElementById('tab-content-defaults');
    if (!container || document.getElementById('setting-allow-open-file')) return;
    const row = document.createElement('div');
    row.id = 'setting-row-open-file';
    row.className = "flex items-center justify-between py-2.5 border-t border-gray-200 dark:border-gray-700 mt-2";
    row.innerHTML = `
        <div class="pr-4">
            <div class="text-xs font-semibold text-gray-800 dark:text-gray-200">Allow "Open File" on Host</div>
            <div class="text-[11px] text-gray-500">Launch files in host desktop app (keep off if port-forwarding over 127.0.0.1)</div>
        </div>
        <label class="relative inline-flex items-center cursor-pointer flex-shrink-0">
            <input type="checkbox" id="setting-allow-open-file" class="sr-only peer">
            <div class="w-9 h-5 bg-gray-200 peer-focus:outline-none rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
        </label>
    `;
    container.appendChild(row);
    const chk = row.querySelector('#setting-allow-open-file');
    chk.checked = isOpenFileSettingEnabled();
    chk.onchange = (e) => {
        setOpenFileSetting(e.target.checked);
        showToast(e.target.checked ? "Enabled 'Open File' option" : "Disabled 'Open File' option");
    };
}

async function openCurrentOriginalFile() {
    if (!currentDocPath) {
        showToast("No document selected");
        return;
    }
    if (!isOpenFileSettingEnabled()) {
        showToast("Opening files on host is disabled in Settings");
        return;
    }
    const btn = document.getElementById('btn-open-original');
    const origHtml = btn ? btn.innerHTML : '';
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="animate-spin inline-block">⏳</span> Opening...`;
    }
    try {
        const res = await fetch(apiUrl('/api/document/open'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                collection: currentDocCollection,
                path: currentDocPath
            })
        });
        const data = await res.json();
        if (res.ok && data.status === 'success') {
            showToast(data.message || "Opened file in system viewer ✓");
        } else if (res.status === 403 || data.status === 'disabled') {
            showToast(data.error || "Opening files on host is disabled in settings");
        } else {
            showToast(data.error || data.message || "Failed to open file in system viewer");
        }
    } catch (e) {
        showToast(`Error opening file: ${e.message}`);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = origHtml;
        }
    }
}

function copyCurrentDocumentContent() {
    const content = document.getElementById('slide-content');
    if (!content) return;
    navigator.clipboard.writeText(content.innerText || '').then(() => {
        showToast("Document text copied to clipboard ✓");
    });
}

function findTargetElement(container, targetText) {
    if (!container || !targetText || !targetText.trim()) return null;

    const cleanLines = targetText
        .split('\n')
        .map(l => l.trim())
        .filter(l => l && !l.startsWith('(...') && !l.startsWith('qmd://') && !l.startsWith('#'));

    const candidateText = (cleanLines.length > 0 ? cleanLines.join(' ') : targetText)
        .replace(/[#*_`~>\[\]\(\)]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();

    if (!candidateText) return null;

    const elements = Array.from(container.querySelectorAll('p, h1, h2, h3, h4, h5, h6, li, pre, tr, blockquote'));

    for (let len = Math.min(candidateText.length, 60); len >= 20; len -= 10) {
        const sample = candidateText.slice(0, len).toLowerCase();
        for (const el of elements) {
            const elText = (el.textContent || '').replace(/\s+/g, ' ').toLowerCase();
            if (elText.includes(sample)) return el;
        }
    }

    const words = candidateText.split(' ').filter(w => w.length > 2).slice(0, 5);
    if (words.length >= 2) {
        const phrase = words.join(' ').toLowerCase();
        for (const el of elements) {
            const elText = (el.textContent || '').replace(/\s+/g, ' ').toLowerCase();
            if (elText.includes(phrase)) return el;
        }
    }

    return null;
}

async function openDocument(collection, path, targetText) {
    currentDocCollection = collection || '';
    currentDocPath = path || '';
    const drawer = document.getElementById('slide-over');
    drawer.classList.remove('hidden');
    void drawer.offsetWidth;
    document.getElementById('slide-backdrop').classList.remove('opacity-0');
    document.getElementById('slide-panel').classList.remove('translate-x-full');
    document.getElementById('slide-uri').textContent = collection ? `qmd://${collection}/${path}` : path;
    const content = document.getElementById('slide-content');
    content.innerHTML = '<div class="text-gray-500 py-16 text-center">Loading document...</div>';

    try {
        const res = await fetch(apiUrl(`/api/document?collection=${encodeURIComponent(collection)}&path=${encodeURIComponent(path)}${featureStates.redact_pii ? '&redact_pii=true' : ''}`));
        const data = await res.json();
        document.getElementById('slide-title').textContent = data.title || path;
        if (data.collection) currentDocCollection = data.collection;
        content.innerHTML = marked.parse(data.content || '');
        updateOpenFileButtonVisibility(data.allow_open);

        if (targetText && targetText.trim()) {
            setTimeout(() => {
                const targetEl = findTargetElement(content, targetText);
                if (targetEl) {
                    targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    targetEl.classList.add('bg-yellow-100', 'dark:bg-yellow-900/40', 'rounded', 'transition-all', 'duration-300');
                    setTimeout(() => {
                        targetEl.classList.remove('bg-yellow-100', 'dark:bg-yellow-900/40');
                    }, 2500);
                }
            }, 120);
        }
    } catch(e) {
        content.innerHTML = `<div class="text-red-500 p-4">Error loading document</div>`;
    }
}

function closeDocument() {
    document.getElementById('slide-backdrop').classList.add('opacity-0');
    document.getElementById('slide-panel').classList.add('translate-x-full');
    setTimeout(() => document.getElementById('slide-over').classList.add('hidden'), 200);
}

// Collection Tree Slide-Over Drawer
function openTreeDrawer() {
    const d = document.getElementById('tree-drawer');
    d.classList.remove('hidden');
    void d.offsetWidth;
    document.getElementById('tree-backdrop').classList.remove('opacity-0');
    document.getElementById('tree-panel').classList.remove('-translate-x-full');
    loadCollectionTree();
}

function closeTreeDrawer() {
    document.getElementById('tree-backdrop').classList.add('opacity-0');
    document.getElementById('tree-panel').classList.add('-translate-x-full');
    setTimeout(() => document.getElementById('tree-drawer').classList.add('hidden'), 200);
}

function toggleDocTree() {
    openTreeDrawer();
}

async function loadCollectionTree(pattern = '') {
    const container = document.getElementById('tree-content');
    container.innerHTML = '<div class="text-gray-500 py-6 text-center">Loading tree...</div>';
    try {
        const url = pattern ? apiUrl(`/api/collections/tree?pattern=${encodeURIComponent(pattern)}`) : apiUrl('/api/collections/tree');
        const res = await fetch(url);
        const data = await res.json();
        container.innerHTML = '';
        (Array.isArray(data) ? data : [data]).forEach(item => {
            const div = document.createElement('div');
            div.className = "bg-gray-50 dark:bg-[#202124] rounded-lg p-3 border border-gray-200 dark:border-gray-700";
            div.innerHTML = `<div class="font-bold text-blue-600 mb-2">${escapeHtml(item.collection)}/</div><div class="tree-inner pl-1 space-y-1"></div>`;
            function render(node, parent) {
                (node.children || []).forEach(child => {
                    const r = document.createElement('div');
                    if (child.type === 'directory') {
                        r.innerHTML = `<div class="text-amber-600 font-semibold py-0.5">📁 ${escapeHtml(child.name)}/</div><div class="pl-3 border-l border-gray-200 dark:border-gray-700 ml-1.5 space-y-1"></div>`;
                        render(child, r.children[1]);
                    } else {
                        r.className = "hover:text-blue-500 cursor-pointer py-0.5 truncate";
                        r.textContent = `📄 ${child.name}`;
                        r.onclick = () => openDocument(item.collection, child.path, '');
                    }
                    parent.appendChild(r);
                });
            }
            render(item.tree || {}, div.querySelector('.tree-inner'));
            container.appendChild(div);
        });
    } catch(e) { container.innerHTML = `<div class="text-red-500 p-2">Error loading tree</div>`; }
}

// Global Escape Listener
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeScopePicker();
        closeSessionModal();
        closeBatchModal();
        closeSettings();
        closeDocument();
        closeTreeDrawer();
    }
});

// Initialize button visibility according to setting on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => updateOpenFileButtonVisibility());
} else {
    updateOpenFileButtonVisibility();
}