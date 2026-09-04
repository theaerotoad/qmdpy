// Base API URL Helper for Reverse Proxy / Subpath Support
function apiUrl(path) {
    const base = (window.QMD_BASE_URL || '').replace(/\/+$/, '');
    const cleanPath = path.startsWith('/') ? path : '/' + path;
    return `${base}${cleanPath}`;
}

// Application State & Configuration
let activeMode = 'discover';
let activeTab = 'all'; // 'all' | 'passages' | 'documents'
let toolsOpen = false;
let lastRawJson = null;
let lastRawXml = '';
let lastSearchType = 'discover';
let currentSessionId = null;
let currentExcludedCount = 0;
let selectedScopes = [];
let featureStates = { rerank: true, exclude_seen: false, redact_pii: false };
let treeFilterTimer = null, scopeFilterTimer = null;

const DEFAULTS_KEY = 'qmd_search_defaults';
const SESSION_STORAGE_KEY = 'qmd_active_session_id';
const THEME_STORAGE_KEY = 'qmd_theme_mode';
const QUERY_HISTORY_KEY = 'qmd_query_history';

// HTML & XML Escaping / Highlighting Helpers
function escapeHtml(s) {
    return String(s || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function escapeXmlAttr(s) {
    return escapeHtml(s);
}

function highlightKeywords(text, query) {
    if (!query || !text) return text;
    const terms = query.replace(/[^\w\s]/g, ' ').split(/\s+/).filter(t => t.length > 1);
    if (!terms.length) return text;
    const reg = new RegExp(`(${terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi');
    return text.replace(reg, '<mark class="g-highlight">$1</mark>');
}

// Past Query History & Autocomplete Management
function getQueryHistory() {
    try {
        return JSON.parse(localStorage.getItem(QUERY_HISTORY_KEY) || '[]');
    } catch (e) {
        return [];
    }
}

function saveQueryToHistory(queryStr) {
    const q = (queryStr || '').trim();
    if (!q) return;
    let history = getQueryHistory().filter(item => item.toLowerCase() !== q.toLowerCase());
    history.unshift(q);
    if (history.length > 50) history = history.slice(0, 50);
    localStorage.setItem(QUERY_HISTORY_KEY, JSON.stringify(history));
    updateQueryHistoryDatalist();
}

function updateQueryHistoryDatalist() {
    const datalist = document.getElementById('past-queries-list');
    if (!datalist) return;
    const history = getQueryHistory();
    datalist.innerHTML = history.map(item => `<option value="${escapeXmlAttr(item)}"></option>`).join('');
}

// Load & Apply Default Preferences
function applyDefaultPreferences(customDefaults = null) {
    const defaults = customDefaults || JSON.parse(localStorage.getItem(DEFAULTS_KEY) || '{}');
    if (defaults.limit) {
        document.querySelectorAll('.limit-select').forEach(sel => sel.value = defaults.limit);
    }
    if (defaults.mode) {
        setMode(defaults.mode);
    }
    featureStates.rerank = defaults.rerank !== undefined ? !!defaults.rerank : true;
    featureStates.exclude_seen = defaults.exclude_seen !== undefined ? !!defaults.exclude_seen : false;
    featureStates.redact_pii = defaults.redact_pii !== undefined ? !!defaults.redact_pii : false;
    updateFeatureButtonsUI();
}

function loadSettingsModalValues() {
    const defaults = JSON.parse(localStorage.getItem(DEFAULTS_KEY) || '{}');
    const limEl = document.getElementById('setting-default-limit');
    const modEl = document.getElementById('setting-default-mode');
    const rkEl = document.getElementById('setting-default-rerank');
    const snEl = document.getElementById('setting-default-seen');
    const piEl = document.getElementById('setting-default-pii');
    if (limEl) limEl.value = defaults.limit || '10';
    if (modEl) modEl.value = defaults.mode || 'discover';
    if (rkEl) rkEl.checked = defaults.rerank !== undefined ? !!defaults.rerank : true;
    if (snEl) snEl.checked = !!defaults.exclude_seen;
    if (piEl) piEl.checked = !!defaults.redact_pii;
}

function saveDefaults() {
    const defaults = {
        limit: document.getElementById('setting-default-limit').value,
        mode: document.getElementById('setting-default-mode').value,
        rerank: document.getElementById('setting-default-rerank').checked,
        exclude_seen: document.getElementById('setting-default-seen').checked,
        redact_pii: document.getElementById('setting-default-pii').checked
    };
    localStorage.setItem(DEFAULTS_KEY, JSON.stringify(defaults));
    applyDefaultPreferences(defaults);
    showToast("Preferences saved");
}

function toggleFeature(name, forceVal = null) {
    featureStates[name] = forceVal !== null ? forceVal : !featureStates[name];
    updateFeatureButtonsUI();
}

function updateFeatureButtonsUI() {
    const rerankEl = document.getElementById('filter-rerank');
    const seenEl = document.getElementById('filter-exclude-seen');
    const redactEl = document.getElementById('filter-redact');
    if (rerankEl) rerankEl.checked = !!featureStates.rerank;
    if (seenEl) seenEl.checked = !!featureStates.exclude_seen;
    if (redactEl) redactEl.checked = !!featureStates.redact_pii;
}

// Smart Query Parser for Cheat Codes
function parseQueryDirectives(rawText) {
    let text = rawText || '';
    const directives = {
        cleanQuery: '',
        title: null,
        path: null,
        collection: null,
        lex: null,
        limit: null
    };

    const titleMatch = text.match(/(?:title|t):(?:"([^"]+)"|'([^']+)'|(\S+))/i);
    if (titleMatch) { directives.title = titleMatch[1] || titleMatch[2] || titleMatch[3]; text = text.replace(titleMatch[0], ' '); }

    const pathMatch = text.match(/(?:path|file|p):(?:"([^"]+)"|'([^']+)'|(\S+))/i);
    if (pathMatch) { directives.path = pathMatch[1] || pathMatch[2] || pathMatch[3]; text = text.replace(pathMatch[0], ' '); }

    const colMatch = text.match(/(?:col|in|c):(?:"([^"]+)"|'([^']+)'|(\S+))/i);
    if (colMatch) { directives.collection = colMatch[1] || colMatch[2] || colMatch[3]; text = text.replace(colMatch[0], ' '); }

    const lexMatch = text.match(/(?:lex|fts|l):(?:"([^"]+)"|'([^']+)'|(\S+))/i);
    if (lexMatch) { directives.lex = lexMatch[1] || lexMatch[2] || lexMatch[3]; text = text.replace(lexMatch[0], ' '); }

    const limitMatch = text.match(/(?:limit|n):(\d+)/i);
    if (limitMatch) { directives.limit = parseInt(limitMatch[1]); text = text.replace(limitMatch[0], ' '); }

    const piiMatch = text.match(/(?:pii):(?:"(on|off|true|false)"|'(on|off|true|false)'|(on|off|true|false))/i);
    if (piiMatch) { toggleFeature('redact_pii', ['on', 'true'].includes((piiMatch[1]||piiMatch[2]||piiMatch[3]).toLowerCase())); text = text.replace(piiMatch[0], ' '); }

    const seenMatch = text.match(/(?:seen):(?:"(exclude|off|on|true|false)"|'(exclude|off|on|true|false)'|(exclude|off|on|true|false))/i);
    if (seenMatch) { toggleFeature('exclude_seen', ['exclude', 'on', 'true'].includes((seenMatch[1]||seenMatch[2]||seenMatch[3]).toLowerCase())); text = text.replace(seenMatch[0], ' '); }

    const rrMatch = text.match(/(?:rerank|rr):(?:"(on|off|true|false)"|'(on|off|true|false)'|(on|off|true|false))/i);
    if (rrMatch) { toggleFeature('rerank', ['on', 'true'].includes((rrMatch[1]||rrMatch[2]||rrMatch[3]).toLowerCase())); text = text.replace(rrMatch[0], ' '); }

    directives.cleanQuery = text.replace(/\s+/g, ' ').trim();
    return directives;
}

// View Mode Switching
function setMode(mode) {
    activeMode = mode;
}

// App View State Handling
function setAppState(state) {
    const container = document.getElementById('app-container');
    if (state === 'serp') {
        container.classList.remove('state-hero');
        container.classList.add('state-serp');
        const resWrapper = document.getElementById('results-wrapper');
        if (resWrapper) resWrapper.classList.remove('hidden');
    } else {
        container.classList.remove('state-serp');
        container.classList.add('state-hero');
        const resWrapper = document.getElementById('results-wrapper');
        if (resWrapper) resWrapper.classList.add('hidden');
    }
}

function returnToHero() {
    setAppState('hero');
    generateNewSession();
    document.getElementById('hero-query').value = '';
    document.getElementById('serp-query').value = '';
    document.getElementById('results').innerHTML = '';
    const footer = document.getElementById('results-footer');
    if (footer) footer.classList.add('hidden');
    document.getElementById('hero-query').focus();
}

// Navigation Tabs
function setSearchTab(tabName) {
    activeTab = tabName;
    ['all', 'passages', 'documents'].forEach(tab => {
        const btn = document.getElementById(`tab-${tab}`);
        if (!btn) return;
        if (tab === tabName) {
            btn.className = "h-10 border-b-2 border-blue-600 text-blue-600 dark:text-blue-400 font-medium flex items-center gap-1.5 px-2";
        } else {
            btn.className = "h-10 border-b-2 border-transparent hover:text-gray-900 dark:hover:text-white flex items-center gap-1.5 px-2 font-medium";
        }
    });

    if (tabName === 'documents') {
        activeMode = 'discover';
    } else if (tabName === 'all' || tabName === 'passages') {
        activeMode = 'search';
    }

    const currentQuery = document.getElementById('serp-query').value.trim();
    if (currentQuery) {
        handleFormSubmit(new Event('submit'), 'serp');
    }
}

// Tools Drawer Toggle
function toggleToolsDrawer() {
    const drawer = document.getElementById('tools-drawer');
    const chevron = document.getElementById('tools-chevron');
    const btn = document.getElementById('tools-toggle-btn');
    if (!drawer) return;

    toolsOpen = !toolsOpen;
    if (toolsOpen) {
        drawer.classList.remove('hidden');
        if (chevron) chevron.classList.add('rotate-180');
        if (btn) btn.classList.add('bg-gray-100', 'dark:bg-[#303134]');
    } else {
        drawer.classList.add('hidden');
        if (chevron) chevron.classList.remove('rotate-180');
        if (btn) btn.classList.remove('bg-gray-100', 'dark:bg-[#303134]');
    }
}

function onToolsLimitChange(val) {
    document.querySelectorAll('.limit-select').forEach(sel => sel.value = val);
    const query = document.getElementById('serp-query').value.trim();
    if (query) handleFormSubmit(new Event('submit'), 'serp');
}

function resetToolsFilters() {
    document.querySelectorAll('.limit-select').forEach(sel => sel.value = "10");
    toggleFeature('rerank', true);
    toggleFeature('exclude_seen', false);
    toggleFeature('redact_pii', false);
    showToast("Filters reset to default");
    const query = document.getElementById('serp-query').value.trim();
    if (query) handleFormSubmit(new Event('submit'), 'serp');
}

// Suggestion & Query Helpers
function runSampleQuery(q) {
    const heroInput = document.getElementById('hero-query');
    const serpInput = document.getElementById('serp-query');
    if (heroInput) heroInput.value = q;
    if (serpInput) serpInput.value = q;
    handleFormSubmit(new Event('submit'), 'hero');
}

function clearQuery() {
    const input = document.getElementById('serp-query');
    if (input) {
        input.value = '';
        input.focus();
    }
}

// Dropdown Helper
function toggleDropdown(id) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('hidden');
}

// Toast Notifications
let toastTimeout;
function showToast(message) {
    const toast = document.getElementById('toast');
    const msg = document.getElementById('toast-message');
    if (!toast || !msg) return;
    msg.textContent = message;
    toast.classList.remove('translate-y-20', 'opacity-0');
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => dismissToast(), 3500);
}

function dismissToast() {
    const toast = document.getElementById('toast');
    if (toast) toast.classList.add('translate-y-20', 'opacity-0');
}

// Theme Handling
function applyTheme(mode) {
    localStorage.setItem(THEME_STORAGE_KEY, mode);
    const isDark = mode === 'dark' || (mode === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
    document.documentElement.classList.toggle('dark', isDark);
    document.querySelectorAll('.theme-option-btn').forEach(btn => {
        btn.classList.toggle('border-blue-500', btn.dataset.theme === mode);
    });
}

function toggleTheme() {
    const isDark = document.documentElement.classList.contains('dark');
    applyTheme(isDark ? 'light' : 'dark');
    showToast(isDark ? "Switched to Light mode" : "Switched to Dark mode");
}

// Close Dropdowns on Click Outside
window.addEventListener('click', (e) => {
    if (!e.target.closest('#session-menu') && !e.target.closest('button[onclick*="session-menu"]')) {
        const menu = document.getElementById('session-menu');
        if (menu && !menu.classList.contains('hidden')) menu.classList.add('hidden');
    }
});