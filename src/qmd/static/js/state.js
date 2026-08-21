// Application State & Configuration
let activeMode = 'discover';
let lastRawJson = null;
let lastRawXml = '';
let lastSearchType = 'discover';
let currentSessionId = null;
let currentExcludedCount = 0;
let selectedScopes = [];
let featureStates = { rerank: false, exclude_seen: false, redact_pii: false };
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
    return text.replace(reg, '<mark class="qmd-highlight">$1</mark>');
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
    featureStates.rerank = defaults.rerank !== undefined ? !!defaults.rerank : false;
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
    if (rkEl) rkEl.checked = !!defaults.rerank;
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
}

function toggleFeature(name, forceVal = null) {
    featureStates[name] = forceVal !== null ? forceVal : !featureStates[name];
    updateFeatureButtonsUI();
}

function updateFeatureButtonsUI() {
    const mapping = {
        rerank: '.feature-btn-rerank',
        exclude_seen: '.feature-btn-seen',
        redact_pii: '.feature-btn-pii'
    };
    for (const [key, selector] of Object.entries(mapping)) {
        const isActive = !!featureStates[key];
        document.querySelectorAll(selector).forEach(btn => {
            if (isActive) {
                btn.className = `${selector.replace('.', '')} bg-blue-50 dark:bg-blue-950/60 border border-blue-400 dark:border-blue-700 text-blue-600 dark:text-blue-300 font-medium px-3 py-1.5 rounded-xl transition flex items-center gap-1 shadow-sm`;
            } else {
                btn.className = `${selector.replace('.', '')} bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 px-3 py-1.5 rounded-xl transition flex items-center gap-1 shadow-sm`;
            }
        });
    }
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
    const hQ = document.getElementById('hero-query');
    const sQ = document.getElementById('serp-query');
    if (mode === 'discover') {
        if (hQ) hQ.placeholder = "Discover top documents (e.g. title:\"space\", pii:off)...";
        if (sQ) sQ.placeholder = "Discover top documents...";
    } else {
        if (hQ) hQ.placeholder = "Search notes in full document reading order...";
        if (sQ) sQ.placeholder = "Search documents...";
    }
}

// App View State Handling
function setAppState(state) {
    document.getElementById('app-container').className = state === 'serp' ? 'state-serp min-h-screen flex flex-col' : 'state-hero min-h-screen flex flex-col';
}

function returnToHero() {
    setAppState('hero');
    generateNewSession();
    document.getElementById('hero-query').value = '';
    document.getElementById('serp-query').value = '';
    document.getElementById('results').innerHTML = '';
    document.getElementById('results-toolbar').classList.add('hidden');
    document.getElementById('hero-query').focus();
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
