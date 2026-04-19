/**
 * MatPad — frontend application
 *
 * Layout:
 *   matrix-defn-bar  — compact chip per matrix (name, rows×cols, remove), + Add Matrix
 *   operation-bar    — expression textarea with autocomplete + Compute / Clear
 *   panel-input      — matrix value editors (grid or text)
 *   panel-results    — computation output
 *   panel-caps       — collapsible capabilities reference (default collapsed)
 *
 * Persistence: localStorage  mp_panel_<id>  ("1" = collapsed)
 */
(() => {
'use strict';

// ─── State ───────────────────────────────────────────────────────────────────
const state = {
  matrices: {},    // { A: {rows, cols, mode:"grid"|"numpy", values:[[…]]}, … }
  operations: [],  // catalog from /api/operations
};

// ─── DOM helpers ─────────────────────────────────────────────────────────────
const $  = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];
const el = (tag, cls, html = '') => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html) e.innerHTML = html;
  return e;
};

// ─── Formatting ───────────────────────────────────────────────────────────────
function fmtNumber(v) {
  if (typeof v === 'object' && v !== null && 're' in v) {
    const re = fmtScalar(v.re), im = fmtScalar(Math.abs(v.im));
    return v.im < 0 ? `${re}−${im}i` : `${re}+${im}i`;
  }
  return fmtScalar(v);
}
function fmtScalar(v) {
  if (typeof v !== 'number') return String(v);
  if (!isFinite(v)) return String(v);
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(5);
}

// ─── Panel collapse / expand ─────────────────────────────────────────────────
const LS_KEY = id => `mp_panel_${id}`;

function initPanels() {
  $$('.panel[data-panel-id]').forEach(panel => {
    const id   = panel.dataset.panelId;
    const body = $(`#panel-${id}-body`);
    if (!body) return;
    const defaultCollapsed = id === 'caps';
    const stored   = localStorage.getItem(LS_KEY(id));
    const collapsed = stored !== null ? stored === '1' : defaultCollapsed;
    applyPanelState(panel, body, collapsed);
  });
}

function applyPanelState(panel, body, collapsed) {
  panel.dataset.collapsed = collapsed ? 'true' : 'false';
  const header = $('.panel-header', panel);
  if (header) header.setAttribute('aria-expanded', String(!collapsed));
  body.hidden = collapsed;
}

window.togglePanel = function(id) {
  const panel = $(`#panel-${id}`);
  const body  = $(`#panel-${id}-body`);
  if (!panel || !body) return;
  const wasCollapsed = panel.dataset.collapsed === 'true';
  applyPanelState(panel, body, !wasCollapsed);
  localStorage.setItem(LS_KEY(id), wasCollapsed ? '0' : '1');
};

document.addEventListener('keydown', e => {
  if ((e.key === 'Enter' || e.key === ' ') && e.target.classList.contains('panel-header')) {
    e.preventDefault();
    const panel = e.target.closest('.panel[data-panel-id]');
    if (panel) togglePanel(panel.dataset.panelId);
  }
});

// ─── Matrix definition bar ────────────────────────────────────────────────────

/** Returns the next unused matrix letter (A…Z). */
function nextMatrixName() {
  const used = new Set(Object.keys(state.matrices));
  for (let i = 0; i < 26; i++) {
    const name = String.fromCharCode(65 + i);
    if (!used.has(name)) return name;
  }
  return null;
}

function addMatrix() {
  const name = nextMatrixName();
  if (!name) return;
  state.matrices[name] = { rows: 2, cols: 2, mode: 'grid', values: [['',''],['','']] };
  renderMatrixChips();
  renderMatrixInputs();
}

function removeMatrix(name) {
  if (Object.keys(state.matrices).length <= 1) return; // keep at least one
  delete state.matrices[name];
  const card = $(`#matrix-card-${name}`);
  if (card) card.remove();
  renderMatrixChips();
  const names = Object.keys(state.matrices);
  if (!names.length) {
    $('#matrices-container').innerHTML = '<p class="empty-hint">Add a matrix using the bar above.</p>';
  }
}

function updateMatrixDim(name, type, rawVal) {
  const m = state.matrices[name];
  if (!m) return;
  const val = Math.max(1, Math.min(20, parseInt(rawVal) || 1));
  const oldRows = m.rows, oldCols = m.cols;
  m[type] = val; // 'rows' or 'cols'

  // Resize values array, preserving existing cells
  const newValues = Array.from({ length: m.rows }, (_, r) =>
    Array.from({ length: m.cols }, (_, c) => m.values[r]?.[c] ?? '')
  );
  m.values = newValues;

  // Update the chip input to the clamped value
  const chipInput = $(`.matrix-chip[data-name="${name}"] .chip-dim[data-type="${type}"]`);
  if (chipInput) chipInput.value = val;

  // Rebuild the matrix card only if dimensions actually changed
  if ((type === 'rows' && val !== oldRows) || (type === 'cols' && val !== oldCols)) {
    const card = $(`#matrix-card-${name}`);
    if (card) {
      card.replaceWith(buildMatrixCard(name, m));
    }
  }
}

/** Rename a matrix. Returns the accepted name (new on success, old on failure). */
function renameMatrix(oldName, newName) {
  newName = newName.trim();
  // Must be a valid identifier and not already in use
  if (!newName || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(newName)) return oldName;
  if (newName === oldName) return oldName;
  if (state.matrices[newName]) return oldName; // duplicate

  // Rebuild state, preserving insertion order
  const entries = Object.entries(state.matrices);
  state.matrices = {};
  for (const [k, v] of entries) {
    state.matrices[k === oldName ? newName : k] = v;
  }

  // Update the chip's identity so dim/remove handlers stay correct
  const chip = $(`.matrix-chip[data-name="${oldName}"]`);
  if (chip) chip.dataset.name = newName;

  // Rebuild the matrix card
  const card = $(`#matrix-card-${oldName}`);
  if (card) card.replaceWith(buildMatrixCard(newName, state.matrices[newName]));

  return newName;
}

/** Build a single matrix chip element (no innerHTML — closures read chip.dataset.name). */
function buildChip(name) {
  const m    = state.matrices[name];
  const chip = el('div', 'matrix-chip');
  chip.dataset.name = name;

  // ── Editable name ───────────────────────────────────────────────────────
  const nameInp = document.createElement('input');
  nameInp.type  = 'text';
  nameInp.className = 'chip-name-input';
  nameInp.value = name;
  nameInp.size  = Math.max(1, name.length);
  nameInp.setAttribute('aria-label', 'Matrix name');
  nameInp.title = 'Click to rename (letters, digits, underscore; must start with a letter)';

  nameInp.addEventListener('input', () => {
    // Live visual feedback: mark invalid while typing
    const v = nameInp.value.trim();
    const occupied = v !== chip.dataset.name && state.matrices[v];
    const badIdent = v && !/^[A-Za-z_][A-Za-z0-9_]*$/.test(v);
    nameInp.classList.toggle('invalid', !!(occupied || badIdent));
  });

  nameInp.addEventListener('blur', () => {
    const currentName = chip.dataset.name;
    const accepted = renameMatrix(currentName, nameInp.value);
    nameInp.value = accepted;
    nameInp.size  = Math.max(1, accepted.length);
    nameInp.classList.remove('invalid');
  });

  nameInp.addEventListener('keydown', e => {
    if (e.key === 'Enter')  { e.preventDefault(); nameInp.blur(); }
    if (e.key === 'Escape') { nameInp.value = chip.dataset.name; nameInp.blur(); }
  });

  // ── Dimension inputs ─────────────────────────────────────────────────────
  const rowsInp = document.createElement('input');
  rowsInp.type  = 'number';
  rowsInp.className   = 'chip-dim';
  rowsInp.dataset.type = 'rows';
  rowsInp.min   = 1; rowsInp.max = 20;
  rowsInp.value = m.rows;
  rowsInp.title = 'Rows';

  const colsInp = document.createElement('input');
  colsInp.type  = 'number';
  colsInp.className   = 'chip-dim';
  colsInp.dataset.type = 'cols';
  colsInp.min   = 1; colsInp.max = 20;
  colsInp.value = m.cols;
  colsInp.title = 'Columns';

  [rowsInp, colsInp].forEach(inp => {
    // Read chip.dataset.name at event time so renames are transparent
    inp.addEventListener('change', () => updateMatrixDim(chip.dataset.name, inp.dataset.type, inp.value));
    inp.addEventListener('keydown', e => { if (e.key === 'Enter') inp.blur(); });
  });

  // ── Remove button ─────────────────────────────────────────────────────────
  const removeBtn = document.createElement('button');
  removeBtn.className = 'chip-remove';
  removeBtn.title     = `Remove matrix`;
  removeBtn.textContent = '×';
  removeBtn.addEventListener('click', () => removeMatrix(chip.dataset.name));

  chip.appendChild(nameInp);
  chip.appendChild(rowsInp);
  chip.appendChild(el('span', 'chip-sep', '×'));
  chip.appendChild(colsInp);
  chip.appendChild(removeBtn);
  return chip;
}

function renderMatrixChips() {
  const container = $('#matrix-chips');
  container.innerHTML = '';
  Object.keys(state.matrices).forEach(name => container.appendChild(buildChip(name)));
}

window.removeMatrix = removeMatrix;

// ─── Matrix input panel ────────────────────────────────────────────��──────────
function renderMatrixInputs() {
  const container = $('#matrices-container');
  container.innerHTML = '';
  const names = Object.keys(state.matrices);
  if (!names.length) {
    container.innerHTML = '<p class="empty-hint">Add a matrix using the bar above.</p>';
    return;
  }
  names.forEach(name => container.appendChild(buildMatrixCard(name, state.matrices[name])));
}

function buildMatrixCard(name, m) {
  const card = el('div', 'matrix-input-card');
  card.id = `matrix-card-${name}`;
  card.innerHTML = `
    <div class="matrix-card-header">
      <div>
        <span class="matrix-card-title">${name}</span>
        <span class="matrix-card-dims">${m.rows} × ${m.cols}</span>
      </div>
      <div class="input-mode-toggle">
        <button class="mode-btn ${m.mode === 'grid' ? 'active' : ''}"
                onclick="setMatrixMode('${name}','grid')">Grid</button>
        <button class="mode-btn ${m.mode === 'numpy' ? 'active' : ''}"
                onclick="setMatrixMode('${name}','numpy')">Text</button>
      </div>
    </div>
    <div class="matrix-card-body" id="matrix-body-${name}"></div>
  `;
  renderMatrixBody(name, m, $(`#matrix-body-${name}`, card));
  return card;
}

function renderMatrixBody(name, m, container) {
  container.innerHTML = '';
  container.appendChild(m.mode === 'grid' ? buildGridEditor(name, m) : buildNumpyEditor(name, m));
}

/** Set an input cell's inline width to fit its current value. */
function syncCellWidth(inp) {
  const len = Math.max(6, (inp.value || '').length + 1);
  inp.style.width = `calc(${len} * 1ch + 10px)`;
}

function buildGridEditor(name, m) {
  const frame = el('div', 'matrix-frame');
  const grid  = el('div', 'matrix-grid');
  grid.style.gridTemplateColumns = `repeat(${m.cols}, max-content)`;

  for (let r = 0; r < m.rows; r++) {
    for (let c = 0; c < m.cols; c++) {
      const inp = document.createElement('input');
      inp.type = 'text';
      inp.inputMode = 'numeric';
      inp.className = 'cell-input' + (r % 2 === 1 ? ' row-guide' : '');
      inp.dataset.row = r;
      inp.dataset.col = c;
      inp.dataset.matrix = name;
      inp.value = m.values[r]?.[c] ?? '';
      inp.setAttribute('aria-label', `${name}[${r+1},${c+1}]`);
      syncCellWidth(inp);
      inp.addEventListener('input', onCellInput);
      inp.addEventListener('keydown', onCellKeydown);
      // On blur: evaluate any expression and format to 5 decimal places
      inp.addEventListener('blur', () => {
        const raw = inp.value.trim();
        if (!raw || raw === '-') return;
        try {
          const formatted = fmtScalar(evalMathExpr(raw));
          inp.value = formatted;
          if (state.matrices[name]) state.matrices[name].values[r][c] = formatted;
          inp.classList.remove('invalid');
        } catch (_) { /* keep raw value; invalid class already set */ }
        syncCellWidth(inp);
      });
      grid.appendChild(inp);
    }
  }
  frame.appendChild(grid);
  return frame;
}

function onCellInput(e) {
  const inp  = e.target;
  const name = inp.dataset.matrix;
  const r    = parseInt(inp.dataset.row);
  const c    = parseInt(inp.dataset.col);
  const val  = inp.value.trim();
  if (!state.matrices[name]) return;
  state.matrices[name].values[r][c] = val;
  let valid = val === '' || val === '-';
  if (!valid) {
    try { evalMathExpr(val); valid = true; } catch (_) {
      valid = isFinite(parseFloat(val)); // allow partial input like "1."
    }
  }
  inp.classList.toggle('invalid', !valid);
  syncCellWidth(inp);
}

function onCellKeydown(e) {
  const inp  = e.target;
  const name = inp.dataset.matrix;
  const m    = state.matrices[name];
  if (!m) return;
  const r = parseInt(inp.dataset.row);
  const c = parseInt(inp.dataset.col);
  let nr = r, nc = c;
  if      (e.key === 'ArrowRight' || (e.key === 'Tab' && !e.shiftKey))  { if (c < m.cols-1) nc=c+1; else if (r<m.rows-1){nr=r+1;nc=0;} }
  else if (e.key === 'ArrowLeft'  || (e.key === 'Tab' && e.shiftKey))   { if (c > 0) nc=c-1; else if (r>0){nr=r-1;nc=m.cols-1;} }
  else if (e.key === 'ArrowDown'  || e.key === 'Enter')                  { if (r < m.rows-1) nr=r+1; }
  else if (e.key === 'ArrowUp')                                          { if (r > 0) nr=r-1; }
  else return;
  if (nr !== r || nc !== c) {
    e.preventDefault();
    const next = $(`input[data-matrix="${name}"][data-row="${nr}"][data-col="${nc}"]`);
    if (next) next.focus();
  }
}

function buildNumpyEditor(name, m) {
  const wrap = el('div', 'numpy-input-wrapper');
  const ta   = el('textarea', 'numpy-textarea');
  ta.placeholder = `[[1, 0], [0, 1]]  or  1 0; 0 1  or plain paste`;
  ta.rows = Math.min(m.rows + 1, 8);
  ta.value = valuesToNumpyText(m.values);
  ta.id = `numpy-ta-${name}`;

  const btn    = el('button', 'btn btn-primary numpy-parse-btn', 'Apply');
  const errDiv = el('div', 'numpy-error');
  errDiv.hidden = true;

  btn.addEventListener('click', () => {
    const result = parseNumpyText(ta.value);
    if (result.error) {
      errDiv.textContent = result.error;
      errDiv.hidden = false;
    } else {
      errDiv.hidden = true;
      const rows = result.data.length;
      const cols = result.data[0].length;
      state.matrices[name].values = result.data;
      state.matrices[name].rows   = rows;
      state.matrices[name].cols   = cols;
      // Sync chip dims
      const chipRows = $(`.matrix-chip[data-name="${name}"] .chip-dim[data-type="rows"]`);
      const chipCols = $(`.matrix-chip[data-name="${name}"] .chip-dim[data-type="cols"]`);
      if (chipRows) chipRows.value = rows;
      if (chipCols) chipCols.value = cols;
      const card = $(`#matrix-card-${name}`);
      if (card) card.replaceWith(buildMatrixCard(name, state.matrices[name]));
    }
  });

  wrap.appendChild(ta);
  wrap.appendChild(btn);
  wrap.appendChild(errDiv);
  return wrap;
}

function valuesToNumpyText(values) {
  if (!values || !values.length) return '';
  return '[' + values.map(row => '[' + row.map(v => v === '' ? '0' : v).join(', ') + ']').join(', ') + ']';
}

// ─── Math expression evaluator ───────────────────────────────────────────────
// Ordered longest-first so e.g. "atan2" is replaced before "atan".
const _MATH_SUBS = [
  ['atan2', 'Math.atan2'], ['log10', 'Math.log10'], ['log2',  'Math.log2'],
  ['sqrt',  'Math.sqrt'],  ['cbrt',  'Math.cbrt'],  ['hypot', 'Math.hypot'],
  ['asin',  'Math.asin'],  ['acos',  'Math.acos'],  ['atan',  'Math.atan'],
  ['sinh',  'Math.sinh'],  ['cosh',  'Math.cosh'],  ['tanh',  'Math.tanh'],
  ['exp',   'Math.exp'],   ['log',   'Math.log'],   ['abs',   'Math.abs'],
  ['ceil',  'Math.ceil'],  ['floor', 'Math.floor'], ['round', 'Math.round'],
  ['sign',  'Math.sign'],  ['pow',   'Math.pow'],
  ['sin',   'Math.sin'],   ['cos',   'Math.cos'],   ['tan',   'Math.tan'],
  ['pi',    'Math.PI'],    ['e',     'Math.E'],
];

function evalMathExpr(expr) {
  expr = String(expr).trim();
  let safe = expr;
  for (const [name, repl] of _MATH_SUBS) {
    safe = safe.replace(new RegExp(`\\b${name}\\b`, 'g'), repl);
  }
  safe = safe.replace(/\^/g, '**');
  // After substitution only Math.xxx, digits, operators, parens and spaces should remain
  if (/[a-zA-Z_$]/.test(safe.replace(/\bMath\.[a-zA-Z0-9]+\b/g, ''))) {
    throw new Error(`Unknown identifier in: "${expr}"`);
  }
  // eslint-disable-next-line no-new-func
  const v = Function('"use strict"; return (' + safe + ')')();
  if (typeof v !== 'number' || !isFinite(v)) throw new Error(`"${expr}" is not a finite number`);
  return v;
}

/** Split a string by top-level commas (respects parentheses and bracket depth). */
function splitByComma(str) {
  const parts = [];
  let depth = 0, start = 0;
  for (let i = 0; i < str.length; i++) {
    if (str[i] === '(' || str[i] === '[') depth++;
    else if (str[i] === ')' || str[i] === ']') depth--;
    else if (str[i] === ',' && depth === 0) {
      parts.push(str.slice(start, i).trim());
      start = i + 1;
    }
  }
  const last = str.slice(start).trim();
  if (last) parts.push(last);
  return parts;
}

/**
 * Parse a [[expr, …], [expr, …]] literal where cell values may be math
 * expressions (e.g. 2/sqrt(5), pi/4, cos(pi/3)).
 */
function parseExprArrayLiteral(text) {
  text = text.trim();
  if (!text.startsWith('[') || !text.endsWith(']')) throw new Error('Expected [...] array');
  const inner = text.slice(1, -1).trim();
  if (!inner.startsWith('[')) {
    // 1-D: [expr, expr, …]
    return [splitByComma(inner).filter(Boolean).map(evalMathExpr)];
  }
  // 2-D: [[row], [row], …] — locate row boundaries by tracking only [ / ]
  const rows = [];
  let i = 0;
  while (i < inner.length) {
    while (i < inner.length && /[\s,]/.test(inner[i])) i++;
    if (i >= inner.length) break;
    if (inner[i] !== '[') throw new Error(`Unexpected character "${inner[i]}" at position ${i}`);
    let j = i + 1, depth = 1;
    while (j < inner.length && depth > 0) {
      if (inner[j] === '[') depth++;
      else if (inner[j] === ']') depth--;
      j++;
    }
    rows.push(splitByComma(inner.slice(i + 1, j - 1)).filter(Boolean).map(evalMathExpr));
    i = j;
  }
  return rows;
}

// Regex for plain number tokens (used when no expressions are present)
const NUM_RE = /-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g;

function parseNumpyText(text) {
  // Normalize typographic minus/dash variants to ASCII hyphen-minus
  text = text.trim().replace(/[\u2212\u2013\u2014]/g, '-');
  try {
    let rows;
    if (text.startsWith('[')) {
      const cleaned = text
        .replace(/\barray\s*\(/g, '')
        .replace(/\)\s*$/, '')
        .replace(/'/g, '"');
      // Fast path: pure JSON numbers
      let parsed = null;
      try { parsed = JSON.parse(cleaned); } catch (_) { /* fall through */ }
      if (parsed !== null && Array.isArray(parsed)) {
        rows = Array.isArray(parsed[0]) ? parsed : [parsed];
      } else {
        // Expression-aware path
        rows = parseExprArrayLiteral(cleaned);
      }
    } else {
      const sep = (text.includes('\n') && !text.includes(';')) ? '\n' : ';';
      rows = text.split(sep).map(row => {
        row = row.trim();
        if (!row) return [];
        // If the row contains letters or commas it may have expressions
        if (/[a-zA-Z,]/.test(row)) {
          return splitByComma(row).filter(Boolean).map(evalMathExpr);
        }
        return (row.match(NUM_RE) || []).map(Number);
      }).filter(row => row.length > 0);
    }
    if (!rows || !rows.length) throw new Error('No data found');
    const cols = rows[0].length;
    if (cols === 0) throw new Error('Matrix cannot have 0 columns');
    if (rows.some(r => r.length !== cols)) throw new Error('All rows must have the same number of columns');
    if (rows.some(r => r.some(v => isNaN(v)))) throw new Error('Non-numeric values detected');
    return { data: rows.map(r => r.map(v => fmtScalar(typeof v === 'number' ? v : parseFloat(v)))) };
  } catch (e) {
    return { error: `Parse error: ${e.message}` };
  }
}

window.setMatrixMode = function(name, mode) {
  if (!state.matrices[name]) return;
  state.matrices[name].mode = mode;
  const card = $(`#matrix-card-${name}`);
  if (card) card.replaceWith(buildMatrixCard(name, state.matrices[name]));
};

// ─── Collect matrix values for API ───────────────────────────────────────────
function collectMatrices() {
  const result = {}, errors = [];
  for (const [name, m] of Object.entries(state.matrices)) {
    const grid = [];
    let ok = true;
    for (let r = 0; r < m.rows; r++) {
      const row = [];
      for (let c = 0; c < m.cols; c++) {
        const raw = (m.values[r]?.[c] ?? '').toString().trim();
        if (raw === '') { row.push(0); continue; }
        try {
          row.push(evalMathExpr(raw));
        } catch (_) {
          const v = parseFloat(raw);
          if (!isFinite(v)) { errors.push(`${name}[${r+1},${c+1}]: invalid value "${raw}"`); ok = false; }
          else row.push(v);
        }
      }
      grid.push(row);
    }
    if (ok) result[name] = grid;
  }
  return { matrices: result, errors };
}

// ─── Compute ──────────────────────────────────────────────────────────────────
async function compute() {
  const btn        = $('#btn-compute');
  const expression = $('#op-expression').value;
  if (!expression.trim()) { showError('Please enter an operation expression.'); return; }

  const { matrices, errors } = collectMatrices();
  if (errors.length) { showError(errors.join('\n')); return; }

  btn.disabled = true;
  btn.textContent = 'Computing…';

  try {
    const resp = await fetch('/api/compute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ matrices, expression }),
    });
    const data = await resp.json();
    if (data.error) {
      showError(data.error);
    } else {
      renderResults(data.results);
      // Capture snapshot for session save (all computations since last save)
      _unsavedSessions.push({
        timestamp:  new Date().toISOString(),
        matrices:   matrices,
        expression: expression,
        results:    data.results,
      });
      updateSaveButtonState();
      const panel = $('#panel-results');
      if (panel && panel.dataset.collapsed === 'true') togglePanel('results');
    }
  } catch (e) {
    showError(`Network error: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<svg viewBox="0 0 20 20" width="14" height="14"><path d="M4 4l12 6-12 6V4z" fill="currentColor"/></svg> Compute';
  }
}

function showError(msg) {
  const container = $('#results-container');
  container.innerHTML = '';
  const block = el('div', 'result-block');
  block.innerHTML = `
    <div class="result-header"><span class="result-line-badge">Error</span></div>
    <div class="result-body"><div class="result-error">${escHtml(msg)}</div></div>`;
  container.appendChild(block);
  const panel = $('#panel-results');
  if (panel && panel.dataset.collapsed === 'true') togglePanel('results');
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ─── LaTeX export ─────────────────────────────────────────────────────────────

function fmtLatexNum(v) {
  if (typeof v === 'object' && v !== null && 're' in v) {
    const re = fmtScalar(v.re), im = fmtScalar(Math.abs(v.im));
    return v.im < 0 ? `${re} - ${im}i` : `${re} + ${im}i`;
  }
  return fmtScalar(typeof v === 'number' ? v : parseFloat(v));
}

function matrixToLatex(values) {
  const rows = values.map(row => row.map(fmtLatexNum).join(' & ')).join(' \\\\\n    ');
  return `\\begin{bmatrix}\n    ${rows}\n  \\end{bmatrix}`;
}

function vectorToLatex(values) {
  const flat = Array.isArray(values[0]) ? values.flat() : values;
  return `\\begin{bmatrix}\n    ${flat.map(fmtLatexNum).join(' \\\\\n    ')}\n  \\end{bmatrix}`;
}

function resultValueToLatex(res) {
  if (!res) return '';
  switch (res.type) {
    case 'scalar':  return fmtLatexNum(res.value);
    case 'boolean': return `\\text{${res.value ? 'true' : 'false'}}`;
    case 'vector':  return vectorToLatex(res.value);
    case 'matrix':  return matrixToLatex(res.value);
    case 'multi_output':
      return Object.entries(res.outputs)
        .map(([k, v]) => `  ${k} &= ${resultValueToLatex(v)}`)
        .join(' \\\\\n');
    default: return String(res.value ?? '');
  }
}

function resultsToLatex(results) {
  return results
    .filter(r => !r.error && r.result)
    .map(r => {
      const expr = r.expr.trim();
      if (r.result.type === 'multi_output') {
        return `% ${expr}\n\\begin{aligned}\n${resultValueToLatex(r.result)}\n\\end{aligned}`;
      }
      return `${expr} = ${resultValueToLatex(r.result)}`;
    })
    .join('\n\n');
}

// Stored for the copy button
let _lastResults = [];

function setCopyLatexEnabled(on) {
  const btn = $('#btn-copy-latex');
  if (btn) btn.disabled = !on;
}

window.copyResultsAsLatex = function() {
  const latex = resultsToLatex(_lastResults);
  if (!latex) return;
  const btn   = $('#btn-copy-latex');
  const label = $('#copy-latex-label');
  navigator.clipboard.writeText(latex).then(() => {
    btn.classList.add('copied');
    label.textContent = 'Copied!';
    setTimeout(() => { btn.classList.remove('copied'); label.textContent = 'Copy LaTeX'; }, 2000);
  }).catch(() => {
    // Fallback for older browsers
    const ta = document.createElement('textarea');
    ta.value = latex;
    ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    btn.classList.add('copied');
    label.textContent = 'Copied!';
    setTimeout(() => { btn.classList.remove('copied'); label.textContent = 'Copy LaTeX'; }, 2000);
  });
};

// ─── Session save ─────────────────────────────────────────────────────────────

// Feature detection
const _hasFSA = typeof window.showDirectoryPicker === 'function';

// Queue of snapshots captured after each successful compute
let _unsavedSessions = [];
// FileSystemDirectoryHandle (Chrome/Edge native picker)
let _dirHandle = null;
// Folder path string used when FSA is unavailable
let _serverFolderPath = 'sessions';

// ── Warn before leaving with unsaved work ─────────────────────────────────────
window.addEventListener('beforeunload', e => {
  if (_unsavedSessions.length > 0) {
    e.preventDefault();
    e.returnValue = ''; // required for Chrome to show the dialog
  }
});

// LaTeX document skeleton (mirrors backend _TEX_PREAMBLE / _TEX_END)
const _TEX_PREAMBLE = `\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{geometry}
\\geometry{margin=1in}

\\title{MatPad Session Log}
\\date{}

\\begin{document}
\\maketitle

`;
const _TEX_END = '\\end{document}';

function escLatex(s) {
  return String(s)
    .replace(/\\/g, '\\textbackslash{}')
    .replace(/[&%$#_{}~^]/g, c => `\\${c}`)
    .replace(/</g, '\\textless{}')
    .replace(/>/g, '\\textgreater{}');
}

function _sessionResultsToLatex(results) {
  return results
    .filter(r => !r.error && r.result)
    .map(r => {
      const expr = r.expr.trim();
      if (r.result.type === 'multi_output') {
        return `\\[\n% ${escLatex(expr)}\n\\begin{aligned}\n${resultValueToLatex(r.result)}\n\\end{aligned}\n\\]`;
      }
      return `\\[\n${expr} = ${resultValueToLatex(r.result)}\n\\]`;
    })
    .join('\n\n');
}

function _sessionEntryToLatex(session) {
  const dateStr = session.timestamp.replace('T', ' ').slice(0, 19);

  const matrixEntries = Object.entries(session.matrices);
  let matrixLatex = '';
  if (matrixEntries.length > 0) {
    const parts = matrixEntries.map(([n, rows]) => `${escLatex(n)} = ${matrixToLatex(rows)}`);
    matrixLatex = `\\subsection*{Matrices}\n\\[\n${parts.join(', \\quad\n')}\n\\]\n\n`;
  }

  const expr = session.expression.trim();
  const exprLatex = expr
    ? `\\subsection*{Expressions}\n\\begin{verbatim}\n${expr}\n\\end{verbatim}\n\n`
    : '';

  const hasResults = session.results.some(r => !r.error && r.result);
  const resultsLatex = hasResults
    ? `\\subsection*{Results}\n${_sessionResultsToLatex(session.results)}\n\n`
    : '';

  return `\\section*{MatPad Session --- ${dateStr}}\n\n${matrixLatex}${exprLatex}${resultsLatex}\\hrule\n`;
}

function _allUnsavedToLatex() {
  return _unsavedSessions.map(_sessionEntryToLatex).join('\n\n');
}

function _sanitiseFilename(name) {
  const n = (name || 'matpad-sessions').replace(/[^a-zA-Z0-9_\-.]/g, '_') || 'matpad-sessions';
  return n.endsWith('.tex') ? n : n + '.tex';
}

function updateSaveButtonState() {
  const btn   = $('#btn-save-session');
  const label = $('#save-session-label');
  if (!btn) return;
  const n = _unsavedSessions.length;
  btn.disabled = n === 0;
  if (label) {
    label.textContent = n > 1 ? `Save ${n} Sessions` : 'Save Session';
  }
}

function showSessionToast(msg, type = 'success') {
  let toast = document.getElementById('session-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'session-toast';
    document.body.appendChild(toast);
  }
  toast.className = `session-toast session-toast--${type}`;
  toast.textContent = msg;
  void toast.offsetWidth; // force reflow so transition re-fires
  toast.classList.add('session-toast--visible');
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.remove('session-toast--visible'), 3200);
}

// ── Folder picker — unified button (FSA native dialog or prompt fallback) ─────
window.pickFolder = async function() {
  if (_hasFSA) {
    try {
      _dirHandle = await window.showDirectoryPicker({ mode: 'readwrite', startIn: 'documents' });
      _updateFolderLabel(_dirHandle.name);
    } catch (e) {
      if (e.name !== 'AbortError') showSessionToast(`Folder error: ${e.message}`, 'error');
    }
  } else {
    // Browsers without File System Access API: ask for a path via prompt
    const input = prompt('Enter folder path for session file:', _serverFolderPath);
    if (input !== null) {
      _serverFolderPath = input.trim() || 'sessions';
      _updateFolderLabel(_serverFolderPath);
    }
  }
};

function _updateFolderLabel(name) {
  const span = $('#folder-display-text');
  const btn  = $('#btn-choose-folder');
  if (!span) return;
  span.textContent = name || 'Choose folder…';
  const chosen = !!(name && name !== 'Choose folder…');
  btn && btn.classList.toggle('folder-chosen', chosen);
}

// ── Write via File System Access API ──────────────────────────────────────────
async function _saveSessionFSA(filename, content) {
  let existingContent = '';
  try {
    const fh   = await _dirHandle.getFileHandle(filename);
    const file = await fh.getFile();
    // Normalise to \n so the _TEX_END search/replace works regardless of
    // whether the file was written on Windows (\r\n) or Unix (\n)
    existingContent = (await file.text()).replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  } catch { /* new file */ }

  const newContent = existingContent
    ? (existingContent.includes(_TEX_END)
        ? existingContent.replace(_TEX_END, content + '\n\n' + _TEX_END)
        : existingContent + '\n' + content + '\n')
    : _TEX_PREAMBLE + content + '\n\n' + _TEX_END + '\n';

  const fh       = await _dirHandle.getFileHandle(filename, { create: true });
  const writable = await fh.createWritable();
  await writable.write(newContent);
  await writable.close();

  showSessionToast(`Saved → ${_dirHandle.name}/${filename}`, 'success');
}

// ── Write via server (non-FSA browsers) ───────────────────────────────────────
async function _saveSessionServer(filename, content) {
  const resp = await fetch('/api/save-session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, content, folder: _serverFolderPath || null }),
  });
  const data = await resp.json();
  if (data.error) throw new Error(data.error);
  showSessionToast(`Saved → ${data.display}`, 'success');
}

// ── Main save entry point ──────────────────────────────────────────────────────
window.saveSession = async function() {
  if (_unsavedSessions.length === 0) return;

  // On FSA browsers ensure a folder is chosen first (opens picker if not yet set)
  if (_hasFSA && !_dirHandle) {
    await window.pickFolder();
    if (!_dirHandle) return; // user cancelled the picker
  }

  const btn      = $('#btn-save-session');
  const filename = _sanitiseFilename(($('#session-filename').value || '').trim());
  const content  = _allUnsavedToLatex();

  btn.disabled = true;
  const origHTML = btn.innerHTML;
  btn.querySelector('#save-session-label').textContent = 'Saving…';

  try {
    if (_hasFSA && _dirHandle) {
      await _saveSessionFSA(filename, content);
    } else {
      await _saveSessionServer(filename, content);
    }
    _unsavedSessions = [];
  } catch (e) {
    showSessionToast(`Save failed: ${e.message}`, 'error');
  } finally {
    btn.innerHTML = origHTML;
    updateSaveButtonState();
  }
};

// ─── Render results ───────────────────────────────────────────────────────────
function renderResults(results) {
  _lastResults = results;
  setCopyLatexEnabled(results.some(r => !r.error && r.result));
  const container = $('#results-container');
  container.innerHTML = '';
  if (!results.length) { container.innerHTML = '<p class="empty-hint">No expressions to evaluate.</p>'; return; }

  results.forEach(r => {
    const block = el('div', 'result-block');
    block.innerHTML = `
      <div class="result-header">
        <span class="result-line-badge">Line ${r.line}</span>
        <span class="result-expr">${escHtml(r.expr)}</span>
      </div>
      <div class="result-body" id="result-body-${r.line}"></div>`;
    container.appendChild(block);
    const body = $(`#result-body-${r.line}`, block);
    if (r.error) {
      body.innerHTML = `<div class="result-error">${escHtml(r.error)}</div>`;
    } else {
      renderResultValue(r.result, body);
    }
  });
}

function renderResultValue(res, container) {
  if (!res) { container.innerHTML = '<em class="empty-hint">No result</em>'; return; }
  switch (res.type) {
    case 'scalar':       renderScalar(res.value, container);       break;
    case 'boolean':      renderBoolean(res.value, container);      break;
    case 'vector':       renderVector(res.value, container);       break;
    case 'matrix':       renderMatrix(res.value, container);       break;
    case 'multi_output': renderMultiOutput(res.outputs, container);break;
    default:
      container.innerHTML = `<div class="result-scalar"><span class="result-scalar-value">${escHtml(String(res.value))}</span></div>`;
  }
}

function renderScalar(value, container) {
  const d = el('div', 'result-scalar');
  d.innerHTML = `<span class="result-scalar-label">scalar</span>
                 <span class="result-scalar-value">${escHtml(fmtNumber(value))}</span>`;
  container.appendChild(d);
}

function renderBoolean(value, container) {
  const d = el('div', 'result-scalar');
  const cls = value ? 'result-bool-true' : 'result-bool-false';
  d.innerHTML = `<span class="result-scalar-label">boolean</span>
                 <span class="result-scalar-value ${cls}">${value ? 'true' : 'false'}</span>`;
  container.appendChild(d);
}

function renderVector(values, container) {
  container.appendChild(el('div', 'result-vector-label', 'vector'));
  const frame = el('div', 'result-matrix-frame');
  const grid  = el('div', 'result-matrix-grid');
  grid.style.gridTemplateColumns = 'max-content';
  const flat = Array.isArray(values[0]) ? values.flat() : values;
  flat.forEach(v => grid.appendChild(el('div', 'result-cell', escHtml(fmtNumber(v)))));
  frame.appendChild(grid);
  container.appendChild(frame);
}

function renderMatrix(values, container, label = 'matrix') {
  container.appendChild(el('div', 'result-matrix-label', label));
  const frame = el('div', 'result-matrix-frame');
  const grid  = el('div', 'result-matrix-grid');
  const cols  = values.length ? values[0].length : 0;
  grid.style.gridTemplateColumns = `repeat(${cols}, max-content)`;
  values.forEach((row, r) =>
    row.forEach(v => grid.appendChild(
      el('div', 'result-cell' + (r % 2 === 1 ? ' row-guide' : ''), escHtml(fmtNumber(v)))
    ))
  );
  frame.appendChild(grid);
  container.appendChild(frame);
}

function renderMultiOutput(outputs, container) {
  const section = el('div', 'multi-output-section');
  for (const [key, res] of Object.entries(outputs)) {
    const item = el('div', 'multi-output-item');
    item.innerHTML = `<div class="multi-output-item-label">${escHtml(key)}</div>`;
    renderResultValue(res, item);
    section.appendChild(item);
  }
  container.appendChild(section);
}

// ─── Autocomplete ─────────────────────────────────────────────────────────────
const acState    = { active: -1, items: [] };
const acDropdown = document.getElementById('autocomplete-dropdown');
const opTextarea = document.getElementById('op-expression');

function getWordAtCursor(ta) {
  const pos = ta.selectionStart, text = ta.value;
  let start = pos;
  while (start > 0 && /[A-Za-z_0-9]/.test(text[start - 1])) start--;
  return { word: text.slice(start, pos), start, end: pos };
}

function showAutocomplete(matches) {
  acDropdown.innerHTML = '';
  acState.items  = matches;
  acState.active = -1;
  matches.forEach((op, i) => {
    const item = el('div', 'autocomplete-item');
    item.dataset.index = i;
    item.innerHTML = `<span class="ac-name">${escHtml(op.name)}</span>
      ${op.operator ? `<span class="ac-op">${escHtml(op.operator)}</span>` : ''}
      <span class="ac-desc">${escHtml(op.description)}</span>`;
    item.addEventListener('mousedown', e => { e.preventDefault(); insertCompletion(op.name); });
    acDropdown.appendChild(item);
  });
  acDropdown.hidden = !matches.length;
}

function hideAutocomplete() { acDropdown.hidden = true; acState.active = -1; }

function insertCompletion(name) {
  const { start, end } = getWordAtCursor(opTextarea);
  const before = opTextarea.value.slice(0, start);
  const after  = opTextarea.value.slice(end);
  opTextarea.value = before + name + '(' + after;
  const pos = before.length + name.length + 1;
  opTextarea.setSelectionRange(pos, pos);
  opTextarea.focus();
  hideAutocomplete();
}

function updateAutocomplete() {
  const { word } = getWordAtCursor(opTextarea);
  if (!word) { hideAutocomplete(); return; }
  const lower   = word.toLowerCase();
  const matches = state.operations.filter(op => op.name.toLowerCase().startsWith(lower));
  showAutocomplete(matches);
}

opTextarea.addEventListener('input', updateAutocomplete);
opTextarea.addEventListener('click', updateAutocomplete);
opTextarea.addEventListener('blur', () => setTimeout(hideAutocomplete, 150));
opTextarea.addEventListener('keydown', e => {
  if (acDropdown.hidden) return;
  const items = $$('.autocomplete-item', acDropdown);
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    acState.active = Math.min(acState.active + 1, items.length - 1);
    items.forEach((it, i) => it.classList.toggle('active', i === acState.active));
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    acState.active = Math.max(acState.active - 1, 0);
    items.forEach((it, i) => it.classList.toggle('active', i === acState.active));
  } else if (e.key === 'Enter' || e.key === 'Tab') {
    if (acState.active >= 0 && acState.items[acState.active]) {
      e.preventDefault();
      insertCompletion(acState.items[acState.active].name);
    } else if (e.key === 'Enter') {
      hideAutocomplete();
    }
  } else if (e.key === 'Escape') {
    hideAutocomplete();
  }
});

// ─── Capabilities panel / fuzzy search ───────────────────────────────────────
function renderCapabilitiesTable(ops) {
  const container = $('#caps-table-container');
  container.innerHTML = '';
  const table = el('table', 'caps-table');
  table.innerHTML = `<thead><tr><th>Function</th><th>Operator</th><th>Description</th></tr></thead>`;
  const tbody = document.createElement('tbody');
  ops.forEach(op => {
    const tr = document.createElement('tr');
    tr.dataset.name = op.name.toLowerCase();
    tr.dataset.op   = (op.operator ?? '').toLowerCase();
    tr.dataset.desc = op.description.toLowerCase();
    tr.innerHTML = `
      <td><span class="caps-fn-name">${escHtml(op.name)}</span></td>
      <td>${op.operator ? `<span class="caps-op-sym">${escHtml(op.operator)}</span>` : '—'}</td>
      <td>${escHtml(op.description)}</td>`;
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  container.appendChild(table);
}

function fuzzyScore(haystack, needle) {
  if (!needle) return 1;
  const h = haystack.toLowerCase(), n = needle.toLowerCase();
  if (h.includes(n)) return 1;
  let hi = 0, ni = 0, score = 0;
  while (hi < h.length && ni < n.length) { if (h[hi] === n[ni]) { score++; ni++; } hi++; }
  return ni === n.length ? score / n.length : 0;
}

$('#caps-search').addEventListener('input', e => {
  const q = e.target.value.trim();
  $$('.caps-table tbody tr').forEach(tr => {
    const score = fuzzyScore(`${tr.dataset.name} ${tr.dataset.op} ${tr.dataset.desc}`, q);
    tr.classList.toggle('caps-hidden', score < 0.6 && q.length > 0);
  });
});

// ─── Load operations catalog ──────────────────────────────────────────────────
async function loadOperations() {
  try {
    const resp = await fetch('/api/operations');
    state.operations = await resp.json();
    renderCapabilitiesTable(state.operations);
  } catch (e) {
    console.error('Failed to load operations catalog', e);
  }
}

// ─── Wire buttons ─────────────────────────────────────────────────────────────
$('#btn-add-matrix').addEventListener('click', addMatrix);
$('#btn-compute').addEventListener('click', compute);
$('#btn-clear-results').addEventListener('click', () => {
  _lastResults = [];
  setCopyLatexEnabled(false);
  $('#results-container').innerHTML = '<p class="empty-hint">Results will appear here after computation.</p>';
});

// ─── Boot ─────────────────────────────────────────────────────────────────────
function boot() {
  initPanels();
  loadOperations();
  // Default: A and B at 2×2
  state.matrices = {
    A: { rows: 2, cols: 2, mode: 'grid', values: [['',''],['','']] },
    B: { rows: 2, cols: 2, mode: 'grid', values: [['',''],['','']] },
  };
  renderMatrixChips();
  renderMatrixInputs();
  // Seed the folder label for non-FSA browsers (shows default path immediately)
  if (!_hasFSA) _updateFolderLabel(_serverFolderPath);
  updateSaveButtonState();
}

document.addEventListener('DOMContentLoaded', boot);

})();
