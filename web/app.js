/**
 * Math Explorer 3D & RRSI Self-Improvement Frontend
 * Uses Three.js / 3d-force-graph for interactive WebGL graph visualization.
 */

let Graph = null;
let rawGraphData = { nodes: [], links: [] };
let activeFilter = 'all';
let isPhysicsActive = true;
let graphShape = '';
let graphRequest = 0;
let graphFitTicks = 0;
let harnessRequest = 0;
let harnessSignature = '';
let evolutionEntries = [];
let selectedEvolution = 'baseline';
let traceRunId = null;
let lastTraceSeq = 0;
let traceRequestBusy = false;

// DOM Elements
const elGraph = document.getElementById('3d-graph');
const elBadgeGen = document.getElementById('badge-gen');
const elBadgeNodes = document.getElementById('badge-nodes');
const elBadgeLinks = document.getElementById('badge-links');
const elBadgeBudget = document.getElementById('badge-budget');

const elInspector = document.getElementById('inspector-panel');
const elInspType = document.getElementById('insp-type');
const elInspGen = document.getElementById('insp-gen');
const elInspTitle = document.getElementById('insp-title');
const elInspTime = document.getElementById('insp-time');
const elInspBody = document.getElementById('insp-body');
const elTraceView = document.getElementById('trace-view');
const elNodeView = document.getElementById('inspector-view');
const elTraceList = document.getElementById('trace-events');
const elTraceTitle = document.getElementById('trace-title');
const elTraceState = document.getElementById('trace-state');
const elTraceClock = document.getElementById('trace-clock');
const elTraceProgress = document.getElementById('trace-progress-fill');

const elQueryInput = document.getElementById('query-input');
const elEngineSelect = document.getElementById('engine-select');
const elForceArxivCheck = document.getElementById('force-arxiv-check');
const elBtnExplore = document.getElementById('btn-explore');
const elExploreProgress = document.getElementById('explore-progress');

const elBtnRRSIStep = document.getElementById('btn-rrsi-step');
const elRRSIGenVal = document.getElementById('rrsi-gen-val');
const elRRSIBudgetVal = document.getElementById('rrsi-budget-val');
const elRRSIBudgetFill = document.getElementById('rrsi-budget-fill');
const elRRSIInvariantsVal = document.getElementById('rrsi-invariants-val');
const elRRSIHistoryList = document.getElementById('rrsi-history-list');
const elConfigTable = document.getElementById('config-table');

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  initGraph();
  setupUIEvents();
  fetchGraph();
  fetchHarness();
  fetchBenchmark();
  fetchTrace();
  setInterval(() => { fetchGraph(); fetchHarness(); }, 10000);
  setInterval(fetchTrace, 1000);
});

function showRailView(view) {
  const trace = view === 'trace';
  elTraceView.hidden = !trace;
  elNodeView.hidden = trace;
  for (const [id, selected] of [['tab-trace', trace], ['tab-node', !trace]]) {
    const tab = document.getElementById(id);
    tab.classList.toggle('active', selected);
    tab.setAttribute('aria-pressed', String(selected));
  }
  elInspector.classList.add('open');
}

async function fetchTrace() {
  if (traceRequestBusy) return;
  traceRequestBusy = true;
  try {
    const res = await fetch('/api/trace', { cache: 'no-store' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    renderTrace(await res.json());
  } catch {
    elTraceState.className = 'trace-state error';
    elTraceState.innerHTML = '<span class="trace-light"></span> DISCONNECTED';
  } finally {
    traceRequestBusy = false;
  }
}

function renderTrace(data) {
  if (data.id !== traceRunId) {
    traceRunId = data.id;
    lastTraceSeq = 0;
    elTraceList.replaceChildren();
  }
  const follow = elTraceList.scrollHeight - elTraceList.scrollTop - elTraceList.clientHeight < 80;
  for (const event of data.events || []) {
    if (event.seq <= lastTraceSeq) continue;
    const row = document.createElement('li');
    const stage = ['task', 'interaction', 'preflight', 'route', 'retrieve', 'tool', 'solve', 'verify', 'graph', 'error'].includes(event.stage) ? event.stage : 'task';
    row.className = 'trace-entry trace-' + stage;
    const dot = document.createElement('span');
    dot.className = 'trace-entry-dot';
    const content = document.createElement('div');
    const meta = document.createElement('span');
    meta.className = 'trace-entry-meta';
    meta.textContent = new Date(event.ts * 1000).toLocaleTimeString() + '  ·  ' + event.stage.toUpperCase();
    const message = document.createElement('strong');
    message.textContent = event.message;
    content.append(meta, message);
    if (event.detail) {
      const detail = document.createElement('small');
      detail.textContent = event.detail;
      content.append(detail);
    }
    renderMath(content);
    row.append(dot, content);
    elTraceList.append(row);
    lastTraceSeq = event.seq;
  }
  while (elTraceList.childElementCount > 100) elTraceList.firstElementChild.remove();
  if (!elTraceList.childElementCount) {
    const empty = document.createElement('li');
    empty.className = 'trace-empty';
    empty.textContent = 'Submit a question to watch the solver work.';
    elTraceList.append(empty);
  }
  if (follow) elTraceList.scrollTop = elTraceList.scrollHeight;
  elTraceTitle.textContent = data.title || 'Waiting for a task';
  renderMath(elTraceTitle);
  elTraceState.className = 'trace-state ' + (data.status || 'idle');
  elTraceState.innerHTML = '<span class="trace-light"></span> ' + (data.status || 'idle').toUpperCase();
  const latest = data.events?.at(-1);
  elTraceClock.textContent = latest ? new Date(latest.ts * 1000).toLocaleTimeString() : '—';
  const progress = { task: 8, interaction: 10, preflight: 18, route: 34, retrieve: 50, tool: 65, solve: 72, verify: 88, graph: 96, error: 100 };
  const reached = Math.max(0, ...(data.events || []).map(event => progress[event.stage] || 0));
  elTraceProgress.style.width = (data.status === 'done' ? 100 : reached) + '%';
}

/**
 * Initialize 3D Force Graph via Three.js
 */
function initGraph() {
  Graph = ForceGraph3D()(elGraph)
    .backgroundColor('#070a11')
    .nodeRelSize(1)
    .nodeVal(n => n.val || 12)
    .nodeColor(n => n.color || '#00e5ff')
    .nodeLabel(n => escapeHtml(`[${(n.type || '').toUpperCase()}] ${n.label || n.title || n.id}`))
    .nodeOpacity(0.92)
    .nodeResolution(24)
    .warmupTicks(200)
    .cooldownTicks(20)
    .linkWidth(link => link.particles ? 2.5 : 1.2)
    .linkColor(link => link.color || '#546e7a')
    .linkCurvature(link => link.curvature || 0.0)
    .linkDirectionalParticles(link => link.particles ? 3 : 0)
    .linkDirectionalParticleWidth(2.5)
    .linkDirectionalParticleSpeed(0.006)
    .linkDirectionalParticleColor(link => link.color || '#00e5ff')
    .onNodeClick(node => {
      // Zoom camera smoothly to node
      const distance = 160;
      const distRatio = 1 + distance / (Math.hypot(node.x, node.y, node.z) || 1);
      Graph.cameraPosition(
        { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
        node,
        1400
      );
      showInspector(node);
    })
    .onBackgroundClick(() => {
      hideInspector();
    })
    .onEngineTick(() => {
      if (graphFitTicks && --graphFitTicks === 0) fitGraph();
    });

  // Initial camera orientation
  Graph.cameraPosition(defaultCamera(), { x: 0, y: 0, z: 0 });
  const resizeGraph = () => Graph.width(elGraph.clientWidth).height(elGraph.clientHeight);
  resizeGraph();
  window.addEventListener('resize', resizeGraph);
}

function defaultCamera() {
  return { x: 0, y: elGraph.clientWidth < 600 ? 0 : 150, z: elGraph.clientWidth < 600 ? 900 : 450 };
}

function fitGraph() {
  const bounds = Graph.getGraphBbox();
  if (!bounds) return;
  const center = Object.fromEntries(['x', 'y', 'z'].map(axis => [axis, (bounds[axis][0] + bounds[axis][1]) / 2]));
  const padding = elGraph.clientWidth > 900 ? 80 : 40;
  const width = Math.max(1, elGraph.clientWidth - 2 * padding);
  const height = Math.max(1, elGraph.clientHeight - 2 * padding);
  const span = Math.max((bounds.x[1] - bounds.x[0]) * elGraph.clientHeight / width,
    (bounds.y[1] - bounds.y[0]) * elGraph.clientHeight / height, 80);
  const distance = span / (2 * Math.tan(Graph.camera().fov * Math.PI / 360));
  Graph.cameraPosition({ x: center.x, y: center.y, z: bounds.z[1] + distance * .75 + 40 }, center, 600);
}

/**
 * Fetch Full Graph Data from FastAPI Backend
 */
async function fetchGraph() {
  const request = ++graphRequest;
  try {
    const res = await fetch('/api/graph');
    if (!res.ok) return;
    const data = await res.json();
    if (request !== graphRequest) return;
    const oldNodes = new Map(rawGraphData.nodes.map(node => [node.id, node]));
    const nodes = data.nodes.map(node => {
      const existing = oldNodes.get(node.id);
      return existing ? Object.assign(existing, node) : node;
    });
    const shape = JSON.stringify([data.nodes.map(node => node.id), data.links.map(link => [link.source, link.target, link.label])]);
    if (shape !== graphShape) {
      const byId = new Map(nodes.map(node => [node.id, node]));
      for (const node of nodes) {
        if (oldNodes.has(node.id)) continue;
        const edge = data.links.find(link => (link.source === node.id && Number.isFinite(byId.get(link.target)?.x)) ||
          (link.target === node.id && Number.isFinite(byId.get(link.source)?.x)));
        const anchor = edge && byId.get(edge.source === node.id ? edge.target : edge.source);
        if (anchor) {
          node.x = anchor.x + (Math.random() - 0.5) * 24;
          node.y = anchor.y + (Math.random() - 0.5) * 24;
          node.z = anchor.z + (Math.random() - 0.5) * 24;
        }
      }
      rawGraphData = { ...data, nodes };
      graphShape = shape;
      applyFilter();
    } else {
      rawGraphData = { ...data, nodes };
    }
    updateTelemetry(data);
  } catch (err) {
    console.error('Failed to load graph data:', err);
  }
}

/**
 * Fetch Current Harness Configuration from Backend
 */
async function fetchHarness() {
  const request = ++harnessRequest;
  try {
    const res = await fetch('/api/harness');
    if (!res.ok) return;
    const cfg = await res.json();
    if (request !== harnessRequest) return;
    const signature = JSON.stringify([cfg.generation, cfg.history?.length, cfg.rejected_attempts?.length]);
    if (signature === harnessSignature) return;
    harnessSignature = signature;
    renderHarnessConfig(cfg);
  } catch (err) {
    console.error('Failed to load harness config:', err);
  }
}

/**
 * Update Header Badges & RRSI Metrics
 */
function updateTelemetry(data) {
  const stats = data.stats || {};
  elBadgeNodes.textContent = `${stats.total_nodes || data.nodes.length} NODES`;
  elBadgeLinks.textContent = `${stats.total_links || data.links.length} EDGES`;

  // Find latest harness node
  const harnessNodes = data.nodes.filter(n => n.type === 'harness_state');
  if (harnessNodes.length > 0) {
    harnessNodes.sort((a, b) => (b.generation || 0) - (a.generation || 0));
    const latest = harnessNodes[0];
    const gen = latest.generation || 0;
    const budget = (latest.data && latest.data.annealed_budget !== undefined) ? latest.data.annealed_budget : 1.0;
    
    elBadgeGen.textContent = `GEN ${gen}`;
    elBadgeBudget.textContent = `BUDGET: ${budget.toFixed(2)}`;
    elRRSIGenVal.textContent = `Gen ${gen}`;
    elRRSIBudgetVal.textContent = budget.toFixed(3);
    elRRSIBudgetFill.style.width = `${Math.min(100, Math.max(10, budget * 100))}%`;
  }
}

/**
 * Apply Category Filters (All, Theorems, Proofs, RRSI)
 */
function applyFilter() {
  if (!rawGraphData || !rawGraphData.nodes) return;

  const mathTypes = ['theorem', 'lemma', 'definition', 'conjecture', 'method'];
  const proofTypes = ['query', 'jev_decision', 'document', 'pageindex_node', 'violetto_proof', 'astra_proof', 'sympy_proof', 'z3_proof', 'verification', 'symbolic_verification'];
  const rrsiTypes = ['harness_state', 'mutation_proposal', 'critic_eval', 'pruner_decision', 'invariant_test'];

  let filteredNodes = rawGraphData.nodes;
  if (activeFilter === 'math') {
    filteredNodes = rawGraphData.nodes.filter(n => mathTypes.includes(n.type));
  } else if (activeFilter === 'knowledge') {
    filteredNodes = rawGraphData.nodes.filter(n => proofTypes.includes(n.type));
  } else if (activeFilter === 'rrsi') {
    filteredNodes = rawGraphData.nodes.filter(n => rrsiTypes.includes(n.type));
  }
  if (activeFilter !== 'all') {
    filteredNodes = filteredNodes.map(({ x, y, z, vx, vy, vz, ...node }) => node);
  }

  const nodeIds = new Set(filteredNodes.map(n => n.id));
  const filteredLinks = rawGraphData.links.filter(l => {
    const src = typeof l.source === 'object' ? l.source.id : l.source;
    const tgt = typeof l.target === 'object' ? l.target.id : l.target;
    return nodeIds.has(src) && nodeIds.has(tgt);
  }).map(l => ({ ...l, source: l.source.id || l.source, target: l.target.id || l.target }));

  Graph.graphData({ nodes: filteredNodes, links: filteredLinks });
  graphFitTicks = 5;
}

/**
 * Display Node Details in Right Inspector
 */
function showInspector(node) {
  showRailView('node');
  elInspType.textContent = (node.type || 'NODE').toUpperCase();
  elInspType.style.background = node.color || '#00e5ff';
  elInspGen.textContent = node.generation !== undefined ? `GEN ${node.generation}` : 'MATH THEOREM';
  elInspTitle.textContent = node.title || node.label || node.id;
  
  const dateStr = node.timestamp ? new Date(node.timestamp * 1000).toLocaleString() : 'Foundational';
  elInspTime.textContent = `Node: ${node.id} | Recorded: ${dateStr}`;

  let html = '';
  const d = node.data || {};

  if (node.type === 'query') {
    html = `
      <div class="inspector-section">
        <div class="ins-label">Full Query Input</div>
        <p><strong>${escapeHtml(d.query || node.title)}</strong></p>
      </div>
      <div class="inspector-section">
        <div class="ins-label">Harness used for this task</div>
        <p>Generation ${escapeHtml(String(node.generation ?? 0))}</p>
        <button type="button" class="inspect-evolution" data-generation="${Number(node.generation) || 0}">Inspect harness evolution →</button>
      </div>
    `;
  } else if (node.type === 'jev_decision') {
    const diff = d.difficulty_score !== undefined ? d.difficulty_score.toFixed(2) : 'N/A';
    const arxivProb = d.needs_arxiv_prob !== undefined ? (d.needs_arxiv_prob * 100).toFixed(1) + '%' : 'N/A';
    html = `
      <div class="inspector-section">
        <div class="ins-label">System One Triage Classification</div>
        <p><strong>Domain:</strong> ${escapeHtml(d.domain || 'Pure Mathematics')}</p>
        <p><strong>Difficulty Score:</strong> ${diff} / 4.0</p>
        <p><strong>arXiv Literature Needed:</strong> ${arxivProb}</p>
        <p><strong>Recommended Solver:</strong> <code>${escapeHtml(d.recommended_engine || 'local_violetto')}</code></p>
      </div>
    `;
  } else if (node.type === 'document' || node.type === 'pageindex_node') {
    html = `
      <div class="inspector-section">
        <div class="ins-label">Literature Node (PageIndex Vectorless Hierarchy)</div>
        <p><strong>Title:</strong> ${escapeHtml(node.title)}</p>
        ${d.doc_name ? `<p><strong>Corpus Document:</strong> ${escapeHtml(d.doc_name)}</p>` : ''}
        ${d.jev_score ? `<p><strong>JEV Relevance:</strong> ${(d.jev_score * 100).toFixed(1)}%</p>` : ''}
      </div>
      <div class="inspector-section">
        <div class="ins-label">Excerpt / Content</div>
        <pre>${escapeHtml(d.text || d.content || JSON.stringify(d, null, 2))}</pre>
      </div>
    `;
  } else if (['violetto_proof', 'astra_proof', 'sympy_proof', 'z3_proof'].includes(node.type)) {
    const isAstra = node.type === 'astra_proof';
    const metrics = d.metrics || {};
    html = `
      <div class="inspector-section">
        <div class="ins-label">Solver Execution Metrics</div>
        <p><strong>Engine:</strong> ${escapeHtml(d.engine || (isAstra ? 'codex_astra' : 'local_violetto'))}</p>
        ${metrics.tokens_used ? `<p><strong>Tokens Used:</strong> ${escapeHtml(String(metrics.tokens_used))}</p>` : ''}
        ${metrics.execution_time_sec ? `<p><strong>Execution Latency:</strong> ${metrics.execution_time_sec}s</p>` : ''}
      </div>
      <div class="inspector-section">
        <div class="ins-label">Mathematical Proof & Reasoning Trace</div>
        <div class="math-proof">${escapeHtml(d.solution || 'No solution trace.')}</div>
      </div>
    `;
  } else if (node.type === 'verification') {
    const verification = d.ensemble || {};
    html = `
      <div class="inspector-section">
        <div class="ins-label">Answer Verification</div>
        <p><strong>Status:</strong> ${escapeHtml(verification.status || 'UNVERIFIED')}</p>
        <p>${escapeHtml(verification.rejection_reason || 'Final answer checked against an exact reference. This does not certify the full proof.')}</p>
      </div>
      <pre>${escapeHtml(JSON.stringify(d, null, 2))}</pre>
    `;
  } else if (node.type === 'harness_state') {
    html = `
      <div class="inspector-section">
        <div class="ins-label">Agent Harness Parameters (Gen ${node.generation})</div>
        <pre>${escapeHtml(JSON.stringify(d, null, 2))}</pre>
      </div>
    `;
  } else if (node.type === 'mutation_proposal') {
    html = `
      <div class="inspector-section">
        <div class="ins-label">RRSI Proposed Mutation</div>
        <p><strong>Target Component:</strong> <code>${escapeHtml(d.component || '')}</code></p>
        <p><strong>Change Delta:</strong> ${escapeHtml(d.change_summary || '')}</p>
        <p><strong>Annealed Budget:</strong> ${(d.budget || 1.0).toFixed(3)}</p>
      </div>
      <div class="inspector-section">
        <div class="ins-label">Hypothesis</div>
        <p><em>"${escapeHtml(d.hypothesis || '')}"</em></p>
      </div>
    `;
  } else if (node.type === 'critic_eval') {
    html = `
      <div class="inspector-section">
        <div class="ins-label">RRSI Critic Evaluation</div>
        <p><strong>Verdict:</strong> <span class="badge ${d.verdict === 'ACCEPT' ? 'badge-links' : 'badge-gen'}">${escapeHtml(d.verdict || 'ACCEPT')}</span></p>
        <p><strong>Generalization Score:</strong> ${(d.generalization_score || 0).toFixed(2)}</p>
        <p><strong>Rationale:</strong> ${escapeHtml(d.reason || '')}</p>
      </div>
    `;
  } else if (node.type === 'pruner_decision') {
    html = `
      <div class="inspector-section">
        <div class="ins-label">RRSI Pruner (Pareto Frontier)</div>
        <p><strong>Action:</strong> <span class="badge badge-budget">${escapeHtml(d.action || 'KEEP')}</span></p>
        <p><strong>Pareto Utility:</strong> ${(d.utility_score || 0).toFixed(2)}</p>
        <p><strong>Rationale:</strong> ${escapeHtml(d.reason || '')}</p>
      </div>
    `;
  } else if (node.type === 'invariant_test') {
    const passed = d.passed || 0;
    const total = d.total || 0;
    html = `
      <div class="inspector-section">
        <div class="ins-label">Mathematical Invariant Regression Tests</div>
        <p><strong>Status:</strong> ${passed} / ${total} Invariants Passed</p>
      </div>
      <div class="inspector-section">
        <div class="ins-label">Test Case Invariants</div>
        <pre>${escapeHtml(JSON.stringify(d.details || d, null, 2))}</pre>
      </div>
    `;
  } else if (node.type === 'symbolic_verification') {
    const status = d.status || 'UNVERIFIED';
    const isSound = status === 'VERIFIED';
    const casV = d.cas_checks ? d.cas_checks.valid : d.valid_steps;
    const casT = d.cas_checks ? d.cas_checks.total : d.total_steps_checked;
    const smtV = d.smt_checks ? d.smt_checks.valid : 0;
    const smtT = d.smt_checks ? d.smt_checks.total : 0;
    const cexs = d.smt_checks ? d.smt_checks.counterexamples : [];

    html = `
      <div class="inspector-section">
        <div class="ins-label">Deterministic Verification Ensemble (SymPy + Z3)</div>
        <p><strong>Status:</strong> <span class="badge ${isSound ? 'badge-links' : 'badge-gen'}">${escapeHtml(status)}</span></p>
        <p>${escapeHtml(d.rejection_reason || 'Final answer checked against an exact reference; full proof not certified.')}</p>
        <p><strong>SymPy CAS Equalities:</strong> ${casV}/${casT} Valid</p>
        ${smtT > 0 ? `<p><strong>Z3 SMT Congruences:</strong> ${smtV}/${smtT} Valid</p>` : ''}
        ${d.extracted_answer ? `<p><strong>Extracted Boxed Answer:</strong> <code>${escapeHtml(d.extracted_answer)}</code></p>` : ''}
        ${d.ground_truth !== undefined && d.ground_truth !== null ? `<p><strong>Ground Truth:</strong> <code>${escapeHtml(d.ground_truth)}</code> (${d.ground_truth_matched === null ? 'UNCONFIRMED' : d.ground_truth_matched ? 'MATCHED' : 'MISMATCH'})</p>` : ''}
      </div>
      ${cexs && cexs.length > 0 ? `
        <div class="inspector-section">
          <div class="ins-label text-danger">Z3 Counterexample Detected</div>
          <pre>${escapeHtml(JSON.stringify(cexs, null, 2))}</pre>
        </div>
      ` : ''}
      <div class="inspector-section">
        <div class="ins-label">Verified Mathematical Step Proofs</div>
        <pre>${escapeHtml(JSON.stringify(d.steps || d.verified_steps || [], null, 2))}</pre>
      </div>
    `;
  } else if (['theorem', 'lemma', 'definition', 'conjecture', 'method'].includes(node.type)) {
    html = `
      <div class="inspector-section">
        <div class="ins-label">Mathematical ${node.type.toUpperCase()}</div>
        <h3 style="color: #fff; margin: 4px 0 10px 0;">${escapeHtml(node.title)}</h3>
        <p style="font-size: 13px; line-height: 1.5;">${escapeHtml(node.description || d.content || '')}</p>
        ${node.centrality !== undefined ? `<p style="margin-top: 8px;"><strong>NetworkX PageRank Centrality:</strong> ${node.centrality}</p>` : ''}
      </div>
    `;
  } else {
    html = `<pre>${escapeHtml(JSON.stringify(d, null, 2))}</pre>`;
  }

  elInspBody.innerHTML = html;

  renderMath(elInspBody);
}

function renderMath(element) {
  if (window.renderMathInElement) {
    try {
      renderMathInElement(element, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\[', right: '\\]', display: true },
          { left: '\\(', right: '\\)', display: false }
        ],
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'option', 'code'],
        throwOnError: false
      });
    } catch (e) {
      console.warn('KaTeX render error:', e);
    }
  }
}

function hideInspector() {
  showRailView('trace');
}

/**
 * Render Active Harness Config Table & History
 */
function renderHarnessConfig(cfg) {
  const lastInvariants = cfg.history?.at(-1)?.invariants;
  if (lastInvariants) {
    elRRSIInvariantsVal.textContent = `${lastInvariants.passed} / ${lastInvariants.total} ${lastInvariants.all_passed ? 'PASSED' : 'FAILED'}`;
    elRRSIInvariantsVal.classList.toggle('text-success', !!lastInvariants.all_passed);
  }
  const fields = [
    ['prompt_system_style', 'Prompt Framing'],
    ['jev_difficulty_threshold', 'Difficulty Routing Cutoff'],
    ['jev_arxiv_threshold', 'arXiv Retrieval Probability'],
    ['top_k_retrieval', 'Tree Retrieval Top-K'],
    ['violetto_temperature', 'Violetto Temperature'],
    ['violetto_top_k', 'Violetto Top-K Sampling'],
    ['search_strategy', 'Tree Search Strategy'],
    ['annealed_budget', 'Annealed Edit Budget B(t)']
  ];

  let html = '';
  for (const [key, label] of fields) {
    const val = cfg[key] !== undefined ? cfg[key] : 'default';
    html += `
      <div class="config-row">
        <span class="config-key">${label}</span>
        <span class="config-val">${escapeHtml(String(val))}</span>
      </div>
    `;
  }
  elConfigTable.innerHTML = html;

  const wasLatest = selectedEvolution === evolutionEntries.at(-1)?.key || evolutionEntries.length === 0;
  evolutionEntries = [
    { key: 'baseline', generation: 0, status: 'baseline', timestamp: 0 },
    ...(cfg.history || []).map(h => ({ ...h, key: `g${h.generation}`, status: 'accepted' })),
    ...(cfg.rejected_attempts || []).map((h, i) => ({ ...h, key: `r${i}` }))
  ].sort((a, b) => a.timestamp - b.timestamp);
  if (wasLatest || !evolutionEntries.some(h => h.key === selectedEvolution)) {
    selectedEvolution = evolutionEntries.at(-1).key;
  }
  renderEvolution();
}

function renderEvolution() {
  elRRSIHistoryList.innerHTML = evolutionEntries.map(entry => `
    <button type="button" class="evolution-point ${entry.status === 'accepted' ? 'accepted' : entry.status === 'baseline' ? 'baseline' : 'rejected'} ${entry.key === selectedEvolution ? 'selected' : ''}"
      data-entry="${entry.key}" aria-pressed="${entry.key === selectedEvolution}">
      <span class="evolution-dot" aria-hidden="true"></span>
      <span class="evolution-point-label">${entry.status === 'baseline' ? 'H₀' : entry.status === 'accepted' ? `G${entry.generation}` : '×'}</span>
      <small>${entry.status === 'baseline' ? 'Base' : entry.status === 'accepted' ? 'Kept' : 'Blocked'}</small>
    </button>
  `).join('');

  const scored = evolutionEntries.filter(entry => entry.status === 'accepted' && Number.isFinite(entry.critic?.generalization_score));
  document.getElementById('rrsi-score-chart').innerHTML = scored.length ? `
    <div class="chart-label">Recorded critic score · accepted generations</div>
    <svg viewBox="0 0 320 76" role="img" aria-label="Recorded critic score across accepted generations">
      <path d="M 12 64 H 308" class="chart-axis"/>
      <polyline class="chart-line" points="${scored.map((entry, i) => `${12 + i * 296 / Math.max(1, scored.length - 1)},${64 - Math.max(0, Math.min(1, entry.critic.generalization_score)) * 50}`).join(' ')}"/>
      ${scored.map((entry, i) => `<circle cx="${12 + i * 296 / Math.max(1, scored.length - 1)}" cy="${64 - Math.max(0, Math.min(1, entry.critic.generalization_score)) * 50}" r="${entry.key === selectedEvolution ? 5 : 3}" class="${entry.key === selectedEvolution ? 'chart-active' : 'chart-dot'}"><title>G${entry.generation}: ${entry.critic.generalization_score.toFixed(2)}</title></circle>`).join('')}
    </svg>
  ` : '';

  const index = evolutionEntries.findIndex(entry => entry.key === selectedEvolution);
  document.getElementById('rrsi-prev').disabled = index <= 0;
  document.getElementById('rrsi-next').disabled = index >= evolutionEntries.length - 1;
  const entry = evolutionEntries[index];
  const checked = entry.invariants;
  elRRSIInvariantsVal.textContent = checked?.total ? `${checked.passed} / ${checked.total} PASSED` : 'Not run';
  elRRSIInvariantsVal.classList.toggle('text-success', checked?.total > 0 && checked.passed === checked.total);
  const detail = document.getElementById('rrsi-generation-detail');
  if (entry.status === 'baseline') {
    detail.innerHTML = '<div class="evolution-status">H₀ · Baseline</div><h3>Starting harness</h3><p>Initial configuration before the first RRSI proposal. Select a later point to see exactly what changed and why it was kept or blocked.</p>';
  } else {
    const p = entry.proposal || {};
    const c = entry.critic || {};
    const pruner = entry.pruner || {};
    const invariants = entry.invariants || {};
    const accepted = entry.status === 'accepted';
    const change = p.field ? `<div class="evolution-diff"><span>${escapeHtml(p.field)}</span><del>${escapeHtml(String(p.old_val ?? '—'))}</del><span aria-hidden="true">→</span><ins>${escapeHtml(String(p.new_val ?? '—'))}</ins></div>` : '';
    const measure = Number.isFinite(c.delta_accuracy) ? `<p>Accuracy Δ: ${(c.delta_accuracy * 100).toFixed(1)} pts${Number.isFinite(c.delta_utility) ? ` · Utility Δ: ${c.delta_utility.toFixed(3)}` : ''}</p>` : '';
    detail.innerHTML = `
      <div class="evolution-status ${accepted ? 'kept' : 'blocked'}">${accepted ? `G${entry.generation} · Kept` : `Candidate for G${entry.generation} · Blocked`}</div>
      <h3>${escapeHtml(p.component || 'Harness proposal')}</h3>
      ${change}
      <div class="evolution-stage"><span>01 · Proposal</span><p>${escapeHtml(p.hypothesis || p.change_summary || 'No hypothesis recorded.')}</p>${Number.isFinite(p.budget) ? `<small>Edit budget ${p.budget.toFixed(3)}</small>` : ''}</div>
      <div class="evolution-stage"><span>02 · Invariants</span><p>${invariants.total !== undefined ? `${invariants.passed}/${invariants.total} passed` : 'Not reached'}</p></div>
      <div class="evolution-stage"><span>03 · Critic</span><p>${escapeHtml(c.reason || 'Not reached')}</p>${measure}</div>
      <div class="evolution-stage"><span>04 · Selection</span><p>${escapeHtml(pruner.reason || (entry.status === 'pruned_zero_delta' ? 'No parameter change to keep.' : accepted ? 'Change accepted.' : 'Candidate rejected before selection.'))}</p></div>
    `;
  }
  elRRSIHistoryList.querySelector('.selected')?.scrollIntoView({ block: 'nearest', inline: 'center' });
}

function selectEvolution(key, focusGraph = false) {
  if (!evolutionEntries.some(entry => entry.key === key)) return;
  selectedEvolution = key;
  renderEvolution();
  document.querySelector('.evolution-heading').scrollIntoView({ block: 'start', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  if (!focusGraph) return;
  const entry = evolutionEntries.find(item => item.key === key);
  if (entry.status !== 'accepted' && entry.status !== 'baseline') return;
  const generation = entry.generation;
  const node = rawGraphData.nodes.find(n => n.id === `harness_gen_${generation}`);
  if (!node) return;
  if (activeFilter !== 'rrsi') {
    activeFilter = 'rrsi';
    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.filter === 'rrsi'));
    applyFilter();
  }
  if (Number.isFinite(node.x) && Number.isFinite(node.y) && Number.isFinite(node.z)) {
    Graph.cameraPosition({ x: node.x, y: node.y, z: node.z + 170 }, node, 1100);
  }
}

/**
 * Event Listeners & Controls
 */
function setupUIEvents() {
  elRRSIHistoryList.addEventListener('click', event => {
    const point = event.target.closest('[data-entry]');
    if (point) {
      selectEvolution(point.dataset.entry, true);
      elRRSIHistoryList.querySelector('.selected')?.focus();
    }
  });
  elRRSIHistoryList.addEventListener('keydown', event => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    document.getElementById(event.key === 'ArrowLeft' ? 'rrsi-prev' : 'rrsi-next').click();
    elRRSIHistoryList.querySelector('.selected')?.focus();
  });
  document.getElementById('rrsi-prev').addEventListener('click', () => {
    const index = evolutionEntries.findIndex(entry => entry.key === selectedEvolution);
    if (index > 0) selectEvolution(evolutionEntries[index - 1].key, true);
  });
  document.getElementById('rrsi-next').addEventListener('click', () => {
    const index = evolutionEntries.findIndex(entry => entry.key === selectedEvolution);
    if (index < evolutionEntries.length - 1) selectEvolution(evolutionEntries[index + 1].key, true);
  });
  elInspector.addEventListener('click', event => {
    const link = event.target.closest('.inspect-evolution');
    if (!link) return;
    document.querySelector('[data-tab="tab-rrsi"]').click();
    selectEvolution(Number(link.dataset.generation) ? `g${link.dataset.generation}` : 'baseline', true);
  });
  // Tabs Switcher
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add('active');
      if (btn.dataset.tab === 'tab-rrsi') {
        elRRSIHistoryList.querySelector('.selected')?.scrollIntoView({ block: 'nearest', inline: 'center' });
      }
    });
  });

  // Filter Buttons
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (activeFilter === btn.dataset.filter) return;
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeFilter = btn.dataset.filter;
      applyFilter();
    });
  });

  // Preset Chips
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      elQueryInput.value = chip.dataset.q;
    });
  });

  // Right panel navigation
  document.getElementById('btn-close-inspector').addEventListener('click', hideInspector);
  document.getElementById('tab-trace').addEventListener('click', () => showRailView('trace'));
  document.getElementById('tab-node').addEventListener('click', () => showRailView('node'));
  document.getElementById('btn-open-trace').addEventListener('click', () => showRailView('trace'));
  document.getElementById('btn-close-panel').addEventListener('click', () => elInspector.classList.remove('open'));

  // Dispatch Exploration
  elBtnExplore.addEventListener('click', async () => {
    const query = elQueryInput.value.trim();
    if (!query) {
      alert('Please enter a mathematical conjecture or query.');
      return;
    }

    const engineVal = elEngineSelect.value;
    const engine = engineVal === 'auto' ? null : engineVal;
    const forceArxiv = elForceArxivCheck.checked;

    // Show spinner & progress steps
    elBtnExplore.disabled = true;
    elBtnExplore.querySelector('.spinner').classList.remove('hidden');
    elBtnExplore.querySelector('.btn-text').textContent = 'Exploring...';
    elExploreProgress.classList.remove('hidden');
    showRailView('trace');
    elTraceTitle.textContent = query;

    elExploreProgress.setAttribute('aria-busy', 'true');
    document.getElementById('explore-status').textContent = 'Exploration in progress · follow the live trace on the right.';

    try {
      const res = await fetch('/api/explore', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, engine, force_arxiv: forceArxiv })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Exploration failed');
      }

      const result = await res.json();
      await fetchTrace();
      document.getElementById('explore-status').textContent = 'Exploration complete.';

      // Refresh graph and locate newly created query node
      await fetchGraph();

      if (result.query_node_id) {
        const proofLink = rawGraphData.links.find(l => l.label === 'dispatched_to' && (l.source.id || l.source) === result.query_node_id.replace('query_', 'jev_'));
        const proofId = proofLink && (proofLink.target.id || proofLink.target);
        const targetNode = rawGraphData.nodes.find(n => n.id === (proofId || result.query_node_id));
        if (targetNode) {
          Graph.cameraPosition(
            { x: targetNode.x * 1.5, y: targetNode.y * 1.5, z: targetNode.z + 180 },
            targetNode,
            1500
          );
        }
      }
    } catch (err) {
      await fetchTrace();
      document.getElementById('explore-status').textContent = `Exploration failed: ${err.message}`;
      alert(`Exploration Error: ${err.message}`);
    } finally {
      elExploreProgress.setAttribute('aria-busy', 'false');
      elBtnExplore.disabled = false;
      elBtnExplore.querySelector('.spinner').classList.add('hidden');
      elBtnExplore.querySelector('.btn-text').textContent = 'Dispatch Exploration';
    }
  });

  // RRSI Step Execution
  elBtnRRSIStep.addEventListener('click', async () => {
    elBtnRRSIStep.disabled = true;
    elBtnRRSIStep.querySelector('.spinner').classList.remove('hidden');
    elBtnRRSIStep.querySelector('.btn-text').textContent = 'Mutating Harness...';

    try {
      const res = await fetch('/api/rrsi/step', { method: 'POST' });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'RRSI step failed');
      }

      const data = await res.json();
      await fetchGraph();
      await fetchHarness();

      if (data.harness_node_id) {
        const targetNode = rawGraphData.nodes.find(n => n.id === data.harness_node_id);
        if (targetNode) {
          showInspector(targetNode);
          Graph.cameraPosition(
            { x: targetNode.x * 1.5, y: targetNode.y * 1.5, z: targetNode.z + 180 },
            targetNode,
            1500
          );
        }
      }
      if (!data.success) {
        document.querySelector('[data-tab="tab-rrsi"]').click();
      }
    } catch (err) {
      alert(`RRSI Step Error: ${err.message}`);
    } finally {
      elBtnRRSIStep.disabled = false;
      elBtnRRSIStep.querySelector('.spinner').classList.add('hidden');
      elBtnRRSIStep.querySelector('.btn-text').textContent = 'Execute RRSI Evolution Step';
    }
  });

  // 3D Controls
  document.getElementById('btn-reset-cam').addEventListener('click', () => {
    fitGraph();
  });

  document.getElementById('btn-toggle-physics').addEventListener('click', () => {
    if (isPhysicsActive) {
      Graph.pauseAnimation();
      isPhysicsActive = false;
      document.getElementById('btn-toggle-physics').textContent = 'Resume';
    } else {
      Graph.resumeAnimation();
      isPhysicsActive = true;
      document.getElementById('btn-toggle-physics').textContent = 'Freeze';
    }
  });

  document.getElementById('btn-reset-graph').addEventListener('click', async () => {
    if (!confirm('Reset graph to baseline seed?')) return;
    try {
      const res = await fetch('/api/graph/reset', { method: 'POST' });
      if (!res.ok) throw new Error((await res.json()).detail || 'Graph reset failed');
      await fetchGraph();
      await fetchHarness();
      hideInspector();
    } catch (err) {
      alert(`Graph reset error: ${err.message}`);
    }
  });
  // Run Benchmark Suite Button
  const btnRunBm = document.getElementById('btn-run-benchmark');
  if (btnRunBm) {
    btnRunBm.addEventListener('click', async () => {
      btnRunBm.disabled = true;
      btnRunBm.querySelector('.spinner').classList.remove('hidden');
      btnRunBm.querySelector('.btn-text').textContent = 'Evaluating Portfolio...';
      try {
        const res = await fetch('/api/benchmark/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        });
        if (!res.ok) throw new Error('Benchmark execution failed');
        await fetchBenchmark();
        await fetchGraph();
      } catch (err) {
        alert('Benchmark error: ' + err.message);
      } finally {
        btnRunBm.disabled = false;
        btnRunBm.querySelector('.spinner').classList.add('hidden');
        btnRunBm.querySelector('.btn-text').textContent = 'Run Benchmark Suite';
      }
    });
  }
}

async function fetchBenchmark() {
  const elAcc = document.getElementById('bm-accuracy');
  const elStep = document.getElementById('bm-step-pass');
  const elLedger = document.getElementById('benchmark-ledger');
  try {
    const res = await fetch('/api/benchmark');
    if (!res.ok) throw new Error('Benchmark results unavailable');
    const data = await res.json();
    if (data.results && data.results.length > 0) {
      if (elAcc) elAcc.textContent = `${(data.overall_accuracy * 100).toFixed(1)}% (${data.solved_correctly}/${data.total_problems})`;
      if (elStep) elStep.textContent = `${(data.symbolic_avg_pass_rate * 100).toFixed(1)}%`;
      
      let html = '';
      for (const r of data.results) {
        html += `
          <div class="history-item">
            <span class="gen-tag">${escapeHtml(r.id)}</span>
            <strong>${r.correct === true ? 'PASS' : r.correct === false ? 'FAIL' : 'UNCONFIRMED'}</strong>
            <div style="color: #94a3b8; margin-top: 2px;">
              Engine: <code>${escapeHtml(r.engine_used)}</code> &bull; Ans: ${escapeHtml(r.extracted_answer || 'N/A')} &bull; Latency: ${escapeHtml(r.latency_sec)}s
            </div>
          </div>
        `;
      }
      if (elLedger) elLedger.innerHTML = html;
    } else {
      if (elAcc) elAcc.textContent = '—';
      if (elStep) elStep.textContent = '—';
      if (elLedger) elLedger.textContent = 'No benchmark run yet.';
    }
  } catch (err) {
    if (elAcc) elAcc.textContent = '—';
    if (elStep) elStep.textContent = '—';
    if (elLedger) elLedger.textContent = 'Benchmark results unavailable.';
    console.error('Failed to load benchmark:', err);
  }
}

function escapeHtml(text) {
  if (typeof text !== 'string') text = String(text);
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
  return text.replace(/[&<>"']/g, m => map[m]);
}
