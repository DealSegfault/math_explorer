/**
 * Math Explorer 3D & RRSI Self-Improvement Frontend
 * Uses Three.js / 3d-force-graph for interactive WebGL graph visualization.
 */

let Graph = null;
let rawGraphData = { nodes: [], links: [] };
let activeFilter = 'all';
let isPhysicsActive = true;

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
  setInterval(fetchGraph, 10000); // Polling background updates
});

/**
 * Initialize 3D Force Graph via Three.js
 */
function initGraph() {
  Graph = ForceGraph3D()(elGraph)
    .backgroundColor('#070a11')
    .nodeRelSize(1)
    .nodeVal(n => n.val || 12)
    .nodeColor(n => n.color || '#00e5ff')
    .nodeLabel(n => `[${(n.type || '').toUpperCase()}] ${n.label || n.title || n.id}`)
    .nodeOpacity(0.92)
    .nodeResolution(24)
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
      const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
      Graph.cameraPosition(
        { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
        node,
        1400
      );
      showInspector(node);
    })
    .onBackgroundClick(() => {
      hideInspector();
    });

  // Enable subtle Bloom / Three.js scene tuning
  const scene = Graph.scene();
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0x00e5ff, 0.8);
  dirLight.position.set(100, 200, 100);
  scene.add(dirLight);

  // Initial camera orientation
  Graph.cameraPosition({ x: 0, y: 150, z: 450 });
}

/**
 * Fetch Full Graph Data from FastAPI Backend
 */
async function fetchGraph() {
  try {
    const res = await fetch('/api/graph');
    if (!res.ok) return;
    const data = await res.json();
    rawGraphData = data;
    applyFilter();
    updateTelemetry(data);
  } catch (err) {
    console.error('Failed to load graph data:', err);
  }
}

/**
 * Fetch Current Harness Configuration from Backend
 */
async function fetchHarness() {
  try {
    const res = await fetch('/api/harness');
    if (!res.ok) return;
    const cfg = await res.json();
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
  const proofTypes = ['query', 'jev_decision', 'document', 'pageindex_node', 'violetto_proof', 'astra_proof', 'verification', 'symbolic_verification', 'counterexample'];
  const rrsiTypes = ['harness_state', 'mutation_proposal', 'critic_eval', 'pruner_decision', 'invariant_test'];

  let filteredNodes = rawGraphData.nodes;
  if (activeFilter === 'math') {
    filteredNodes = rawGraphData.nodes.filter(n => mathTypes.includes(n.type));
  } else if (activeFilter === 'knowledge') {
    filteredNodes = rawGraphData.nodes.filter(n => proofTypes.includes(n.type));
  } else if (activeFilter === 'rrsi') {
    filteredNodes = rawGraphData.nodes.filter(n => rrsiTypes.includes(n.type));
  }

  const nodeIds = new Set(filteredNodes.map(n => n.id));
  const filteredLinks = rawGraphData.links.filter(l => {
    const src = typeof l.source === 'object' ? l.source.id : l.source;
    const tgt = typeof l.target === 'object' ? l.target.id : l.target;
    return nodeIds.has(src) && nodeIds.has(tgt);
  });

  Graph.graphData({ nodes: filteredNodes, links: filteredLinks });
}

/**
 * Display Node Details in Right Inspector
 */
function showInspector(node) {
  elInspector.classList.add('open');
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
    const paperUrl = /^https:\/\/arxiv\.org\/abs\/\d{4}\.\d{4,5}$/.test(d.abs_url || '') ? d.abs_url : null;
    html = `
      <div class="inspector-section">
        <div class="ins-label">Literature Node (PageIndex Vectorless Hierarchy)</div>
        <p><strong>Title:</strong> ${escapeHtml(node.title)}</p>
        ${d.doc_name ? `<p><strong>Corpus Document:</strong> ${escapeHtml(d.doc_name)}</p>` : ''}
        ${d.jev_score ? `<p><strong>JEV Relevance:</strong> ${(d.jev_score * 100).toFixed(1)}%</p>` : ''}
        ${paperUrl ? `<p><a href="${paperUrl}" target="_blank" rel="noopener noreferrer">Open arXiv paper</a></p>` : ''}
      </div>
      <div class="inspector-section">
        <div class="ins-label">Excerpt / Content</div>
        <pre>${escapeHtml(d.text || d.content || JSON.stringify(d, null, 2))}</pre>
      </div>
    `;
  } else if (node.type === 'violetto_proof' || node.type === 'astra_proof') {
    const isAstra = node.type === 'astra_proof';
    const metrics = d.metrics || {};
    html = `
      <div class="inspector-section">
        <div class="ins-label">Solver Execution Metrics</div>
        <p><strong>Engine:</strong> ${isAstra ? 'Codex CLI (gpt-6-astra xhigh)' : 'Limite 1B Violetto (Apple Silicon MPS)'}</p>
        ${metrics.tokens_used ? `<p><strong>Tokens Used:</strong> ${metrics.tokens_used}</p>` : ''}
        ${metrics.generation_time_sec ? `<p><strong>Execution Latency:</strong> ${metrics.generation_time_sec}s</p>` : ''}
      </div>
      <div class="inspector-section">
        <div class="ins-label">Mathematical Proof & Reasoning Trace</div>
        <pre class="math-proof">${escapeHtml(d.solution || 'No solution trace.')}</pre>
      </div>
    `;
  } else if (node.type === 'verification') {
    const plaus = d.is_plausible ? (d.is_plausible.noul * 100).toFixed(1) + '%' : 'N/A';
    const rigor = d.rigor_score ? d.rigor_score.score.toFixed(1) + ' / 2.0' : 'N/A';
    html = `
      <div class="inspector-section">
        <div class="ins-label">JEV Confidence Gate Verification</div>
        <p><strong>Mathematical Plausibility:</strong> ${plaus}</p>
        <p><strong>Formal Rigor Score:</strong> ${rigor}</p>
      </div>
      <div class="inspector-section">
        <div class="ins-label">Raw Verification Payload</div>
        <pre>${escapeHtml(JSON.stringify(d, null, 2))}</pre>
      </div>
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
    const status = d.status || (d.is_formally_sound ? 'STEP_CHECKED' : 'UNVERIFIED');
    const casV = d.cas_checks ? d.cas_checks.valid : d.valid_steps;
    const casT = d.cas_checks ? d.cas_checks.total : d.total_steps_checked;
    const smtV = d.smt_checks ? d.smt_checks.valid : 0;
    const smtT = d.smt_checks ? d.smt_checks.total : 0;
    const cexs = d.smt_checks ? d.smt_checks.counterexamples : [];

    html = `
      <div class="inspector-section">
        <div class="ins-label">Deterministic Verification Ensemble (SymPy + Z3)</div>
        <p><strong>Status:</strong> <span class="badge ${status === 'REFUTED' ? 'badge-gen' : 'badge-links'}">${escapeHtml(status)}</span></p>
        <p><strong>SymPy CAS Equalities:</strong> ${casV}/${casT} Valid</p>
        ${smtT > 0 ? `<p><strong>Z3 SMT Congruences:</strong> ${smtV}/${smtT} Valid</p>` : ''}
        ${d.extracted_answer ? `<p><strong>Extracted Boxed Answer:</strong> <code>${escapeHtml(d.extracted_answer)}</code></p>` : ''}
        ${d.ground_truth !== undefined && d.ground_truth !== null ? `<p><strong>Ground Truth:</strong> <code>${escapeHtml(d.ground_truth)}</code> (${d.ground_truth_matched ? 'MATCHED' : 'MISMATCH'})</p>` : ''}
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
    const candidateUrl = /^https:\/\/arxiv\.org\/abs\/\d{4}\.\d{4,5}$/.test(d.source || '') ? d.source : null;
    html = `
      <div class="inspector-section">
        <div class="ins-label">Mathematical ${node.type.toUpperCase()}</div>
        <h3 style="color: #fff; margin: 4px 0 10px 0;">${escapeHtml(node.title)}</h3>
        <p style="font-size: 13px; line-height: 1.5;">${escapeHtml(d.statement || node.description || d.content || '')}</p>
        ${d.status ? `<p><strong>Status:</strong> ${escapeHtml(d.status)}</p>` : ''}
        ${d.check ? `<p><strong>Machine check:</strong> ${escapeHtml(d.check.status)}</p>` : ''}
        ${candidateUrl ? `<p><a href="${candidateUrl}" target="_blank" rel="noopener noreferrer">Source arXiv</a></p>` : ''}
        ${node.centrality !== undefined ? `<p style="margin-top: 8px;"><strong>NetworkX PageRank Centrality:</strong> ${node.centrality}</p>` : ''}
      </div>
    `;
  } else {
    html = `<pre>${escapeHtml(JSON.stringify(d, null, 2))}</pre>`;
  }

  elInspBody.innerHTML = html;

  // Render KaTeX math equations if KaTeX is present
  if (window.renderMathInElement) {
    try {
      renderMathInElement(elInspBody, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\[', right: '\\]', display: true },
          { left: '\\(', right: '\\)', display: false }
        ],
        throwOnError: false
      });
    } catch (e) {
      console.warn('KaTeX render error:', e);
    }
  }
}

function hideInspector() {
  elInspector.classList.remove('open');
}

/**
 * Render Active Harness Config Table & History
 */
function renderHarnessConfig(cfg) {
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
        <span class="config-val">${val}</span>
      </div>
    `;
  }
  elConfigTable.innerHTML = html;

  // Render History
  const history = cfg.history || [];
  if (history.length === 0) {
    elRRSIHistoryList.innerHTML = '<p class="hint-text">No self-improvement iterations yet.</p>';
  } else {
    let histHtml = '';
    for (let i = history.length - 1; i >= 0; i--) {
      const h = history[i];
      const p = h.proposal || {};
      const c = h.critic || {};
      histHtml += `
        <div class="history-item">
          <span class="gen-tag">G${h.generation}</span>
          <strong>${escapeHtml(p.change_summary || p.component || '')}</strong>
          <div style="color: #94a3b8; margin-top: 2px;">
            Critic: ${c.verdict || 'ACCEPT'} (${(c.generalization_score || 0).toFixed(2)}) &bull; Invariants: ${h.invariants ? h.invariants.passed : 4}/4
          </div>
        </div>
      `;
    }
    elRRSIHistoryList.innerHTML = histHtml;
  }
}

/**
 * Event Listeners & Controls
 */
function setupUIEvents() {
  // Tabs Switcher
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add('active');
    });
  });

  // Filter Buttons
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
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

  // Close Inspector Button
  document.getElementById('btn-close-inspector').addEventListener('click', hideInspector);

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

    resetProgressSteps();
    setStepActive('step-jev');

    try {
      setTimeout(() => setStepActive('step-tree'), 300);
      setTimeout(() => setStepActive('step-solve'), 1200);

      const res = await fetch('/api/explore', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, engine, force_arxiv: forceArxiv })
      });

      setStepActive('step-verify');

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Exploration failed');
      }

      const result = await res.json();
      setAllStepsDone();

      // Refresh graph and locate newly created query node
      await fetchGraph();

      if (result.query_node_id) {
        const targetNode = rawGraphData.nodes.find(n => n.id === result.query_node_id);
        if (targetNode) {
          showInspector(targetNode);
          Graph.cameraPosition(
            { x: targetNode.x * 1.5, y: targetNode.y * 1.5, z: targetNode.z + 180 },
            targetNode,
            1500
          );
        }
      }
    } catch (err) {
      alert(`Exploration Error: ${err.message}`);
    } finally {
      elBtnExplore.disabled = false;
      elBtnExplore.querySelector('.spinner').classList.add('hidden');
      elBtnExplore.querySelector('.btn-text').textContent = 'Dispatch Exploration';
    }
  });

  document.getElementById('btn-scan-arxiv').addEventListener('click', async (event) => {
    const button = event.currentTarget;
    const status = document.getElementById('scan-arxiv-status');
    button.disabled = true;
    status.textContent = 'Scanning recent papers…';
    try {
      const response = await fetch('/api/conjecture/scan', { method: 'POST' });
      if (!response.ok) throw new Error((await response.json()).detail || 'arXiv scan failed');
      const result = await response.json();
      await fetchGraph();
      status.textContent = `${result.new_papers} new papers, ${result.new_candidates} candidate statements`;
    } catch (error) {
      status.textContent = `Scan failed: ${error.message}`;
    } finally {
      button.disabled = false;
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
    Graph.cameraPosition({ x: 0, y: 150, z: 450 }, { x: 0, y: 0, z: 0 }, 1200);
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
      await fetch('/api/graph/reset', { method: 'POST' });
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
  try {
    const res = await fetch('/api/benchmark');
    if (!res.ok) return;
    const data = await res.json();
    if (data.results && data.results.length > 0) {
      const elAcc = document.getElementById('bm-accuracy');
      const elStep = document.getElementById('bm-step-pass');
      const elLedger = document.getElementById('benchmark-ledger');
      if (elAcc) elAcc.textContent = `${(data.overall_accuracy * 100).toFixed(1)}% (${data.solved_correctly}/${data.total_problems})`;
      if (elStep) elStep.textContent = `${(data.symbolic_avg_pass_rate * 100).toFixed(1)}%`;
      
      let html = '';
      for (const r of data.results) {
        html += `
          <div class="history-item">
            <span class="gen-tag">${escapeHtml(r.id)}</span>
            <strong>${r.correct ? '<span style="color:#00e676">PASS</span>' : '<span style="color:#ff1744">FAIL</span>'}</strong>
            <div style="color: #94a3b8; margin-top: 2px;">
              Engine: <code>${escapeHtml(r.engine_used)}</code> &bull; Ans: ${escapeHtml(r.extracted_answer || 'N/A')} &bull; Latency: ${r.latency_sec}s
            </div>
          </div>
        `;
      }
      if (elLedger) elLedger.innerHTML = html;
    }
  } catch (err) {
    console.error('Failed to load benchmark:', err);
  }
}

function resetProgressSteps() {
  document.querySelectorAll('.progress-step').forEach(step => {
    step.classList.remove('active', 'done');
  });
}

function setStepActive(stepId) {
  const step = document.getElementById(stepId);
  if (step) {
    document.querySelectorAll('.progress-step').forEach(s => s.classList.remove('active'));
    step.classList.add('active');
  }
}

function setAllStepsDone() {
  document.querySelectorAll('.progress-step').forEach(step => {
    step.classList.remove('active');
    step.classList.add('done');
  });
}

function escapeHtml(text) {
  if (typeof text !== 'string') text = String(text);
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
  return text.replace(/[&<>"']/g, m => map[m]);
}
