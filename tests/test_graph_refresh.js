const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const elements = new Map();
const graph = { calls: 0, graphData(data) { this.calls++; this.data = data; } };
let payload = { nodes: [{ id: 'a', type: 'query' }], links: [] };
const context = vm.createContext({
  document: {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, { style: {}, classList: { add() {}, remove() {}, toggle() {} }, setAttribute() {} });
      return elements.get(id);
    },
    addEventListener() {}
  },
  fetch: async () => ({ ok: true, json: async () => structuredClone(payload) }),
  graph,
  window: {},
  console,
  Math
});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8'), context);
vm.runInContext('Graph = graph', context);

(async () => {
  await vm.runInContext('fetchGraph()', context);
  assert.equal(graph.calls, 1);
  vm.runInContext('rawGraphData.nodes[0].x = 10; rawGraphData.nodes[0].y = 20; rawGraphData.nodes[0].z = 30', context);

  await vm.runInContext('fetchGraph()', context);
  assert.equal(graph.calls, 1, 'identical polls must not restart the layout');
  assert.equal(vm.runInContext('rawGraphData.nodes[0].x', context), 10);

  payload = { nodes: [{ id: 'a', type: 'query' }, { id: 'b', type: 'verification' }], links: [{ source: 'a', target: 'b', label: 'checked_by' }] };
  await vm.runInContext('fetchGraph()', context);
  assert.equal(graph.calls, 2);
  assert.equal(graph.data.nodes[0].x, 10, 'existing positions must survive additions');
  assert.ok(Math.abs(graph.data.nodes[1].x - 10) <= 12, 'new nodes start near a connected node');
  payload.nodes.push({ id: 'c', type: 'theorem' });
  await vm.runInContext('fetchGraph()', context);
  vm.runInContext('activeFilter = "math"; applyFilter()', context);
  assert.equal(graph.data.nodes.length, 1);
  assert.equal(graph.data.nodes[0].id, 'c');
  assert.notEqual(graph.data.nodes[0], vm.runInContext('rawGraphData.nodes.find(n => n.id === "c")', context), 'filtered layout must not move full graph nodes');
  vm.runInContext('showInspector({ id: "q", type: "query", title: "Proof task", generation: 3, data: { query: "Proof task" } })', context);
  assert.match(elements.get('insp-body').innerHTML, /data-generation="3"/);
  console.log('graph refresh: ok');
})().catch(error => { console.error(error); process.exitCode = 1; });
