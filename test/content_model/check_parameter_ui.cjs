// DOM/canvas contract checks; this does not substitute for browser layout QA.
const fs = require('fs'), vm = require('vm'), assert = require('assert'), path = require('path');
const out = process.argv[2];
const html = fs.readFileSync(path.join(__dirname, 'parameter_ui.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const fixture = JSON.parse(fs.readFileSync(path.join(out, 'ui-fixtures.json'), 'utf8'));
const checks = [];
const flush = async () => { for (let i = 0; i < 15; i++) await Promise.resolve(); };
(async () => {
  for (const width of [900, 360]) {
    let strokes = 0, generation = 0, settings = fixture.states.bendable.settings;
    const requests = [], callbacks = new Map();
    function element(dataset = {}) { return { dataset, style: {}, listeners: {}, value: '0', checked: true,
      classList: { toggle() {} }, setAttribute() {}, addEventListener(k, fn) { this.listeners[k] = fn; },
      getBoundingClientRect: () => ({ width, height: 350 }), setPointerCapture() {} }; }
    const ids = Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map(m => [m[1], element()]));
    const modes = ['squeeze_y', 'squeeze_z', 'sag'].map(mode => element({ mode }));
    const presets = ['bendable', 'compressible'].map(preset => element({ preset }));
    const finite = (...values) => { assert(values.every(Number.isFinite), 'finite draw coordinates'); };
    const context2d = { clearRect: finite, setTransform: finite, beginPath() {}, closePath() {},
      moveTo: finite, lineTo: finite, fillRect: finite, fill() {}, stroke() { strokes++; },
      setLineDash() {}, fillText(text, x, y) { finite(x, y); } };
    ids.scene.getContext = ids.chart.getContext = () => context2d;
    const fetch = async (url, options) => {
      if (options) {
        const body = JSON.parse(options.body); requests.push({ url, body });
        if (url === '/api/config') { settings = body; generation++; }
        return { ok: true, json: async () => ({ generation }) };
      }
      const value = url === '/api/meta' ? fixture.meta : {
        ...fixture.states[settings.category], settings, generation, requested_generation: generation
      };
      return { ok: true, json: async () => value };
    };
    const document = { getElementById(id) { assert(ids[id], `missing ${id}`); return ids[id]; },
      querySelectorAll(q) { return q === '[data-mode]' ? modes : presets; } };
    vm.runInNewContext(script, { document, window: { devicePixelRatio: 1 }, fetch,
      setTimeout(fn) { const id = callbacks.size + 1; callbacks.set(id, fn); return id; },
      clearTimeout(id) { callbacks.delete(id); }, ResizeObserver: class { observe() {} } });
    await flush();
    assert(ids['model-name'].textContent.includes('易弯'));
    assert(strokes > 10);
    ids.ky.value = '250'; ids.ky.listeners.input(); ids.ky.listeners.change(); await flush();
    assert(requests.at(-1).body.ky < .5 && requests.at(-1).body.ky > fixture.meta.presets.bendable.ranges.ky[0]);
    assert(ids['ky-value'].textContent.includes('N/mm'));
    ids.ky.value = '1000'; ids.ky.listeners.input(); ids.ky.listeners.change(); await flush();
    assert.strictEqual(requests.at(-1).body.ky, fixture.meta.presets.bendable.ranges.ky[1]);
    ids.ky.value = '0'; ids.ky.listeners.input(); ids.ky.listeners.change(); await flush();
    assert.strictEqual(requests.at(-1).body.ky, fixture.meta.presets.bendable.ranges.ky[0]);
    presets[1].listeners.click(); await flush();
    assert.strictEqual(requests.at(-1).body.category, 'compressible');
    modes[2].listeners.click(); await flush();
    assert.strictEqual(requests.at(-1).body.mode, 'sag');
    assert.strictEqual(ids.force.disabled, true);
    await ids.pause.listeners.click(); await flush();
    assert.strictEqual(requests.at(-1).url, '/api/pause');
    ids.scene.listeners.pointerdown({ clientX: 100, clientY: 100, pointerId: 1 });
    ids.scene.listeners.pointermove({ clientX: 130, clientY: 120 });
    ids.scene.listeners.pointerup();
    ids.scene.listeners.wheel({ deltaY: 120, preventDefault() {} });
    ids['view-reset'].listeners.click();
    assert(ids['status'].textContent && !ids['metric-height'].textContent.includes('NaN'));
    checks.push({ width, sliders: true, presets: true, modes: true, pause: true,
      rotate_zoom_reset: true, finite_canvas_coordinates: true });
  }
  const report = { browser_test: false, checks };
  fs.writeFileSync(path.join(out, 'ui-contract-check.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
})().catch(e => { console.error(e); process.exitCode = 1; });
