(() => {
  const root = document.getElementById('kappa-demo');
  if (!root) return;
  const plot = document.getElementById('kappa-space');
  const preview = document.getElementById('kappa-preview');
  const spread = document.getElementById('kappa-spread');
  const reset = document.getElementById('kappa-reset');
  const shapeButtons = Array.from(root.querySelectorAll('[data-kappa-shape]'));
  const state = { mean: [0.5, 1, 0.5], spread: 0.14, yaw: -0.72, pitch: 0.58, selected: null };
  const names = ['Cube', 'Cuboid', 'Cylinder', 'Sphere', 'Pillow'];
  const axisNames = ['Size', 'Shape', 'Compliance'];
  const colors = ['#1d4fd6', '#b7895a', '#24724a'];
  const clamp = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, x));
  const mix = (a, b, t) => a + (b - a) * t;
  let seed = 2027;
  const random = () => ((seed = (1664525 * seed + 1013904223) >>> 0) + 1) / 4294967297;
  const gaussian = () => Math.sqrt(-2 * Math.log(random())) * Math.cos(2 * Math.PI * random());
  const noise = Array.from({ length: 52 }, () => [gaussian(), gaussian(), gaussian()]);
  const samples = () => noise.map(v => v.map((n, i) => clamp(state.mean[i] + n * state.spread)));
  const output = id => document.getElementById(id);
  let handles = [], dots = [], center = null, drag = null, queued = false;

  function canvasContext(canvas) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, rect.width), height = Math.max(1, rect.height);
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, width, height };
  }

  function rotate(v, yaw, pitch) {
    const x = Math.cos(yaw) * v[0] - Math.sin(yaw) * v[1];
    const y = Math.sin(yaw) * v[0] + Math.cos(yaw) * v[1];
    return [x, Math.sin(pitch) * y - Math.cos(pitch) * v[2], Math.cos(pitch) * y + Math.sin(pitch) * v[2]];
  }

  function projection(width, height) {
    const scale = Math.min(width * 0.46, height * 0.52);
    return v => {
      const r = rotate(v.map(x => x - 0.5), state.yaw, state.pitch);
      return [width / 2 + r[0] * scale, height / 2 + 12 + r[1] * scale, r[2]];
    };
  }

  function line(ctx, points, color, width = 1) {
    ctx.beginPath();
    points.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  }

  function circle(ctx, p, radius, fill, stroke) {
    ctx.beginPath(); ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
    ctx.fillStyle = fill; ctx.fill();
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
  }

  function drawDistribution(points) {
    const { ctx, width, height } = canvasContext(plot);
    const project = projection(width, height);
    ctx.font = '9px "IBM Plex Mono", monospace';
    const corners = Array.from({ length: 8 }, (_, i) => [i & 1 ? 1 : 0, i & 2 ? 1 : 0, i & 4 ? 1 : 0]);
    corners.forEach((v, i) => {
      for (let axis = 0; axis < 3; axis++) {
        const j = i ^ (1 << axis);
        if (i < j) line(ctx, [project(v), project(corners[j])], '#d9dcd7');
      }
    });
    const origin = project([0, 0, 0]);
    axisNames.forEach((name, axis) => {
      const end = [0, 0, 0]; end[axis] = 1.13;
      const p = project(end);
      line(ctx, [origin, p], colors[axis], 1.2);
      ctx.fillStyle = colors[axis]; ctx.fillText(name, p[0] + 5, p[1] + 4);
    });
    for (let plane = 0; plane < 3; plane++) {
      const ring = Array.from({ length: 41 }, (_, i) => {
        const a = i / 40 * Math.PI * 2, v = state.mean.slice();
        v[(plane + 1) % 3] += Math.cos(a) * state.spread * 1.6;
        v[(plane + 2) % 3] += Math.sin(a) * state.spread * 1.6;
        return project(v);
      });
      line(ctx, ring, '#1d4fd62b');
    }
    dots = points.map((v, i) => ({ p: project(v), i })).sort((a, b) => a.p[2] - b.p[2]);
    dots.forEach(({ p, i }) => circle(ctx, p, state.selected === i ? 5 : 2.7, state.selected === i ? '#b7895a' : '#1d4fd68a'));
    center = project(state.mean);
    handles = axisNames.map((name, axis) => {
      const v = state.mean.slice(); v[axis] += state.mean[axis] > 0.8 ? -0.20 : 0.20;
      const p = project(v);
      line(ctx, [center, p], colors[axis], 2);
      circle(ctx, p, 4.5, colors[axis], '#fff');
      return { p, axis };
    });
    circle(ctx, center, 7, '#1d4fd6', '#fff');
    ctx.fillStyle = '#1d4fd6'; ctx.fillText('μ', center[0] + 10, center[1] - 8);
  }

  function shapePoint(raw, kind) {
    const [x, y, z] = raw;
    if (kind === 0) return [x * 36, y * 36, z * 36];
    if (kind === 1) return [x * 64, y * 31, z * 22];
    if (kind === 2) {
      const r = Math.hypot(x, y), f = r ? Math.max(Math.abs(x), Math.abs(y)) / r : 0;
      return [x * f * 37, y * f * 37, z * 40];
    }
    const length = Math.hypot(x, y, z) || 1;
    if (kind === 3) return raw.map(v => v / length * 42);
    return raw.map((v, i) => mix(v, v / length, 0.38) * [63, 38, 24][i]);
  }

  function parcelGeometry(parameters) {
    const location = parameters[1] * 4, first = Math.floor(location), second = Math.min(4, first + 1);
    const fraction = location - first, size = 0.67 + parameters[0] * 0.75, soft = parameters[2];
    const vertex = raw => {
      const a = shapePoint(raw, first), b = shapePoint(raw, second);
      const v = a.map((x, i) => mix(x, b[i], fraction) * size);
      const halfLength = mix([36, 64, 37, 42, 63][first], [36, 64, 37, 42, 63][second], fraction) * size;
      const position = Math.abs(v[0]) / halfLength;
      v[2] *= 1 - soft * 0.25 * Math.exp(-position * position * 8);
      v[2] -= soft * 23 * Math.pow(position, 1.6);
      v[1] *= 1 + soft * 0.08;
      return v;
    };
    const faces = [], n = 5;
    for (let axis = 0; axis < 3; axis++) {
      for (const sign of [-1, 1]) {
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
          const corners = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
          faces.push(corners.map(([u, v]) => {
            const raw = [0, 0, 0]; raw[axis] = sign;
            raw[(axis + 1) % 3] = u / n * 2 - 1;
            raw[(axis + 2) % 3] = v / n * 2 - 1;
            return vertex(raw);
          }));
        }
      }
    }
    const top = mix([36, 22, 40, 42, 24][first], [36, 22, 40, 42, 24][second], fraction) * size * (1 - soft * 0.25);
    return { faces, top };
  }

  function drawParcel(ctx, parameters, cx, cy, scale, gripper) {
    const { faces, top } = parcelGeometry(parameters);
    const project = v => {
      const p = rotate(v, -0.63, 0.68);
      return [cx + p[0] * scale, cy + p[1] * scale, p[2]];
    };
    ctx.beginPath(); ctx.ellipse(cx, cy + 50 * scale, 62 * scale, 12 * scale, 0, 0, Math.PI * 2);
    ctx.fillStyle = '#161b220b'; ctx.fill();
    faces.map(face => ({ points: face.map(project), shade: face.reduce((v, p) => v + p[2], 0) / 4 }))
      .sort((a, b) => a.points.reduce((n, p) => n + p[2], 0) - b.points.reduce((n, p) => n + p[2], 0))
      .forEach(({ points, shade }) => {
        const light = clamp(0.89 + shade / 180, 0.67, 1.15);
        ctx.beginPath(); points.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath();
        ctx.fillStyle = `rgb(${Math.round(187 * light)},${Math.round(145 * light)},${Math.round(99 * light)})`;
        ctx.fill(); ctx.strokeStyle = '#79552d28'; ctx.lineWidth = 0.45; ctx.stroke();
      });
    if (gripper) {
      for (const x of [-9, 9]) line(ctx, [project([x, 0, top + 31]), project([x, 0, top - 1])], '#25303d', 5 * scale);
      line(ctx, [project([-9, 0, top + 31]), project([9, 0, top + 31])], '#25303d', 5 * scale);
      circle(ctx, project([0, 0, top]), 2.6 * scale, '#1d4fd6');
    }
  }

  function shapeLabel(value) {
    const location = value * 4, near = Math.round(location);
    if (Math.abs(location - near) < 0.12) return names[near];
    return `${names[Math.floor(location)]} / ${names[Math.ceil(location)]}`;
  }

  function drawPreview(points) {
    const { ctx, width, height } = canvasContext(preview);
    const parameters = state.selected === null ? state.mean : points[state.selected];
    const scale = Math.min(width / 335, height / 210, 1.24);
    drawParcel(ctx, parameters, width / 2, height * 0.38, scale, true);
    ctx.font = '8px "IBM Plex Mono", monospace'; ctx.fillStyle = '#59606b'; ctx.textAlign = 'center';
    ctx.fillText('SAMPLES FROM THIS DISTRIBUTION', width / 2, height * 0.72);
    [4, 19, 37].forEach((index, i) => {
      const x = width * (0.21 + i * 0.29);
      drawParcel(ctx, points[index], x, height * 0.86, scale * 0.37, false);
      ctx.fillStyle = '#59606b'; ctx.fillText(names[Math.round(points[index][1] * 4)], x, height - 3);
    });
    output('kappa-preview-title').textContent = state.selected === null ? 'Mean parcel' : `Sample ${String(state.selected + 1).padStart(2, '0')}`;
    output('kappa-size').textContent = parameters[0].toFixed(2);
    output('kappa-shape').textContent = shapeLabel(parameters[1]);
    output('kappa-compliance').textContent = parameters[2].toFixed(2);
  }

  function render() {
    queued = false;
    const points = samples();
    drawDistribution(points); drawPreview(points);
    output('kappa-mean').textContent = `μ = (${state.mean.map(x => x.toFixed(2)).join(', ')})`;
    output('kappa-spread-value').textContent = state.spread.toFixed(2);
    shapeButtons.forEach(button => button.setAttribute('aria-pressed', String(Math.abs(Number(button.dataset.kappaShape) - state.mean[1]) < 0.03)));
    root.dataset.mean = state.mean.join(',');
  }

  function schedule() {
    if (!queued) { queued = true; requestAnimationFrame(render); }
  }

  function pointer(event) {
    const rect = plot.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }
  const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

  plot.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    const p = pointer(event), rect = plot.getBoundingClientRect(), project = projection(rect.width, rect.height);
    const origin = project(state.mean);
    const basis = [0, 1, 2].map(axis => { const v = state.mean.slice(); v[axis] += 1; const q = project(v); return [q[0] - origin[0], q[1] - origin[1]]; });
    const handle = handles.find(h => distance(h.p, p) < 10);
    const onCenter = center && distance(center, p) < 11;
    const selected = dots.reduce((best, dot) => distance(dot.p, p) < (best ? distance(best.p, p) : 8) ? dot : best, null);
    drag = { p, mean: state.mean.slice(), yaw: state.yaw, pitch: state.pitch, basis, mode: onCenter ? 'plane' : handle ? 'axis' : 'orbit', axis: handle?.axis };
    state.selected = onCenter || handle ? null : selected ? selected.i : state.selected;
    plot.setPointerCapture(event.pointerId); schedule();
  });

  plot.addEventListener('pointermove', event => {
    if (!drag) return;
    const p = pointer(event), dx = p[0] - drag.p[0], dy = p[1] - drag.p[1];
    if (drag.mode === 'axis') {
      const b = drag.basis[drag.axis];
      state.mean[drag.axis] = clamp(drag.mean[drag.axis] + (dx * b[0] + dy * b[1]) / (b[0] * b[0] + b[1] * b[1]));
    } else if (drag.mode === 'plane') {
      const [a, b] = drag.basis, determinant = a[0] * b[1] - b[0] * a[1];
      state.mean[0] = clamp(drag.mean[0] + (dx * b[1] - b[0] * dy) / determinant);
      state.mean[1] = clamp(drag.mean[1] + (a[0] * dy - dx * a[1]) / determinant);
    } else {
      state.yaw = drag.yaw + dx * 0.008;
      state.pitch = clamp(drag.pitch + dy * 0.006, 0.28, 1.12);
    }
    schedule();
  });
  const release = () => { drag = null; };
  plot.addEventListener('pointerup', release);
  plot.addEventListener('pointercancel', release);
  plot.addEventListener('keydown', event => {
    const changes = { ArrowLeft: [0, -0.03], ArrowRight: [0, 0.03], ArrowDown: [1, -0.03], ArrowUp: [1, 0.03], PageDown: [2, -0.03], PageUp: [2, 0.03] };
    if (!changes[event.key]) return;
    event.preventDefault(); const [axis, amount] = changes[event.key];
    state.mean[axis] = clamp(state.mean[axis] + amount); state.selected = null; schedule();
  });
  spread.addEventListener('input', () => { state.spread = clamp(Number(spread.value), 0.03, 0.30); schedule(); });
  shapeButtons.forEach(button => button.addEventListener('click', () => { state.mean[1] = Number(button.dataset.kappaShape); state.selected = null; schedule(); }));
  reset.addEventListener('click', () => { Object.assign(state, { mean: [0.5, 1, 0.5], spread: 0.14, yaw: -0.72, pitch: 0.58, selected: null }); spread.value = '0.14'; schedule(); });
  new ResizeObserver(schedule).observe(root);
  if (document.fonts) document.fonts.ready.then(schedule);
  schedule();
})();
