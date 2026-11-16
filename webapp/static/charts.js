/* Small canvas charts (no external libraries, so the lab works offline).
 *
 *   LineChart     loss curves (live, many series, hover crosshair)
 *   BarChart      horizontal bars (feature importance)
 *   MFChart       membership functions of one input, with a patient marker
 *   ScatterChart  runs: fidelity vs number of conditions
 *
 * Colours come from the CSS custom properties, so charts follow the theme.
 */
const Charts = (() => {
  const registry = new Set();
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const FONT = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
  const tooltip = () => document.getElementById("tooltip");

  function niceTicks(lo, hi, count = 4) {
    if (!(hi > lo)) { hi = lo + 1; }
    const raw = (hi - lo) / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(+v.toFixed(10));
    return ticks;
  }
  const fmt = (v) => (Number.isInteger(v) ? String(v) : Math.abs(v) >= 100 ? v.toFixed(0)
    : Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3));

  function showTip(evt, html) {
    const tip = tooltip();
    tip.innerHTML = html;
    tip.classList.add("show");
    const pad = 14;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    const r = tip.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }
  const hideTip = () => tooltip().classList.remove("show");
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  class Base {
    constructor(el) {
      this.el = el;
      el.innerHTML = "";
      this.canvas = document.createElement("canvas");
      el.appendChild(this.canvas);
      this.ctx = this.canvas.getContext("2d");
      this.hover = null;
      this.observer = new ResizeObserver(() => this.draw());
      this.observer.observe(el);
      this.canvas.addEventListener("pointermove", (e) => { this.hover = this.pos(e); this.draw(); this.onHover(e); });
      this.canvas.addEventListener("pointerleave", () => { this.hover = null; this.draw(); hideTip(); });
      registry.add(this);
    }
    destroy() { this.observer.disconnect(); registry.delete(this); }
    pos(e) { const r = this.canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
    prepare() {
      const dpr = window.devicePixelRatio || 1;
      const w = this.el.clientWidth, h = this.el.clientHeight;
      if (!w || !h) return null;
      if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
        this.canvas.width = Math.round(w * dpr);
        this.canvas.height = Math.round(h * dpr);
      }
      const ctx = this.ctx;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.font = FONT;
      return { ctx, w, h };
    }
    axes(ctx, box, xTicks, yTicks, sx, sy, xLabel, yLabel, xFmt = fmt, yFmt = fmt) {
      ctx.strokeStyle = css("--grid");
      ctx.lineWidth = 1;
      ctx.fillStyle = css("--muted");
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (const t of yTicks) {
        const y = Math.round(sy(t)) + 0.5;
        ctx.beginPath(); ctx.moveTo(box.l, y); ctx.lineTo(box.r, y); ctx.stroke();
        ctx.fillText(yFmt(t), box.l - 6, y);
      }
      ctx.strokeStyle = css("--axis");
      ctx.beginPath(); ctx.moveTo(box.l, box.b + 0.5); ctx.lineTo(box.r, box.b + 0.5); ctx.stroke();
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (const t of xTicks) ctx.fillText(xFmt(t), sx(t), box.b + 5);
      ctx.fillStyle = css("--ink-2");
      if (xLabel) { ctx.fillText(xLabel, (box.l + box.r) / 2, box.b + 19); }
      if (yLabel) {
        ctx.save(); ctx.translate(11, (box.t + box.b) / 2); ctx.rotate(-Math.PI / 2);
        ctx.textBaseline = "middle"; ctx.fillText(yLabel, 0, 0); ctx.restore();
      }
    }
    onHover() {}
  }

  // --------------------------------------------------------------- line
  class LineChart extends Base {
    constructor(el, opts = {}) { super(el); this.opts = opts; this.series = []; }
    setSeries(series) { this.series = series; this.draw(); }
    domain() {
      const pts = this.series.flatMap((s) => s.points);
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]).filter((v) => v != null);
      return {
        x0: this.opts.xMin ?? Math.min(...xs, 0), x1: this.opts.xMax ?? Math.max(...xs, 1),
        y0: this.opts.yMin ?? Math.min(...ys), y1: this.opts.yMax ?? Math.max(...ys),
      };
    }
    draw() {
      const p = this.prepare();
      if (!p) return;
      const { ctx, w, h } = p;
      if (!this.series.some((s) => s.points.length)) {
        ctx.fillStyle = css("--muted"); ctx.textAlign = "center";
        ctx.fillText(this.opts.empty || "Waiting for training…", w / 2, h / 2);
        return;
      }
      const d = this.domain();
      let { y0, y1 } = d;
      const padY = (y1 - y0) * 0.08 || 0.05;
      if (this.opts.yMin == null) y0 -= padY;
      if (this.opts.yMax == null) y1 += padY;
      const box = { l: this.opts.yLabel ? 62 : 46, r: w - 12, t: 22, b: h - 34 };
      const sx = (x) => box.l + ((x - d.x0) / (d.x1 - d.x0 || 1)) * (box.r - box.l);
      const sy = (y) => box.b - ((y - y0) / (y1 - y0 || 1)) * (box.b - box.t);
      this.scale = { sx, sy, box, d };
      this.axes(ctx, box, niceTicks(d.x0, d.x1, 5).filter((t) => t >= d.x0 && t <= d.x1),
        niceTicks(y0, y1, 4).filter((t) => t >= y0 && t <= y1), sx, sy, this.opts.xLabel, this.opts.yLabel);

      for (const s of this.series) {
        if (!s.points.length) continue;
        ctx.strokeStyle = css(s.color);
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        ctx.setLineDash(s.dash ? [5, 4] : []);
        ctx.beginPath();
        s.points.forEach(([x, y], i) => (i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y))));
        ctx.stroke();
        ctx.setLineDash([]);
      }
      // legend (two or more series): swatch + name, top right
      if (this.series.length > 1) {
        ctx.textAlign = "left"; ctx.textBaseline = "middle";
        let x = box.r;
        for (const s of [...this.series].reverse()) {
          const tw = ctx.measureText(s.name).width;
          x -= tw + 24;
          ctx.fillStyle = css(s.color); ctx.fillRect(x, 8, 12, 3);
          ctx.fillStyle = css("--ink-2"); ctx.fillText(s.name, x + 16, 10);
        }
      }
      if (this.hover && this.hover.x >= box.l && this.hover.x <= box.r) {
        ctx.strokeStyle = css("--axis"); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(this.hover.x + 0.5, box.t); ctx.lineTo(this.hover.x + 0.5, box.b); ctx.stroke();
      }
    }
    onHover(evt) {
      if (!this.scale || !this.hover) return;
      const { box, d } = this.scale;
      if (this.hover.x < box.l || this.hover.x > box.r) { hideTip(); return; }
      const x = d.x0 + ((this.hover.x - box.l) / (box.r - box.l)) * (d.x1 - d.x0);
      const rows = [];
      for (const s of this.series) {
        if (!s.points.length) continue;
        let best = s.points[0];
        for (const pt of s.points) if (Math.abs(pt[0] - x) < Math.abs(best[0] - x)) best = pt;
        if (best[1] != null) rows.push(`<div class="row"><span class="sw" style="background:${css(s.color)}"></span><b>${fmt(best[1])}</b> ${esc(s.name)}</div>`);
      }
      showTip(evt, `<div>${esc(this.opts.xLabel || "x")} ${Math.round(x)}</div>${rows.join("")}`);
    }
  }

  // ---------------------------------------------------------------- bars
  class BarChart extends Base {
    constructor(el, opts = {}) { super(el); this.opts = opts; this.rows = []; }
    setRows(rows) { this.rows = rows; this.draw(); }
    draw() {
      const p = this.prepare();
      if (!p) return;
      const { ctx, w, h } = p;
      if (!this.rows.length) {
        ctx.fillStyle = css("--muted"); ctx.textAlign = "center"; ctx.fillText("Waiting for the network…", w / 2, h / 2);
        return;
      }
      const labelW = Math.max(...this.rows.map((r) => ctx.measureText(r.label).width)) + 10;
      const box = { l: labelW, r: w - 54, t: 4, b: h - 4 };
      const max = Math.max(...this.rows.map((r) => r.value)) || 1;
      const band = (box.b - box.t) / this.rows.length;
      this.layout = { box, band };
      this.rows.forEach((r, i) => {
        const y = box.t + i * band;
        const bw = (r.value / max) * (box.r - box.l);
        const hov = this.hover && this.hover.y >= y && this.hover.y < y + band;
        ctx.fillStyle = css(r.highlight ? "--s1" : "--axis");
        ctx.globalAlpha = hov ? 0.8 : 1;
        const bh = Math.min(16, band - 6);
        ctx.beginPath();
        ctx.roundRect(box.l, y + (band - bh) / 2, Math.max(bw, 2), bh, [0, 4, 4, 0]);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.fillStyle = css("--ink-2"); ctx.textAlign = "right"; ctx.textBaseline = "middle";
        ctx.fillText(r.label, labelW - 8, y + band / 2);
        ctx.textAlign = "left";
        ctx.fillText(this.opts.format ? this.opts.format(r) : fmt(r.value), box.l + bw + 6, y + band / 2);
      });
    }
    onHover(evt) {
      if (!this.layout) return;
      const i = Math.floor((this.hover.y - this.layout.box.t) / this.layout.band);
      const r = this.rows[i];
      if (r && r.tip) showTip(evt, r.tip); else hideTip();
    }
  }

  // ------------------------------------------------- membership functions
  function memberships(feature, x, terms) {
    const n = terms.length;
    return terms.map((t, i) => {
      let d = x - t.center;
      if (i === 0 && d < 0) d = 0;            // "Low" is a shoulder: 1 below its centre
      if (i === n - 1 && d > 0) d = 0;        // "High" is a shoulder: 1 above its centre
      return Math.exp(-0.5 * (d / t.sigma) ** 2);
    });
  }
  const termColor = (i, n) => (n === 2 ? ["--t0", "--t2"][i] : ["--t0", "--t1", "--t2"][i]);

  class MFChart extends Base {
    constructor(el) { super(el); this.feature = null; this.initial = null; this.marker = null; }
    set(feature, initial = null, marker = null) {
      this.feature = feature; this.initial = initial; this.marker = marker; this.draw();
    }
    draw() {
      const p = this.prepare();
      if (!p || !this.feature) return;
      const { ctx, w, h } = p;
      const f = this.feature;
      const box = { l: 30, r: w - 10, t: 30, b: h - 22 };
      const sx = (x) => box.l + ((x - f.lo) / (f.hi - f.lo || 1)) * (box.r - box.l);
      const sy = (y) => box.b - y * (box.b - box.t);
      this.scale = { sx, box };
      this.axes(ctx, box, niceTicks(f.lo, f.hi, 4).filter((t) => t >= f.lo && t <= f.hi), [0, 0.5, 1], sx, sy,
        null, null, fmt, (v) => v.toFixed(1));
      const N = 160;
      const draw = (feat, dashed) => {
        feat.terms.forEach((t, i) => {
          ctx.strokeStyle = css(termColor(i, feat.terms.length));
          ctx.lineWidth = dashed ? 1 : 2;
          ctx.globalAlpha = dashed ? 0.6 : 1;
          ctx.setLineDash(dashed ? [4, 3] : []);
          ctx.beginPath();
          for (let k = 0; k <= N; k++) {
            const x = f.lo + (k / N) * (f.hi - f.lo);
            const y = memberships(feat, x, feat.terms)[i];
            k ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y));
          }
          ctx.stroke();
        });
        ctx.setLineDash([]); ctx.globalAlpha = 1;
      };
      if (this.initial) draw(this.initial, true);
      draw(f, false);
      // term names at their centres; a name that would collide with the previous one moves up a row
      ctx.textBaseline = "bottom"; ctx.textAlign = "center";
      const rightEdge = [-Infinity, -Infinity];          // last label end on each of the two rows
      f.terms.forEach((t, i) => {
        const tw = ctx.measureText(t.name).width;
        const x = Math.min(Math.max(sx(t.center), box.l + tw / 2), box.r - tw / 2);
        const row = x - tw / 2 > rightEdge[0] + 4 ? 0 : 1;
        rightEdge[row] = x + tw / 2;
        ctx.fillStyle = css(termColor(i, f.terms.length));
        ctx.fillText(t.name, x, box.t - 3 - row * 12);
      });
      if (this.marker != null) {
        const x = sx(Math.min(Math.max(this.marker, f.lo), f.hi));
        ctx.strokeStyle = css("--s2"); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x, box.t); ctx.lineTo(x, box.b); ctx.stroke();
        memberships(f, this.marker, f.terms).forEach((m, i) => {
          ctx.fillStyle = css(termColor(i, f.terms.length));
          ctx.beginPath(); ctx.arc(x, sy(m), 4, 0, 2 * Math.PI); ctx.fill();
          ctx.strokeStyle = css("--card"); ctx.lineWidth = 2; ctx.stroke();
        });
      }
    }
    onHover(evt) {
      if (!this.scale || !this.feature) return;
      const { box } = this.scale;
      const f = this.feature;
      if (this.hover.x < box.l || this.hover.x > box.r) { hideTip(); return; }
      const x = f.lo + ((this.hover.x - box.l) / (box.r - box.l)) * (f.hi - f.lo);
      const mu = memberships(f, x, f.terms);
      showTip(evt, `<div><b>${f.name} = ${fmt(x)}</b> ${esc(f.unit || "")}</div>` + f.terms.map((t, i) =>
        `<div class="row"><span class="sw" style="background:${css(termColor(i, f.terms.length))}"></span><b>${mu[i].toFixed(2)}</b> ${t.name}</div>`).join(""));
    }
  }

  // ------------------------------------------------------------- scatter
  class ScatterChart extends Base {
    constructor(el, opts = {}) {
      super(el); this.opts = opts; this.points = [];
      this.canvas.addEventListener("click", () => { const p = this.nearest(); if (p && opts.onClick) opts.onClick(p); });
    }
    setPoints(points) { this.points = points; this.draw(); }
    nearest() {
      if (!this.hover || !this.scale) return null;
      let best = null, bd = 14;
      for (const p of this.points) {
        const d = Math.hypot(this.scale.sx(p.x) - this.hover.x, this.scale.sy(p.y) - this.hover.y);
        if (d < bd) { bd = d; best = p; }
      }
      return best;
    }
    draw() {
      const p = this.prepare();
      if (!p) return;
      const { ctx, w, h } = p;
      if (!this.points.length) {
        ctx.fillStyle = css("--muted"); ctx.textAlign = "center"; ctx.fillText("Train a run to see it here.", w / 2, h / 2);
        return;
      }
      const xs = this.points.map((q) => q.x), ys = this.points.map((q) => q.y);
      const x1 = Math.max(...xs) * 1.1 + 2, y0 = Math.min(...ys) - 0.02, y1 = Math.min(1, Math.max(...ys) + 0.02);
      const box = { l: 46, r: w - 14, t: 14, b: h - 34 };
      const sx = (x) => box.l + (x / x1) * (box.r - box.l);
      const sy = (y) => box.b - ((y - y0) / (y1 - y0)) * (box.b - box.t);
      this.scale = { sx, sy };
      this.axes(ctx, box, niceTicks(0, x1, 6).filter((t) => t <= x1), niceTicks(y0, y1, 4).filter((t) => t >= y0 && t <= y1),
        sx, sy, this.opts.xLabel, this.opts.yLabel, fmt, (v) => (v * 100).toFixed(0) + "%");
      const near = this.nearest();
      for (const q of this.points) {
        const r = q === near ? 7 : 5.5;
        ctx.beginPath(); ctx.arc(sx(q.x), sy(q.y), r, 0, 2 * Math.PI);
        ctx.fillStyle = q.hollow ? css("--card") : css(q.color || "--s1");
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = q.hollow ? css(q.color || "--s1") : css("--card");
        ctx.stroke();
        if (q.label) {
          ctx.fillStyle = css("--ink-2"); ctx.textAlign = "left"; ctx.textBaseline = "middle";
          ctx.fillText(q.label, sx(q.x) + 9, sy(q.y));
        }
      }
    }
    onHover(evt) {
      const p = this.nearest();
      this.canvas.style.cursor = p ? "pointer" : "default";
      if (p) showTip(evt, p.tip); else hideTip();
    }
  }

  function redrawAll() { for (const c of registry) c.draw(); }

  return { LineChart, BarChart, MFChart, ScatterChart, memberships, termColor, redrawAll, css, esc, fmt };
})();
