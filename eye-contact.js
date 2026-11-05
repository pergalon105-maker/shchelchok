const TASKS_VERSION = '0.10.14'; // как в основной программе
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}`;
const MODEL_URL =
'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';
const EYES = [
{ contour: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], corners: [33, 133], lids: [159, 145] },
{ contour: [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466], corners: [263, 362], lids: [386, 374] },
];
const IRISES = [468, 473]; // центр радужки, следующие 4 точки — её край
const NOSE = 1, CHEEK_A = 234, CHEEK_B = 454, CHIN = 152;
export const CFG = {
kx: 1.0, ky: 1.5,              // компенсация поворота/наклона головы (сколько сдвигать зрачок)
defaultU0: 0.09, defaultV0: -0.07, defaultYaw0: 0.0, defaultPitch0: null, // без калибровки
maxU: 0.26, minV: -0.14, maxV: 0.12, // куда вообще можно ставить зрачок внутри глаза
irisScale: 1.06,               // радиус радужки чуть больше точек MediaPipe (захват лимба)
holeScale: 1.18,               // сколько белка перерисовать на старом месте
lidInset: 0.07,                // насколько отступать внутрь от контура век (доля радиуса радужки)
openLo: 0.16, openHi: 0.23,    // моргание: глаз закрыт → эффект 0
yawLo: 0.14, yawHi: 0.2,       // поворот головы ~20–27° → эффект гаснет
pitchLo: 0.12, pitchHi: 0.18,  // наклон головы (после калибровки)
shiftLo: 1.3, shiftHi: 1.9,    // сдвиг больше ~1.5 радиусов радужки = человек смотрит в сторону специально
fadeSpeed: 5,                  // как быстро эффект появляется/гаснет (в секунду)
minSize: 10,                   // глаз меньше 10 px — слишком далеко, не трогаем
};
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const smooth = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; };
class OneEuro {
constructor(minCutoff = 1.2, beta = 0.05, dCutoff = 1.0) { Object.assign(this, { minCutoff, beta, dCutoff }); this.x = null; }
static a(cutoff, dt) { const r = 2 * Math.PI * cutoff * dt; return r / (r + 1); }
filter(x, dt) {
if (this.x === null || !(dt > 0)) { this.x = x; this.dx = 0; return x; }
const dx = (x - this.x) / dt;
this.dx += OneEuro.a(this.dCutoff, dt) * (dx - this.dx);
const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
this.x += OneEuro.a(cutoff, dt) * (x - this.x);
return this.x;
}
reset() { this.x = null; }
}
function polyDist(px, py, poly) {
let inside = false, best = Infinity;
for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
const [xi, yi] = poly[i], [xj, yj] = poly[j];
if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
const ex = xj - xi, ey = yj - yi;
const t = clamp(((px - xi) * ex + (py - yi) * ey) / (ex * ex + ey * ey || 1), 0, 1);
const dx = px - (xi + t * ex), dy = py - (yi + t * ey);
const d = dx * dx + dy * dy;
if (d < best) best = d;
}
best = Math.sqrt(best);
return inside ? best : -best;
}
function rowSpan(y, poly, x) {
const xs = [];
for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
const [xi, yi] = poly[i], [xj, yj] = poly[j];
if ((yi > y) !== (yj > y)) xs.push(xi + ((y - yi) * (xj - xi)) / (yj - yi));
}
xs.sort((a, b) => a - b);
let span = null;
for (let k = 0; k + 1 < xs.length; k += 2) {
if (!span || (x >= xs[k] && x <= xs[k + 1])) span = [xs[k], xs[k + 1]];
}
return span;
}
export class EyeContact {
static async create(opts = {}) {
const vision = opts.vision ?? (await import( `${CDN}/vision_bundle.mjs`));
const fileset = await vision.FilesetResolver.forVisionTasks(opts.wasmPath ?? `${CDN}/wasm`);
const make = (delegate) =>
vision.FaceLandmarker.createFromOptions(fileset, {
baseOptions: { modelAssetPath: opts.modelPath ?? MODEL_URL, delegate },
runningMode: 'VIDEO',
numFaces: 1,
outputFacialTransformationMatrixes: false,
outputFaceBlendshapes: false,
});
let landmarker;
try { landmarker = await make(opts.delegate ?? 'GPU'); } catch { landmarker = await make('CPU'); }
return new EyeContact(landmarker);
}
constructor(landmarker) {
this.landmarker = landmarker;
this.enabled = true;
this.strength = 1;
this.active = 0;              // текущая сила после предохранителей (0..1), плавная
this.status = { face: false, active: 0, reason: 'старт', calibrating: false, calibrated: false };
this.lastTs = -1;
this.fShift = [new OneEuro(1.0, 0.04), new OneEuro(1.0, 0.04)];
this.fPts = new Map();
this.calib = null;
this.calibSamples = null;
this.debug = null;
this.forceShift = null;       // для тестов: [dx, dy] в радиусах радужки
try {
const saved = JSON.parse(localStorage.getItem('eyeContactCalib') || 'null');
if (saved && Array.isArray(saved.eyes)) { this.calib = saved; this.status.calibrated = true; }
} catch {  }
}
calibrate(ms = 1500) {
this.calibSamples = { until: null, ms, eyes: [[], []], yaw: [], pitch: [] };
this.status.calibrating = true;
}
resetCalibration() {
this.calib = null;
this.status.calibrated = false;
try { localStorage.removeItem('eyeContactCalib'); } catch {}
}
process(canvas, ts, o = {}) {
if (ts <= this.lastTs) ts = this.lastTs + 1;
const dt = this.lastTs < 0 ? 1 / 30 : (ts - this.lastTs) / 1000;
this.lastTs = ts;
this.debug = null;
const W = canvas.width, H = canvas.height;
if (!this.enabled && !this.status.calibrating) {
this.active = 0;
this.status = { ...this.status, face: false, active: 0, reason: 'выключено' };
return;
}
let res;
try { res = this.landmarker.detectForVideo(o.source ?? canvas, ts); } catch (e) { this.status.reason = 'ошибка: ' + e.message; return; }
const lm = res.faceLandmarks?.[0];
if (!lm || lm.length < 478) {
this.fade(0, dt);
this.fShift.forEach((f) => f.reset());
this.fPts.clear();
this.status = { ...this.status, face: false, active: this.active, reason: 'лица нет' };
return;
}
const P = (i) => {
const x = lm[i].x * W, y = lm[i].y * H;
let f = this.fPts.get(i);
if (!f) { f = [new OneEuro(2.5, 0.3), new OneEuro(2.5, 0.3)]; this.fPts.set(i, f); }
return [f[0].filter(x, dt), f[1].filter(y, dt)];
};
const nose = P(NOSE), ca = P(CHEEK_A), cb = P(CHEEK_B), chin = P(CHIN);
const left = Math.min(ca[0], cb[0]), right = Math.max(ca[0], cb[0]);
const yaw = (nose[0] - left) / Math.max(1, right - left) - 0.5;  // >0: лицо повёрнуто к правому краю кадра
const eyes = EYES.map((E) => {
const poly = E.contour.map(P);
let [a, b] = E.corners.map(P);
if (a[0] > b[0]) [a, b] = [b, a];
const ox = (a[0] + b[0]) / 2, oy = (a[1] + b[1]) / 2;
const w = Math.hypot(b[0] - a[0], b[1] - a[1]);
const ex = [(b[0] - a[0]) / w, (b[1] - a[1]) / w], ey = [-ex[1], ex[0]];
const [up, lo] = E.lids.map(P);
const open = Math.hypot(up[0] - lo[0], up[1] - lo[1]) / w;
return { poly, ox, oy, w, ex, ey, open };
});
const irises = IRISES.map((c) => {
const ctr = P(c);
const r = [1, 2, 3, 4].reduce((s, k) => { const p = P(c + k); return s + Math.hypot(p[0] - ctr[0], p[1] - ctr[1]); }, 0) / 4;
return { c: ctr, r };
});
for (const e of eyes) {
let best = null, bd = Infinity;
for (const ir of irises) { const d = Math.hypot(ir.c[0] - e.ox, ir.c[1] - e.oy); if (d < bd) { bd = d; best = ir; } }
e.iris = best;
const dx = best.c[0] - e.ox, dy = best.c[1] - e.oy;
e.u = (dx * e.ex[0] + dy * e.ex[1]) / e.w;
e.v = (dx * e.ey[0] + dy * e.ey[1]) / e.w;
}
const eyeMidY = (eyes[0].oy + eyes[1].oy) / 2;
const pitch = (nose[1] - eyeMidY) / Math.max(1, chin[1] - eyeMidY); // больше = голова наклонена вниз
if (this.calibSamples) {
const s = this.calibSamples;
if (s.until === null) s.until = ts + s.ms;
if (eyes.every((e) => e.open > CFG.openHi)) {
eyes.forEach((e, k) => s.eyes[k].push([e.u, e.v]));
s.yaw.push(yaw); s.pitch.push(pitch);
}
if (ts >= s.until) {
if (s.yaw.length >= 8) {
this.calib = {
eyes: s.eyes.map((arr) => [median(arr.map((p) => p[0])), median(arr.map((p) => p[1]))]),
yaw0: median(s.yaw), pitch0: median(s.pitch),
};
try { localStorage.setItem('eyeContactCalib', JSON.stringify(this.calib)); } catch {}
this.status.calibrated = true;
this.status.reason = 'калибровка готова';
} else {
this.status.reason = 'калибровка не удалась — глаза не было видно';
}
this.calibSamples = null;
this.status.calibrating = false;
}
}
const cal = this.calib;
const yaw0 = cal ? cal.yaw0 : CFG.defaultYaw0;
const pitch0 = cal ? cal.pitch0 : CFG.defaultPitch0;
const dYaw = yaw - yaw0, dPitch = pitch0 == null ? 0 : pitch - pitch0;
const faceX = (eyes[0].ox + eyes[1].ox) / 2;
const shifts = eyes.map((e, k) => {
const outward = Math.sign(e.ox - faceX) || 1;
const u0 = cal ? cal.eyes[k][0] : CFG.defaultU0 * outward, v0 = cal ? cal.eyes[k][1] : CFG.defaultV0;
const ut = clamp(u0 - CFG.kx * dYaw, -CFG.maxU, CFG.maxU);
const vt = clamp(v0 - CFG.ky * dPitch, CFG.minV, CFG.maxV);
const du = (ut - e.u) * e.w, dv = (vt - e.v) * e.w;
return [du * e.ex[0] + dv * e.ey[0], du * e.ex[1] + dv * e.ey[1]];
});
const R = (eyes[0].iris.r + eyes[1].iris.r) / 2 * CFG.irisScale;
let sx = (shifts[0][0] + shifts[1][0]) / 2, sy = (shifts[0][1] + shifts[1][1]) / 2;
if (this.forceShift) { sx = this.forceShift[0] * R; sy = this.forceShift[1] * R; }
sx = this.fShift[0].filter(sx, dt); sy = this.fShift[1].filter(sy, dt);
const open = Math.min(eyes[0].open, eyes[1].open);
const mag = Math.hypot(sx, sy) / Math.max(1, R);
let safe = 1, reason = 'работает';
const gate = (v, why) => { if (v < safe) { safe = v; if (v < 0.5) reason = why; } };
gate(smooth(CFG.openLo, CFG.openHi, open), 'моргание / глаза прикрыты');
gate(1 - smooth(CFG.yawLo, CFG.yawHi, Math.abs(dYaw)), 'голова сильно повёрнута');
if (pitch0 != null) gate(1 - smooth(CFG.pitchLo, CFG.pitchHi, Math.abs(dPitch)), 'голова сильно наклонена');
if (!this.forceShift) gate(1 - smooth(CFG.shiftLo, CFG.shiftHi, mag), 'смотрит далеко в сторону');
gate(smooth(CFG.minSize, CFG.minSize * 1.6, Math.min(eyes[0].w, eyes[1].w)), 'лицо слишком далеко');
if (this.status.calibrating) { safe = 0; reason = 'калибровка: смотри в камеру'; }
if (!this.enabled) safe = 0;
this.fade(safe * clamp(this.strength, 0, 1), dt);
this.status = { ...this.status, face: true, active: this.active, reason, yaw, pitch };
this.debug = { eyes, shift: [sx, sy], R };
const kx = sx * this.active, ky = sy * this.active;
if (Math.hypot(kx, ky) < 0.25) return;
const ctx = canvas.getContext('2d', { willReadFrequently: true });
for (const e of eyes) this.warpEye(ctx, W, H, e, R, kx, ky);
}
fade(target, dt) {
const step = CFG.fadeSpeed * dt;
this.active += clamp(target - this.active, -step, step);
}
warpEye(ctx, W, H, e, R, sx, sy) {
const poly = e.poly;
const [cx, cy] = e.iris.c;
const tx = cx + sx, ty = cy + sy;
let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
for (const [x, y] of poly) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
const m = R * 1.4 + 2;
x0 = Math.max(0, Math.floor(Math.min(x0, cx - m, tx - m)));
y0 = Math.max(0, Math.floor(Math.min(y0, cy - m, ty - m)));
x1 = Math.min(W, Math.ceil(Math.max(x1, cx + m, tx + m)));
y1 = Math.min(H, Math.ceil(Math.max(y1, cy + m, ty + m)));
const bw = x1 - x0, bh = y1 - y0;
if (bw < 4 || bh < 4) return;
const img = ctx.getImageData(x0, y0, bw, bh);
const src = new Uint8ClampedArray(img.data);
const dst = img.data;
const lp = poly.map(([x, y]) => [x - x0, y - y0]);
const icx = cx - x0, icy = cy - y0, itx = tx - x0, ity = ty - y0;
const base = 0.6, extra = Math.max(0.2, R * CFG.lidInset - base);
const inside = new Float32Array(bw * bh);
for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) inside[y * bw + x] = polyDist(x + 0.5, y + 0.5, lp) - base;
const sample = (x, y, out) => {
x = clamp(x - 0.5, 0, bw - 1.001); y = clamp(y - 0.5, 0, bh - 1.001);
const xi = x | 0, yi = y | 0, fx = x - xi, fy = y - yi;
const i00 = (yi * bw + xi) * 4, i10 = i00 + 4, i01 = i00 + bw * 4, i11 = i01 + 4;
for (let c = 0; c < 3; c++) {
out[c] = (src[i00 + c] * (1 - fx) + src[i10 + c] * fx) * (1 - fy) + (src[i01 + c] * (1 - fx) + src[i11 + c] * fx) * fy;
}
};
const insideAt = (x, y) => {
const xi = clamp(Math.round(x - 0.5), 0, bw - 1), yi = clamp(Math.round(y - 0.5), 0, bh - 1);
return inside[yi * bw + xi];
};
const hole = R * CFG.holeScale / CFG.irisScale;
const bg = new Float32Array(bw * bh * 3);
for (let i = 0, n = bw * bh; i < n; i++) { bg[i * 3] = src[i * 4]; bg[i * 3 + 1] = src[i * 4 + 1]; bg[i * 3 + 2] = src[i * 4 + 2]; }
const L = [0, 0, 0], Rt = [0, 0, 0], tmp = [0, 0, 0];
const holeMask = new Float32Array(bw * bh);
const scl = [0, 0, 0]; let sclN = 0;
for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
const d = Math.hypot(x + 0.5 - icx, y + 0.5 - icy);
if (inside[y * bw + x] > 0.5 && d > hole * 1.1 && d < hole * 2.2) { const i = (y * bw + x) * 4; scl[0] += src[i]; scl[1] += src[i + 1]; scl[2] += src[i + 2]; sclN++; }
}
if (sclN) for (let c = 0; c < 3; c++) scl[c] /= sclN;
for (let y = Math.max(0, Math.floor(icy - hole - 1)); y < Math.min(bh, Math.ceil(icy + hole + 1)); y++) {
const py = y + 0.5, dy = py - icy;
const half = Math.sqrt(Math.max(0, hole * hole - dy * dy)) + 1;
const span = rowSpan(py, lp, icx);
if (!span) continue;
const a = icx - half - 1, b = icx + half + 1;
const okL = a > span[0] + base + 0.5, okR = b < span[1] - base - 0.5;
const avg = (x, out, dir) => {
out[0] = out[1] = out[2] = 0; let n = 0;
for (let k = 0; k < 3; k++) for (let j = -1; j <= 1; j++) {
const xx = x + dir * k, yy = py + j;
if (insideAt(xx, yy) > 0.3) { sample(xx, yy, tmp); out[0] += tmp[0]; out[1] += tmp[1]; out[2] += tmp[2]; n++; }
}
if (!n) sample(x, py, out); else for (let c = 0; c < 3; c++) out[c] /= n;
};
if (okL) avg(a, L, -1);
if (okR) avg(b, Rt, 1);
if (!okL && !okR) { if (!sclN) continue; L.splice(0, 3, ...scl); Rt.splice(0, 3, ...scl); }
else if (!okL) L.splice(0, 3, ...Rt);
else if (!okR) Rt.splice(0, 3, ...L);
for (let x = Math.max(0, Math.floor(a)); x < Math.min(bw, Math.ceil(b)); x++) {
const px = x + 0.5;
const d = Math.hypot(px - icx, dy);
const w = smooth(hole + 0.8, hole - 0.8, d);
if (w <= 0) continue;
const t = clamp((px - a) / (b - a), 0, 1);
const i = (y * bw + x) * 3;
for (let c = 0; c < 3; c++) bg[i + c] = bg[i + c] * (1 - w) + (L[c] * (1 - t) + Rt[c] * t) * w;
holeMask[y * bw + x] = w;
}
}
for (let pass = 0; pass < 3; pass++) {
const cp = bg.slice();
for (let y = 1; y < bh - 1; y++) for (let x = 0; x < bw; x++) {
const w = holeMask[y * bw + x];
if (w <= 0) continue;
const i = (y * bw + x) * 3, u = i - bw * 3, d = i + bw * 3;
const wu = inside[(y - 1) * bw + x] > 0 ? 1 : 0, wd = inside[(y + 1) * bw + x] > 0 ? 1 : 0;
for (let c = 0; c < 3; c++) bg[i + c] = (cp[i + c] * 2 + cp[u + c] * wu + cp[d + c] * wd) / (2 + wu + wd);
}
}
const NB = Math.max(4, Math.ceil(R));
const bins = Array.from({ length: NB }, () => []);
for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
if (inside[y * bw + x] - extra <= 1) continue;
const r = Math.hypot(x + 0.5 - icx, y + 0.5 - icy);
if (r >= R) continue;
const i = (y * bw + x) * 4;
bins[Math.min(NB - 1, (r / R * NB) | 0)].push([src[i], src[i + 1], src[i + 2], src[i] + src[i + 1] + src[i + 2]]);
}
const prof = bins.map((b) => {
if (!b.length) return null;
b.sort((p, q) => p[3] - q[3]);
const n = Math.max(1, Math.ceil(b.length * 0.6)), m = [0, 0, 0];
for (let k = 0; k < n; k++) for (let c = 0; c < 3; c++) m[c] += b[k][c];
return m.map((v) => v / n);
});
for (let k = 0; k < NB; k++) if (!prof[k]) { // пустые кольца — от ближайших соседей
for (let s = 1; s < NB; s++) { const p = prof[k - s] || prof[k + s]; if (p) { prof[k] = p; break; } }
}
const profS = prof.map((p, k) => {
if (!p) return null;
const a = prof[k - 1] || p, b2 = prof[k + 1] || p;
return [0, 1, 2].map((c) => (a[c] + 2 * p[c] + b2[c]) / 4);
});
const profileAt = (r) => profS[Math.min(NB - 1, Math.max(0, (r / R * NB) | 0))];
const col = [0, 0, 0];
const rr = R + 1;
for (let y = Math.max(0, Math.floor(ity - rr)); y < Math.min(bh, Math.ceil(ity + rr)); y++) {
for (let x = Math.max(0, Math.floor(itx - rr)); x < Math.min(bw, Math.ceil(itx + rr)); x++) {
const px = x + 0.5, py = y + 0.5;
const d = Math.hypot(px - itx, py - ity);
const a = smooth(R + 0.7, R - 0.9, d) * smooth(-1.0, 1.5, insideAt(px, py) - extra);
if (a <= 0) continue;
const qx = px - (itx - icx), qy = py - (ity - icy);
const vis = smooth(0.3, 2.0, insideAt(qx, qy) - extra);
const pc = profileAt(Math.hypot(qx - icx, qy - icy));
if (vis > 0 && !pc) sample(qx, qy, col);
else if (vis > 0) { sample(qx, qy, col); for (let c = 0; c < 3; c++) col[c] = col[c] * vis + pc[c] * (1 - vis); }
else if (pc) { col[0] = pc[0]; col[1] = pc[1]; col[2] = pc[2]; }
else continue;
const i = (y * bw + x) * 3;
for (let c = 0; c < 3; c++) bg[i + c] = bg[i + c] * (1 - a) + col[c] * a;
}
}
for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
const k = smooth(0, 1.2, inside[y * bw + x]);
if (k <= 0) continue;
const i = y * bw + x;
for (let c = 0; c < 3; c++) dst[i * 4 + c] = src[i * 4 + c] * (1 - k) + bg[i * 3 + c] * k;
}
ctx.putImageData(img, x0, y0);
}
drawDebug(ctx) {
const d = this.debug;
if (!d) return;
ctx.save();
ctx.lineWidth = 1;
for (const e of d.eyes) {
ctx.strokeStyle = 'rgba(0,255,120,.8)';
ctx.beginPath(); e.poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); ctx.stroke();
ctx.strokeStyle = 'rgba(255,200,0,.9)';
ctx.beginPath(); ctx.arc(e.iris.c[0], e.iris.c[1], d.R, 0, Math.PI * 2); ctx.stroke();
ctx.strokeStyle = 'rgba(255,60,60,.9)';
ctx.beginPath(); ctx.moveTo(e.iris.c[0], e.iris.c[1]); ctx.lineTo(e.iris.c[0] + d.shift[0], e.iris.c[1] + d.shift[1]); ctx.stroke();
}
ctx.restore();
}
close() { this.landmarker?.close?.(); }
}
