// «Взгляд в камеру» v2: запоминаем глаза, когда человек смотрит в объектив,
// и в каждом кадре вставляем их на место настоящих (с веками), подгоняя по положению, размеру и свету.
const TASKS_VERSION = '0.10.14';
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}`;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';
const EYES = [{ corners: [33, 133], lids: [159, 145] }, { corners: [263, 362], lids: [386, 374] }];
const NOSE = 1, CHEEK_A = 234, CHEEK_B = 454, CHIN = 152;
const STORE_KEY = 'eyeSwapRef.v1';
export const CFG = {
  rx: 0.8,            // полуширина вставки (доля ширины глаза)
  ry: 0.5,            // полувысота вставки
  inner: 0.55,        // до этой доли радиуса — полная замена, дальше мягкий переход
  openLo: 0.55, openHi: 0.8,   // моргание: открытость относительно запомненной
  yawLo: 0.12, yawHi: 0.2,     // поворот головы относительно запомненного
  pitchLo: 0.12, pitchHi: 0.2, // наклон головы
  fadeSpeed: 6,
  calibMs: 2000,
};
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const smooth = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
class OneEuro {
  constructor(minCutoff = 1.5, beta = 0.2, dCutoff = 1.0) { Object.assign(this, { minCutoff, beta, dCutoff }); this.x = null; }
  static a(c, dt) { const r = 2 * Math.PI * c * dt; return r / (r + 1); }
  filter(x, dt) {
    if (this.x === null || !(dt > 0)) { this.x = x; this.dx = 0; return x; }
    const dx = (x - this.x) / dt;
    this.dx += OneEuro.a(this.dCutoff, dt) * (dx - this.dx);
    this.x += OneEuro.a(this.minCutoff + this.beta * Math.abs(this.dx), dt) * (x - this.x);
    return this.x;
  }
  reset() { this.x = null; }
}

export class EyeContact {
  static async create(opts = {}) {
    const vision = opts.vision ?? (await import(`${CDN}/vision_bundle.mjs`));
    const fileset = await vision.FilesetResolver.forVisionTasks(`${CDN}/wasm`);
    const make = (delegate) => vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate }, runningMode: 'VIDEO', numFaces: 1,
    });
    let lmk;
    try { lmk = await make('GPU'); } catch { lmk = await make('CPU'); }
    const e = new EyeContact(lmk);
    await e.loadRef();
    return e;
  }

  constructor(landmarker) {
    this.landmarker = landmarker;
    this.enabled = true;
    this.strength = 1;
    this.active = 0;
    this.status = { face: false, active: 0, reason: 'старт', calibrating: false, calibrated: false };
    this.lastTs = -1;
    this.f = new Map();
    this.ref = null;
    this.cal = null;
    this.gain = [[1, 1, 1], [1, 1, 1]];
    this.tmp = document.createElement('canvas');
    this.tctx = this.tmp.getContext('2d', { willReadFrequently: true });
  }

  // ---------- запоминание
  calibrate(ms = CFG.calibMs) {
    this.cal = { until: null, ms, best: null, score: -1e9 };
    this.status.calibrating = true;
  }
  resetCalibration() {
    this.ref = null; this.status.calibrated = false;
    try { localStorage.removeItem(STORE_KEY); } catch {}
  }
  async loadRef() {
    try {
      const s = JSON.parse(localStorage.getItem(STORE_KEY) || 'null');
      if (!s) return;
      const img = new Image(); img.src = s.img; await img.decode();
      const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
      c.getContext('2d').drawImage(img, 0, 0);
      this.ref = { ...s, img: c };
      this.status.calibrated = true;
    } catch { this.ref = null; }
  }
  saveRef() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify({ ...this.ref, img: this.ref.img.toDataURL('image/jpeg', 0.92) })); } catch {}
  }

  // ---------- геометрия лица
  geom(lm, W, H, dt, smoothPts) {
    const P = (i) => {
      const x = lm[i].x * W, y = lm[i].y * H;
      if (!smoothPts) return [x, y];
      let f = this.f.get(i);
      if (!f) { f = [new OneEuro(), new OneEuro()]; this.f.set(i, f); }
      return [f[0].filter(x, dt), f[1].filter(y, dt)];
    };
    const nose = P(NOSE), ca = P(CHEEK_A), cb = P(CHEEK_B), chin = P(CHIN);
    const l = Math.min(ca[0], cb[0]), r = Math.max(ca[0], cb[0]);
    const yaw = (nose[0] - l) / Math.max(1, r - l) - 0.5;
    const eyes = EYES.map((E) => {
      let [a, b] = E.corners.map(P);
      if (a[0] > b[0]) [a, b] = [b, a];
      const w = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const [up, lo] = E.lids.map(P);
      return { c: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], ang: Math.atan2(b[1] - a[1], b[0] - a[0]), w, open: Math.hypot(up[0] - lo[0], up[1] - lo[1]) / Math.max(1, w) };
    });
    const mid = [(eyes[0].c[0] + eyes[1].c[0]) / 2, (eyes[0].c[1] + eyes[1].c[1]) / 2];
    const faceH = Math.hypot(chin[0] - mid[0], chin[1] - mid[1]);
    const pitch = (nose[1] - mid[1]) / Math.max(1, chin[1] - mid[1]);
    return { eyes, yaw, pitch, faceH };
  }

  fade(target, dt) { const s = CFG.fadeSpeed * dt; this.active += clamp(target - this.active, -s, s); }

  process(canvas, ts) {
    if (ts <= this.lastTs) ts = this.lastTs + 1;
    const dt = this.lastTs < 0 ? 1 / 30 : (ts - this.lastTs) / 1000;
    this.lastTs = ts;
    const W = canvas.width, H = canvas.height;
    if (!this.enabled && !this.status.calibrating) { this.active = 0; this.status = { ...this.status, face: false, active: 0, reason: 'выключено' }; return; }
    let res;
    try { res = this.landmarker.detectForVideo(canvas, ts); } catch (e) { this.status.reason = 'ошибка'; return; }
    const lm = res.faceLandmarks?.[0];
    if (!lm) {
      this.fade(0, dt); this.f.clear();
      this.status = { ...this.status, face: false, active: this.active, reason: 'лица нет' };
      return;
    }
    const g = this.geom(lm, W, H, dt, true);

    // Запоминание: выбираем самый удачный кадр (глаза открыты, голова прямо)
    if (this.cal) {
      const c = this.cal;
      if (c.until === null) c.until = ts + c.ms;
      const raw = this.geom(lm, W, H, dt, false);
      const score = Math.min(raw.eyes[0].open, raw.eyes[1].open) - Math.abs(raw.yaw) * 0.5;
      if (score > c.score && Math.min(raw.eyes[0].open, raw.eyes[1].open) > 0.18) {
        c.score = score;
        const ew = Math.max(raw.eyes[0].w, raw.eyes[1].w);
        const xs = raw.eyes.map((e) => e.c[0]), ys = raw.eyes.map((e) => e.c[1]);
        const ox = Math.max(0, Math.floor(Math.min(...xs) - ew * 1.6)), oy = Math.max(0, Math.floor(Math.min(...ys) - ew * 1.2));
        const ox2 = Math.min(W, Math.ceil(Math.max(...xs) + ew * 1.6)), oy2 = Math.min(H, Math.ceil(Math.max(...ys) + ew * 1.2));
        const img = c.best?.img || document.createElement('canvas');
        img.width = ox2 - ox; img.height = oy2 - oy;
        img.getContext('2d').drawImage(canvas, ox, oy, img.width, img.height, 0, 0, img.width, img.height);
        c.best = { img, ox, oy, eyes: raw.eyes, yaw: raw.yaw, pitch: raw.pitch, faceH: raw.faceH };
      }
      if (ts >= c.until) {
        if (c.best) {
          this.ref = c.best; this.saveRef();
          this.status.calibrated = true; this.status.reason = 'глаза запомнены';
        } else this.status.reason = 'не получилось — глаз не было видно';
        this.cal = null; this.status.calibrating = false;
      }
    }

    const ref = this.ref;
    let safe = 1, reason = 'работает';
    const gate = (v, why) => { if (v < safe) { safe = v; if (v < 0.5) reason = why; } };
    if (!ref) { safe = 0; reason = 'нажми «Запомнить глаза»'; }
    else {
      const openRatio = Math.min(g.eyes[0].open / Math.max(0.05, ref.eyes[0].open), g.eyes[1].open / Math.max(0.05, ref.eyes[1].open));
      gate(smooth(CFG.openLo, CFG.openHi, openRatio), 'моргание');
      gate(1 - smooth(CFG.yawLo, CFG.yawHi, Math.abs(g.yaw - ref.yaw)), 'голова сильно повёрнута');
      gate(1 - smooth(CFG.pitchLo, CFG.pitchHi, Math.abs(g.pitch - ref.pitch)), 'голова сильно наклонена');
      gate(smooth(12, 20, Math.min(g.eyes[0].w, g.eyes[1].w)), 'лицо слишком далеко');
    }
    if (this.status.calibrating) { safe = 0; reason = 'смотри прямо в объектив'; }
    this.fade(safe * clamp(this.strength, 0, 1), dt);
    this.status = { ...this.status, face: true, active: this.active, reason };
    if (!ref || this.active < 0.02) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const sy = g.faceH / Math.max(1, ref.faceH);
    for (let k = 0; k < 2; k++) this.pasteEye(ctx, W, H, g.eyes[k], ref.eyes[k], ref, sy, k);
  }

  // Вставить запомненный глаз на место текущего
  pasteEye(ctx, W, H, cur, re, ref, sy, k) {
    const sx = cur.w / Math.max(1, re.w);
    const rx = CFG.rx * cur.w, ry = CFG.ry * re.w * sy;
    const R = Math.hypot(rx, ry);
    const bx = Math.max(0, Math.floor(cur.c[0] - R)), by = Math.max(0, Math.floor(cur.c[1] - R));
    const bw = Math.min(W, Math.ceil(cur.c[0] + R)) - bx, bh = Math.min(H, Math.ceil(cur.c[1] + R)) - by;
    if (bw < 6 || bh < 6) return;
    const t = this.tmp, tc = this.tctx;
    if (t.width < bw || t.height < bh) { t.width = Math.max(t.width, bw); t.height = Math.max(t.height, bh); }
    tc.setTransform(1, 0, 0, 1, 0, 0);
    tc.globalCompositeOperation = 'source-over';
    tc.clearRect(0, 0, t.width, t.height);
    // ref-кадр → текущий: центр глаза, поворот, масштаб (по ширине глаза и высоте лица)
    tc.translate(cur.c[0] - bx, cur.c[1] - by);
    tc.rotate(cur.ang);
    tc.scale(sx, sy);
    tc.rotate(-re.ang);
    tc.translate(-re.c[0], -re.c[1]);
    tc.imageSmoothingQuality = 'high';
    tc.drawImage(ref.img, ref.ox, ref.oy);
    tc.setTransform(1, 0, 0, 1, 0, 0);

    // Подгонка цвета/света: сравниваем кольцо кожи вокруг глаза (без самого глаза)
    const A = tc.getImageData(0, 0, bw, bh), B = ctx.getImageData(bx, by, bw, bh);
    const a = A.data, b = B.data;
    const cs = Math.cos(cur.ang), sn = Math.sin(cur.ang);
    const sa = [0, 0, 0], sb = [0, 0, 0]; let n = 0;
    const mask = new Float32Array(bw * bh);
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      const dx = x + bx + 0.5 - cur.c[0], dy = y + by + 0.5 - cur.c[1];
      const u = (dx * cs + dy * sn) / rx, v = (-dx * sn + dy * cs) / ry;
      const d = Math.sqrt(u * u + v * v);
      const i = (y * bw + x) * 4;
      mask[y * bw + x] = 1 - smooth(CFG.inner, 1, d);
      if (d > 0.75 && d < 1.0 && a[i + 3] > 250) { sa[0] += a[i]; sa[1] += a[i + 1]; sa[2] += a[i + 2]; sb[0] += b[i]; sb[1] += b[i + 1]; sb[2] += b[i + 2]; n++; }
    }
    const gk = this.gain[k];
    if (n > 20) for (let c = 0; c < 3; c++) gk[c] += (clamp(sb[c] / Math.max(1, sa[c]), 0.7, 1.4) - gk[c]) * 0.25;
    const al = this.active;
    for (let p = 0, i = 0; p < bw * bh; p++, i += 4) {
      const m = mask[p] * al * (a[i + 3] / 255);
      if (m <= 0) continue;
      b[i] = b[i] * (1 - m) + Math.min(255, a[i] * gk[0]) * m;
      b[i + 1] = b[i + 1] * (1 - m) + Math.min(255, a[i + 1] * gk[1]) * m;
      b[i + 2] = b[i + 2] * (1 - m) + Math.min(255, a[i + 2] * gk[2]) * m;
    }
    ctx.putImageData(B, bx, by);
  }

  close() { this.landmarker?.close?.(); }
}
