const MP_VERSION = '0.10.14';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODELS = {
multiclass: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite',
selfie: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite',
hand: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
};
const params = new URLSearchParams(location.search);
const TEST_MODE = params.has('test');           // для отладки: без MediaPipe и без микрофона
const IS_MOBILE = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || (navigator.maxTouchPoints > 1 && /Mac/.test(navigator.userAgent));
const WAVE_TIME = 1.7;      // за сколько волна проходит всё тело
const NOISE_DELAY = 0.30;   // насколько рваный край (разброс задержки)
const LIFE_MIN = 0.9, LIFE_MAX = 1.75;
const EFFECT_T = WAVE_TIME + NOISE_DELAY * 2 + LIFE_MAX + 0.05;
const COOLDOWN_MS = 1500;
const ASM_SKIP = 0.9;       // возврат начинается сразу с летящей пыли (без паузы)
const $ = (id) => document.getElementById(id);
const video = $('video');
const canvas = $('view');
const ui = {
start: $('start'), startBtn: $('startBtn'), startNote: $('startNote'),
hud: $('hud'), bar: $('bar'), toasts: $('toasts'),
overlay: $('overlay'), ovNum: $('ovNum'), ovMsg: $('ovMsg'), ovSub: $('ovSub'),
earPill: $('earPill'), earTxt: $('earTxt'), meterFill: $('meterFill'),
handPill: $('handPill'), handTxt: $('handTxt'), fps: $('fps'),
rec: $('rec'), recTime: $('recTime'),
snapBtn: $('snapBtn'), bgBtn: $('bgBtn'), mirrorBtn: $('mirrorBtn'), recBtn: $('recBtn'), recLabel: $('recLabel'),
fsBtn: $('fsBtn'), cleanBtn: $('cleanBtn'), sens: $('sens'), sensOut: $('sensOut'),
needHand: $('needHand'), fxSound: $('fxSound'), cleanHint: $('cleanHint'),
};
const store = {
get(k, d) { try { const v = localStorage.getItem('snap.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
set(k, v) { try { localStorage.setItem('snap.' + k, JSON.stringify(v)); } catch {} },
};
const settings = {
mirror: store.get('mirror', true),
sens: store.get('sens', 6),
needHand: store.get('needHand', true),
fxSound: store.get('fxSound', true),
};
function toast(text, kind = '', ms = 3200) {
const el = document.createElement('div');
el.className = 'toast ' + kind;
el.textContent = text;
ui.toasts.appendChild(el);
while (ui.toasts.children.length > 3) ui.toasts.firstChild.remove();
setTimeout(() => { el.style.transition = 'opacity .4s'; el.style.opacity = '0'; setTimeout(() => el.remove(), 450); }, ms);
}
let lastToastKey = '', lastToastAt = 0;
function toastOnce(key, text, kind, ms) {
const now = performance.now();
if (key === lastToastKey && now - lastToastAt < 4000) return;
lastToastKey = key; lastToastAt = now;
toast(text, kind, ms);
}
function overlay(num, msg, sub) {
if (num == null && !msg) { ui.overlay.classList.remove('show'); return; }
ui.ovNum.textContent = num ?? '';
ui.ovMsg.textContent = msg ?? '';
ui.ovSub.textContent = sub ?? '';
ui.overlay.classList.add('show');
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const app = {
started: false,
vw: 0, vh: 0,               // размер видео
sw: 0, sh: 0,               // размер для сегментации
mode: 'visible',            // visible | dissolving | hidden | assembling
effectStart: 0,
origin: [0.5, 0.4],
wind: [0.4, -0.1],
spread: 1.2,
lastTriggerAt: -1e9,
bgReady: false,
capturing: false,
segmenter: null, segKind: 'none', segPersonFromBg: true,
hands: null,
audioCtx: null, micStream: null, recDest: null, fxBus: null, detector: null,
handHistory: [],            // {t, hands:[{r, pt:[x,y]}]}
pendingSnap: null,
frame: 0,
};
ui.startBtn.addEventListener('click', start);
async function start() {
if (app.started) return;
ui.startBtn.disabled = true;
ui.startNote.classList.remove('err');
try {
if (!window.isSecureContext) throw new Error('Нужно открыть сайт по https:// — иначе браузер не даст камеру.');
if (!navigator.mediaDevices?.getUserMedia) throw new Error('Этот браузер не умеет работать с камерой. Открой в Chrome, Edge или Safari.');
ui.startNote.textContent = 'Разреши доступ к камере и микрофону…';
const AC = window.AudioContext || window.webkitAudioContext;
app.audioCtx = new AC({ latencyHint: 'interactive' });
app.audioCtx.resume?.();
const stream = await navigator.mediaDevices.getUserMedia({
video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
audio: TEST_MODE ? false : { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
}).catch((e) => {
if (e && (e.name === 'NotAllowedError' || e.name === 'SecurityError')) throw new Error('Доступ к камере/микрофону запрещён. Разреши его в настройках сайта (значок замка в адресной строке) и обнови страницу.');
if (e && e.name === 'NotFoundError') throw new Error('Камера или микрофон не найдены.');
if (e && e.name === 'NotReadableError') throw new Error('Камера занята другой программой (Zoom, OBS?). Закрой её и обнови страницу.');
throw e;
});
app.micStream = stream;
video.srcObject = stream;
await video.play().catch(() => {});
await waitForVideo();
ui.startNote.textContent = 'Загружаю нейросети (первый раз — до 20 секунд)…';
setupGL();
setupAudioGraph(stream);
await loadModels();
app.started = true;
ui.start.style.transition = 'opacity .5s'; ui.start.style.opacity = '0';
setTimeout(() => ui.start.remove(), 520);
ui.hud.hidden = false; ui.bar.hidden = false;
applySettingsUI();
requestWakeLock();
requestAnimationFrame(loop);
await sleep(400);
captureBackground();
} catch (err) {
console.error(err);
ui.startNote.textContent = (err && err.message) || String(err);
ui.startNote.classList.add('err');
ui.startBtn.disabled = false;
}
}
function waitForVideo() {
return new Promise((resolve, reject) => {
const t0 = performance.now();
(function check() {
if (video.readyState >= 2 && video.videoWidth > 0) return resolve();
if (performance.now() - t0 > 10000) return reject(new Error('Камера не отдаёт картинку. Проверь, не занята ли она другой программой.'));
requestAnimationFrame(check);
})();
});
}
async function loadModels() {
if (TEST_MODE) { app.segKind = 'fallback'; return; }
let vision;
for (const url of [`${MP_BASE}/vision_bundle.mjs`, `${MP_BASE}/+esm`]) {
try { vision = await import(url); if (vision?.FilesetResolver) break; } catch (e) { console.warn('MediaPipe import failed', url, e); }
}
if (!vision?.FilesetResolver) {
app.segKind = 'fallback';
toast('Не удалось загрузить нейросети (нет интернета?). Работаю в упрощённом режиме.', 'warn', 6000);
return;
}
const { FilesetResolver, ImageSegmenter, HandLandmarker } = vision;
let fileset;
try { fileset = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`); } catch (e) {
console.warn(e); app.segKind = 'fallback';
toast('Не удалось загрузить нейросети — упрощённый режим.', 'warn', 6000); return;
}
const makeSeg = async (kind, delegate) => ImageSegmenter.createFromOptions(fileset, {
baseOptions: { modelAssetPath: MODELS[kind], delegate },
runningMode: 'VIDEO', outputCategoryMask: false, outputConfidenceMasks: true,
});
const makeHands = (delegate) => HandLandmarker.createFromOptions(fileset, {
baseOptions: { modelAssetPath: MODELS.hand, delegate },
runningMode: 'VIDEO', numHands: 2,
minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
});
const withFallback = async (fn) => { try { return await fn('GPU'); } catch (e) { console.warn('GPU delegate failed, CPU', e); return fn('CPU'); } };
const firstKind = IS_MOBILE ? 'selfie' : 'multiclass';
const [seg, hands] = await Promise.allSettled([
withFallback((d) => makeSeg(firstKind, d)),
withFallback((d) => makeHands(d)),
]);
if (seg.status === 'fulfilled') {
setSegmenter(seg.value, firstKind);
app._makeSeg = (kind) => withFallback((d) => makeSeg(kind, d));
} else {
console.warn(seg.reason);
app.segKind = 'fallback';
toast('Сегментация недоступна — упрощённый режим.', 'warn', 6000);
}
if (hands.status === 'fulfilled') app.hands = hands.value;
else {
console.warn(hands.reason);
toast('Распознавание руки недоступно — срабатываю только по звуку.', 'warn', 6000);
}
}
function setSegmenter(seg, kind) {
app.segmenter = seg;
app.segKind = kind;
let labels = [];
try { labels = seg.getLabels() || []; } catch {}
app.segPersonFromBg = kind === 'multiclass' || (labels[0] && /background/i.test(labels[0]));
app._segLabels = labels;
app.segTimes = [];
}
let segCanvas, segCtx, conf, maskS, tmpA, D2, D3, maskU8, segRGB, bgLow, gain = [1, 1, 1];
function allocSeg() {
const aspect = app.vw / app.vh;
const base = IS_MOBILE ? 256 : 320;
if (aspect >= 1) { app.sw = base; app.sh = Math.round(base / aspect); } else { app.sh = base; app.sw = Math.round(base * aspect); }
const N = app.sw * app.sh;
segCanvas = document.createElement('canvas');
segCanvas.width = app.sw; segCanvas.height = app.sh;
segCtx = segCanvas.getContext('2d', { willReadFrequently: true });
conf = new Float32Array(N); maskS = new Float32Array(N); tmpA = new Float32Array(N);
D2 = new Float32Array(N); D3 = new Float32Array(N);
maskU8 = new Uint8Array(N * 4);
bgLow = null;
}
function maxFilter(src, dst, r, w, h) {
for (let y = 0; y < h; y++) {
const row = y * w;
for (let x = 0; x < w; x++) {
let m = 0; const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r);
for (let k = x0; k <= x1; k++) { const v = src[row + k]; if (v > m) m = v; }
tmpA[row + x] = m;
}
}
for (let x = 0; x < w; x++) {
for (let y = 0; y < h; y++) {
let m = 0; const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
for (let k = y0; k <= y1; k++) { const v = tmpA[k * w + x]; if (v > m) m = v; }
dst[y * w + x] = m;
}
}
}
function minFilter(src, dst, r, w, h) {
for (let y = 0; y < h; y++) {
const row = y * w;
for (let x = 0; x < w; x++) {
let m = 1; const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r);
for (let k = x0; k <= x1; k++) { const v = src[row + k]; if (v < m) m = v; }
tmpA[row + x] = m;
}
}
for (let x = 0; x < w; x++) {
for (let y = 0; y < h; y++) {
let m = 1; const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
for (let k = y0; k <= y1; k++) { const v = tmpA[k * w + x]; if (v < m) m = v; }
dst[y * w + x] = m;
}
}
}
function readMaskInto(mpMask, invert) {
const data = mpMask.getAsFloat32Array();
const mw = mpMask.width, mh = mpMask.height;
const w = app.sw, h = app.sh;
if (mw === w && mh === h) {
if (invert) for (let i = 0; i < data.length; i++) conf[i] = 1 - data[i];
else conf.set(data);
} else {
for (let y = 0; y < h; y++) {
const sy = Math.min(mh - 1, Math.floor((y + 0.5) * mh / h));
for (let x = 0; x < w; x++) {
const sx = Math.min(mw - 1, Math.floor((x + 0.5) * mw / w));
const v = data[sy * mw + sx];
conf[y * w + x] = invert ? 1 - v : v;
}
}
}
}
let segTs = 0, handTs = 0;
let invVotes = 0, invFrames = 0;
function checkInverted() {
if (invFrames > 90) return;
invFrames++;
const w = app.sw, h = app.sh, pw = Math.max(3, Math.round(w * 0.06)), ph = Math.max(3, Math.round(h * 0.08));
let s = 0, n = 0;
for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) { s += conf[y * w + x] + conf[y * w + (w - 1 - x)]; n += 2; }
if (s / n > 0.75) invVotes++;
if (invFrames === 90 && invVotes > 70) { app.segPersonFromBg = !app.segPersonFromBg; console.warn('Маска перевёрнута — исправляю'); }
}
function runSegmentation(now) {
const w = app.sw, h = app.sh;
if (app.segmenter) {
const ts = Math.max(segTs + 1, Math.round(now)); segTs = ts;
const t0 = performance.now();
try {
app.segmenter.segmentForVideo(segCanvas, ts, (res) => {
const masks = res.confidenceMasks;
if (masks && masks.length) {
const invert = masks.length > 1 ? true : app.segPersonFromBg;
readMaskInto(masks[0], invert);
if (masks.length === 1) checkInverted();
}
});
} catch (e) { console.warn('seg error', e); }
const dt = performance.now() - t0;
if (app.segKind === 'multiclass' && app._makeSeg && !app._switching) {
app.segTimes.push(dt);
if (app.segTimes.length > 45) {
app.segTimes.shift();
const avg = app.segTimes.reduce((a, b) => a + b, 0) / app.segTimes.length;
if (avg > 34) {
app._switching = true;
app._makeSeg('selfie').then((s) => { const old = app.segmenter; setSegmenter(s, 'selfie'); try { old.close(); } catch {} })
.catch((e) => console.warn(e));
}
}
}
} else if (app.segKind === 'fallback') {
if (!bgLow) { conf.fill(0); return; }
const px = segRGB;
for (let i = 0, j = 0; i < w * h; i++, j += 4) {
const d = (Math.abs(px[j] - bgLow[i * 3] * gain[0]) + Math.abs(px[j + 1] - bgLow[i * 3 + 1] * gain[1]) + Math.abs(px[j + 2] - bgLow[i * 3 + 2] * gain[2])) / 765;
const t = clamp((d - 0.045) / 0.09, 0, 1);
conf[i] = t * t * (3 - 2 * t);
}
minFilter(conf, D2, 1, w, h);
maxFilter(D2, conf, 2, w, h);
}
}
function processMask() {
const w = app.sw, h = app.sh, N = w * h;
const hiddenish = app.mode !== 'visible';
const up = hiddenish ? 0.9 : 0.7, down = hiddenish ? 0.22 : 0.45;
for (let i = 0; i < N; i++) {
const c = conf[i], s = maskS[i];
maskS[i] = s + (c - s) * (c > s ? up : down);
}
const r2 = Math.max(2, Math.round(w * 0.014));
const r3 = Math.max(6, Math.round(w * 0.045));
maxFilter(maskS, D2, r2, w, h);
maxFilter(D2, D3, r3 - r2, w, h);
for (let i = 0, j = 0; i < N; i++, j += 4) {
maskU8[j] = maskS[i] * 255;
maskU8[j + 1] = D2[i] * 255;
maskU8[j + 2] = D3[i] * 255;
maskU8[j + 3] = 255;
}
if (bgLow && segRGB) {
let sv0 = 0, sv1 = 0, sv2 = 0, sb0 = 0, sb1 = 0, sb2 = 0, n = 0;
for (let i = 0, j = 0; i < N; i++, j += 4) {
const z = D3[i];
if (z < 0.02) {
const k = i * 3;
bgLow[k] += (segRGB[j] - bgLow[k]) * 0.05;
bgLow[k + 1] += (segRGB[j + 1] - bgLow[k + 1]) * 0.05;
bgLow[k + 2] += (segRGB[j + 2] - bgLow[k + 2]) * 0.05;
} else if (D2[i] < 0.05 && (i & 1) === 0) {
const k = i * 3;
sv0 += segRGB[j]; sv1 += segRGB[j + 1]; sv2 += segRGB[j + 2];
sb0 += bgLow[k]; sb1 += bgLow[k + 1]; sb2 += bgLow[k + 2]; n++;
}
}
if (n > 200 && sb0 > 0 && sb1 > 0 && sb2 > 0) {
const g = [sv0 / sb0, sv1 / sb1, sv2 / sb2];
for (let c = 0; c < 3; c++) gain[c] += (clamp(g[c], 0.75, 1.35) - gain[c]) * 0.08;
} else {
for (let c = 0; c < 3; c++) gain[c] += (1 - gain[c]) * 0.05;
}
}
}
function runHands(now) {
if (!app.hands) return;
const ts = Math.max(handTs + 1, Math.round(now)); handTs = ts;
let res;
try { res = app.hands.detectForVideo(video, ts); } catch (e) { console.warn('hand error', e); return; }
const list = [];
for (const lm of res.landmarks || []) {
const d = (a, b) => Math.hypot((lm[a].x - lm[b].x) * app.vw / app.vh, lm[a].y - lm[b].y);
const palm = Math.max(1e-4, d(0, 9));
const r = d(4, 12) / palm;              // большой ↔ средний палец
list.push({ r, pt: [(lm[4].x + lm[12].x) / 2, (lm[4].y + lm[12].y) / 2], wrist: [lm[0].x, lm[0].y] });
}
app.handHistory.push({ t: now, hands: list });
while (app.handHistory.length && now - app.handHistory[0].t > 2500) app.handHistory.shift();
}
function handVisibleRecently(now, ms = 350) {
for (let i = app.handHistory.length - 1; i >= 0; i--) {
const h = app.handHistory[i];
if (now - h.t > ms) break;
if (h.hands.length) return true;
}
return false;
}
function findSnapHand(onset) {
let best = null, seen = false;
for (const e of app.handHistory) {
if (e.t < onset - 950 || e.t > onset + 220) continue;
for (const h of e.hands) {
seen = true;
if (h.r < 0.55 && (!best || h.r < best.r)) best = h;
}
}
return { seen, best };
}
const WORKLET_SRC = `
class SnapDetector extends AudioWorkletProcessor {
constructor() {
super();
const fc = 2400, Q = Math.SQRT1_2, w0 = 2 * Math.PI * fc / sampleRate, al = Math.sin(w0) / (2 * Q), cw = Math.cos(w0);
const a0 = 1 + al;
this.b0 = (1 + cw) / 2 / a0; this.b1 = -(1 + cw) / a0; this.b2 = (1 + cw) / 2 / a0;
this.a1 = -2 * cw / a0; this.a2 = (1 - al) / a0;
this.x1 = this.x2 = this.y1 = this.y2 = 0;
this.hist = new Float32Array(16); this.hi = 0;
this.floor = 1e-7; this.sens = 6; this.block = 0; this.refractory = 0;
this.cand = null; this.lvl = 0; this.lvlN = 0; this.warm = 0;
this.port.onmessage = (e) => { if (e.data && e.data.sens) this.sens = e.data.sens; };
}
process(inputs) {
const ch = inputs[0] && inputs[0][0];
if (!ch) return true;
let hp = 0, full = 0;
const { b0, b1, b2, a1, a2 } = this;
let { x1, x2, y1, y2 } = this;
for (let i = 0; i < ch.length; i++) {
const x = ch[i];
const y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
x2 = x1; x1 = x; y2 = y1; y1 = y;
hp += y * y; full += x * x;
}
this.x1 = x1; this.x2 = x2; this.y1 = y1; this.y2 = y2;
hp /= ch.length; full /= ch.length;
this.block++;
let ref = 0; for (let k = 4; k < 16; k++) ref += this.hist[(this.hi - k + 32) % 16]; ref /= 12;
this.hist[this.hi] = hp; this.hi = (this.hi + 1) % 16;
if (hp < this.floor) this.floor = this.floor * 0.9 + hp * 0.1; else this.floor = this.floor * 0.9993 + hp * 0.0007;
this.floor = Math.max(this.floor, 1e-10);
this.warm++;
const s = this.sens;
const ratio = Math.pow(10, (33 - s * 2.6) / 10);           // во сколько раз громче фона
const absMin = Math.pow(10, (-38 - s * 2.4) / 10);         // абсолютный минимум энергии
const blockSec = ch.length / sampleRate;
if (this.refractory > 0) this.refractory -= blockSec;
if (this.cand) {
const c = this.cand; c.n++;
if (c.n <= 6) { if (hp > c.peak) c.peak = hp; }
else if (c.n >= 22 && c.n <= 34) { c.tail += hp; c.tn++; }
if (c.n > 34) {
const tail = c.tail / Math.max(1, c.tn);
if (tail < c.peak * 0.06) {
this.port.postMessage({ type: 'snap', ago: c.n * blockSec, db: 10 * Math.log10(c.peak / this.floor) });
this.refractory = 0.25;
} else {
this.port.postMessage({ type: 'reject', why: 'long' });
}
this.cand = null;
}
} else if (this.refractory <= 0 && this.warm > 150) {
const base = Math.max(ref, this.floor);
if (hp > this.floor * ratio && hp > absMin && hp > base * 7 && hp / (full + 1e-12) > 0.10) {
this.cand = { peak: hp, n: 0, tail: 0, tn: 0 };
}
}
const lvl = 10 * Math.log10((hp + 1e-12) / (this.floor * ratio));
if (lvl > this.lvl) this.lvl = lvl;
if (++this.lvlN >= 18) { this.port.postMessage({ type: 'level', v: this.lvl }); this.lvl = -99; this.lvlN = 0; }
return true;
}
}
registerProcessor('snap-detector', SnapDetector);
`;
async function setupAudioGraph(stream) {
const ctx = app.audioCtx;
app.recDest = ctx.createMediaStreamDestination();
app.fxBus = ctx.createGain();
app.fxBus.gain.value = 0.9;
app.fxBus.connect(app.recDest);
app.fxOut = ctx.createGain();
app.fxOut.gain.value = settings.fxSound ? 1 : 0;
app.fxBus.connect(app.fxOut);
app.fxOut.connect(ctx.destination);
if (!stream.getAudioTracks().length) { setEar(false, 'нет микрофона'); return; }
const mic = ctx.createMediaStreamSource(stream);
mic.connect(app.recDest);           // микрофон идёт в запись видео (но не в колонки)
try {
const url = URL.createObjectURL(new Blob([WORKLET_SRC], { type: 'application/javascript' }));
await ctx.audioWorklet.addModule(url);
const node = new AudioWorkletNode(ctx, 'snap-detector', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
const mute = ctx.createGain(); mute.gain.value = 0;
mic.connect(node); node.connect(mute); mute.connect(ctx.destination);
node.port.postMessage({ sens: settings.sens });
node.port.onmessage = onDetectorMessage;
app.detector = node;
setEar(true);
} catch (e) {
console.warn('AudioWorklet недоступен', e);
setEar(false, 'не слышу');
toast('Микрофон не работает в этом браузере — щёлкай пробелом или тапом.', 'warn', 6000);
}
}
function setEar(on, txt) {
ui.earPill.classList.toggle('on', on);
ui.earTxt.textContent = txt || 'слышу щелчки';
}
let earFlashT = 0;
function onDetectorMessage(e) {
const m = e.data;
if (m.type === 'level') {
const pct = clamp((m.v + 25) / 30, 0, 1) * 100;
ui.meterFill.style.width = pct.toFixed(0) + '%';
} else if (m.type === 'snap') {
const onset = performance.now() - m.ago * 1000 - 15;
ui.earPill.classList.add('flash');
clearTimeout(earFlashT); earFlashT = setTimeout(() => ui.earPill.classList.remove('flash'), 350);
onSnapHeard(onset);
}
}
function canTrigger(now) {
if (!app.bgReady || app.capturing) return false;
if (app.mode === 'dissolving' || app.mode === 'assembling') return false;
return now - app.lastTriggerAt > COOLDOWN_MS;
}
function onSnapHeard(onset) {
if (app.pendingSnap) return;
const now = performance.now();
if (!app.bgReady) { toastOnce('nobg', 'Сначала нужно снять фон.', 'warn'); return; }
if (!canTrigger(now)) return;
const wait = Math.max(0, onset + 230 - now);
app.pendingSnap = setTimeout(() => {
app.pendingSnap = null;
const t = performance.now();
if (!canTrigger(t)) return;
const useHand = settings.needHand && app.hands;
const { seen, best } = findSnapHand(onset);
if (useHand) {
if (!seen) { toastOnce('nohand', 'Слышу щелчок, но не вижу руку ✋ — подними руку в кадр.', 'warn'); return; }
if (!best) { toastOnce('nopinch', 'Рука видна, но щелчка пальцами не заметил — попробуй ещё раз.', 'warn'); return; }
}
trigger(best ? best.pt : null);
}, wait);
}
function trigger(origin) {
const now = performance.now();
if (!app.bgReady) { toastOnce('nobg', 'Сначала нужно снять фон.', 'warn'); return; }
if (!canTrigger(now)) return;
if (app.mode === 'visible') startEffect('dissolving', origin);
else if (app.mode === 'hidden') startEffect('assembling', origin);
}
function bodyStats() {
const w = app.sw, h = app.sh;
let n = 0, sx = 0, sy = 0, minY = 1;
for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
if (maskS[y * w + x] > 0.5) { n++; sx += x; sy += y; if (y / h < minY) minY = y / h; }
}
if (!n) return null;
return { area: n / (w * h), cx: (sx / n + 0.5) / w, cy: (sy / n + 0.5) / h, top: minY };
}
function startEffect(kind, origin) {
const st = bodyStats();
if (!st || st.area < 0.004) { toastOnce('noperson', 'Не вижу человека в кадре.', 'warn'); return; }
const aspect = app.vw / app.vh;
if (!origin) origin = [st.cx + (Math.random() < 0.5 ? -1 : 1) * 0.08, st.top + (st.cy - st.top) * 0.6];
let maxD = 0.05;
const w = app.sw, h = app.sh;
for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 2) {
if (maskS[y * w + x] > 0.5) {
const dx = ((x + 0.5) / w - origin[0]) * aspect, dy = (y + 0.5) / h - origin[1];
const d = Math.hypot(dx, dy); if (d > maxD) maxD = d;
}
}
app.spread = WAVE_TIME / maxD;
app.origin = origin;
let dir = Math.sign(st.cx - origin[0]);
if (Math.abs(st.cx - origin[0]) < 0.02) dir = Math.random() < 0.5 ? -1 : 1;
app.wind = [dir * 0.42, -0.16];
gl_snapshot();
app.mode = kind;
app.effectStart = performance.now();
app.lastTriggerAt = app.effectStart;
playDustSound(kind === 'assembling');
}
let noiseBuf = null;
function playDustSound(reverse) {
const ctx = app.audioCtx; if (!ctx || !app.fxBus) return;
if (!noiseBuf) {
noiseBuf = ctx.createBuffer(1, ctx.sampleRate * 4, ctx.sampleRate);
const d = noiseBuf.getChannelData(0);
let b = 0;
for (let i = 0; i < d.length; i++) {
const wN = Math.random() * 2 - 1;
b = 0.97 * b + 0.03 * wN;
d[i] = wN * 0.6 + b * 3 + (Math.random() < 0.0015 ? (Math.random() * 2 - 1) * 2.2 : 0); // шорох + треск
}
}
const t = ctx.currentTime + 0.01, dur = reverse ? EFFECT_T - ASM_SKIP - 0.1 : Math.min(3.4, EFFECT_T - 0.2);
const src = ctx.createBufferSource(); src.buffer = noiseBuf;
const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.Q.value = 0.8;
const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 6000;
const g = ctx.createGain();
src.connect(bp); bp.connect(lp); lp.connect(g); g.connect(app.fxBus);
const peak = 0.33;
if (!reverse) {
bp.frequency.setValueAtTime(2600, t); bp.frequency.exponentialRampToValueAtTime(500, t + dur);
g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(peak, t + 0.25);
g.gain.setValueAtTime(peak, t + 0.9); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
} else {
bp.frequency.setValueAtTime(500, t); bp.frequency.exponentialRampToValueAtTime(2600, t + dur);
g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(peak, t + dur - 0.9);
g.gain.setValueAtTime(peak, t + dur - 0.25); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
}
src.start(t, Math.random() * 0.5); src.stop(t + dur + 0.05);
}
async function captureBackground() {
if (app.capturing) return;
if (app.mode !== 'visible') { app.mode = 'visible'; }
app.capturing = true;
ui.bgBtn.disabled = true;
try {
for (let i = 3; i >= 1; i--) {
overlay(i, 'Отойди из кадра', 'Сниму пустую комнату — это нужно, чтобы ты мог исчезнуть');
await sleep(1000);
}
overlay('📸', 'Снимаю фон… не возвращайся', '');
const W = Math.min(app.vw, 1280), H = Math.round(W * app.vh / app.vw);
const c = document.createElement('canvas'); c.width = W; c.height = H;
const cx = c.getContext('2d', { willReadFrequently: true });
const frames = [], masks = [];
const NF = 13;
let lastVT = -1;
for (let k = 0; k < NF; k++) {
const t0 = performance.now();
while (video.currentTime === lastVT && performance.now() - t0 < 200) await sleep(10);
lastVT = video.currentTime;
cx.drawImage(video, 0, 0, W, H);
frames.push(cx.getImageData(0, 0, W, H).data);
masks.push(app.segKind === 'fallback' ? null : Float32Array.from(maskS));
await sleep(40);
}
overlay('⏳', 'Обрабатываю…', '');
await sleep(20);
const out = new Uint8Array(W * H * 4);
const vals = new Float32Array(NF);
const sw = app.sw, sh = app.sh;
let holes = 0;
const holeMap = new Uint8Array(sw * sh);
const usable = new Uint8Array(NF);
for (let y = 0; y < H; y++) {
const my = Math.min(sh - 1, Math.floor(y * sh / H));
for (let x = 0; x < W; x++) {
const mx = Math.min(sw - 1, Math.floor(x * sw / W));
const mi = my * sw + mx;
let nu = 0;
for (let f = 0; f < NF; f++) { const ok = !masks[f] || masks[f][mi] < 0.25; usable[f] = ok ? 1 : 0; if (ok) nu++; }
const useAll = nu < 3;
if (useAll && !holeMap[mi]) { holeMap[mi] = 1; holes++; }
const p = (y * W + x) * 4;
for (let ch = 0; ch < 3; ch++) {
let n = 0;
for (let f = 0; f < NF; f++) {
if (!useAll && !usable[f]) continue;
const v = frames[f][p + ch];
let j = n++;
while (j > 0 && vals[j - 1] > v) { vals[j] = vals[j - 1]; j--; }
vals[j] = v;
}
out[p + ch] = vals[n >> 1];
}
out[p + 3] = 255;
}
if ((y & 63) === 0) await sleep(0);
}
gl_uploadBackground(out, W, H);
bgLow = new Float32Array(sw * sh * 3);
for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
const sx = Math.min(W - 1, Math.floor((x + 0.5) * W / sw)), sy = Math.min(H - 1, Math.floor((y + 0.5) * H / sh));
const p = (sy * W + sx) * 4, k = (y * sw + x) * 3;
bgLow[k] = out[p]; bgLow[k + 1] = out[p + 1]; bgLow[k + 2] = out[p + 2];
}
gain = [1, 1, 1];
app.bgReady = true;
overlay('✅', 'Готово! Возвращайся и щёлкни пальцами', '');
await sleep(1300);
overlay(null);
if (holes / (sw * sh) > 0.01) toast('Ты был в кадре, когда снимался фон. Он дочистится сам, когда ты подвигаешься, — или нажми «Переснять фон».', 'warn', 7000);
} finally {
app.capturing = false;
ui.bgBtn.disabled = false;
}
}
let gl, progComp, progBg, progPart, texVideo, texSnap, texMask, texSnapMask, bgTex = [], bgFbo = [], bgIdx = 0, bgW = 0, bgH = 0;
let partVao, partCount = 0, quadVao, maxPointSize = 64;
const GLSL_NOISE = `
uniform vec2 uOrigin; uniform float uAspect; uniform float uSpread;
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.0-2.0*f);
return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y); }
float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++){ s += a*vnoise(p); p = p*2.03 + vec2(17.1, 9.3); a *= 0.5; } return s; }
float delayAt(vec2 uv){
vec2 d = (uv - uOrigin) * vec2(uAspect, 1.0);
float n = fbm(uv * vec2(uAspect, 1.0) * 7.0) - 0.47;
return length(d) * uSpread + n * ${(NOISE_DELAY * 2).toFixed(3)} + ${NOISE_DELAY.toFixed(3)};
}`;
const VS_QUAD = `#version 300 es
in vec2 aPos; uniform float uMirror; uniform float uFbo; out vec2 vUv;
void main(){
vUv = vec2(aPos.x*0.5+0.5, uFbo > 0.5 ? aPos.y*0.5+0.5 : 0.5-aPos.y*0.5);
if (uMirror > 0.5) vUv.x = 1.0 - vUv.x;
gl_Position = vec4(aPos, 0.0, 1.0);
}`;
const FS_COMP = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 outColor;
uniform sampler2D uVideo, uBg, uMask;
uniform int uMode; uniform float uTau, uDil; uniform vec3 uGain; uniform float uHasBg;
${GLSL_NOISE}
void main(){
vec2 uv = vUv;
vec3 v = texture(uVideo, uv).rgb;
if (uHasBg < 0.5 || uMode == 0) { outColor = vec4(v, 1.0); return; }
vec4 mk = texture(uMask, uv);
float zone = mk.b;
vec3 b = texture(uBg, uv).rgb * mix(vec3(1.0), uGain, smoothstep(0.0, 0.3, zone));
float m = mix(mk.r, mk.g, uDil);
m = smoothstep(0.22, 0.58, m);
float diff = length(v - b);
float halo = smoothstep(0.02, 0.25, zone) * smoothstep(0.09, 0.2, diff) * uDil;
float rep = clamp(max(m, halo), 0.0, 1.0);
float hide = 1.0, x = 1.0;
if (uMode == 1 || uMode == 3) { float d = delayAt(uv); x = uTau - d; hide = smoothstep(-0.035, 0.035, x); }
vec3 col = mix(v, b, rep * hide);
if (uMode == 1 || uMode == 3) {
float body = smoothstep(0.35, 0.6, mk.r) * (1.0 - hide);
float pre = smoothstep(-0.32, 0.0, x);
col = mix(col, col * vec3(0.42, 0.36, 0.32), pre * body * 0.75);
float edge = exp(-pow((x + 0.03) / 0.045, 2.0)) * body;
float flick = 0.75 + 0.5 * vnoise(uv * 90.0 + uTau * 7.0);
col += vec3(1.0, 0.46, 0.14) * edge * 0.95 * flick;
}
outColor = vec4(col, 1.0);
}`;
const FS_BG = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 outColor;
uniform sampler2D uVideo, uBg, uMask; uniform float uRate, uSeed;
float h(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
void main(){
vec3 b = texture(uBg, vUv).rgb;
vec3 v = texture(uVideo, vUv).rgb;
float z = texture(uMask, vUv).b;
float w = uRate * (1.0 - smoothstep(0.004, 0.05, z));
vec3 o = mix(b, v, w);
o += (vec3(h(gl_FragCoord.xy + uSeed), h(gl_FragCoord.yx + uSeed*1.7), h(gl_FragCoord.xy*1.3 + uSeed*2.3)) - 0.5) / 255.0 * step(1e-5, w);
outColor = vec4(o, 1.0);
}`;
const VS_PART = `#version 300 es
precision highp float;
in vec2 aHome; in vec4 aRand;
uniform sampler2D uSnap, uSnapMask;
uniform float uTau, uMirror, uCell, uMaxPt; uniform vec2 uWind;
out vec4 vCol; out float vSoft;
${GLSL_NOISE}
vec2 curl(vec2 p){
float e = 0.05;
float a = vnoise(p + vec2(0.0, e)) + 0.5*vnoise(2.1*(p + vec2(0.0, e)));
float b = vnoise(p - vec2(0.0, e)) + 0.5*vnoise(2.1*(p - vec2(0.0, e)));
float c = vnoise(p + vec2(e, 0.0)) + 0.5*vnoise(2.1*(p + vec2(e, 0.0)));
float d = vnoise(p - vec2(e, 0.0)) + 0.5*vnoise(2.1*(p - vec2(e, 0.0)));
return vec2(a - b, -(c - d)) / (2.0 * e);
}
void cull(){ gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; vCol = vec4(0.0); vSoft = 0.0; }
void main(){
float m = texture(uSnapMask, aHome).r;
if (m < 0.55) { cull(); return; }
float d = delayAt(aHome) - 0.02 + aRand.x * 0.06;
float life = ${LIFE_MIN.toFixed(2)} + aRand.y * ${(LIFE_MAX - LIFE_MIN).toFixed(2)};
float local = uTau - d;
if (local < 0.0 || local > life) { cull(); return; }
float p = local / life;
float gust = p * p * 1.15 + p * 0.22;
vec2 wind = uWind * (0.55 + aRand.z * 0.9);
vec2 off = wind * gust;
off.y -= (0.03 + aRand.w * 0.10) * p * p;
vec2 cp = aHome * vec2(uAspect, 1.0) * 2.5 + vec2(uTau * 0.25, aRand.x * 3.0);
off += curl(cp) * 0.018 * p * (0.6 + aRand.w);
float ang = aRand.z * 6.2832 + local * (3.0 + aRand.w * 6.0);
off += vec2(cos(ang), sin(ang)) * (0.004 + 0.01 * aRand.y) * p;
off.x /= uAspect;
vec2 pos = aHome + off;
float x = uMirror > 0.5 ? 1.0 - pos.x : pos.x;
gl_Position = vec4(x * 2.0 - 1.0, 1.0 - pos.y * 2.0, 0.0, 1.0);
vec3 c = texture(uSnap, aHome).rgb;
float lum = dot(c, vec3(0.299, 0.587, 0.114));
vec3 ash = vec3(0.07, 0.065, 0.06) + vec3(0.55, 0.52, 0.5) * lum * 0.45;
vec3 col = mix(c, ash, smoothstep(0.04, 0.5, p));
float ember = step(0.72, aRand.w) * exp(-local * 9.0);
col += vec3(1.0, 0.45, 0.12) * ember * 0.9;
float alpha = (1.0 - smoothstep(0.55, 1.0, p)) * 0.95;
float size = uCell * (1.25 + aRand.z * 0.9) * (1.0 - 0.65 * p);
gl_PointSize = clamp(size, 1.0, uMaxPt);
vSoft = clamp(size, 1.0, 8.0);
vCol = vec4(col * alpha, alpha);
}`;
const FS_PART = `#version 300 es
precision highp float;
in vec4 vCol; in float vSoft; out vec4 outColor;
void main(){
float r = length(gl_PointCoord - 0.5) * 2.0;
float a = 1.0 - smoothstep(1.0 - 1.6 / vSoft, 1.0, r);
if (a <= 0.0) discard;
outColor = vCol * a;
}`;
function compile(type, src) {
const s = gl.createShader(type);
gl.shaderSource(s, src); gl.compileShader(s);
if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
const log = gl.getShaderInfoLog(s);
console.error(log, src.split('\n').map((l, i) => (i + 1) + ': ' + l).join('\n'));
throw new Error('Ошибка шейдера: ' + log);
}
return s;
}
function program(vs, fs, attribs) {
const p = gl.createProgram();
gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
attribs.forEach((a, i) => gl.bindAttribLocation(p, i, a));
gl.linkProgram(p);
if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('Ошибка программы: ' + gl.getProgramInfoLog(p));
const u = {};
const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(p, i); u[info.name] = gl.getUniformLocation(p, info.name); }
return { p, u };
}
function makeTex(filter = gl.LINEAR) {
const t = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, t);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
return t;
}
function setupGL() {
gl = canvas.getContext('webgl2', { alpha: false, antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: false, powerPreference: 'high-performance', desynchronized: false });
if (!gl) throw new Error('Этот браузер не поддерживает WebGL2. Обнови браузер (Chrome, Edge, Safari 15+).');
canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); toast('Видеокарта перезапустилась — обнови страницу.', 'err', 10000); });
progComp = program(VS_QUAD, FS_COMP, ['aPos']);
progBg = program(VS_QUAD, FS_BG, ['aPos']);
progPart = program(VS_PART, FS_PART, ['aHome', 'aRand']);
maxPointSize = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1] || 64;
quadVao = gl.createVertexArray();
gl.bindVertexArray(quadVao);
const qb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, qb);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);
texVideo = makeTex(); texSnap = makeTex(); texMask = makeTex(); texSnapMask = makeTex();
gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
resize();
}
function resize() {
app.vw = video.videoWidth; app.vh = video.videoHeight;
canvas.width = app.vw; canvas.height = app.vh;
allocSeg();
maskS.fill(0);
buildParticles();
app.bgReady = false;
bgTex.forEach((t) => gl.deleteTexture(t)); bgFbo.forEach((f) => gl.deleteFramebuffer(f));
bgTex = []; bgFbo = [];
fitCanvas();
}
function fitCanvas() {
const vw = window.innerWidth, vh = window.innerHeight;
const s = Math.min(vw / app.vw, vh / app.vh);
canvas.style.width = Math.floor(app.vw * s) + 'px';
canvas.style.height = Math.floor(app.vh * s) + 'px';
}
window.addEventListener('resize', () => { if (app.vw) fitCanvas(); });
function buildParticles() {
const aspect = app.vw / app.vh;
const target = IS_MOBILE ? 38000 : 72000;     // частиц на весь кадр (рисуются только те, что на теле)
const pw = Math.round(Math.sqrt(target * aspect)), ph = Math.round(pw / aspect);
partCount = pw * ph;
const home = new Float32Array(partCount * 2), rnd = new Float32Array(partCount * 4);
let i = 0;
for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++, i++) {
home[i * 2] = (x + 0.15 + Math.random() * 0.7) / pw;
home[i * 2 + 1] = (y + 0.15 + Math.random() * 0.7) / ph;
rnd[i * 4] = Math.random(); rnd[i * 4 + 1] = Math.random(); rnd[i * 4 + 2] = Math.random(); rnd[i * 4 + 3] = Math.random();
}
app.cellPx = app.vw / pw;
if (partVao) gl.deleteVertexArray(partVao);
partVao = gl.createVertexArray();
gl.bindVertexArray(partVao);
const b1 = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b1); gl.bufferData(gl.ARRAY_BUFFER, home, gl.STATIC_DRAW);
gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
const b2 = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b2); gl.bufferData(gl.ARRAY_BUFFER, rnd, gl.STATIC_DRAW);
gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);
}
function gl_uploadBackground(rgba, w, h) {
bgW = w; bgH = h;
bgTex.forEach((t) => gl.deleteTexture(t)); bgFbo.forEach((f) => gl.deleteFramebuffer(f));
bgTex = []; bgFbo = [];
for (let k = 0; k < 2; k++) {
const t = makeTex();
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
const f = gl.createFramebuffer();
gl.bindFramebuffer(gl.FRAMEBUFFER, f);
gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
bgTex.push(t); bgFbo.push(f);
}
gl.bindFramebuffer(gl.FRAMEBUFFER, null);
bgIdx = 0;
}
function gl_snapshot() {
gl.bindTexture(gl.TEXTURE_2D, texSnap);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
gl.bindTexture(gl.TEXTURE_2D, texSnapMask);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, app.sw, app.sh, 0, gl.RGBA, gl.UNSIGNED_BYTE, maskU8);
}
function bindTex(unit, tex, loc) { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(loc, unit); }
function setNoiseUniforms(u) {
gl.uniform2f(u.uOrigin, app.origin[0], app.origin[1]);
gl.uniform1f(u.uAspect, app.vw / app.vh);
gl.uniform1f(u.uSpread, app.spread);
}
const MODE_ID = { visible: 0, dissolving: 1, hidden: 2, assembling: 3 };
let bgSeed = 0;
function render(tau, dil) {
if (app.bgReady && bgTex.length) {
const src = bgIdx, dst = 1 - bgIdx;
gl.bindFramebuffer(gl.FRAMEBUFFER, bgFbo[dst]);
gl.viewport(0, 0, bgW, bgH);
gl.disable(gl.BLEND);
gl.useProgram(progBg.p);
gl.uniform1f(progBg.u.uMirror, 0);
gl.uniform1f(progBg.u.uFbo, 1);
gl.uniform1f(progBg.u.uRate, 0.045);
gl.uniform1f(progBg.u.uSeed, (bgSeed = (bgSeed + 17.13) % 997));
bindTex(0, texVideo, progBg.u.uVideo);
bindTex(1, bgTex[src], progBg.u.uBg);
bindTex(2, texMask, progBg.u.uMask);
gl.bindVertexArray(quadVao);
gl.drawArrays(gl.TRIANGLES, 0, 3);
bgIdx = dst;
}
gl.bindFramebuffer(gl.FRAMEBUFFER, null);
gl.viewport(0, 0, canvas.width, canvas.height);
gl.disable(gl.BLEND);
gl.useProgram(progComp.p);
const u = progComp.u;
gl.uniform1f(u.uMirror, settings.mirror ? 1 : 0);
gl.uniform1f(u.uFbo, 0);
gl.uniform1i(u.uMode, MODE_ID[app.mode]);
gl.uniform1f(u.uTau, tau);
gl.uniform1f(u.uDil, dil);
gl.uniform3f(u.uGain, gain[0], gain[1], gain[2]);
gl.uniform1f(u.uHasBg, app.bgReady && bgTex.length ? 1 : 0);
setNoiseUniforms(u);
bindTex(0, texVideo, u.uVideo);
bindTex(1, bgTex.length ? bgTex[bgIdx] : texVideo, u.uBg);
bindTex(2, texMask, u.uMask);
gl.bindVertexArray(quadVao);
gl.drawArrays(gl.TRIANGLES, 0, 3);
if (app.mode === 'dissolving' || app.mode === 'assembling') {
gl.enable(gl.BLEND);
gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
gl.useProgram(progPart.p);
const p = progPart.u;
setNoiseUniforms(p);
gl.uniform1f(p.uTau, tau);
gl.uniform1f(p.uMirror, settings.mirror ? 1 : 0);
gl.uniform1f(p.uCell, app.cellPx);
gl.uniform1f(p.uMaxPt, maxPointSize);
gl.uniform2f(p.uWind, app.wind[0], app.wind[1]);
bindTex(0, texSnap, p.uSnap);
bindTex(1, texSnapMask, p.uSnapMask);
gl.bindVertexArray(partVao);
gl.drawArrays(gl.POINTS, 0, partCount);
gl.disable(gl.BLEND);
}
gl.bindVertexArray(null);
}
let lastVT = -1, fpsN = 0, fpsT = performance.now(), hasRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype, newFrame = false;
if (hasRVFC) {
const onVF = () => { newFrame = true; video.requestVideoFrameCallback(onVF); };
video.requestVideoFrameCallback(onVF);
}
const smooth01 = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
function loop(now) {
requestAnimationFrame(loop);
if (video.readyState < 2) return;
if (video.videoWidth !== app.vw || video.videoHeight !== app.vh) resize();
const fresh = hasRVFC ? newFrame : video.currentTime !== lastVT;
if (fresh) {
newFrame = false; lastVT = video.currentTime;
app.frame++;
segCtx.drawImage(video, 0, 0, app.sw, app.sh);
segRGB = segCtx.getImageData(0, 0, app.sw, app.sh).data;
runSegmentation(now);
processMask();
if (app.frame % 2 === 0 || app.mode === 'hidden' || app.mode === 'visible') runHands(now);
gl.bindTexture(gl.TEXTURE_2D, texVideo);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
gl.bindTexture(gl.TEXTURE_2D, texMask);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, app.sw, app.sh, 0, gl.RGBA, gl.UNSIGNED_BYTE, maskU8);
fpsN++;
}
const el = (now - app.effectStart) / 1000;
let tau = 0, dil = 0;
if (app.mode === 'dissolving') {
tau = el; dil = smooth01(0, 1.6, tau);
if (el >= EFFECT_T) { app.mode = 'hidden'; app.lastTriggerAt = Math.max(app.lastTriggerAt, now - COOLDOWN_MS + 400); }
} else if (app.mode === 'assembling') {
tau = EFFECT_T - ASM_SKIP - el; dil = smooth01(0, 1.6, tau);
if (tau <= 0) { app.mode = 'visible'; app.lastTriggerAt = Math.max(app.lastTriggerAt, now - COOLDOWN_MS + 400); }
} else if (app.mode === 'hidden') { tau = EFFECT_T; dil = 1; }
render(tau, dil);
if (now - fpsT >= 1000) {
ui.fps.textContent = Math.round(fpsN * 1000 / (now - fpsT)) + ' FPS';
fpsN = 0; fpsT = now;
}
if (app.hands) {
const hv = handVisibleRecently(now);
ui.handPill.classList.toggle('on', hv);
ui.handTxt.textContent = hv ? 'рука видна' : 'руки не видно';
} else if (app.frame === 5) { ui.handTxt.textContent = 'рука: выкл'; }
}
let recorder = null, recChunks = [], recStart = 0, recTimer = 0;
function pickMime() {
const c = ['video/mp4;codecs=avc1.42E01E,mp4a.40.2', 'video/mp4;codecs=avc1,mp4a', 'video/mp4', 'video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm'];
for (const m of c) if (window.MediaRecorder && MediaRecorder.isTypeSupported?.(m)) return m;
return '';
}
function toggleRecord() {
if (recorder) { recorder.stop(); return; }
if (!window.MediaRecorder || !canvas.captureStream) { toast('Этот браузер не умеет записывать видео.', 'err'); return; }
const vs = canvas.captureStream(30);
const tracks = [...vs.getVideoTracks()];
if (app.recDest) tracks.push(...app.recDest.stream.getAudioTracks());
const mime = pickMime();
try {
recorder = new MediaRecorder(new MediaStream(tracks), { mimeType: mime || undefined, videoBitsPerSecond: 8_000_000, audioBitsPerSecond: 128_000 });
} catch (e) {
recorder = new MediaRecorder(new MediaStream(tracks));
}
recChunks = [];
recorder.ondataavailable = (e) => { if (e.data && e.data.size) recChunks.push(e.data); };
recorder.onstop = () => {
const type = recorder.mimeType || mime || 'video/webm';
const ext = /mp4/.test(type) ? 'mp4' : 'webm';
const blob = new Blob(recChunks, { type });
const a = document.createElement('a');
const d = new Date(), pad = (n) => String(n).padStart(2, '0');
a.download = `shchelchok-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}.${ext}`;
a.href = URL.createObjectURL(blob);
document.body.appendChild(a); a.click(); a.remove();
setTimeout(() => URL.revokeObjectURL(a.href), 60000);
toast(`Видео сохранено (${(blob.size / 1048576).toFixed(1)} МБ) — смотри «Загрузки».`, '', 5000);
recorder = null;
vs.getVideoTracks().forEach((t) => t.stop());
clearInterval(recTimer);
ui.rec.classList.remove('show');
ui.recBtn.classList.remove('recording');
ui.recLabel.textContent = 'Записать видео';
};
recorder.start(1000);
recStart = performance.now();
ui.rec.classList.add('show');
ui.recBtn.classList.add('recording');
ui.recLabel.textContent = 'Остановить';
const upd = () => { const s = Math.floor((performance.now() - recStart) / 1000); ui.recTime.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };
upd(); recTimer = setInterval(upd, 500);
}
function applySettingsUI() {
ui.mirrorBtn.classList.toggle('active', settings.mirror);
ui.sens.value = settings.sens; ui.sensOut.textContent = settings.sens;
ui.needHand.checked = settings.needHand;
ui.fxSound.checked = settings.fxSound;
}
ui.snapBtn.addEventListener('click', () => manualSnap(null));
$('setBtn').addEventListener('click', () => {
const r = $('setRow'), h = $('hint'); r.hidden = !r.hidden; h.hidden = r.hidden;
$('setBtn').classList.toggle('active', !r.hidden);
});
let dimT = 0;
function undim() { ui.bar.classList.remove('dim'); clearTimeout(dimT); dimT = setTimeout(() => { if ($('setRow').hidden) ui.bar.classList.add('dim'); }, 6000); }
['pointermove', 'pointerdown', 'keydown'].forEach((ev) => window.addEventListener(ev, undim, { passive: true }));
ui.bgBtn.addEventListener('click', () => captureBackground());
ui.mirrorBtn.addEventListener('click', toggleMirror);
ui.recBtn.addEventListener('click', toggleRecord);
ui.fsBtn.addEventListener('click', toggleFullscreen);
ui.cleanBtn.addEventListener('click', () => setClean(true));
ui.sens.addEventListener('input', () => {
settings.sens = +ui.sens.value; ui.sensOut.textContent = settings.sens; store.set('sens', settings.sens);
app.detector?.port.postMessage({ sens: settings.sens });
});
ui.needHand.addEventListener('change', () => { settings.needHand = ui.needHand.checked; store.set('needHand', settings.needHand); });
ui.fxSound.addEventListener('change', () => {
settings.fxSound = ui.fxSound.checked; store.set('fxSound', settings.fxSound);
if (app.fxOut) app.fxOut.gain.value = settings.fxSound ? 1 : 0;
});
function toggleMirror() { settings.mirror = !settings.mirror; store.set('mirror', settings.mirror); applySettingsUI(); }
function manualSnap(origin) {
if (!app.started) return;
if (!app.bgReady) { toastOnce('nobg', 'Сначала нужно снять фон.', 'warn'); return; }
if (!canTrigger(performance.now())) return;
if (!origin) {
const last = [...app.handHistory].reverse().find((e) => e.hands.length && performance.now() - e.t < 600);
if (last) origin = last.hands.reduce((a, b) => (a.r < b.r ? a : b)).pt;
}
trigger(origin);
}
function toggleFullscreen() {
const d = document;
if (d.fullscreenElement || d.webkitFullscreenElement) { (d.exitFullscreen || d.webkitExitFullscreen).call(d); return; }
const el = d.documentElement;
const req = el.requestFullscreen || el.webkitRequestFullscreen;
if (!req) { toast('На iPhone: «Поделиться» → «На экран Домой» — тогда откроется на весь экран.', '', 6000); return; }
Promise.resolve(req.call(el)).catch(() => toast('Полный экран недоступен в этом браузере.', 'warn'));
}
let cleanHintT = 0;
function setClean(on) {
document.body.classList.toggle('clean', on);
clearTimeout(cleanHintT);
ui.cleanHint.classList.toggle('show', on);
if (on) cleanHintT = setTimeout(() => ui.cleanHint.classList.remove('show'), 3500);
}
window.addEventListener('keydown', (e) => {
if (!app.started || e.target.tagName === 'INPUT' && e.target.type !== 'range' && e.target.type !== 'checkbox') return;
const k = e.key.toLowerCase();
if (e.code === 'Space') { e.preventDefault(); manualSnap(null); }
else if (k === 'h' || k === 'р') setClean(!document.body.classList.contains('clean'));
else if (k === 'escape') setClean(false);
else if (k === 'f' || k === 'а') toggleFullscreen();
else if (k === 'm' || k === 'ь') toggleMirror();
else if (k === 'b' || k === 'и') captureBackground();
else if (k === 'r' || k === 'к') toggleRecord();
});
document.addEventListener('click', (e) => { const b = e.target.closest?.('button'); if (b) b.blur(); });
let pressT = 0, longPressed = false, pressXY = null;
const stage = $('stage');
stage.addEventListener('pointerdown', (e) => {
longPressed = false; pressXY = [e.clientX, e.clientY];
clearTimeout(pressT);
pressT = setTimeout(() => { longPressed = true; if (document.body.classList.contains('clean')) setClean(false); }, 700);
});
stage.addEventListener('pointerup', (e) => {
clearTimeout(pressT);
if (longPressed || !pressXY || Math.hypot(e.clientX - pressXY[0], e.clientY - pressXY[1]) > 20) return;
const r = canvas.getBoundingClientRect();
let x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
if (x < 0 || x > 1 || y < 0 || y > 1) { manualSnap(null); return; }
if (settings.mirror) x = 1 - x;
manualSnap([x, y]);
});
stage.addEventListener('pointercancel', () => clearTimeout(pressT));
stage.addEventListener('contextmenu', (e) => e.preventDefault());
let wakeLock = null;
async function requestWakeLock() {
try { wakeLock = await navigator.wakeLock?.request('screen'); } catch {}
}
document.addEventListener('visibilitychange', () => {
if (document.visibilityState === 'visible') { requestWakeLock(); app.audioCtx?.resume?.(); }
});
(function startDust() {
const box = $('dust'); if (!box) return;
const c = document.createElement('canvas'); box.appendChild(c);
const x = c.getContext('2d');
let W = 0, H = 0; const P = [];
const fit = () => { W = c.width = box.clientWidth * devicePixelRatio; H = c.height = box.clientHeight * devicePixelRatio; c.style.width = '100%'; c.style.height = '100%'; };
fit(); window.addEventListener('resize', fit);
for (let i = 0; i < 140; i++) P.push({ x: Math.random(), y: Math.random(), s: Math.random(), v: 0.2 + Math.random() });
(function f(t) {
if (!document.body.contains(c)) return;
x.clearRect(0, 0, W, H);
for (const p of P) {
p.x += 0.0006 * p.v; p.y -= 0.00025 * p.v + Math.sin(t / 900 + p.s * 9) * 0.0002;
if (p.x > 1.02) p.x = -0.02; if (p.y < -0.02) p.y = 1.02;
x.globalAlpha = 0.25 + 0.5 * p.s;
x.fillStyle = p.s > 0.85 ? '#ffb36b' : '#8a7f72';
x.beginPath(); x.arc(p.x * W, p.y * H, (0.6 + p.s * 1.8) * devicePixelRatio, 0, 6.283); x.fill();
}
requestAnimationFrame(f);
})(0);
})();
if ('serviceWorker' in navigator && location.protocol === 'https:') {
navigator.serviceWorker.register('sw.js').catch(() => {});
}
window.__snap = { app, trigger: (o) => trigger(o), manualSnap, settings };
