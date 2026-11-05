const VERSION = 'v1';
const SHELL = `shell-${VERSION}`;
const MODELS = 'models-v1';
const SHELL_FILES = ['./', './index.html', './app.js', './manifest.webmanifest', './icon-192.png', './icon-512.png'];
self.addEventListener('install', (e) => {
e.waitUntil(caches.open(SHELL).then((c) => c.addAll(SHELL_FILES)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== SHELL && k !== MODELS).map((k) => caches.delete(k))))
.then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
const req = e.request;
if (req.method !== 'GET') return;
const url = new URL(req.url);
if (url.hostname === 'cdn.jsdelivr.net' || url.hostname === 'storage.googleapis.com') {
e.respondWith(caches.open(MODELS).then(async (c) => {
const hit = await c.match(req);
if (hit) return hit;
const res = await fetch(req);
if (res.ok) c.put(req, res.clone());
return res;
}));
return;
}
if (url.origin === self.location.origin) {
e.respondWith(fetch(req).then((res) => {
if (res.ok) { const copy = res.clone(); caches.open(SHELL).then((c) => c.put(req, copy)); }
return res;
}).catch(() => caches.match(req).then((r) => r || caches.match('./index.html'))));
}
});
