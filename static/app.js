/* Drowsiness Detection UI: webcam capture -> /api/frame -> overlay, stats, alarm */
(() => {
const $ = id => document.getElementById(id);
const STATE_TEXT = {
  awake:["Awake","Eyes are open — all good."], closing:["Eyes closed","Eyes shut — watching the timer…"],
  drowsy:["DROWSY!","Eyes closed too long. Wake up!"], no_face:["No face","Look at the camera so a face can be detected."],
  idle:["Waiting","Start the camera to begin monitoring."]};
const SEND_WIDTH = 480, MIN_GAP_MS = 110;

/* ---------------- tabs ---------------- */
document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === t));
  document.querySelectorAll(".panel").forEach(p => p.classList.toggle("active", p.id === "tab-" + t.dataset.tab));
  if (t.dataset.tab === "model") loadModel();
});

/* ---------------- live monitor ---------------- */
const video = $("video"), overlay = $("overlay"), octx = overlay.getContext("2d");
const grab = document.createElement("canvas"), gctx = grab.getContext("2d");
const alarm = $("alarm");
let stream = null, running = false, sid = Math.random().toString(36).slice(2);
let samples = [], lastState = "idle", lastRes = null, lat = [], lastTick = 0, fpsEMA = 0;

const setState = s => { document.body.dataset.state = s; };
const fmtTime = s => `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,"0")}`;
const clock = () => new Date().toLocaleTimeString([], {hour12:false});
function log(msg, bad) {
  const ul = $("log"); if (ul.firstChild && ul.firstChild.classList.contains("muted")) ul.innerHTML = "";
  const li = document.createElement("li"); if (bad) li.className = "bad";
  li.innerHTML = `<time>${clock()}</time><span>${msg}</span>`; ul.prepend(li);
  while (ul.children.length > 60) ul.lastChild.remove();
}
const closedS = () => +$("rClosed").value, thr = () => +$("rThr").value;

function syncSettings() {
  $("vClosed").textContent = closedS().toFixed(1) + " s"; $("vThr").textContent = thr().toFixed(2);
  $("closedOf").textContent = "/ " + closedS().toFixed(1) + " s"; $("thrMark").style.left = thr() * 100 + "%";
}
$("rClosed").oninput = $("rThr").oninput = syncSettings; syncSettings();

async function start() {
  $("camError").hidden = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:960}, height:{ideal:720}, facingMode:"user"}, audio:false});
  } catch (e) {
    $("camError").hidden = false;
    $("camError").textContent = "Could not open the camera: " + (e.message || e.name) +
      ". Allow camera access in the browser address bar, close other apps using the camera, and use http://127.0.0.1 or localhost.";
    return;
  }
  video.srcObject = stream; await video.play();
  alarm.muted = true; alarm.play().then(() => { alarm.pause(); alarm.currentTime = 0; alarm.muted = false; }).catch(() => { alarm.muted = false; });
  $("idleMsg").style.display = "none"; $("btnStart").disabled = true; $("btnStop").disabled = false; $("btnReset").disabled = false;
  running = true; samples = []; await fetch(`/api/reset?sid=${sid}`, {method:"POST"});
  log("Monitoring started"); loop();
}
function stop() {
  running = false; if (stream) stream.getTracks().forEach(t => t.stop()); stream = null; video.srcObject = null;
  alarm.pause(); alarm.currentTime = 0; setState("idle"); lastState = "idle";
  $("idleMsg").style.display = ""; $("btnStart").disabled = false; $("btnStop").disabled = true; $("btnReset").disabled = true;
  $("statusBadge").textContent = "IDLE"; octx.clearRect(0, 0, overlay.width, overlay.height);
  $("statusBig").textContent = STATE_TEXT.idle[0]; $("statusSub").textContent = STATE_TEXT.idle[1]; log("Monitoring stopped");
}
$("btnStart").onclick = start; $("btnStop").onclick = stop;
$("btnReset").onclick = async () => { await fetch(`/api/reset?sid=${sid}`, {method:"POST"}); samples = []; ["sBlinks","sAlerts"].forEach(i => $(i).textContent = 0); $("sPerclos").textContent = "0%"; log("Statistics reset"); };

async function loop() {
  while (running) {
    const t0 = performance.now();
    if (video.videoWidth) {
      grab.width = SEND_WIDTH; grab.height = Math.round(SEND_WIDTH * video.videoHeight / video.videoWidth);
      gctx.drawImage(video, 0, 0, grab.width, grab.height);
      const blob = await new Promise(r => grab.toBlob(r, "image/jpeg", 0.72));
      try {
        const r = await fetch(`/api/frame?sid=${sid}&closed_s=${closedS()}&thr=${thr()}`, {method:"POST", headers:{"Content-Type":"image/jpeg"}, body:blob});
        if (!r.ok) throw new Error("HTTP " + r.status);
        if (running) render(await r.json(), performance.now() - t0);
      } catch (e) {
        if (running) { $("camError").hidden = false; $("camError").textContent = "Lost connection to the server: " + e.message; await new Promise(r => setTimeout(r, 1000)); }
      }
    }
    const wait = MIN_GAP_MS - (performance.now() - t0); if (wait > 0) await new Promise(r => setTimeout(r, wait));
  }
}

function drawOverlay(res) {
  const w = overlay.width = overlay.clientWidth, h = overlay.height = overlay.clientHeight;
  octx.clearRect(0, 0, w, h);
  // the video is object-fit:cover, so map normalised frame coords to the displayed area
  const vw = video.videoWidth, vh = video.videoHeight; if (!vw) return;
  const s = Math.max(w / vw, h / vh), ox = (w - vw * s) / 2, oy = (h - vh * s) / 2;
  const R = b => [ox + b[0] * vw * s, oy + b[1] * vh * s, b[2] * vw * s, b[3] * vh * s];
  if (res.face) { octx.strokeStyle = "rgba(125,211,252,.55)"; octx.lineWidth = 1.5; octx.setLineDash([6, 5]); octx.strokeRect(...R(res.face)); octx.setLineDash([]); }
  if ($("optBoxes").checked) res.eyes.forEach(e => {
    octx.strokeStyle = e.p_open >= thr() ? "#34d399" : "#f43f5e"; octx.lineWidth = 2.5;
    const [x, y, bw, bh] = R(e.box); octx.beginPath(); octx.roundRect(x, y, bw, bh, 6); octx.stroke();
  });
}

function render(res, ms) {
  lastRes = res; const st = res.state;
  fpsEMA = fpsEMA ? .85 * fpsEMA + .15 * (1000 / Math.max(ms, 1)) : 1000 / Math.max(ms, 1);
  $("camError").hidden = true;
  drawOverlay(res);
  setState(st); $("statusBadge").textContent = st === "no_face" ? "NO FACE" : st === "closing" ? "EYES CLOSED" : st.toUpperCase();
  $("statusBig").textContent = STATE_TEXT[st][0]; $("statusSub").textContent = STATE_TEXT[st][1];
  const p = res.p_open; $("pOpenVal").textContent = p == null ? "–" : p.toFixed(2);
  $("pOpenFill").style.width = (p == null ? 0 : p * 100) + "%";
  const cf = res.closed_for || 0; $("closedFor").textContent = cf.toFixed(1) + " s";
  const frac = Math.min(cf / closedS(), 1); $("closedBar").style.width = frac * 100 + "%"; $("closedBar").style.background = st === "drowsy" ? "var(--bad)" : "var(--warn)";
  $("sBlinks").textContent = res.blinks; $("sPerclos").textContent = Math.round(res.perclos * 100) + "%";
  $("sAlerts").textContent = res.drowsy_events; $("sTime").textContent = fmtTime(res.session_s);
  $("sPerf").textContent = `${fpsEMA.toFixed(0)} fps · server ${res.latency_ms} ms`;
  // alarm
  if (res.alarm && $("optSound").checked) { if (alarm.paused) alarm.play().catch(() => {}); } else if (!alarm.paused) { alarm.pause(); alarm.currentTime = 0; }
  document.title = res.alarm ? "⚠ DROWSY! – Drowsiness Detection" : "Drowsiness Detection";
  // log
  if (res.new_event) log("Drowsiness alert — eyes closed for " + cf.toFixed(1) + " s", true);
  else if (st !== lastState && (st === "no_face" || lastState === "no_face") && lastState !== "idle") log(st === "no_face" ? "Face lost" : "Face found");
  lastState = st;
  samples.push([performance.now(), p]); const cut = performance.now() - 60000; while (samples.length && samples[0][0] < cut) samples.shift();
  drawTimeline();
}

function drawTimeline() {
  const c = $("timeline"), dpr = devicePixelRatio || 1, W = c.clientWidth, H = 150;
  c.width = W * dpr; c.height = H * dpr; const g = c.getContext("2d"); g.scale(dpr, dpr);
  const pad = {l:30, r:8, t:8, b:18}, pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  g.font = "11px " + getComputedStyle(document.body).fontFamily; g.fillStyle = "#8196b8"; g.strokeStyle = "#1f2d49"; g.lineWidth = 1;
  [0, .5, 1].forEach(v => { const y = pad.t + ph * (1 - v); g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke(); g.fillText(v.toFixed(1), 2, y + 4); });
  g.fillText("−60 s", pad.l, H - 3); g.fillText("now", W - pad.r - 22, H - 3);
  const ty = pad.t + ph * (1 - thr()); g.strokeStyle = "#f43f5e99"; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(pad.l, ty); g.lineTo(W - pad.r, ty); g.stroke(); g.setLineDash([]);
  const now = performance.now(), X = t => pad.l + pw * (1 - (now - t) / 60000), Y = p => pad.t + ph * (1 - p);
  g.lineWidth = 2; g.strokeStyle = "#2dd4bf"; g.beginPath(); let pen = false;
  samples.forEach(([t, p]) => { if (p == null) { pen = false; return; } pen ? g.lineTo(X(t), Y(p)) : g.moveTo(X(t), Y(p)); pen = true; }); g.stroke();
}

/* ---------------- upload tab ---------------- */
function dropzone(zone, input, handler) {
  input.onchange = () => input.files[0] && handler(input.files[0]);
  ["dragover","dragenter"].forEach(e => zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.add("over"); }));
  ["dragleave","drop"].forEach(e => zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.remove("over"); }));
  zone.addEventListener("drop", ev => ev.dataTransfer.files[0] && handler(ev.dataTransfer.files[0]));
}
async function post(url, file) { const fd = new FormData(); fd.append("file", file); const r = await fetch(url, {method:"POST", body:fd}); const j = await r.json(); if (!r.ok) throw new Error(j.error || r.status); return j; }

dropzone($("dropImg"), $("fileImg"), async f => {
  try {
    const j = await post(`/api/analyze-image?thr=${thr()}`, f);
    $("imgResult").hidden = false; $("imgOut").src = j.image;
    const v = $("imgVerdict"); v.textContent = j.verdict; v.style.color = j.state === "awake" ? "var(--ok)" : j.state === "no_face" ? "var(--muted)" : "var(--bad)";
    $("imgDetail").textContent = j.p_open == null ? "No face detected — try a clearer, front-facing photo." : "P(open) = " + j.p_open.toFixed(2) + "  ·  per eye: " + j.eyes.map(e => e.p_open.toFixed(2)).join(" / ");
  } catch (e) { alert(e.message); }
});
dropzone($("dropVid"), $("fileVid"), async f => {
  $("vidBusy").hidden = false; $("vidResult").hidden = true;
  try {
    const j = await post(`/api/analyze-video?thr=${thr()}&closed_s=${closedS()}`, f);
    $("vidResult").hidden = false;
    const card = (l, v) => `<div class="card stat"><div class="label">${l}</div><div class="num">${v}</div></div>`;
    $("vidStats").innerHTML = card("Duration", fmtTime(j.duration)) + card("Eyes closed", j.closed_pct + "%") + card("Blinks", j.blinks) + card("Drowsy events", j.drowsy_events);
    $("vidFrame").src = j.frame || ""; $("vidFrame").hidden = !j.frame; drawVideoChart(j);
  } catch (e) { alert(e.message); } finally { $("vidBusy").hidden = true; }
});
function drawVideoChart(j) {
  const c = $("vidChart"), dpr = devicePixelRatio || 1, W = c.clientWidth, H = 130; c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d"); g.scale(dpr, dpr); const T = Math.max(j.duration, 1), pl = 28, pw = W - pl - 6, ph = H - 24;
  const X = t => pl + pw * t / T, Y = p => 6 + ph * (1 - p);
  g.fillStyle = "#f43f5e33"; j.episodes.forEach(([a, b]) => g.fillRect(X(a), 6, Math.max(X(b) - X(a), 2), ph));
  g.strokeStyle = "#1f2d49"; g.fillStyle = "#8196b8"; g.font = "11px sans-serif"; [0, .5, 1].forEach(v => { g.beginPath(); g.moveTo(pl, Y(v)); g.lineTo(W - 6, Y(v)); g.stroke(); g.fillText(v.toFixed(1), 2, Y(v) + 4); });
  g.strokeStyle = "#f43f5e99"; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(pl, Y(j.closed_threshold)); g.lineTo(W - 6, Y(j.closed_threshold)); g.stroke(); g.setLineDash([]);
  g.strokeStyle = "#2dd4bf"; g.lineWidth = 2; g.beginPath(); let pen = false;
  j.timeline.forEach(([t, p]) => { if (p == null) { pen = false; return; } pen ? g.lineTo(X(t), Y(p)) : g.moveTo(X(t), Y(p)); pen = true; }); g.stroke();
  g.fillStyle = "#8196b8"; g.fillText("0 s", pl, H - 4); g.fillText(fmtTime(T), W - 34, H - 4);
}

/* ---------------- model tab ---------------- */
let modelLoaded = false;
async function loadModel() {
  if (modelLoaded) return;
  const box = $("modelBody");
  const r = await fetch("/api/model-info"); if (!r.ok) { box.innerHTML = '<p class="muted">No metrics found — run <code>python train.py</code>.</p>'; return; }
  const m = await r.json(), t = m.test, d = m.dataset, c = t.confusion, pct = v => (v * 100).toFixed(1) + "%";
  const card = (l, v, s) => `<div class="card stat"><div class="label">${l}</div><div class="num">${v}</div>${s ? `<div class="muted">${s}</div>` : ""}</div>`;
  box.innerHTML = `
  <div class="m-grid">${card("Test accuracy", pct(t.accuracy), "held-out " + d.test + " images")}${card("Parameters", m.params.toLocaleString(), "trained in " + m.train_seconds + " s")}
    ${card("Dataset", d.total.toLocaleString(), d.open + " open · " + d.closed + " closed")}${card("Split", d.train + " / " + d.val + " / " + d.test, "train / val / test")}</div>
  <div class="m-two">
    <div class="card"><div class="card-head"><h2>Training curves</h2></div><canvas id="curves" height="230" style="width:100%"></canvas></div>
    <div class="card"><div class="card-head"><h2>Test confusion matrix</h2></div>
      <div class="cm"><div class="h"></div><div class="h">pred. closed</div><div class="h">pred. open</div>
        <div class="h">true closed</div><div class="v good">${c.tn}</div><div class="v ${c.fp ? "badc" : ""}">${c.fp}</div>
        <div class="h">true open</div><div class="v ${c.fn ? "badc" : ""}">${c.fn}</div><div class="v good">${c.tp}</div></div>
      <table class="t" style="margin-top:16px"><tr><th>class</th><th>precision</th><th>recall</th><th>F1</th></tr>
        ${["open","closed"].map(k => `<tr><td>${k}</td><td>${pct(t[k].precision)}</td><td>${pct(t[k].recall)}</td><td>${pct(t[k].f1)}</td></tr>`).join("")}</table></div>
  </div>
  <div class="m-two">
    <div class="card"><div class="card-head"><h2>CNN architecture</h2></div><ol class="arch">${m.architecture.map(a => `<li>${a}</li>`).join("")}</ol></div>
    <div class="card"><div class="card-head"><h2>Dataset samples</h2></div><img class="m-img" src="/models/dataset_samples.png" alt="samples" onerror="this.remove()">
      <p class="muted" style="margin-top:10px">Top: open eyes · bottom: closed eyes. ${d.image_size[1]}×${d.image_size[0]} grayscale crops. Training uses random rotation, scale, shift, flip, brightness/contrast, blur and noise.</p></div>
  </div>`;
  drawCurves(m.history); modelLoaded = true;
}
function drawCurves(h) {
  const c = $("curves"), dpr = devicePixelRatio || 1, W = c.clientWidth, H = 230; c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d"); g.scale(dpr, dpr); const n = h.loss.length, pl = 36, pr = 10;
  const panel = (x0, w, title, a, b, lo, hi) => {
    g.fillStyle = "#8196b8"; g.font = "12px sans-serif"; g.fillText(title, x0 + pl, 14);
    const X = i => x0 + pl + (w - pl - pr) * i / Math.max(n - 1, 1), Y = v => 24 + (H - 44) * (1 - (v - lo) / (hi - lo));
    g.strokeStyle = "#1f2d49"; for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4, y = Y(v); g.beginPath(); g.moveTo(x0 + pl, y); g.lineTo(x0 + w - pr, y); g.stroke(); g.fillText(v.toFixed(2), x0, y + 4); }
    [[a, "#2dd4bf"], [b, "#fbbf24"]].forEach(([s, col]) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); s.forEach((v, i) => i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))); g.stroke(); });
    g.fillText("epoch →", x0 + w - 56, H - 4);
  };
  const half = W / 2;
  panel(0, half, "Accuracy   (teal = train, amber = validation)", h.accuracy, h.val_accuracy, Math.min(...h.accuracy, ...h.val_accuracy, .9) - .02, 1);
  panel(half, half, "Loss", h.loss, h.val_loss, 0, Math.max(...h.loss, ...h.val_loss));
}
})();
