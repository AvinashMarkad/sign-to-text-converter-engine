(() => {
  const $ = (id) => document.getElementById(id);
  const video = $("camera");
  const canvas = $("capture-canvas");
  const context = canvas.getContext("2d");
  const sessionId = crypto.randomUUID();
  const ui = {
    apiStatus: $("api-status"), apiStatusText: $("api-status-text"),
    stage: $("video-stage"), feed: $("feed-indicator"),
    prediction: $("prediction-label"), confidence: $("prediction-confidence"),
    letter: $("current-letter"), letterState: $("letter-state"),
    word: $("current-word"), sentence: $("sentence-text"), wordCount: $("word-count"),
    hint: $("camera-hint"), start: $("start-button"), stop: $("stop-button"),
  };
  let stream = null;
  let running = false;
  let busy = false;
  let timer = null;

  async function checkHealth() {
    try {
      const response = await fetch("/api/health", { cache: "no-store" });
      const health = await response.json();
      ui.apiStatus.className = `api-status ${health.ready ? "ready" : "error"}`;
      ui.apiStatusText.textContent = health.ready ? "Model ready" : "Model unavailable";
      if (!health.ready && health.model_error) ui.hint.textContent = health.model_error;
    } catch {
      ui.apiStatus.className = "api-status error";
      ui.apiStatusText.textContent = "Backend offline";
    }
  }

  function escapeHtml(value) {
    return value.replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
    })[char]);
  }

  function render(data) {
    if (data.label) {
      const blank = data.label === "blank";
      ui.prediction.textContent = blank ? "—" : data.label;
      ui.confidence.textContent = blank ? "No hand" : `${Math.round(data.confidence * 100)}% confidence`;
    }
    if (data.stable_label !== undefined) {
      const active = Boolean(data.stable_label && data.stable_label !== "blank");
      ui.letter.textContent = active ? data.stable_label : "—";
      if (data.letter_accepted && active) {
        ui.letterState.textContent = `Added ${data.stable_label} · change handshape`;
      } else if (active && data.awaiting_release) {
        ui.letterState.textContent = "Letter added · change handshape";
      } else {
        ui.letterState.textContent = active ? "Stabilizing sign…" : "Waiting for a sign";
      }
      ui.letterState.classList.toggle("detected", active);
    }
    const word = data.current_word || "";
    const sentence = data.sentence || data.transcript || "";
    ui.word.innerHTML = word ? escapeHtml(word) : '<span class="empty-word">Your letters will collect here</span>';
    ui.sentence.innerHTML = sentence ? escapeHtml(sentence) : '<span class="empty-sentence">Your letters and words will appear here.</span>';
    const completedCount = (data.completed_sentence || "").trim().split(/\s+/).filter(Boolean).length;
    const count = completedCount + (word ? 1 : 0);
    ui.wordCount.textContent = `${count} WORD${count === 1 ? "" : "S"}`;
  }

  function captureBlob() {
    if (!video.videoWidth || !video.videoHeight) return Promise.resolve(null);
    // Match the original training ROI: x=10..50% of the mirrored frame and
    // y=10..50% of its width. The raw webcam frame is unmirrored, so crop its
    // left half and mirror that crop before sending it to the model.
    const side = Math.min(video.videoWidth * 0.5 - 10, video.videoHeight - 20);
    const left = 10;
    const top = 10;
    canvas.width = canvas.height = 320;
    context.save();
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, left, top, side, side, 0, 0, 320, 320);
    context.restore();
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.78));
  }

  async function sendFrame() {
    if (!running || busy) return;
    busy = true;
    try {
      const blob = await captureBlob();
      if (!blob) return;
      const response = await fetch(`/api/sessions/${sessionId}/frame`, {
        method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Frame recognition failed");
      render(data);
    } catch (error) {
      ui.apiStatus.className = "api-status error";
      ui.apiStatusText.textContent = "Recognition error";
      ui.letterState.textContent = error.message || "Could not read this frame";
    } finally {
      busy = false;
      if (running) timer = window.setTimeout(sendFrame, 40);
    }
  }

  async function startCamera() {
    if (running) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      ui.hint.textContent = "Camera access needs HTTPS or localhost";
      return;
    }
    ui.start.disabled = true;
    ui.hint.textContent = "Requesting camera access…";
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      });
      video.srcObject = stream;
      await video.play();
      running = true;
      ui.stage.classList.add("active");
      ui.feed.classList.add("on");
      ui.feed.innerHTML = "<i></i> CAMERA ON";
      ui.start.textContent = "Camera running";
      ui.stop.disabled = false;
      ui.hint.textContent = "Keep hand inside the guide; change shape for each letter";
      sendFrame();
    } catch (error) {
      ui.hint.textContent = error.name === "NotAllowedError"
        ? "Camera permission was blocked. Allow it in your browser settings."
        : "Could not start the camera. Check that it is connected.";
      ui.start.disabled = false;
    }
  }

  async function sendAction(action) {
    try {
      const response = await fetch(`/api/sessions/${sessionId}/action`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not update transcript");
      render(data);
    } catch (error) {
      ui.letterState.textContent = error.message || "Could not update transcript";
    }
  }

  async function stopCamera() {
    running = false;
    window.clearTimeout(timer);
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null;
    video.srcObject = null;
    ui.stage.classList.remove("active");
    ui.feed.classList.remove("on");
    ui.feed.innerHTML = "<i></i> CAMERA OFF";
    ui.start.disabled = false;
    ui.start.innerHTML = '<span class="button-play">▶</span> Start camera';
    ui.stop.disabled = true;
    while (busy) await new Promise((resolve) => window.setTimeout(resolve, 25));
    ui.hint.textContent = "Preview here · cropped frames go to the backend";
    await sendAction("commit");
  }

  ui.start.addEventListener("click", startCamera);
  ui.stop.addEventListener("click", stopCamera);
  $("space-button").addEventListener("click", () => sendAction("space"));
  $("backspace-button").addEventListener("click", () => sendAction("backspace"));
  $("clear-button").addEventListener("click", () => sendAction("clear"));
  window.addEventListener("beforeunload", () => {
    if (stream) stream.getTracks().forEach((track) => track.stop());
  });
  checkHealth();
})();
