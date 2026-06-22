import init, { EdgeVisionPipeline } from "../pkg/edge_vision.js";

const frameCanvas = document.querySelector("#frameCanvas");
const overlayCanvas = document.querySelector("#overlayCanvas");
const frameCtx = frameCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");

const els = {
  fps: document.querySelector("#fps"),
  latency: document.querySelector("#latency"),
  detections: document.querySelector("#detections"),
  tracks: document.querySelector("#tracks"),
  frameId: document.querySelector("#frameId"),
  trackList: document.querySelector("#trackList"),
  captureBar: document.querySelector("#captureBar"),
  preprocessBar: document.querySelector("#preprocessBar"),
  detectionBar: document.querySelector("#detectionBar"),
  trackingBar: document.querySelector("#trackingBar"),
  captureMs: document.querySelector("#captureMs"),
  preprocessMs: document.querySelector("#preprocessMs"),
  detectionMs: document.querySelector("#detectionMs"),
  trackingMs: document.querySelector("#trackingMs"),
};

const colors = {
  person: "#ffcf4a",
  vehicle: "#38d3ff",
  package: "#7cff9b",
};

function drawFrame(snapshot) {
  const image = frameCtx.createImageData(snapshot.width, snapshot.height);

  for (let index = 0; index < snapshot.grayscale.length; index += 1) {
    const value = snapshot.grayscale[index];
    const heat = snapshot.motion_heatmap[index];
    const offset = index * 4;
    image.data[offset] = Math.min(255, value + heat * 0.4);
    image.data[offset + 1] = Math.min(255, value + heat * 0.18);
    image.data[offset + 2] = Math.min(255, value + 24);
    image.data[offset + 3] = 255;
  }

  const temp = new OffscreenCanvas(snapshot.width, snapshot.height);
  temp.getContext("2d").putImageData(image, 0, 0);
  frameCtx.imageSmoothingEnabled = false;
  frameCtx.clearRect(0, 0, frameCanvas.width, frameCanvas.height);
  frameCtx.drawImage(temp, 0, 0, frameCanvas.width, frameCanvas.height);
}

function drawOverlay(snapshot) {
  const sx = overlayCanvas.width / snapshot.width;
  const sy = overlayCanvas.height / snapshot.height;
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  for (const region of snapshot.calibration.regions) {
    overlayCtx.strokeStyle = "rgba(255,255,255,0.22)";
    overlayCtx.lineWidth = 2;
    overlayCtx.setLineDash([7, 7]);
    overlayCtx.strokeRect(
      region.bounds.x * sx,
      region.bounds.y * sy,
      region.bounds.width * sx,
      region.bounds.height * sy,
    );
    overlayCtx.setLineDash([]);
    overlayCtx.fillStyle = "rgba(255,255,255,0.68)";
    overlayCtx.font = "600 13px Inter, system-ui";
    overlayCtx.fillText(region.name, region.bounds.x * sx + 8, region.bounds.y * sy + 18);
  }

  for (const detection of snapshot.detections) {
    const color = colors[detection.class_name] || "#ffffff";
    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = 3;
    overlayCtx.strokeRect(
      detection.bbox.x * sx,
      detection.bbox.y * sy,
      detection.bbox.width * sx,
      detection.bbox.height * sy,
    );
    overlayCtx.fillStyle = color;
    overlayCtx.font = "700 14px Inter, system-ui";
    overlayCtx.fillText(
      `${detection.class_name} ${(detection.confidence * 100).toFixed(0)}%`,
      detection.bbox.x * sx,
      Math.max(18, detection.bbox.y * sy - 7),
    );
  }

  for (const track of snapshot.tracks) {
    const color = colors[track.class_name] || "#ffffff";
    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = 2;
    overlayCtx.beginPath();
    overlayCtx.moveTo(track.centroid.x * sx, track.centroid.y * sy);
    overlayCtx.lineTo(
      (track.centroid.x + track.velocity.x * 7) * sx,
      (track.centroid.y + track.velocity.y * 7) * sy,
    );
    overlayCtx.stroke();
    overlayCtx.fillStyle = "#0b1117";
    overlayCtx.beginPath();
    overlayCtx.arc(track.centroid.x * sx, track.centroid.y * sy, 12, 0, Math.PI * 2);
    overlayCtx.fill();
    overlayCtx.fillStyle = color;
    overlayCtx.font = "800 12px Inter, system-ui";
    overlayCtx.textAlign = "center";
    overlayCtx.fillText(track.id, track.centroid.x * sx, track.centroid.y * sy + 4);
    overlayCtx.textAlign = "left";
  }
}

function updateMetrics(snapshot) {
  const latency = snapshot.latency;
  els.fps.textContent = latency.fps.toFixed(0);
  els.latency.textContent = `${latency.total_ms.toFixed(1)} ms`;
  els.detections.textContent = snapshot.detections.length;
  els.tracks.textContent = snapshot.tracks.length;
  els.frameId.textContent = `frame ${snapshot.frame_index}`;

  const max = 16;
  const setBar = (bar, label, value) => {
    bar.style.width = `${Math.min(100, (value / max) * 100)}%`;
    label.textContent = `${value.toFixed(1)} ms`;
  };

  setBar(els.captureBar, els.captureMs, latency.capture_ms);
  setBar(els.preprocessBar, els.preprocessMs, latency.preprocess_ms);
  setBar(els.detectionBar, els.detectionMs, latency.detection_ms);
  setBar(els.trackingBar, els.trackingMs, latency.tracking_ms);

  els.trackList.innerHTML = snapshot.tracks
    .map(
      (track) => `
        <article>
          <span class="track-id" style="border-color: ${colors[track.class_name] || "#fff"}">#${track.id}</span>
          <div>
            <strong>${track.class_name}</strong>
            <small>${track.age_frames} frames · ${track.confidence.toFixed(2)} confidence</small>
          </div>
          <b>${Math.hypot(track.velocity.x, track.velocity.y).toFixed(1)} px/f</b>
        </article>
      `,
    )
    .join("");
}

async function main() {
  await init();
  const pipeline = new EdgeVisionPipeline();
  let last = performance.now();

  function animate(now) {
    const dt = Math.min(50, now - last);
    last = now;
    pipeline.tick(dt);
    const snapshot = JSON.parse(pipeline.snapshot_json());
    drawFrame(snapshot);
    drawOverlay(snapshot);
    updateMetrics(snapshot);
    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
}

main();
