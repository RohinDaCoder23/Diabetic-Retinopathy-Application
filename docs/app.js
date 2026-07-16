/* Diabetic Retinopathy Detector — 100% in-browser inference.
 * No server: OpenCV.js does the (training-matched) preprocessing and
 * onnxruntime-web runs the models on the visitor's device. */

const GRADE_URL  = "./models/grade_model.onnx";   // outputs: logits[1,5], cam[1,5,7,7]
const LESION_URL = "./models/lesion_unet.onnx";   // output:  masks[1,4,S,S] (already sigmoid)
const GRADE_SIZE = 224;
const LESION_SIZE = 512;                            // must match the ONNX export size
const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];
const CLASS_NAMES = ["0 — No DR", "1 — Mild", "2 — Moderate", "3 — Severe", "4 — Proliferative DR"];
const LESIONS = [
  { code: "MA", name: "Microaneurysms", color: [255, 0, 0] },
  { code: "HE", name: "Haemorrhages",   color: [255, 140, 0] },
  { code: "EX", name: "Hard exudates",  color: [255, 255, 0] },
  { code: "SE", name: "Soft exudates",  color: [0, 200, 255] },
];

let gradeSession = null, lesionSession = null, cvReady = false;

function $(id){ return document.getElementById(id); }
function status(msg, kind){ const s=$("status"); s.textContent=msg; s.className="status "+(kind||""); }

window.onOpenCvReady = function(){ cvReady = true; status("Ready — upload a fundus image.", "ok"); };

async function ensureSessions(){
  if (typeof ort !== "undefined" && ort.env && ort.env.wasm){
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
  }
  if (!gradeSession){
    status("Loading grade model…");
    gradeSession = await ort.InferenceSession.create(GRADE_URL, { executionProviders: ["wasm"] });
  }
  if (!lesionSession){
    try {
      status("Loading lesion model…");
      lesionSession = await ort.InferenceSession.create(LESION_URL, { executionProviders: ["wasm"] });
    } catch(e){ lesionSession = null; console.warn("lesion model unavailable:", e); }
  }
}

function fileToImage(file){
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => res(img);
    img.onerror = rej;
    img.src = URL.createObjectURL(file);
  });
}
function imageToRGBMat(img){
  const c = document.createElement("canvas");
  const maxSide = 1024;
  let w = img.naturalWidth, h = img.naturalHeight;
  const scale = Math.min(1, maxSide / Math.max(w, h));
  w = Math.round(w * scale); h = Math.round(h * scale);
  c.width = w; c.height = h;
  c.getContext("2d").drawImage(img, 0, 0, w, h);
  const rgba = cv.imread(c);
  const rgb = new cv.Mat();
  cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
  rgba.delete();
  return rgb;
}

function cropFieldOfView(rgb){
  const gray = new cv.Mat();
  cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);
  const mask = new cv.Mat();
  cv.threshold(gray, mask, 7, 255, cv.THRESH_BINARY);
  const rect = cv.boundingRect(mask);
  gray.delete(); mask.delete();
  if (rect.width < 10 || rect.height < 10) return rgb.clone();
  return rgb.roi(rect).clone();
}
function benGraham(rgb, sigma){
  const f = new cv.Mat(); rgb.convertTo(f, cv.CV_32F);
  const blur = new cv.Mat();
  cv.GaussianBlur(f, blur, new cv.Size(0, 0), sigma, sigma, cv.BORDER_DEFAULT);
  const out = new cv.Mat();
  cv.addWeighted(f, 4, blur, -4, 128, out);
  f.delete(); blur.delete();
  const u8 = new cv.Mat(); out.convertTo(u8, cv.CV_8U); out.delete();
  return u8;
}
function resizeRGB(rgb, size){
  const d = new cv.Mat();
  cv.resize(rgb, d, new cv.Size(size, size), 0, 0, cv.INTER_AREA);
  return d;
}
function matToTensor(rgb, size, mean, std){
  const data = rgb.data;
  const out = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let i = 0; i < plane; i++){
    for (let c = 0; c < 3; c++){
      let v = data[i * 3 + c] / 255.0;
      if (mean) v = (v - mean[c]) / std[c];
      out[c * plane + i] = v;
    }
  }
  return new ort.Tensor("float32", out, [1, 3, size, size]);
}
function softmax(arr){
  const m = Math.max(...arr);
  const e = arr.map(v => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}

async function runGrade(rgb){
  const cropped = cropFieldOfView(rgb);
  const bg = benGraham(cropped, 10);
  const small = resizeRGB(bg, GRADE_SIZE);
  const displayBase = resizeRGB(cropped, GRADE_SIZE);
  const tensor = matToTensor(small, GRADE_SIZE, MEAN, STD);

  const feeds = {}; feeds[gradeSession.inputNames[0]] = tensor;
  const out = await gradeSession.run(feeds);
  const names = gradeSession.outputNames;
  let logits = null, cam = null;
  for (const n of names){
    const t = out[n];
    if (t.data.length === CLASS_NAMES.length) logits = t;
    else cam = t;
  }
  const probs = softmax(Array.from(logits.data));
  const predIdx = probs.indexOf(Math.max(...probs));
  let camCanvas = null;
  if (cam){ camCanvas = renderCAM(displayBase, cam, predIdx); }
  cropped.delete(); bg.delete(); small.delete(); displayBase.delete();
  return { probs, predIdx, camCanvas };
}

function renderCAM(displayBaseMat, camTensor, predIdx){
  const [ , C, h, w] = camTensor.dims;
  const per = h * w;
  const map = new cv.Mat(h, w, cv.CV_32F);
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < per; i++){
    let v = camTensor.data[predIdx * per + i];
    if (v < 0) v = 0;
    map.data32F[i] = v;
    if (v < mn) mn = v; if (v > mx) mx = v;
  }
  const rng = (mx - mn) || 1;
  for (let i = 0; i < per; i++) map.data32F[i] = (map.data32F[i] - mn) / rng;
  const up = new cv.Mat();
  cv.resize(map, up, new cv.Size(GRADE_SIZE, GRADE_SIZE), 0, 0, cv.INTER_LINEAR);
  const u8 = new cv.Mat(); up.convertTo(u8, cv.CV_8U, 255.0);
  const heat = new cv.Mat(); cv.applyColorMap(u8, heat, cv.COLORMAP_JET);
  const heatRGB = new cv.Mat(); cv.cvtColor(heat, heatRGB, cv.COLOR_BGR2RGB);
  const blend = new cv.Mat();
  cv.addWeighted(displayBaseMat, 0.6, heatRGB, 0.4, 0, blend);
  const canvas = document.createElement("canvas");
  cv.imshow(canvas, blend);
  map.delete(); up.delete(); u8.delete(); heat.delete(); heatRGB.delete(); blend.delete();
  return canvas;
}

async function runLesion(rgb){
  if (!lesionSession) return null;
  const small = resizeRGB(rgb, LESION_SIZE);
  const tensor = matToTensor(small, LESION_SIZE, null, null);
  const feeds = {}; feeds[lesionSession.inputNames[0]] = tensor;
  const out = await lesionSession.run(feeds);
  const masks = out[lesionSession.outputNames[0]];
  const S = masks.dims[2], plane = S * S;
  const base = resizeRGB(rgb, S);
  const overlay = base.clone();
  const summary = [];
  for (let k = 0; k < LESIONS.length; k++){
    const [r, g, b] = LESIONS[k].color;
    let count = 0;
    for (let i = 0; i < plane; i++){
      if (masks.data[k * plane + i] >= 0.5){
        overlay.data[i * 3]     = Math.round(0.5 * overlay.data[i * 3]     + 0.5 * r);
        overlay.data[i * 3 + 1] = Math.round(0.5 * overlay.data[i * 3 + 1] + 0.5 * g);
        overlay.data[i * 3 + 2] = Math.round(0.5 * overlay.data[i * 3 + 2] + 0.5 * b);
        count++;
      }
    }
    summary.push({ ...LESIONS[k], present: count > 0, areaPct: +(100 * count / plane).toFixed(3) });
  }
  const canvas = document.createElement("canvas");
  cv.imshow(canvas, overlay);
  small.delete(); base.delete(); overlay.delete();
  return { canvas, summary };
}

async function analyze(file){
  if (!cvReady){ status("OpenCV is still loading — try again in a moment.", "err"); return; }
  try {
    $("results").style.display = "none";
    status("Analyzing…");
    await ensureSessions();
    const img = await fileToImage(file);
    $("inputPreview").src = img.src;
    const rgb = imageToRGBMat(img);
    const grade = await runGrade(rgb);
    const lesion = await runLesion(rgb);
    rgb.delete();
    renderResults(grade, lesion);
    status("Done.", "ok");
  } catch(e){
    console.error(e);
    status("Analysis failed: " + (e && e.message ? e.message : e) + ". If the models "
      + "aren't uploaded yet, see the note below.", "err");
  }
}

function renderResults(grade, lesion){
  $("results").style.display = "block";
  const conf = (grade.probs[grade.predIdx] * 100).toFixed(1);
  const referable = grade.predIdx >= 2;
  $("grade").textContent = CLASS_NAMES[grade.predIdx];
  $("confidence").textContent = conf + "% confidence";
  $("referable").textContent = referable ? "Referable (grade ≥ 2)" : "Non-referable (grade < 2)";
  $("referable").className = "pill " + (referable ? "warn" : "ok");
  const bars = grade.probs.map((p, i) =>
    `<div class="bar-row"><span>${CLASS_NAMES[i]}</span>
       <span class="bar"><span style="width:${(p*100).toFixed(1)}%"></span></span>
       <span class="pct">${(p*100).toFixed(1)}%</span></div>`).join("");
  $("probs").innerHTML = bars;
  const camWrap = $("camWrap");
  camWrap.innerHTML = "";
  if (grade.camCanvas){ camWrap.appendChild(grade.camCanvas); $("camSection").style.display = "block"; }
  else { $("camSection").style.display = "none"; }
  const lesionSection = $("lesionSection");
  if (lesion){
    lesionSection.style.display = "block";
    const lw = $("lesionWrap"); lw.innerHTML = ""; lw.appendChild(lesion.canvas);
    $("lesionTable").innerHTML =
      "<tr><th>Sign</th><th>Detected</th><th>Area %</th></tr>" +
      lesion.summary.map(s => {
        const sw = `rgb(${s.color[0]},${s.color[1]},${s.color[2]})`;
        return `<tr><td><span class="dot" style="background:${sw}"></span>${s.name}</td>
          <td>${s.present ? "Yes" : "—"}</td><td>${s.areaPct}</td></tr>`;
      }).join("");
  } else { lesionSection.style.display = "none"; }
}

window.addEventListener("DOMContentLoaded", () => {
  const fi = $("fileInput");
  fi.addEventListener("change", e => { if (e.target.files[0]) analyze(e.target.files[0]); });
  status("Loading OpenCV…");
});
