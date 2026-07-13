// Minimal popup: send the textarea (or the page selection) to the screening API
// and render the stable /screen response. Point API_URL at your deployment.
const API_URL = "http://127.0.0.1:8000/screen";

async function screenText(text) {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error((await res.json()).errors?.join("; ") || res.statusText);
  return res.json();
}

function render(r) {
  const out = document.getElementById("out");
  const cls = r.verdict === "FAKE" ? "fake" : "real";
  const review = r.needs_review ? " · ⚠️ needs human review" : "";
  const techniques = r.signals.manipulation.techniques.length
    ? `<div class="muted">Manipulation: ${r.signals.manipulation.techniques.join(", ")}</div>` : "";
  const authority = r.signals.fabricated_authority.high
    ? `<div class="muted">Fabricated-authority register detected — verify the cited source exists.</div>` : "";
  const evidence = (r.explanation.evidence || [])
    .map(e => `<li>${e.label} match (sim ${e.score}): ${e.snippet}…</li>`).join("");
  out.innerHTML =
    `<div class="verdict ${cls}">${r.verdict} · ${(r.fake_probability * 100).toFixed(0)}% · ${r.confidence}${review}</div>` +
    techniques + authority +
    (evidence ? `<div class="muted">Closest known articles:</div><ul>${evidence}</ul>` : "") +
    `<p class="muted">${r.disclaimer}</p>`;
}

document.getElementById("go").addEventListener("click", async () => {
  const out = document.getElementById("out");
  const text = document.getElementById("text").value.trim();
  if (!text) { out.textContent = "Paste some text first."; return; }
  out.textContent = "Screening…";
  try { render(await screenText(text)); }
  catch (e) { out.textContent = "Error: " + e.message; }
});
