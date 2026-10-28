/**
 * Claim-Level Hallucination Detector — Interactive Frontend Logic
 * Connects UI presets and forms to the Flask backend endpoints.
 */

let PRESETS = {};
let LAST_RESULT = null;

document.addEventListener("DOMContentLoaded", () => {
  initSystemHealth();
  initPresets();
  setupEventListeners();
});

// 1. Fetch system status on load
async function initSystemHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();
    document.getElementById("stat-retriever").textContent = data.retriever_backend || "dense";
    document.getElementById("stat-verifier").textContent = data.verifier_backend || "DeBERTa-v3";
    document.getElementById("stat-corpus").textContent = `${data.corpus_size || 8} Passages`;
  } catch (err) {
    console.warn("Could not fetch system status:", err);
    document.getElementById("stat-retriever").textContent = "all-MiniLM-L6-v2";
    document.getElementById("stat-verifier").textContent = "DeBERTa-v3";
    document.getElementById("stat-corpus").textContent = "8 Reference Docs";
  }
}

// 2. Fetch and render preset demonstration examples
async function initPresets() {
  try {
    const res = await fetch("/api/presets");
    if (res.ok) {
      PRESETS = await res.json();
      selectPreset("factual");
    }
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

function selectPreset(key) {
  const preset = PRESETS[key];
  if (!preset) return;

  document.querySelectorAll(".btn-preset").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.preset === key);
  });

  document.getElementById("preset-info-title").textContent = preset.title;
  document.getElementById("preset-info-desc").textContent = `${preset.description} (Expected: ${preset.expected})`;

  const textarea = document.getElementById("input-text");
  textarea.value = preset.text;
}

// 3. Setup UI event listeners
function setupEventListeners() {
  const presetButtons = document.querySelectorAll(".btn-preset");
  presetButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      selectPreset(btn.dataset.preset);
    });
  });

  const verifyBtn = document.getElementById("btn-verify");
  if (verifyBtn) {
    verifyBtn.addEventListener("click", handleVerification);
  }
}

// 4. API verification pipeline caller
async function handleVerification() {
  const textInput = document.getElementById("input-text");
  const text = textInput ? textInput.value.trim() : "";
  const useScalar = document.getElementById("toggle-scalar").checked;

  if (!text) {
    alert("Please enter a paragraph or select a preset example.");
    return;
  }

  const verifyBtn = document.getElementById("btn-verify");
  const btnText = verifyBtn.querySelector(".btn-text");
  const spinner = document.getElementById("btn-spinner");
  const resultsArea = document.getElementById("results-area");

  // Set loading state
  verifyBtn.disabled = true;
  btnText.textContent = "Verifying Claims...";
  spinner.style.display = "inline-block";

  try {
    const response = await fetch("/api/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        use_scalar: useScalar
      })
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.error || "Verification pipeline failed.");
    }

    const data = await response.json();
    LAST_RESULT = data;
    renderResults(data);
    resultsArea.style.display = "block";
    resultsArea.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    console.error("Verification error:", err);
    alert(`Error running verification pipeline: ${err.message}`);
  } finally {
    verifyBtn.disabled = false;
    btnText.textContent = "Run Verification Pipeline";
    spinner.style.display = "none";
  }
}

// 5. Render results: cards, probability bars, and badges
function renderResults(data) {
  renderSummaryBar(data);
  renderClaimCards(data.claims);
}

// 6. Summary Bar & Score Meter (Milestone 7)
function renderSummaryBar(data) {
  const scoreCircle = document.getElementById("score-circle");
  const scoreRate = document.getElementById("score-rate");
  const headline = document.getElementById("summary-headline");
  const detail = document.getElementById("summary-detail");
  const latency = document.getElementById("summary-latency");

  const supp = data.counts.supported || 0;
  const contra = data.counts.contradicted || 0;
  const unsupp = data.counts.unsupported || 0;
  const total = data.num_claims || 1;
  const pct = data.hallucination_pct;

  scoreRate.textContent = `${pct}%`;
  latency.textContent = data.elapsed_ms || 0;

  // Visual meter color
  scoreCircle.classList.remove("clean", "dirty");
  if (pct === 0) {
    scoreCircle.classList.add("clean");
    headline.textContent = "Fully Factual & Grounded Response";
    detail.textContent = `All ${total} atomic claims are verified and supported by corpus evidence.`;
  } else if (pct === 100) {
    scoreCircle.classList.add("dirty");
    headline.textContent = "High Hallucination Detected (100%)";
    detail.textContent = `All ${total} extracted claims contradict or lack factual grounding.`;
  } else {
    scoreCircle.classList.add("dirty");
    headline.textContent = `Partially Hallucinated Response (${pct}%)`;
    detail.textContent = `${supp} supported, ${contra} contradicted, and ${unsupp} unsupported claims out of ${total}.`;
  }

  // Count breakdown badges
  document.getElementById("count-supported").textContent = supp;
  document.getElementById("count-contradicted").textContent = contra;
  document.getElementById("count-unsupported").textContent = unsupp;
}

function renderClaimCards(claims) {
  const container = document.getElementById("claims-container");
  container.innerHTML = "";

  if (!claims || claims.length === 0) {
    container.innerHTML = "<p class='text-muted'>No claims could be extracted.</p>";
    return;
  }

  claims.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "claim-card";

    // Verdict Badge styling
    let badgeClass = "badge-supported";
    let badgeText = "Supported";
    let icon = "✓";

    if (item.label.toLowerCase() === "contradicted") {
      badgeClass = "badge-contradicted";
      badgeText = "Contradicted";
      icon = "✗";
    } else if (item.label.toLowerCase() === "unsupported") {
      badgeClass = "badge-unsupported";
      badgeText = "Unsupported";
      icon = "?";
    }

    // Probability percentages
    const entPct = (item.entailment * 100).toFixed(1);
    const contraPct = (item.contradiction * 100).toFixed(1);
    const neuPct = (item.neutral * 100).toFixed(1);
    const simPct = (item.similarity * 100).toFixed(1);

    card.innerHTML = `
      <div class="claim-card-header">
        <div>
          <div class="claim-index">Atomic Claim ${idx + 1}</div>
          <div class="claim-text">"${escapeHtml(item.claim)}"</div>
        </div>
        <span class="badge-verdict ${badgeClass}">${icon} ${badgeText}</span>
      </div>

      <div class="evidence-box">
        <div class="evidence-header">
          <span>Top Retrieved Evidence</span>
          <span class="evidence-sim">Cosine Similarity: ${simPct}%</span>
        </div>
        <div class="evidence-quote">"${escapeHtml(item.evidence)}"</div>
      </div>

      <div class="nli-bars-row">
        <div class="nli-bar-group">
          <div class="nli-bar-label">
            <span>Entailment P(ent)</span>
            <span>${entPct}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill bar-fill-ent" style="width: ${entPct}%;"></div>
          </div>
        </div>

        <div class="nli-bar-group">
          <div class="nli-bar-label">
            <span>Contradiction P(contra)</span>
            <span>${contraPct}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill bar-fill-contra" style="width: ${contraPct}%;"></div>
          </div>
        </div>

        <div class="nli-bar-group">
          <div class="nli-bar-label">
            <span>Neutral P(neu)</span>
            <span>${neuPct}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill bar-fill-neu" style="width: ${neuPct}%;"></div>
          </div>
        </div>
      </div>

      <!-- Expandable Audit Trace Receipt (Milestone 6) -->
      <details class="audit-details">
        <summary>🔍 View Factual Audit Trace Receipt</summary>
        <div class="audit-content">
          <div><strong>Claim ID:</strong> c_${idx + 1}</div>
          <div><strong>Premise / Evidence:</strong> "${escapeHtml(item.evidence)}"</div>
          <div><strong>Hypothesis / Claim:</strong> "${escapeHtml(item.claim)}"</div>
          <div><strong>Inference Backend:</strong> ${escapeHtml(item.backend)}</div>
          <div><strong>Entailment Check:</strong> &tau;<sub>ent</sub> &ge; 0.55 &rarr; ${item.entailment >= 0.55 ? "MET" : "NOT MET"} (Score: ${item.entailment})</div>
          <div><strong>Contradiction Check:</strong> &tau;<sub>contra</sub> &ge; 0.30 &rarr; ${item.contradiction >= 0.30 ? "TRIGGERED" : "NOT TRIGGERED"} (Score: ${item.contradiction})</div>
          <div><strong>Retrieval Gate:</strong> &tau;<sub>sim</sub> &ge; 0.05 &rarr; ${item.similarity >= 0.05 ? "PASS" : "FAIL"} (Score: ${item.similarity})</div>
          <div style="margin-top:0.4rem; padding-top:0.4rem; border-top:1px dashed #cbd5e1;">
            <strong>Decision Rationale:</strong> ${getDecisionRationale(item)}
          </div>
        </div>
      </details>
    `;

    container.appendChild(card);
  });
}

function getDecisionRationale(item) {
  if (item.label.toLowerCase() === "contradicted") {
    return "Contradiction threshold reached (P(contra) >= 0.30). Immediate refutation override applied.";
  } else if (item.label.toLowerCase() === "supported") {
    return "Entailment threshold satisfied (P(ent) >= 0.55) and similarity gate bound met (sim >= 0.05).";
  } else {
    return "Evidence neither entailed nor contradicted above calibrated bounds. Classified as Unsupported / NEI.";
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
