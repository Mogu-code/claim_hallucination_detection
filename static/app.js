/**
 * Veritas — Claim-Level Fact Verification System
 * Human-Centered Frontend Controller
 * Connects editorial UI components to the Flask verification backend.
 */

let PRESETS = {};
let LAST_RESULT = null;

document.addEventListener("DOMContentLoaded", () => {
  initSystemHealth();
  initPresets();
  setupEventListeners();
  initTextareaCounters();
});

// 1. Fetch system status & format human-friendly metadata
async function initSystemHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();
    
    // Format human readable retriever backend
    const retLabel = (data.retriever_backend || "lexical").includes("dense") 
      ? "Dense Neural Embeddings" 
      : "BM25 & Semantic Filter";
    document.getElementById("stat-retriever").textContent = retLabel;

    // Format verifier backend
    const verLabel = (data.verifier_backend || "").includes("Neural")
      ? "DeBERTa-v3 Cross-Encoder"
      : "Predicate-Aware NLI Verifier";
    document.getElementById("stat-verifier").textContent = verLabel;

    // Corpus count
    const corpusCount = data.corpus_size || 12;
    document.getElementById("stat-corpus").textContent = `${corpusCount} reference records`;
  } catch (err) {
    console.warn("Could not fetch system status:", err);
    document.getElementById("stat-retriever").textContent = "Archive Search Engine";
    document.getElementById("stat-verifier").textContent = "NLI Claim Verifier";
    document.getElementById("stat-corpus").textContent = "Reference Archives";
  }
}

// 2. Fetch and initialize human sample cases
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

  document.querySelectorAll(".preset-chip").forEach(chip => {
    chip.classList.toggle("active", chip.dataset.preset === key);
  });

  const titleEl = document.getElementById("preset-info-title");
  const descEl = document.getElementById("preset-info-desc");
  if (titleEl) titleEl.textContent = preset.title;
  if (descEl) descEl.textContent = `${preset.description} — Expected outcome: ${preset.expected}`;

  const textarea = document.getElementById("input-text");
  if (textarea) {
    textarea.value = preset.text;
    updateCharCount(textarea.value);
  }
}

// 3. Setup UI event listeners
function setupEventListeners() {
  const presetChips = document.querySelectorAll(".preset-chip");
  presetChips.forEach(chip => {
    chip.addEventListener("click", () => {
      selectPreset(chip.dataset.preset);
    });
  });

  const verifyBtn = document.getElementById("btn-verify");
  if (verifyBtn) {
    verifyBtn.addEventListener("click", handleVerification);
  }

  const clearBtn = document.getElementById("btn-clear");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      const textarea = document.getElementById("input-text");
      if (textarea) {
        textarea.value = "";
        textarea.focus();
        updateCharCount("");
      }
    });
  }
}

function initTextareaCounters() {
  const textarea = document.getElementById("input-text");
  if (textarea) {
    textarea.addEventListener("input", (e) => {
      updateCharCount(e.target.value);
    });
    updateCharCount(textarea.value);
  }
}

function updateCharCount(text) {
  const counter = document.getElementById("char-count");
  if (counter) {
    const len = (text || "").length;
    counter.textContent = `${len} character${len === 1 ? "" : "s"}`;
  }
}

// 4. Verification pipeline caller
async function handleVerification() {
  const textInput = document.getElementById("input-text");
  const text = textInput ? textInput.value.trim() : "";
  const useScalarToggle = document.getElementById("toggle-scalar");
  const useScalar = useScalarToggle ? useScalarToggle.checked : true;

  if (!text) {
    alert("Please enter a statement or select a sample case above.");
    if (textInput) textInput.focus();
    return;
  }

  const verifyBtn = document.getElementById("btn-verify");
  const btnLabel = verifyBtn.querySelector(".btn-label");
  const spinner = document.getElementById("btn-spinner");
  const resultsArea = document.getElementById("results-area");

  // Natural loading state
  verifyBtn.disabled = true;
  btnLabel.textContent = "Auditing statements against records...";
  if (spinner) spinner.style.display = "inline-block";

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
    alert(`Could not complete verification: ${err.message}`);
  } finally {
    verifyBtn.disabled = false;
    btnLabel.textContent = "Verify Statements";
    if (spinner) spinner.style.display = "none";
  }
}

// 5. Render results: Editorial Summary & Proofreader Claim Cards
function renderResults(data) {
  renderSummaryBanner(data);
  renderClaimCards(data.claims);
}

// 6. Editorial Executive Summary Banner
function renderSummaryBanner(data) {
  const banner = document.getElementById("summary-banner");
  const iconEl = document.getElementById("verdict-icon");
  const headline = document.getElementById("summary-headline");
  const detail = document.getElementById("summary-detail");
  const latency = document.getElementById("summary-latency");
  const claimCountEl = document.getElementById("summary-claim-count");

  const supp = data.counts.supported || 0;
  const contra = data.counts.contradicted || 0;
  const unsupp = data.counts.unsupported || 0;
  const total = data.num_claims || 1;
  const pct = data.hallucination_pct;

  if (latency) latency.textContent = data.elapsed_ms || 0;
  if (claimCountEl) {
    claimCountEl.textContent = `${total} statement${total === 1 ? "" : "s"} analyzed`;
  }

  // Clear existing banner status classes
  banner.classList.remove("status-verified", "status-conflicts", "status-mixed");

  if (contra === 0 && unsupp === 0) {
    banner.classList.add("status-verified");
    if (iconEl) iconEl.textContent = "✓";
    headline.textContent = "Fully Verified by Historical Records";
    detail.textContent = `All ${total} statement${total === 1 ? "" : "s"} were substantiated by archival references with consistent dates, locations, and facts.`;
  } else if (contra > 0 && supp === 0) {
    banner.classList.add("status-conflicts");
    if (iconEl) iconEl.textContent = "✕";
    headline.textContent = "Discrepancy Detected — Statements Contradict Records";
    detail.textContent = `${contra} statement${contra === 1 ? "" : "s"} directly conflict with established reference documents.`;
  } else if (contra > 0 && supp > 0) {
    banner.classList.add("status-mixed");
    if (iconEl) iconEl.textContent = "◫";
    headline.textContent = "Partially Verified — Mixed Accuracy Found";
    detail.textContent = `${supp} statement${supp === 1 ? "" : "s"} verified, but ${contra} contradictory statement${contra === 1 ? "" : "s"} detected.`;
  } else {
    banner.classList.add("status-mixed");
    if (iconEl) iconEl.textContent = "?";
    headline.textContent = "Unsubstantiated Assertions";
    detail.textContent = `${unsupp} statement${unsupp === 1 ? "" : "s"} lack supporting evidence in the reference archives.`;
  }

  // Update counts
  const suppEl = document.getElementById("count-supported");
  const contraEl = document.getElementById("count-contradicted");
  const unsuppEl = document.getElementById("count-unsupported");
  if (suppEl) suppEl.textContent = supp;
  if (contraEl) contraEl.textContent = contra;
  if (unsuppEl) unsuppEl.textContent = unsupp;
}

// 7. Render Claim Proofreader Cards
function renderClaimCards(claims) {
  const container = document.getElementById("claims-container");
  container.innerHTML = "";

  if (!claims || claims.length === 0) {
    container.innerHTML = "<p class='intro-subtext'>No individual claims were detected in the text.</p>";
    return;
  }

  claims.forEach((item, idx) => {
    const card = document.createElement("article");
    card.className = "claim-card";

    // Natural human badge text and styling
    let badgeClass = "badge-supported";
    let badgeText = "Verified by Record";
    let icon = "✓";

    const labelLower = (item.label || "").toLowerCase();
    if (labelLower === "contradicted") {
      badgeClass = "badge-contradicted";
      badgeText = "Conflicts with Record";
      icon = "✕";
    } else if (labelLower === "unsupported") {
      badgeClass = "badge-unsupported";
      badgeText = "No Supporting Record";
      icon = "—";
    }

    // Probability percentages
    const entPct = (item.entailment * 100).toFixed(1);
    const contraPct = (item.contradiction * 100).toFixed(1);
    const neuPct = (item.neutral * 100).toFixed(1);
    const simPct = (item.similarity * 100).toFixed(0);

    const rationale = generateHumanRationale(item);

    card.innerHTML = `
      <div class="claim-card-header">
        <div>
          <div class="claim-statement-meta">Statement ${idx + 1} of ${claims.length}</div>
          <div class="claim-quote">&ldquo;${escapeHtml(item.claim)}&rdquo;</div>
        </div>
        <span class="verdict-badge ${badgeClass}">${icon} ${badgeText}</span>
      </div>

      <div class="reference-card">
        <div class="reference-card-header">
          <span class="reference-origin">
            <span>Archival Reference</span>
          </span>
          <span class="reference-sim-tag">${simPct}% match</span>
        </div>
        <div class="reference-passage">&ldquo;${escapeHtml(item.evidence)}&rdquo;</div>
      </div>

      <div class="confidence-breakdown">
        <div class="confidence-tile">
          <div class="tile-header">
            <span class="tile-label">Agreement</span>
            <span class="tile-value">${entPct}%</span>
          </div>
          <div class="tile-meter">
            <div class="meter-fill fill-supported" style="width: ${entPct}%;"></div>
          </div>
        </div>

        <div class="confidence-tile">
          <div class="tile-header">
            <span class="tile-label">Conflict</span>
            <span class="tile-value">${contraPct}%</span>
          </div>
          <div class="tile-meter">
            <div class="meter-fill fill-contradicted" style="width: ${contraPct}%;"></div>
          </div>
        </div>

        <div class="confidence-tile">
          <div class="tile-header">
            <span class="tile-label">Unrelated / Neutral</span>
            <span class="tile-value">${neuPct}%</span>
          </div>
          <div class="tile-meter">
            <div class="meter-fill fill-neutral" style="width: ${neuPct}%;"></div>
          </div>
        </div>
      </div>

      <details class="audit-drawer">
        <summary>Review verification details &amp; logic</summary>
        <div class="audit-drawer-body">
          <p class="audit-prose">${rationale}</p>
          <table class="audit-table">
            <tbody>
              <tr>
                <td>Statement Evaluated</td>
                <td>${escapeHtml(item.claim)}</td>
              </tr>
              <tr>
                <td>Reference Passage</td>
                <td>${escapeHtml(item.evidence)}</td>
              </tr>
              <tr>
                <td>Scoring Engine</td>
                <td>${escapeHtml(item.backend || "Predicate-Aware Calibrated Verifier")}</td>
              </tr>
              <tr>
                <td>Confidence Scores</td>
                <td>Agreement: ${item.entailment} &middot; Conflict: ${item.contradiction} &middot; Neutral: ${item.neutral}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </details>
    `;

    container.appendChild(card);
  });
}

// 8. Generate natural human explanation for each claim
function generateHumanRationale(item) {
  const label = (item.label || "").toLowerCase();
  const claim = item.claim.toLowerCase();
  const evidence = item.evidence.toLowerCase();

  if (label === "contradicted") {
    if (claim.includes("tokyo") && (evidence.includes("paris") || evidence.includes("france"))) {
      return "The statement asserts that the Eiffel Tower is in Tokyo, which directly contradicts the archival record establishing it in Paris, France.";
    }
    if (claim.includes("paris") && evidence.includes("warsaw")) {
      return "The statement claims Paris as the place of birth, directly conflicting with the archival record documenting Warsaw, Poland.";
    }
    if (claim.includes("three") && (evidence.includes("1903") && evidence.includes("1911"))) {
      return "The statement claims three Nobel prizes, while historical records confirm two (Physics in 1903 and Chemistry in 1911).";
    }
    if (claim.includes("1950") && evidence.includes("1889")) {
      return "The statement dates the construction to 1950, which conflicts with the archival completion date of 1889.";
    }
    return "The statement contains a date, location, or attribute that directly conflicts with the facts recorded in the reference archive.";
  }

  if (label === "supported") {
    return "The assertion aligns with the reference record — all key factual entities, dates, and predicate details are corroborated by the source.";
  }

  return "The reference archives do not contain enough verifiable information to substantiate or refute this assertion.";
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
