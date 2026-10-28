/**
 * Claim-Level Hallucination Detector — Interactive Frontend Logic
 * Connects UI presets and forms to the Flask backend endpoints.
 */

let PRESETS = {};

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

// Stubs for rendering functions to be fleshed out in Milestones 5 and 6
function renderResults(data) {
  // Will be implemented in Milestones 5, 6, and 7
}
