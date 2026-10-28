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
      // Load default preset (factual) into the textarea
      selectPreset("factual");
    }
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

function selectPreset(key) {
  const preset = PRESETS[key];
  if (!preset) return;

  // Update active button state
  document.querySelectorAll(".btn-preset").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.preset === key);
  });

  // Update description box
  document.getElementById("preset-info-title").textContent = preset.title;
  document.getElementById("preset-info-desc").textContent = `${preset.description} (Expected: ${preset.expected})`;

  // Populate textarea
  const textarea = document.getElementById("input-text");
  textarea.value = preset.text;
}

// 3. Setup UI event listeners
function setupEventListeners() {
  // Preset button clicks
  const presetButtons = document.querySelectorAll(".btn-preset");
  presetButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      selectPreset(btn.dataset.preset);
    });
  });

  // Verification button click
  const verifyBtn = document.getElementById("btn-verify");
  if (verifyBtn) {
    verifyBtn.addEventListener("click", handleVerification);
  }
}

// Stub for handleVerification to be implemented in Milestone 4
async function handleVerification() {
  // Handled in next milestone
}
