/**
 * static/js/app.js
 * NAAC Accreditation AI Assistant – Frontend Logic
 */

const API_BASE = "";

// ── Mode Management ───────────────────────────────────────────────────────────
let currentMode = "qa";

function setMode(mode) {
    currentMode = mode;
    // Hide all panels
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    document.querySelectorAll(".action-btn").forEach(b => b.classList.remove("active"));

    // Show selected panel
    const panelMap = { qa: "qaPanel", ssr: "ssrPanel", criteria: "criteriaPanel", practice: "practicePanel", upload: "uploadPanel" };
    const btnIndex = ["qa", "ssr", "criteria", "practice", "upload"].indexOf(mode);
    document.getElementById(panelMap[mode]).classList.add("active");
    document.querySelectorAll(".action-btn")[btnIndex]?.classList.add("active");
}

function quickCriteria(num) {
    setMode("criteria");
    document.getElementById("criteriaNum").value = num;
}

// ── Message Rendering ─────────────────────────────────────────────────────────
function addMessage(role, content, sources = []) {
    const container = document.getElementById("chatMessages");

    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role}`;

    const headerDiv = document.createElement("div");
    headerDiv.className = "message-header";
    headerDiv.innerHTML = role === "user"
        ? `<span class="sender">You</span><span class="avatar">👤</span>`
        : `<span class="avatar">🤖</span><span class="sender">NAAC Assistant <small>(IBM Granite)</small></span>`;

    const bodyDiv = document.createElement("div");
    bodyDiv.className = "message-body";
    bodyDiv.innerHTML = formatContent(content);

    msgDiv.appendChild(headerDiv);
    msgDiv.appendChild(bodyDiv);

    if (sources && sources.length > 0) {
        const srcDiv = document.createElement("div");
        srcDiv.className = "sources-block";
        srcDiv.innerHTML = `<strong>📚 Sources:</strong> ${sources.map(s => `[${s.index}] ${s.source_file}${s.page !== "N/A" ? ` (p.${s.page})` : ""}`).join(" · ")}`;
        msgDiv.appendChild(srcDiv);
    }

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function formatContent(text) {
    return text
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/^#{1,3} (.+)$/gm, "<strong>$1</strong>")
        .replace(/\n- /g, "\n• ")
        .replace(/\n/g, "<br>");
}

function addLoadingMessage() {
    const container = document.getElementById("chatMessages");
    const loadDiv = document.createElement("div");
    loadDiv.id = "loadingMsg";
    loadDiv.className = "loading";
    loadDiv.innerHTML = `<div class="spinner"></div> IBM Granite is thinking...`;
    container.appendChild(loadDiv);
    container.scrollTop = container.scrollHeight;
}

function removeLoadingMessage() {
    const el = document.getElementById("loadingMsg");
    if (el) el.remove();
}

function setStatus(text, ok = true) {
    const si = document.getElementById("statusIndicator");
    si.innerHTML = `<span class="dot" style="background:${ok ? "#42be65" : "#fa4d56"}"></span> ${text}`;
}

// ── API Calls ─────────────────────────────────────────────────────────────────
async function apiFetch(endpoint, payload) {
    setStatus("Processing...", true);
    addLoadingMessage();
    try {
        const res = await fetch(`${API_BASE}${endpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        removeLoadingMessage();
        if (!res.ok) throw new Error(data.detail || "Server error");
        setStatus("Ready", true);
        return data;
    } catch (err) {
        removeLoadingMessage();
        setStatus("Error", false);
        addMessage("assistant",
            `**Error:** ${err.message}\n\n` +
            `If you see a credentials error, open .env and set:\n` +
            `  IBM_CLOUD_API_KEY=your_real_key\n` +
            `  IBM_PROJECT_ID=your_real_project_id\n\n` +
            `Then restart the server. The app will work in fallback mode (RAG only) without credentials.`
        );
        return null;
    }
}

// ── Submit Handlers ───────────────────────────────────────────────────────────
async function submitQA() {
    const input = document.getElementById("qaInput");
    const question = input.value.trim();
    if (!question) return;
    addMessage("user", question);
    input.value = "";
    const data = await apiFetch("/api/chat/ask", { question });
    if (data) addMessage("assistant", data.content, data.sources);
}

async function submitSSR() {
    const criterion = document.getElementById("ssrCriterion").value;
    const section = document.getElementById("ssrSection").value.trim() || "General Overview";
    const institution_data = document.getElementById("ssrData").value.trim();
    addMessage("user", `📝 Draft SSR – Criterion ${criterion}: ${section}`);
    const data = await apiFetch("/api/chat/draft-ssr", { criterion, section, institution_data });
    if (data) addMessage("assistant", data.content, data.sources);
}

async function submitCriteria() {
    const criterion_number = document.getElementById("criteriaNum").value;
    addMessage("user", `📋 Get criteria inputs for Criterion ${criterion_number}`);
    const data = await apiFetch("/api/chat/criteria-inputs", { criterion_number });
    if (data) addMessage("assistant", data.content, data.sources);
}

async function submitPractice() {
    const input = document.getElementById("practiceInput");
    const domain = input.value.trim();
    if (!domain) return;
    addMessage("user", `⭐ Document best practice: ${domain}`);
    input.value = "";
    const data = await apiFetch("/api/chat/best-practice", { domain });
    if (data) addMessage("assistant", data.content, data.sources);
}

// ── Document Upload ───────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("fileInput");
    const filePreview = document.getElementById("filePreview");
    const uploadZone = document.getElementById("uploadZone");

    fileInput.addEventListener("change", () => {
        const file = fileInput.files[0];
        if (file) {
            filePreview.textContent = `✅ Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            filePreview.classList.remove("hidden");
        }
    });

    // Drag and drop
    uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.style.borderColor = "#0f62fe"; });
    uploadZone.addEventListener("dragleave", () => { uploadZone.style.borderColor = "#d0d7de"; });
    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = "#d0d7de";
        const file = e.dataTransfer.files[0];
        if (file) {
            const dt = new DataTransfer();
            dt.items.add(file);
            fileInput.files = dt.files;
            filePreview.textContent = `✅ Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            filePreview.classList.remove("hidden");
        }
    });

    // Allow Enter key to submit Q&A
    document.getElementById("qaInput").addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submitQA(); }
    });

    // Init mode
    setMode("qa");
});

async function uploadDocument() {
    const fileInput = document.getElementById("fileInput");
    const category = document.getElementById("uploadCategory").value;
    const file = fileInput.files[0];
    if (!file) { alert("Please select a file first."); return; }

    setStatus("Uploading...", true);
    addLoadingMessage();
    try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("category", category);
        const res = await fetch("/api/ingest/upload", { method: "POST", body: formData });
        const data = await res.json();
        removeLoadingMessage();
        if (!res.ok) throw new Error(data.detail || "Upload failed");
        setStatus("Ready", true);
        addMessage("assistant", `✅ **Document uploaded and indexed!**\n\n📄 File: ${file.name}\n📂 Category: ${category}\n🔢 Chunks added to knowledge base: ${data.chunks_added}`);
        document.getElementById("filePreview").classList.add("hidden");
        fileInput.value = "";
    } catch (err) {
        removeLoadingMessage();
        setStatus("Error", false);
        addMessage("assistant", `⚠️ Upload error: ${err.message}`);
    }
}
