document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("analyze-form");
    const textarea = document.getElementById("ticket_text");
    const charCount = document.getElementById("char-count");
    const clearBtn = document.getElementById("clear-btn");
    const submitBtn = document.getElementById("submit-btn");
    const submitIcon = document.getElementById("submit-icon");
    const spinner = document.getElementById("spinner");
    const resultContainer = document.getElementById("result-container");
    const presetBtns = document.querySelectorAll(".preset-btn");
    const copyBtn = document.getElementById("copy-response-btn");
    const copyText = document.getElementById("copy-text");

    // 1. Character count update
    function updateCharCount() {
        const len = textarea.value.length;
        charCount.textContent = `${len} character${len === 1 ? '' : 's'}`;
    }

    if (textarea) {
        textarea.addEventListener("input", updateCharCount);
    }

    // 2. Preset Quick-Test Buttons
    presetBtns.forEach(btn => {
        btn.addEventListener("click", function () {
            const text = this.getAttribute("data-text");
            textarea.value = text;
            updateCharCount();
            textarea.focus();
            // Automatically trigger analysis
            form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
        });
    });

    // 3. Clear Button
    if (clearBtn) {
        clearBtn.addEventListener("click", function () {
            textarea.value = "";
            updateCharCount();
            resultContainer.classList.add("d-none");
            textarea.focus();
        });
    }

    // 4. Copy Response to Clipboard
    if (copyBtn) {
        copyBtn.addEventListener("click", function () {
            const responseText = document.getElementById("res-suggested-response").innerText;
            if (!responseText) return;

            navigator.clipboard.writeText(responseText).then(() => {
                copyText.textContent = "Copied to Clipboard!";
                copyBtn.classList.replace("btn-outline-primary", "btn-success");
                setTimeout(() => {
                    copyText.textContent = "Copy Response";
                    copyBtn.classList.replace("btn-success", "btn-outline-primary");
                }, 2500);
            }).catch(err => {
                console.error("Failed to copy:", err);
            });
        });
    }

    // 5. Form Submission & Real-time ML Inference
    if (form) {
        form.addEventListener("submit", function (e) {
            e.preventDefault();

            const text = textarea.value.trim();
            if (!text) {
                alert("Please enter a customer support message.");
                return;
            }

            if (text.length < 5) {
                alert("Please enter at least 5 characters for meaningful analysis.");
                return;
            }

            // Set loading state
            submitBtn.disabled = true;
            spinner.classList.remove("d-none");
            submitIcon.classList.add("d-none");

            fetch("/api/analyze", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ ticket_text: text })
            })
            .then(res => {
                if (!res.ok) {
                    return res.json().then(errData => { throw new Error(errData.error || "Inference request failed"); });
                }
                return res.json();
            })
            .then(data => {
                // Populate Result Data
                document.getElementById("res-ticket-id").textContent = data.ticket_id || "N/A";
                document.getElementById("res-category").textContent = data.category;
                document.getElementById("res-cat-conf").textContent = `${data.category_confidence}%`;
                document.getElementById("res-cat-bar").style.width = `${data.category_confidence}%`;

                // Priority Badge & Border Styling
                const prioEl = document.getElementById("res-priority");
                const prioBadge = document.getElementById("res-prio-badge");
                const prioDesc = document.getElementById("res-prio-desc");
                const prioBorder = document.getElementById("prio-card-border");

                prioEl.textContent = data.priority;
                prioBorder.className = "card h-100 border-0 shadow-sm rounded-3 result-metric-card border-top border-4";

                if (data.priority === "High") {
                    prioBorder.classList.add("border-danger");
                    prioBadge.className = "badge bg-danger text-white rounded-pill px-2 py-1";
                    prioBadge.textContent = "Urgent SLA (1-2h)";
                    prioDesc.textContent = "Urgency signals detected. Expedited routing assigned.";
                } else if (data.priority === "Medium") {
                    prioBorder.classList.add("border-warning");
                    prioBadge.className = "badge bg-warning text-dark rounded-pill px-2 py-1";
                    prioBadge.textContent = "Standard SLA (4-6h)";
                    prioDesc.textContent = "Standard queue priority for regular operations.";
                } else {
                    prioBorder.classList.add("border-success");
                    prioBadge.className = "badge bg-success text-white rounded-pill px-2 py-1";
                    prioBadge.textContent = "General SLA (24h)";
                    prioDesc.textContent = "Informational request with standard response window.";
                }

                // Sentiment Badge & Border Styling
                const sentEl = document.getElementById("res-sentiment");
                const sentBadge = document.getElementById("res-sent-badge");
                const sentConf = document.getElementById("res-sent-conf");
                const sentBorder = document.getElementById("sent-card-border");

                sentEl.textContent = data.sentiment;
                sentConf.textContent = `${data.sentiment_confidence}%`;
                sentBorder.className = "card h-100 border-0 shadow-sm rounded-3 result-metric-card border-top border-4";

                if (data.sentiment === "Negative") {
                    sentBorder.classList.add("border-danger");
                    sentBadge.className = "badge bg-danger-subtle text-danger border border-danger rounded-pill px-2 py-1";
                    sentBadge.textContent = "Dissatisfied / Distressed";
                } else if (data.sentiment === "Positive") {
                    sentBorder.classList.add("border-success");
                    sentBadge.className = "badge bg-success-subtle text-success border border-success rounded-pill px-2 py-1";
                    sentBadge.textContent = "Satisfied / Complimentary";
                } else {
                    sentBorder.classList.add("border-secondary");
                    sentBadge.className = "badge bg-secondary-subtle text-secondary border rounded-pill px-2 py-1";
                    sentBadge.textContent = "Objective / Neutral";
                }

                // Suggested Response
                document.getElementById("res-suggested-response").textContent = data.suggested_response;
                document.getElementById("res-model-used").textContent = data.model_used || "Logistic Regression";

                // Reveal results with smooth scroll
                resultContainer.classList.remove("d-none");
                resultContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
            })
            .catch(err => {
                alert("Error during ticket analysis: " + err.message);
            })
            .finally(() => {
                submitBtn.disabled = false;
                spinner.classList.add("d-none");
                submitIcon.classList.remove("d-none");
            });
        });
    }
});
