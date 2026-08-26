document.addEventListener("DOMContentLoaded", function () {
    const statsEl = document.getElementById("stats-data");
    if (!statsEl) return;

    let stats = {};
    try {
        stats = JSON.parse(statsEl.textContent);
    } catch (e) {
        console.error("Failed to parse stats JSON:", e);
        return;
    }

    // 1. Category Bar Chart
    const catCanvas = document.getElementById("categoryChart");
    if (catCanvas) {
        const catData = stats.category_distribution || {};
        const labels = Object.keys(catData);
        const counts = Object.values(catData);

        new Chart(catCanvas, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Tickets",
                    data: counts,
                    backgroundColor: "rgba(0, 120, 212, 0.75)",
                    borderColor: "rgba(0, 120, 212, 1)",
                    borderWidth: 1,
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        padding: 8,
                        backgroundColor: "rgba(15, 23, 42, 0.9)"
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { stepSize: 1, precision: 0, font: { size: 10 } },
                        grid: { color: "rgba(0,0,0,0.04)" }
                    },
                    x: {
                        grid: { display: false },
                        ticks: {
                            maxRotation: 30,
                            minRotation: 0,
                            font: { size: 10 }
                        }
                    }
                }
            }
        });
    }

    // 2. Priority Doughnut Chart
    const prioCanvas = document.getElementById("priorityChart");
    if (prioCanvas) {
        const prioData = stats.priority_distribution || {};
        const labels = ["High", "Medium", "Low"];
        const counts = [prioData["High"] || 0, prioData["Medium"] || 0, prioData["Low"] || 0];

        new Chart(prioCanvas, {
            type: "doughnut",
            data: {
                labels: labels,
                datasets: [{
                    data: counts,
                    backgroundColor: [
                        "rgba(220, 38, 38, 0.85)",   // Red for High
                        "rgba(217, 119, 6, 0.85)",   // Amber for Medium
                        "rgba(22, 163, 74, 0.85)"    // Green for Low
                    ],
                    borderWidth: 2,
                    borderColor: "#ffffff"
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "right",
                        labels: { boxWidth: 10, padding: 8, font: { size: 11 } }
                    }
                },
                cutout: "65%"
            }
        });
    }

    // 3. Sentiment Doughnut Chart
    const sentCanvas = document.getElementById("sentimentChart");
    if (sentCanvas) {
        const sentData = stats.sentiment_distribution || {};
        const labels = ["Negative", "Neutral", "Positive"];
        const counts = [sentData["Negative"] || 0, sentData["Neutral"] || 0, sentData["Positive"] || 0];

        new Chart(sentCanvas, {
            type: "doughnut",
            data: {
                labels: labels,
                datasets: [{
                    data: counts,
                    backgroundColor: [
                        "rgba(239, 68, 68, 0.85)",   // Red for Negative
                        "rgba(148, 163, 184, 0.85)", // Slate for Neutral
                        "rgba(34, 197, 94, 0.85)"    // Green for Positive
                    ],
                    borderWidth: 2,
                    borderColor: "#ffffff"
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "right",
                        labels: { boxWidth: 10, padding: 8, font: { size: 11 } }
                    }
                },
                cutout: "65%"
            }
        });
    }
});
