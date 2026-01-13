// Front-end behavior for modals, form submission, and progress polling.
document.addEventListener("DOMContentLoaded", () => {
    const instructionsModalEl = document.getElementById("instructionsModal");
    const noticeModalEl = document.getElementById("noticeModal");
    const instructionCheck = document.getElementById("instructions-check");
    const privacyCheck = document.getElementById("privacy-check");
    const analyzeForm = document.getElementById("homeForm");
    const analyzeButton = document.getElementById("analyzeBtn");
    const processingOverlay = document.getElementById("processingOverlay");
    const progressBar = document.getElementById("processingProgress");
    const progressStage = document.getElementById("progressStage");
    const ajaxErrors = document.getElementById("ajaxErrors");
    const ajaxErrorsList = document.getElementById("ajaxErrorsList");
    const instructionsKey = "instructionsAccepted";
    const privacyKey = "privacyAccepted";
    const submitLabel = "Analyse Video";
    const submittingLabel = "Submitting...";

    if (!instructionsModalEl || !noticeModalEl || !window.bootstrap) return;

    const instructionsModal = new bootstrap.Modal(instructionsModalEl, {
        backdrop: "static",
        keyboard: false,
    });
    const noticeModal = new bootstrap.Modal(noticeModalEl, {
        backdrop: "static",
        keyboard: false,
    });

    // Enforce required acknowledgements before continuing.
    const requireCheck = (inputEl) => {
        if (!inputEl) return false;
        if (inputEl.checked) {
            inputEl.classList.remove("is-invalid");
            return true;
        }
        inputEl.classList.add("is-invalid");
        inputEl.focus();
        return false;
    };

    const instructionsAccepted = sessionStorage.getItem(instructionsKey) === "true";
    const privacyAccepted = sessionStorage.getItem(privacyKey) === "true";

    if (!instructionsAccepted) {
        instructionsModal.show();
    } else if (!privacyAccepted) {
        noticeModal.show();
    }

    instructionsModalEl.querySelector("[data-next]")?.addEventListener("click", () => {
        if (requireCheck(instructionCheck)) {
            sessionStorage.setItem(instructionsKey, "true");
            instructionsModal.hide();
            if (!privacyAccepted) {
                noticeModal.show();
            }
        }
    });

    noticeModalEl.querySelector("[data-close]")?.addEventListener("click", () => {
        if (requireCheck(privacyCheck)) {
            sessionStorage.setItem(privacyKey, "true");
            noticeModal.hide();
        }
    });

    // Lightweight CSRF helper for Django-style cookies.
    const getCookie = (name) => {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) {
            return parts.pop().split(";").shift();
        }
        return "";
    };

    const showAjaxErrors = (errors) => {
        if (!ajaxErrors || !ajaxErrorsList) return;
        ajaxErrorsList.innerHTML = "";
        errors.forEach((message) => {
            const item = document.createElement("li");
            item.textContent = message;
            ajaxErrorsList.appendChild(item);
        });
        ajaxErrors.classList.remove("d-none");
    };

    const setProgress = (progress, stage) => {
        const safeProgress = Math.max(0, Math.min(100, progress));
        if (progressBar) {
            progressBar.style.width = `${safeProgress.toFixed(1)}%`;
            progressBar.parentElement?.setAttribute("aria-valuenow", String(safeProgress));
        }
        if (progressStage) {
            progressStage.textContent = `${stage} (${Math.round(safeProgress)}%)`;
        }
    };

    const setBusyState = (isBusy) => {
        if (analyzeButton) {
            analyzeButton.disabled = isBusy;
            analyzeButton.textContent = isBusy ? submittingLabel : submitLabel;
        }
        if (processingOverlay) {
            processingOverlay.classList.toggle("d-none", !isBusy);
            processingOverlay.classList.toggle("d-flex", isBusy);
            processingOverlay.setAttribute("aria-hidden", String(!isBusy));
        }
    };

    const handleAjaxFailure = (messages) => {
        showAjaxErrors(messages);
        setBusyState(false);
    };

    const startPollingProgress = (progressUrl, resultsUrl) => {
        const pollInterval = setInterval(async () => {
            try {
                const progressResponse = await fetch(progressUrl, {
                    cache: "no-store",
                    headers: { "X-Requested-With": "XMLHttpRequest" },
                });
                if (!progressResponse.ok) {
                    return;
                }
                const progressData = await progressResponse.json();
                if (progressData.error) {
                    clearInterval(pollInterval);
                    handleAjaxFailure([progressData.error]);
                    return;
                }
                setProgress(progressData.progress || 0, progressData.stage || "Running");
                if (progressData.done) {
                    clearInterval(pollInterval);
                    window.location.href = progressData.results_url || resultsUrl;
                }
            } catch (pollError) {
                clearInterval(pollInterval);
                handleAjaxFailure(["Progress tracking failed. Please refresh."]);
            }
        }, 1000);
    };

    if (analyzeForm) {
        analyzeForm.addEventListener("submit", async (event) => {
            const runUrl = analyzeForm.dataset.runUrl;
            if (!runUrl) {
                return;
            }
            event.preventDefault();
            ajaxErrors?.classList.add("d-none");

            setBusyState(true);
            setProgress(1, "Starting pipeline");

            const formData = new FormData(analyzeForm);
            try {
                const response = await fetch(runUrl, {
                    method: "POST",
                    body: formData,
                    headers: {
                        "X-Requested-With": "XMLHttpRequest",
                        "X-CSRFToken": getCookie("csrftoken"),
                    },
                });
                const data = await response.json();
                if (!response.ok) {
                    handleAjaxFailure(data.errors || ["Unable to start pipeline."]);
                    return;
                }

                const progressUrl = data.progress_url;
                const resultsUrl = data.results_url;

                startPollingProgress(progressUrl, resultsUrl);
            } catch (error) {
                handleAjaxFailure(["Unable to start pipeline. Please try again."]);
            }
        });
    }
});
