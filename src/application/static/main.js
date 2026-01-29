// Front-end behavior for modals, form submission, and progress polling.
document.addEventListener("DOMContentLoaded", () => {
    const instructionsModalEl = document.getElementById("instructionsModal");
    const noticeModalEl = document.getElementById("noticeModal");
    const modelsModalEl = document.getElementById("modelsModal");
    const instructionCheck = document.getElementById("instructions-check");
    const privacyCheck = document.getElementById("privacy-check");
    const analyzeForm = document.getElementById("homeForm");
    const analyzeButton = document.getElementById("analyzeBtn");
    const processingOverlay = document.getElementById("processingOverlay");
    const progressBar = document.getElementById("processingProgress");
    const progressStage = document.getElementById("progressStage");
    const previousRunSelect = document.getElementById("previousRunSelect");
    const openPreviousRun = document.getElementById("openPreviousRun");
    const previousRunTimestamp = document.getElementById("previousRunTimestamp");
    const videoUpload = document.getElementById("videoUpload");
    const ajaxErrors = document.getElementById("ajaxErrors");
    const ajaxErrorsList = document.getElementById("ajaxErrorsList");
    const deleteRunButton = document.getElementById("deleteRunButton");
    const deleteRunForm = document.getElementById("deleteRunForm");
    const missingModelsList = document.getElementById("missingModelsList");
    const downloadModelsButton = document.getElementById("downloadModelsBtn");
    const modelsDownloadError = document.getElementById("modelsDownloadError");
    const modelsDownloadProgress = document.getElementById("modelsDownloadProgress");
    const instructionsKey = "instructionsAccepted";
    const privacyKey = "privacyAccepted";
    const submitLabel = "Analyse Video";
    const submittingLabel = "Submitting...";

    const modelStatusUrl = document.body?.dataset.modelStatusUrl;
    const modelDownloadUrl = document.body?.dataset.modelDownloadUrl;

    const fetchModelStatus = async () => {
        if (!modelStatusUrl) return null;
        const response = await fetch(modelStatusUrl, {
            cache: "no-store",
            headers: { "X-Requested-With": "XMLHttpRequest" },
        });
        if (!response.ok) {
            return null;
        }
        return response.json();
    };

    const renderMissingModels = (missing) => {
        if (!missingModelsList) return;
        missingModelsList.innerHTML = "";
        missing.forEach((name) => {
            const item = document.createElement("li");
            item.textContent = name;
            missingModelsList.appendChild(item);
        });
    };

    const showModelsModalIfMissing = async (modelsModal) => {
        if (!modelsModal || !modelStatusUrl) return;
        const status = await fetchModelStatus();
        if (!status || !Array.isArray(status.missing)) return;
        if (status.missing.length === 0) return;
        renderMissingModels(status.missing);
        modelsDownloadError?.classList.add("d-none");
        if (modelsDownloadProgress) {
            const progressBar = modelsDownloadProgress.querySelector(".progress-bar");
            modelsDownloadProgress.classList.add("d-none");
            modelsDownloadProgress.setAttribute("aria-valuenow", "0");
            if (progressBar) {
                progressBar.style.width = "0%";
            }
        }
        modelsModal.show();
    };

    if (instructionsModalEl && noticeModalEl && window.bootstrap) {
        const instructionsModal = new bootstrap.Modal(instructionsModalEl, {
            backdrop: "static",
            keyboard: false,
        });
        const noticeModal = new bootstrap.Modal(noticeModalEl, {
            backdrop: "static",
            keyboard: false,
        });
        const modelsModal = modelsModalEl
            ? new bootstrap.Modal(modelsModalEl, { backdrop: "static", keyboard: false })
            : null;

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
        } else if (modelsModal) {
            showModelsModalIfMissing(modelsModal);
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
                showModelsModalIfMissing(modelsModal);
            }
        });

        if (downloadModelsButton && modelDownloadUrl && modelsModal) {
            downloadModelsButton.addEventListener("click", async () => {
                downloadModelsButton.disabled = true;
                downloadModelsButton.textContent = "Downloading...";
                modelsDownloadError?.classList.add("d-none");
                let downloadSucceeded = false;
                if (modelsDownloadProgress) {
                    const progressBar = modelsDownloadProgress.querySelector(".progress-bar");
                    modelsDownloadProgress.classList.remove("d-none");
                    if (progressBar) {
                        progressBar.style.width = "0%";
                    }
                    modelsDownloadProgress.setAttribute("aria-valuenow", "0");
                }
                try {
                    const response = await fetch(modelDownloadUrl, {
                        method: "POST",
                        headers: {
                            "X-Requested-With": "XMLHttpRequest",
                            "X-CSRFToken": getCookie("csrftoken"),
                        },
                    });
                    const data = await response.json();
                    if (!response.ok) {
                        const errors = data.errors || ["Download failed."];
                        if (modelsDownloadError) {
                            modelsDownloadError.textContent = errors.join("\n");
                            modelsDownloadError.classList.remove("d-none");
                        }
                    }
                    const progressUrl = data.progress_url || null;
                    if (progressUrl && modelsDownloadProgress) {
                        const progressBar = modelsDownloadProgress.querySelector(".progress-bar");
                        let finished = false;
                        while (!finished) {
                            const progressResponse = await fetch(progressUrl, {
                                cache: "no-store",
                                headers: { "X-Requested-With": "XMLHttpRequest" },
                            });
                            const progressData = await progressResponse.json();
                            if (progressData.error) {
                                throw new Error(progressData.error);
                            }
                            const progress = Number(progressData.progress || 0);
                            modelsDownloadProgress.setAttribute("aria-valuenow", String(progress));
                            if (progressBar) {
                                progressBar.style.width = `${progress}%`;
                            }
                            if (progressData.status === "failed") {
                                const errors = progressData.errors || ["Download failed."];
                                if (modelsDownloadError) {
                                    modelsDownloadError.textContent = errors.join("\n");
                                    modelsDownloadError.classList.remove("d-none");
                                }
                                finished = true;
                            } else if (progressData.status === "completed") {
                                downloadSucceeded = true;
                                finished = true;
                            }
                            if (!finished) {
                                await new Promise((resolve) => setTimeout(resolve, 1000));
                            }
                        }
                    }
                } catch (error) {
                    if (modelsDownloadError) {
                        modelsDownloadError.textContent = "Unable to download models. Check your network.";
                        modelsDownloadError.classList.remove("d-none");
                    }
                }

                if (modelsDownloadProgress) {
                    const progressBar = modelsDownloadProgress.querySelector(".progress-bar");
                    modelsDownloadProgress.setAttribute(
                        "aria-valuenow",
                        downloadSucceeded ? "100" : modelsDownloadProgress.getAttribute("aria-valuenow") || "0"
                    );
                    if (progressBar) {
                        progressBar.style.width = downloadSucceeded
                            ? "100%"
                            : progressBar.style.width || "0%";
                    }
                }
                downloadModelsButton.disabled = false;
                downloadModelsButton.textContent = "Download models";
                await showModelsModalIfMissing(modelsModal);
                const status = await fetchModelStatus();
                if (status && status.missing && status.missing.length === 0) {
                    modelsModal.hide();
                }
            });
        }
    }

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
        if (videoUpload) {
            videoUpload.value = "";
        }

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

    window.addEventListener("pageshow", () => {
        if (videoUpload) {
            videoUpload.value = "";
        }
    });

    if (previousRunSelect && openPreviousRun) {
        const updatePreviousRunButton = () => {
            openPreviousRun.disabled = !previousRunSelect.value;
        };

        updatePreviousRunButton();
        previousRunSelect.addEventListener("change", updatePreviousRunButton);
        openPreviousRun.addEventListener("click", () => {
            const runKey = previousRunSelect.value;
            const template = previousRunSelect.dataset.resultsTemplate;
            if (!runKey || !template) {
                return;
            }
            const encodedRunKey = encodeURIComponent(runKey).replace(/%2F/g, "/");
            const target = template.replace("__RUN_KEY__", encodedRunKey);
            window.location.href = target;
        });
    }

    if (deleteRunForm && deleteRunButton) {
        deleteRunForm.addEventListener("submit", (event) => {
            const message =
                deleteRunButton.dataset.confirmMessage || "Remove stored results for this run?";
            if (!window.confirm(message)) {
                event.preventDefault();
            }
        });
    }

    if (previousRunSelect && previousRunTimestamp) {
        const updateTimestamp = () => {
            const selectedOption = previousRunSelect.selectedOptions[0];
            const modifiedRaw = selectedOption?.dataset.modified;
            if (!modifiedRaw) {
                previousRunTimestamp.textContent = "Last updated: --";
                return;
            }
            const modifiedDate = new Date(Number(modifiedRaw) * 1000);
            if (Number.isNaN(modifiedDate.getTime())) {
                previousRunTimestamp.textContent = "Last updated: --";
                return;
            }
            previousRunTimestamp.textContent = `Last updated: ${modifiedDate.toLocaleString()}`;
        };

        updateTimestamp();
        previousRunSelect.addEventListener("change", updateTimestamp);
    }

    const collapseToggles = document.querySelectorAll("[data-collapse-toggle]");
    const updateCollapseToggle = (button, isExpanded) => {
        const showLabel = button.dataset.labelShow || "Show";
        const hideLabel = button.dataset.labelHide || "Hide";
        const collapsedIcon = button.dataset.iconCollapsed || "bi-chevron-down";
        const expandedIcon = button.dataset.iconExpanded || "bi-chevron-up";
        const label = button.querySelector("[data-collapse-label]");
        const icon = button.querySelector("[data-collapse-icon]");
        const nextLabel = isExpanded ? hideLabel : showLabel;
        button.setAttribute("aria-expanded", String(isExpanded));
        button.setAttribute("aria-label", nextLabel);
        if (label) {
            label.textContent = nextLabel;
        }
        if (icon) {
            icon.classList.remove(collapsedIcon, expandedIcon);
            icon.classList.add(isExpanded ? expandedIcon : collapsedIcon);
        }
    };

    collapseToggles.forEach((button) => {
        const targetSelector = button.getAttribute("data-bs-target");
        if (!targetSelector) return;
        const target = document.querySelector(targetSelector);
        if (!target) return;
        updateCollapseToggle(button, target.classList.contains("show"));
        target.addEventListener("shown.bs.collapse", () => updateCollapseToggle(button, true));
        target.addEventListener("hidden.bs.collapse", () => updateCollapseToggle(button, false));
    });
});
