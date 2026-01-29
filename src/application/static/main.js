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
            const parseDownloadError = (error, response) => {
                // Provide specific error messages based on error type
                if (error?.name === "AbortError" || error?.message?.includes("timeout")) {
                    return "Download timed out. Please check your network connection and try again.";
                }
                if (error?.name === "TypeError" || error?.message?.includes("NetworkError")) {
                    return "Network error. Please check your internet connection and try again.";
                }
                if (response?.status === 404) {
                    return "Model files not found on the server. Please contact the administrator.";
                }
                if (response?.status === 403) {
                    return "Access denied. Please refresh the page and try again.";
                }
                if (response?.status >= 500) {
                    return "Server error. Please try again later or contact the administrator.";
                }
                return error?.message || "Download failed. Please try again.";
            };

            const resetDownloadUI = (succeeded) => {
                if (modelsDownloadProgress) {
                    const progressBar = modelsDownloadProgress.querySelector(".progress-bar");
                    modelsDownloadProgress.setAttribute(
                        "aria-valuenow",
                        succeeded ? "100" : modelsDownloadProgress.getAttribute("aria-valuenow") || "0"
                    );
                    if (progressBar) {
                        progressBar.style.width = succeeded
                            ? "100%"
                            : progressBar.style.width || "0%";
                    }
                }
                downloadModelsButton.disabled = false;
                downloadModelsButton.textContent = succeeded ? "Download models" : "Retry download";
            };

            downloadModelsButton.addEventListener("click", async () => {
                downloadModelsButton.disabled = true;
                downloadModelsButton.textContent = "Downloading...";
                modelsDownloadError?.classList.add("d-none");
                let downloadSucceeded = false;
                let response = null;

                if (modelsDownloadProgress) {
                    const progressBar = modelsDownloadProgress.querySelector(".progress-bar");
                    modelsDownloadProgress.classList.remove("d-none");
                    if (progressBar) {
                        progressBar.style.width = "0%";
                    }
                    modelsDownloadProgress.setAttribute("aria-valuenow", "0");
                }
                try {
                    response = await fetch(modelDownloadUrl, {
                        method: "POST",
                        headers: {
                            "X-Requested-With": "XMLHttpRequest",
                            "X-CSRFToken": getCookie("csrftoken"),
                        },
                    });
                    const data = await response.json();
                    if (!response.ok) {
                        const errorMsg = parseDownloadError(null, response);
                        const details = data.errors ? data.errors.join("\n") : "";
                        if (modelsDownloadError) {
                            modelsDownloadError.innerHTML = `<strong>${errorMsg}</strong>${details ? `<br><small class="text-muted">${details}</small>` : ""}`;
                            modelsDownloadError.classList.remove("d-none");
                        }
                        resetDownloadUI(false);
                        return;
                    }
                    const progressUrl = data.progress_url || null;
                    if (progressUrl && modelsDownloadProgress) {
                        const progressBar = modelsDownloadProgress.querySelector(".progress-bar");
                        let finished = false;
                        let currentFile = "";
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
                            // Show current file being downloaded if available
                            if (progressData.current_file && progressData.current_file !== currentFile) {
                                currentFile = progressData.current_file;
                                downloadModelsButton.textContent = `Downloading ${currentFile}...`;
                            }
                            if (progressData.status === "failed") {
                                const errors = progressData.errors || ["Download failed."];
                                if (modelsDownloadError) {
                                    modelsDownloadError.innerHTML = `<strong>Download failed</strong><br><small class="text-muted">${errors.join("<br>")}</small>`;
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
                    const errorMsg = parseDownloadError(error, response);
                    if (modelsDownloadError) {
                        modelsDownloadError.innerHTML = `<strong>${errorMsg}</strong>`;
                        modelsDownloadError.classList.remove("d-none");
                    }
                }

                resetDownloadUI(downloadSucceeded);
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
        const MAX_CONSECUTIVE_FAILURES = 3;
        const RETRY_TIMEOUT_MS = 30000;
        let consecutiveFailures = 0;
        let lastSuccessTime = Date.now();

        const pollInterval = setInterval(async () => {
            try {
                const progressResponse = await fetch(progressUrl, {
                    cache: "no-store",
                    headers: { "X-Requested-With": "XMLHttpRequest" },
                });
                if (!progressResponse.ok) {
                    consecutiveFailures++;
                    if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
                        const elapsed = Date.now() - lastSuccessTime;
                        if (elapsed > RETRY_TIMEOUT_MS) {
                            clearInterval(pollInterval);
                            handleAjaxFailure([
                                "Connection lost. Unable to track progress.",
                                "Please refresh the page to check if processing completed."
                            ]);
                            return;
                        }
                        setProgress(
                            progressBar ? parseFloat(progressBar.style.width) || 0 : 0,
                            "Connection lost, retrying..."
                        );
                    }
                    return;
                }
                // Reset failure count on success
                consecutiveFailures = 0;
                lastSuccessTime = Date.now();

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
                consecutiveFailures++;
                if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
                    const elapsed = Date.now() - lastSuccessTime;
                    if (elapsed > RETRY_TIMEOUT_MS) {
                        clearInterval(pollInterval);
                        handleAjaxFailure([
                            "Connection lost. Unable to track progress.",
                            "Please refresh the page to check if processing completed."
                        ]);
                        return;
                    }
                    setProgress(
                        progressBar ? parseFloat(progressBar.style.width) || 0 : 0,
                        "Connection lost, retrying..."
                    );
                }
            }
        }, 1000);
    };

    const MAX_VIDEO_DURATION_SECONDS = 60;
    const videoInfo = document.getElementById("videoInfo");
    const videoDurationWarning = document.getElementById("videoDurationWarning");

    const formatFileSize = (bytes) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    const formatDuration = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
    };

    const checkVideoDuration = (file) => {
        return new Promise((resolve) => {
            const video = document.createElement("video");
            video.preload = "metadata";
            video.onloadedmetadata = () => {
                URL.revokeObjectURL(video.src);
                resolve(video.duration);
            };
            video.onerror = () => {
                URL.revokeObjectURL(video.src);
                resolve(null);
            };
            video.src = URL.createObjectURL(file);
        });
    };

    if (videoUpload) {
        videoUpload.addEventListener("change", async () => {
            const file = videoUpload.files?.[0];
            if (!file) {
                if (videoInfo) videoInfo.classList.add("d-none");
                if (videoDurationWarning) videoDurationWarning.classList.add("d-none");
                return;
            }

            // Show file size immediately
            if (videoInfo) {
                videoInfo.textContent = `File size: ${formatFileSize(file.size)}`;
                videoInfo.classList.remove("d-none");
            }

            // Check duration asynchronously
            const duration = await checkVideoDuration(file);
            if (duration !== null) {
                if (videoInfo) {
                    videoInfo.textContent = `${formatFileSize(file.size)} | Duration: ${formatDuration(duration)}`;
                }
                if (duration > MAX_VIDEO_DURATION_SECONDS) {
                    if (videoDurationWarning) {
                        videoDurationWarning.textContent = `Video is ${formatDuration(duration)} long. Maximum duration is 1 minute. The upload may be rejected.`;
                        videoDurationWarning.classList.remove("d-none");
                    }
                } else if (videoDurationWarning) {
                    videoDurationWarning.classList.add("d-none");
                }
            }
        });
    }

    const validateFormInputs = () => {
        const errors = [];
        const heightInput = document.getElementById("height");
        const weightInput = document.getElementById("weight");
        const visibilityInput = document.getElementById("visibilityMin");

        // Clear previous validation states
        [heightInput, weightInput, visibilityInput].forEach((input) => {
            if (input) input.classList.remove("is-invalid");
        });

        // Validate height if provided
        if (heightInput && heightInput.value) {
            const height = parseFloat(heightInput.value);
            if (isNaN(height) || height < 0.5 || height > 2.5) {
                heightInput.classList.add("is-invalid");
                errors.push("Height must be between 0.5 and 2.5 meters.");
            }
        }

        // Validate weight if provided
        if (weightInput && weightInput.value) {
            const weight = parseFloat(weightInput.value);
            if (isNaN(weight) || weight < 10 || weight > 500) {
                weightInput.classList.add("is-invalid");
                errors.push("Weight must be between 10 and 500 kg.");
            }
        }

        // Validate visibility threshold
        if (visibilityInput && visibilityInput.value) {
            const visibility = parseFloat(visibilityInput.value);
            if (isNaN(visibility) || visibility < 0 || visibility > 1) {
                visibilityInput.classList.add("is-invalid");
                errors.push("Visibility threshold must be between 0 and 1.");
            }
        }

        return errors;
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

            // Client-side validation
            const validationErrors = validateFormInputs();
            if (validationErrors.length > 0) {
                showAjaxErrors(validationErrors);
                return;
            }

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
