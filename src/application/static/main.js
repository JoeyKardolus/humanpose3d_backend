document.addEventListener("DOMContentLoaded", () => {
    const instructionsModalEl = document.getElementById("instructionsModal");
    const noticeModalEl = document.getElementById("noticeModal");
    const instructionCheck = document.getElementById("instructions-check");
    const privacyCheck = document.getElementById("privacy-check");
    const analyzeForm = document.getElementById("homeForm");
    const analyzeButton = document.getElementById("analyzeBtn");
    const processingOverlay = document.getElementById("processingOverlay");
    const instructionsKey = "instructionsAccepted";
    const privacyKey = "privacyAccepted";

    if (!instructionsModalEl || !noticeModalEl || !window.bootstrap) return;

    const instructionsModal = new bootstrap.Modal(instructionsModalEl, {
        backdrop: "static",
        keyboard: false,
    });
    const noticeModal = new bootstrap.Modal(noticeModalEl, {
        backdrop: "static",
        keyboard: false,
    });

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

    const instructionsAccepted =
        sessionStorage.getItem(instructionsKey) === "true";
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

    if (analyzeForm) {
        analyzeForm.addEventListener("submit", () => {
            if (analyzeButton) {
                analyzeButton.disabled = true;
                analyzeButton.textContent = "Submitting...";
            }
            if (processingOverlay) {
                processingOverlay.classList.add("is-visible");
                processingOverlay.setAttribute("aria-hidden", "false");
            }
        });
    }
});
