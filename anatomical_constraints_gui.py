#!/usr/bin/env python3
"""Standalone GUI for applying anatomical constraints to MediaPipe CSV files.

This tool provides an interactive interface for post-processing landmark CSV files
with anatomical constraints: bone length consistency, pelvis smoothing, and ground
plane enforcement.

Usage:
    python anatomical_constraints_gui.py
"""

import csv
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from src.anatomical.anatomical_constraints import apply_anatomical_constraints
from src.datastream.data_stream import CSV_HEADERS, LandmarkRecord


def load_csv_to_records(csv_path: Path) -> list[LandmarkRecord]:
    """Load CSV file into list of LandmarkRecord objects.

    Args:
        csv_path: Path to CSV file with MediaPipe landmark data

    Returns:
        List of LandmarkRecord objects

    Raises:
        ValueError: If CSV is missing required columns
    """
    records = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        # Validate headers
        required = {"timestamp_s", "landmark", "x_m", "y_m", "z_m", "visibility"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV missing required columns. Required: {required}, "
                f"found: {set(reader.fieldnames or [])}"
            )

        for row in reader:
            records.append(
                LandmarkRecord(
                    timestamp_s=float(row["timestamp_s"]),
                    landmark=row["landmark"],
                    x_m=float(row["x_m"]),
                    y_m=float(row["y_m"]),
                    z_m=float(row["z_m"]),
                    visibility=float(row["visibility"]),
                )
            )

    return records


def save_records_to_csv(csv_path: Path, records: list[LandmarkRecord]) -> None:
    """Save LandmarkRecord objects to CSV file.

    Args:
        csv_path: Output path for CSV file
        records: List of LandmarkRecord objects to save
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(records, key=lambda r: (r.timestamp_s, r.landmark))

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADERS)
        for record in sorted_records:
            writer.writerow(record.as_csv_row())


class AnatomicalConstraintsGUI(tk.Tk):
    """GUI application for anatomical constraints post-processing."""

    def __init__(self):
        super().__init__()

        self.title("Anatomical Constraints - MediaPipe CSV Post-processing")
        self.geometry("700x340")

        # State variables
        self.input_path = tk.StringVar()
        self.smooth_window = tk.StringVar(value="21")
        self.ground_percentile = tk.StringVar(value="5.0")
        self.ground_margin = tk.StringVar(value="0.02")
        self.foot_visibility = tk.StringVar(value="0.5")

        self.create_widgets()

    def create_widgets(self):
        """Create and layout all GUI widgets."""
        # File selection frame
        frm_file = tk.Frame(self)
        frm_file.pack(fill="x", padx=10, pady=10)

        tk.Label(frm_file, text="Input CSV file:").pack(anchor="w")

        frm_file_inner = tk.Frame(frm_file)
        frm_file_inner.pack(fill="x")

        entry_file = tk.Entry(frm_file_inner, textvariable=self.input_path)
        entry_file.pack(side="left", fill="x", expand=True)

        btn_browse = tk.Button(
            frm_file_inner, text="Browse...", command=self.browse_file
        )
        btn_browse.pack(side="left", padx=5)

        # Parameters frame
        frm_params = tk.Frame(self)
        frm_params.pack(fill="x", padx=10, pady=5)

        tk.Label(
            frm_params,
            text="Parameters:",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))

        tk.Label(frm_params, text="Smoothing window (frames, odd):").grid(
            row=1, column=0, sticky="w", padx=(10, 0)
        )
        tk.Entry(frm_params, textvariable=self.smooth_window, width=12).grid(
            row=1, column=1, sticky="w", padx=5
        )

        tk.Label(frm_params, text="Ground percentile (0-100):").grid(
            row=2, column=0, sticky="w", padx=(10, 0)
        )
        tk.Entry(frm_params, textvariable=self.ground_percentile, width=12).grid(
            row=2, column=1, sticky="w", padx=5
        )

        tk.Label(frm_params, text="Ground margin (meters):").grid(
            row=3, column=0, sticky="w", padx=(10, 0)
        )
        tk.Entry(frm_params, textvariable=self.ground_margin, width=12).grid(
            row=3, column=1, sticky="w", padx=5
        )

        tk.Label(frm_params, text="Foot visibility threshold:").grid(
            row=4, column=0, sticky="w", padx=(10, 0)
        )
        tk.Entry(frm_params, textvariable=self.foot_visibility, width=12).grid(
            row=4, column=1, sticky="w", padx=5
        )

        # Description frame
        frm_desc = tk.Frame(self)
        frm_desc.pack(fill="x", padx=10, pady=5)

        desc_text = (
            "This tool applies anatomical constraints to pose landmarks:\n"
            "• Constant bone lengths (arms and legs) across frames\n"
            "• Pelvis Z-depth smoothing for stability\n"
            "• Ground plane estimation and foot contact enforcement"
        )
        tk.Label(
            frm_desc,
            text=desc_text,
            justify="left",
            anchor="w",
            fg="#555",
        ).pack(fill="x")

        # Status label
        self.status_label = tk.Label(
            self, text="Select a CSV file and click 'Process'", anchor="w", fg="#0066cc"
        )
        self.status_label.pack(fill="x", padx=10, pady=5)

        # Buttons frame
        frm_buttons = tk.Frame(self)
        frm_buttons.pack(fill="x", padx=10, pady=10)

        btn_run = tk.Button(
            frm_buttons,
            text="Process",
            command=self.run_processing,
            bg="#0066cc",
            fg="white",
            padx=20,
        )
        btn_run.pack(side="left")

        btn_quit = tk.Button(frm_buttons, text="Close", command=self.destroy, padx=20)
        btn_quit.pack(side="right")

    def browse_file(self):
        """Open file dialog to select input CSV."""
        path = filedialog.askopenfilename(
            title="Select MediaPipe CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.input_path.set(path)
            self.status_label.config(
                text=f"Selected: {Path(path).name}", fg="#0066cc"
            )

    def run_processing(self):
        """Execute anatomical constraints processing."""
        in_path = self.input_path.get().strip()
        if not in_path:
            messagebox.showerror("Error", "No input file selected")
            return
        if not os.path.isfile(in_path):
            messagebox.showerror("Error", f"File does not exist: {in_path}")
            return

        # Parse parameters
        try:
            smooth_window = int(self.smooth_window.get())
        except ValueError:
            messagebox.showerror("Error", "Smoothing window must be an integer")
            return

        try:
            ground_percentile = float(self.ground_percentile.get())
        except ValueError:
            messagebox.showerror("Error", "Ground percentile must be a number")
            return

        try:
            ground_margin = float(self.ground_margin.get())
        except ValueError:
            messagebox.showerror("Error", "Ground margin must be a number")
            return

        try:
            foot_visibility = float(self.foot_visibility.get())
        except ValueError:
            messagebox.showerror("Error", "Foot visibility threshold must be a number")
            return

        # Suggest output path
        base_path = Path(in_path)
        default_out = base_path.parent / f"{base_path.stem}_anatomical.csv"

        out_path = filedialog.asksaveasfilename(
            title="Save processed file as...",
            defaultextension=".csv",
            initialfile=default_out.name,
            initialdir=default_out.parent,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not out_path:
            return

        self.status_label.config(text="Processing...", fg="#ff6600")
        self.update_idletasks()

        try:
            # Load CSV
            records = load_csv_to_records(Path(in_path))
            n_input = len(records)

            # Apply constraints
            records = apply_anatomical_constraints(
                records,
                smooth_window=smooth_window,
                ground_percentile=ground_percentile,
                foot_visibility_threshold=foot_visibility,
                ground_margin=ground_margin,
            )

            # Save output
            save_records_to_csv(Path(out_path), records)

            # Calculate statistics
            timestamps = sorted(set(r.timestamp_s for r in records))
            landmarks = sorted(set(r.landmark for r in records))

            msg = (
                f"Processing complete!\n\n"
                f"Input records: {n_input}\n"
                f"Output records: {len(records)}\n"
                f"Frames: {len(timestamps)}\n"
                f"Landmarks: {len(landmarks)}\n\n"
                f"Output: {Path(out_path).name}"
            )
            messagebox.showinfo("Success", msg)
            self.status_label.config(text="Processing complete", fg="#00cc00")

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
            self.status_label.config(text="Processing failed", fg="#cc0000")
            return


def main():
    """Launch the GUI application."""
    app = AnatomicalConstraintsGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
