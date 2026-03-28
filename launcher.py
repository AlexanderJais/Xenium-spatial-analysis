"""
launcher.py
-----------
macOS GUI launcher for the Xenium AGED vs ADULT MBH pipeline.

Opens a native macOS-style Tkinter window where you:
  1. Browse to each of the 8 Xenium run folders (4 AGED + 4 ADULT)
  2. Set slide IDs and condition labels
  3. Point to the base panel CSV
  4. Choose output directory
  5. Configure pipeline options (panel mode, DGE method, ROI mode)
  6. Launch the pipeline — progress streams into a live log panel

Run:
    conda activate xenium_dge
    python launcher.py
"""

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, messagebox, scrolledtext, ttk

# ---------------------------------------------------------------------------
# Colour palette (matches Nature figure palette for visual coherence)
# ---------------------------------------------------------------------------
BG          = "#F5F5F5"
PANEL_BG    = "#FFFFFF"
ACCENT_AGED  = "#D55E00"   # orange-red for AGED
ACCENT_ADULT = "#0072B2"   # blue for ADULT
ACCENT_DARK  = "#2C2C2C"
BORDER      = "#DDDDDD"
SUCCESS     = "#009E73"
WARNING     = "#E69F00"
FONT_BODY   = ("SF Pro Text", 12)
FONT_SMALL  = ("SF Pro Text", 10)
FONT_MONO   = ("SF Mono", 10)
FONT_TITLE  = ("SF Pro Display", 16, "bold")
FONT_HEADER = ("SF Pro Text", 11, "bold")

N_AGED  = 4
N_ADULT = 4
N_TOTAL = N_AGED + N_ADULT


class SlideRow:
    """One row in the slide table: condition badge + ID entry + path + browse button."""

    def __init__(self, parent, index: int, condition: str, default_id: str, row: int):
        self.index     = index
        self.condition = condition

        # Condition badge
        badge_colour = ACCENT_AGED if condition == "AGED" else ACCENT_ADULT
        badge = tk.Label(
            parent, text=condition, width=6,
            bg=badge_colour, fg="white",
            font=("SF Pro Text", 10, "bold"),
            relief="flat", pady=2,
        )
        badge.grid(row=row, column=0, padx=(0, 6), pady=3, sticky="w")

        # Slide ID entry
        self.id_var = tk.StringVar(value=default_id)
        id_entry = tk.Entry(
            parent, textvariable=self.id_var, width=10,
            font=FONT_SMALL, relief="solid",
            highlightthickness=1, highlightbackground=BORDER,
        )
        id_entry.grid(row=row, column=1, padx=(0, 6), pady=3, sticky="w")

        # Path display
        self.path_var = tk.StringVar(value="No folder selected")
        path_label = tk.Label(
            parent, textvariable=self.path_var,
            anchor="w", width=48,
            bg=PANEL_BG, fg="#777777",
            font=FONT_SMALL,
        )
        path_label.grid(row=row, column=2, padx=(0, 6), pady=3, sticky="ew")
        self._path_label = path_label

        # Browse button
        browse_btn = tk.Button(
            parent, text="Browse …",
            command=self._browse,
            font=FONT_SMALL,
            bg="#EFEFEF", fg=ACCENT_DARK,
            relief="flat", cursor="hand2",
            padx=10, pady=2,
            activebackground=BORDER,
        )
        browse_btn.grid(row=row, column=3, pady=3, sticky="e")

        # Clear button
        clear_btn = tk.Button(
            parent, text="✕",
            command=self._clear,
            font=("SF Pro Text", 9),
            bg=PANEL_BG, fg="#AAAAAA",
            relief="flat", cursor="hand2",
            padx=4,
            activebackground=BORDER,
        )
        clear_btn.grid(row=row, column=4, pady=3, padx=(2, 0))

    def _browse(self):
        path = filedialog.askdirectory(
            title=f"Select Xenium run folder for {self.id_var.get()}",
            mustexist=True,
        )
        if path:
            self.path_var.set(path)
            self._path_label.config(fg=ACCENT_DARK)

    def _clear(self):
        self.path_var.set("No folder selected")
        self._path_label.config(fg="#777777")

    @property
    def slide_id(self) -> str:
        return self.id_var.get().strip()

    @property
    def path(self) -> str:
        v = self.path_var.get()
        return "" if v == "No folder selected" else v

    @property
    def is_set(self) -> bool:
        return bool(self.path) and Path(self.path).exists()


class XeniumLauncher(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("Xenium DGE Pipeline — AGED vs ADULT MBH")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(860, 720)

        # Try to set a nice macOS-style window size
        self.geometry("960x840")

        # Centre on screen
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - 960) // 2
        y = (self.winfo_screenheight() - 840) // 2
        self.geometry(f"+{x}+{y}")

        self._slide_rows: list[SlideRow] = []
        self._log_queue  = queue.Queue()
        self._proc       = None

        self._build_ui()
        self._poll_log()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Main scrollable canvas
        main_canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        scrollbar   = ttk.Scrollbar(self, orient="vertical", command=main_canvas.yview)
        main_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        main_canvas.pack(side="left", fill="both", expand=True)

        self._content = tk.Frame(main_canvas, bg=BG, padx=28, pady=20)
        self._content_id = main_canvas.create_window(
            (0, 0), window=self._content, anchor="nw"
        )
        self._content.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        main_canvas.bind(
            "<Configure>",
            lambda e: main_canvas.itemconfig(self._content_id, width=e.width)
        )
        # Mouse wheel scrolling
        self.bind_all("<MouseWheel>", lambda e: main_canvas.yview_scroll(-1*(e.delta//120), "units"))

        f = self._content

        # ---- Title ----
        tk.Label(f, text="Xenium DGE Pipeline", font=FONT_TITLE,
                 bg=BG, fg=ACCENT_DARK).pack(anchor="w", pady=(0, 2))
        tk.Label(f, text="AGED vs ADULT mouse brain  ·  Mediobasal Hypothalamus",
                 font=FONT_SMALL, bg=BG, fg="#666666").pack(anchor="w", pady=(0, 16))

        _separator(f)

        # ---- Slide table ----
        self._build_slide_table(f)
        _separator(f)

        # ---- Paths ----
        self._build_paths_section(f)
        _separator(f)

        # ---- Options ----
        self._build_options_section(f)
        _separator(f)

        # ---- Action buttons ----
        self._build_action_buttons(f)

        # ---- Log panel ----
        self._build_log_panel(f)

    def _build_slide_table(self, parent):
        tk.Label(parent, text="Slide folders", font=FONT_HEADER,
                 bg=BG, fg=ACCENT_DARK).pack(anchor="w", pady=(10, 6))
        tk.Label(
            parent,
            text="Browse to each Xenium output directory (the folder that contains "
                 "cell_feature_matrix/, cells.parquet, experiment.xenium).",
            font=FONT_SMALL, bg=BG, fg="#666666", wraplength=820, justify="left",
        ).pack(anchor="w", pady=(0, 10))

        table_frame = tk.Frame(parent, bg=PANEL_BG, relief="solid",
                               highlightthickness=1, highlightbackground=BORDER)
        table_frame.pack(fill="x", pady=(0, 8))

        inner = tk.Frame(table_frame, bg=PANEL_BG, padx=12, pady=8)
        inner.pack(fill="x")
        inner.columnconfigure(2, weight=1)

        # Column headers
        for col, text, w in [
            (0, "Condition", 7), (1, "Slide ID", 10),
            (2, "Run folder", 0), (3, "", 0), (4, "", 0),
        ]:
            tk.Label(inner, text=text, font=("SF Pro Text", 10, "bold"),
                     bg=PANEL_BG, fg="#888888", anchor="w",
                     width=w if w else None,
                     ).grid(row=0, column=col, padx=(0, 6), pady=(0, 4), sticky="w")

        _row_sep = lambda r: tk.Frame(inner, bg=BORDER, height=1).grid(
            row=r, column=0, columnspan=5, sticky="ew", pady=1
        )

        row_num = 1
        for i in range(N_AGED):
            sr = SlideRow(inner, index=i, condition="AGED",
                          default_id=f"AGED_{i+1}", row=row_num)
            self._slide_rows.append(sr)
            row_num += 1
            _row_sep(row_num); row_num += 1

        for i in range(N_ADULT):
            sr = SlideRow(inner, index=N_AGED + i, condition="ADULT",
                          default_id=f"ADULT_{i+1}", row=row_num)
            self._slide_rows.append(sr)
            row_num += 1
            if i < N_ADULT - 1:
                _row_sep(row_num); row_num += 1

    def _build_paths_section(self, parent):
        tk.Label(parent, text="Files and folders", font=FONT_HEADER,
                 bg=BG, fg=ACCENT_DARK).pack(anchor="w", pady=(10, 8))

        grid = tk.Frame(parent, bg=BG)
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)

        # Base panel CSV
        self._panel_csv_var = tk.StringVar(
            value=str(Path(__file__).parent / "data" / "Xenium_mBrain_v1_1_metadata.csv")
        )
        self._add_path_row(grid, 0, "Base panel CSV",
                           self._panel_csv_var, mode="file",
                           filetypes=[("CSV files", "*.csv"), ("All", "*.*")])

        # Output directory
        self._output_dir_var = tk.StringVar(
            value=str(Path.home() / "xenium_dge_output")
        )
        self._add_path_row(grid, 1, "Output directory",
                           self._output_dir_var, mode="dir")

        # ROI cache
        self._roi_cache_var = tk.StringVar(
            value=str(Path(__file__).parent / "roi_cache")
        )
        self._add_path_row(grid, 2, "ROI cache folder",
                           self._roi_cache_var, mode="dir")

    def _add_path_row(self, parent, row, label, var, mode="dir", filetypes=None):
        tk.Label(parent, text=label, font=FONT_SMALL, bg=BG,
                 fg=ACCENT_DARK, width=20, anchor="w",
                 ).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 8))

        entry = tk.Entry(parent, textvariable=var, font=FONT_SMALL,
                         relief="solid", highlightthickness=1,
                         highlightbackground=BORDER)
        entry.grid(row=row, column=1, sticky="ew", pady=4)

        def _browse():
            if mode == "dir":
                p = filedialog.askdirectory(title=f"Select {label}", mustexist=False)
            else:
                p = filedialog.askopenfilename(
                    title=f"Select {label}",
                    filetypes=filetypes or [("All", "*.*")],
                )
            if p:
                var.set(p)

        tk.Button(parent, text="Browse …", command=_browse,
                  font=FONT_SMALL, bg="#EFEFEF", fg=ACCENT_DARK,
                  relief="flat", cursor="hand2", padx=10, pady=2,
                  activebackground=BORDER,
                  ).grid(row=row, column=2, padx=(8, 0), pady=4)

    def _build_options_section(self, parent):
        tk.Label(parent, text="Pipeline options", font=FONT_HEADER,
                 bg=BG, fg=ACCENT_DARK).pack(anchor="w", pady=(10, 8))

        opts = tk.Frame(parent, bg=BG)
        opts.pack(fill="x")

        # Row 1
        r1 = tk.Frame(opts, bg=BG)
        r1.pack(fill="x", pady=3)

        # DGE method
        tk.Label(r1, text="DGE method:", font=FONT_SMALL, bg=BG,
                 fg=ACCENT_DARK, width=20, anchor="w").pack(side="left")
        self._dge_method_var = tk.StringVar(value="stringent_wilcoxon")
        for val, lab in [("stringent_wilcoxon", "★ Stringent Wilcoxon (recommended)"),
                         ("wilcoxon", "Wilcoxon (fast, permissive)"),
                         ("pydeseq2", "PyDESeq2 pseudobulk (n≥8 needed)"),
                         ("cside",    "C-SIDE pseudobulk (per-cell-type)")]:
            tk.Radiobutton(r1, text=lab, variable=self._dge_method_var, value=val,
                           font=FONT_SMALL, bg=BG, fg=ACCENT_DARK,
                           activebackground=BG, selectcolor=BG,
                           ).pack(side="left", padx=(0, 12))

        # Row 2
        r2 = tk.Frame(opts, bg=BG)
        r2.pack(fill="x", pady=3)

        # Panel mode
        tk.Label(r2, text="Panel mode:", font=FONT_SMALL, bg=BG,
                 fg=ACCENT_DARK, width=20, anchor="w").pack(side="left")
        self._panel_mode_var = tk.StringVar(value="partial_union")
        for val, lab in [
            ("intersection",  "Intersection — base only (no custom genes)"),
            ("partial_union", "Partial union — base + shared custom genes ✓"),
            ("union",         "Union — all custom genes (max zero-inflation)"),
        ]:
            tk.Radiobutton(r2, text=lab, variable=self._panel_mode_var, value=val,
                           font=FONT_SMALL, bg=BG, fg=ACCENT_DARK,
                           activebackground=BG, selectcolor=BG,
                           ).pack(side="left", padx=(0, 10))

        # min_slides spinbox (shown inline after the radio buttons)
        tk.Label(r2, text="  min slides:", font=FONT_SMALL, bg=BG,
                 fg=ACCENT_DARK).pack(side="left", padx=(6, 2))
        self._min_slides_var = tk.StringVar(value="2")
        spinbox = tk.Spinbox(
            r2, from_=1, to=8, width=3,
            textvariable=self._min_slides_var,
            font=FONT_SMALL,
            relief="solid", highlightthickness=1,
            highlightbackground=BORDER,
        )
        spinbox.pack(side="left")
        tk.Label(r2, text="(for partial union)",
                 font=("SF Pro Text", 9), bg=BG, fg="#888888",
                 ).pack(side="left", padx=(4, 0))

        # Row 3
        r3 = tk.Frame(opts, bg=BG)
        r3.pack(fill="x", pady=3)

        # ROI drawing mode
        tk.Label(r3, text="ROI draw mode:", font=FONT_SMALL, bg=BG,
                 fg=ACCENT_DARK, width=20, anchor="w").pack(side="left")
        self._roi_mode_var = tk.StringVar(value="polygon")
        for val, lab in [("polygon", "Polygon (click vertices)"),
                         ("lasso",   "Lasso (freehand)"),
                         ("rectangle", "Rectangle")]:
            tk.Radiobutton(r3, text=lab, variable=self._roi_mode_var, value=val,
                           font=FONT_SMALL, bg=BG, fg=ACCENT_DARK,
                           activebackground=BG, selectcolor=BG,
                           ).pack(side="left", padx=(0, 16))

        # Row 4 — checkboxes
        r4 = tk.Frame(opts, bg=BG)
        r4.pack(fill="x", pady=3)

        self._redraw_roi_var  = tk.BooleanVar(value=False)
        self._no_roi_gui_var  = tk.BooleanVar(value=False)
        self._use_cache_var   = tk.BooleanVar(value=True)

        tk.Label(r4, text="", font=FONT_SMALL, bg=BG, width=20).pack(side="left")
        tk.Checkbutton(r4, text="Force redraw ROIs",
                       variable=self._redraw_roi_var,
                       font=FONT_SMALL, bg=BG, fg=ACCENT_DARK,
                       activebackground=BG, selectcolor=BG,
                       ).pack(side="left", padx=(0, 16))
        tk.Checkbutton(r4, text="Skip ROI GUI (use saved/preset)",
                       variable=self._no_roi_gui_var,
                       font=FONT_SMALL, bg=BG, fg=ACCENT_DARK,
                       activebackground=BG, selectcolor=BG,
                       ).pack(side="left", padx=(0, 16))
        tk.Checkbutton(r4, text="Use preprocessing cache",
                       variable=self._use_cache_var,
                       font=FONT_SMALL, bg=BG, fg=ACCENT_DARK,
                       activebackground=BG, selectcolor=BG,
                       ).pack(side="left")

        # Row 5 — numeric params
        r5 = tk.Frame(opts, bg=BG)
        r5.pack(fill="x", pady=6)
        params = [
            ("Leiden resolution:", "0.6",  6, "_leiden_res_var"),
            ("n neighbours:",      "12",   4, "_n_neighbors_var"),
            ("Min counts (QC):",   "10",   5, "_min_counts_var"),
            ("log2FC threshold:",  "1.0",  4, "_log2fc_var"),
            ("adj-p threshold:",   "0.01", 5, "_pval_var"),
        ]
        for label, default, width, attr in params:
            tk.Label(r5, text=label, font=FONT_SMALL, bg=BG, fg=ACCENT_DARK,
                     ).pack(side="left", padx=(0, 4))
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            tk.Entry(r5, textvariable=var, width=width, font=FONT_SMALL,
                     relief="solid", highlightthickness=1,
                     highlightbackground=BORDER,
                     ).pack(side="left", padx=(0, 14))

    def _build_action_buttons(self, parent):
        btn_frame = tk.Frame(parent, bg=BG)
        btn_frame.pack(fill="x", pady=(12, 8))

        # Validate button
        tk.Button(
            btn_frame, text="✓  Validate paths",
            command=self._validate,
            font=("SF Pro Text", 11),
            bg="#EFEFEF", fg=ACCENT_DARK,
            relief="flat", cursor="hand2",
            padx=16, pady=8,
            activebackground=BORDER,
        ).pack(side="left", padx=(0, 10))

        # Save config button
        tk.Button(
            btn_frame, text="💾  Save config",
            command=self._save_config,
            font=("SF Pro Text", 11),
            bg="#EFEFEF", fg=ACCENT_DARK,
            relief="flat", cursor="hand2",
            padx=16, pady=8,
            activebackground=BORDER,
        ).pack(side="left", padx=(0, 10))

        # Load config button
        tk.Button(
            btn_frame, text="📂  Load config",
            command=self._load_config,
            font=("SF Pro Text", 11),
            bg="#EFEFEF", fg=ACCENT_DARK,
            relief="flat", cursor="hand2",
            padx=16, pady=8,
            activebackground=BORDER,
        ).pack(side="left", padx=(0, 28))

        # Launch button (prominent)
        self._launch_btn = tk.Button(
            btn_frame, text="▶   Run Pipeline",
            command=self._launch,
            font=("SF Pro Text", 12, "bold"),
            bg=SUCCESS, fg="white",
            relief="flat", cursor="hand2",
            padx=24, pady=8,
            activebackground="#007A58",
        )
        self._launch_btn.pack(side="left", padx=(0, 10))

        # Stop button
        self._stop_btn = tk.Button(
            btn_frame, text="■  Stop",
            command=self._stop,
            font=("SF Pro Text", 11),
            bg=ACCENT_AGED, fg="white",
            relief="flat", cursor="hand2",
            padx=16, pady=8,
            activebackground="#A03000",
            state="disabled",
        )
        self._stop_btn.pack(side="left")

        # Status label
        self._status_var = tk.StringVar(value="Ready")
        self._status_lbl = tk.Label(
            btn_frame, textvariable=self._status_var,
            font=FONT_SMALL, bg=BG, fg="#888888",
        )
        self._status_lbl.pack(side="right")

    def _build_log_panel(self, parent):
        tk.Label(parent, text="Pipeline log", font=FONT_HEADER,
                 bg=BG, fg=ACCENT_DARK).pack(anchor="w", pady=(12, 4))

        self._log = scrolledtext.ScrolledText(
            parent,
            height=16,
            font=FONT_MONO,
            bg="#1E1E1E", fg="#D4D4D4",
            insertbackground="white",
            relief="flat",
            wrap="word",
            state="disabled",
        )
        self._log.pack(fill="both", expand=True)

        # Colour tags
        self._log.tag_config("ok",    foreground="#4EC9B0")
        self._log.tag_config("warn",  foreground="#CE9178")
        self._log.tag_config("error", foreground="#F44747")
        self._log.tag_config("info",  foreground="#9CDCFE")
        self._log.tag_config("dim",   foreground="#6A9955")

        # Clear log button
        tk.Button(
            parent, text="Clear log",
            command=self._clear_log,
            font=FONT_SMALL,
            bg="#EFEFEF", fg=ACCENT_DARK,
            relief="flat", cursor="hand2",
            padx=10, pady=2,
        ).pack(anchor="e", pady=(4, 0))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _validate(self) -> bool:
        errors = []
        warnings = []

        # Check slide paths
        set_count = sum(1 for sr in self._slide_rows if sr.is_set)
        for sr in self._slide_rows:
            if not sr.is_set:
                warnings.append(f"Slide '{sr.slide_id}': folder not set or does not exist.")
            else:
                # Check for cell_feature_matrix/
                p = Path(sr.path)
                if not (p / "cell_feature_matrix").exists():
                    errors.append(
                        f"Slide '{sr.slide_id}': "
                        f"'cell_feature_matrix/' not found in {p}"
                    )

        # Check base panel CSV
        panel_csv = Path(self._panel_csv_var.get())
        if not panel_csv.exists():
            errors.append(f"Base panel CSV not found: {panel_csv}")

        # Check IDs unique
        ids = [sr.slide_id for sr in self._slide_rows]
        if len(ids) != len(set(ids)):
            errors.append("Slide IDs must be unique.")

        # Report
        for e in errors:
            self._log_line(f"ERROR: {e}", "error")
        for w in warnings:
            self._log_line(f"WARN:  {w}", "warn")

        if errors:
            self._status("Validation failed", "error")
            messagebox.showerror("Validation failed",
                                 "\n".join(errors))
            return False
        elif warnings:
            self._status("Warnings — check log", "warn")
            self._log_line(f"Validation: {set_count}/{N_TOTAL} slides configured. "
                           "Missing slides will be skipped.", "warn")
        else:
            self._status("All paths valid", "ok")
            self._log_line(
                f"Validation passed. {set_count}/{N_TOTAL} slides configured.", "ok"
            )
        return True

    def _launch(self):
        if not self._validate():
            return

        # Build config dict
        cfg = self._collect_config()

        # Write a temp JSON config for the pipeline process
        cfg_path = Path(self._output_dir_var.get()) / ".launcher_config.json"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, indent=2))

        self._log_line("=" * 56, "dim")
        self._log_line("Launching pipeline …", "info")
        self._log_line(f"Config written to: {cfg_path}", "dim")
        self._log_line("=" * 56, "dim")

        # Disable launch, enable stop
        self._launch_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._status("Running …", "info")

        # Run in subprocess so it does not block the GUI
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "run_xenium_mbh.py"),
            "--launcher-config", str(cfg_path),
        ]
        if cfg.get("redraw_roi"):
            cmd.append("--redraw-roi")
        if cfg.get("no_roi_gui"):
            cmd.append("--no-roi-gui")

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._stream_output, daemon=True).start()

    def _stream_output(self):
        for line in self._proc.stdout:
            tag = "ok"   if "ok" in line.lower() or "done" in line.lower() \
                 else "error" if "error" in line.lower() or "fail" in line.lower() \
                 else "warn"  if "warn" in line.lower() \
                 else "info"
            self._log_queue.put((line.rstrip(), tag))
        ret = self._proc.wait()
        self._log_queue.put(("=" * 56, "dim"))
        if ret == 0:
            self._log_queue.put(("Pipeline finished successfully.", "ok"))
            self._log_queue.put(
                (f"Figures saved to: {self._output_dir_var.get()}", "ok")
            )
        else:
            self._log_queue.put((f"Pipeline exited with code {ret}.", "error"))
        self._log_queue.put(("__DONE__", None))

    def _stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._log_line("Pipeline terminated by user.", "warn")
        self._launch_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._status("Stopped", "warn")

    def _poll_log(self):
        try:
            while True:
                line, tag = self._log_queue.get_nowait()
                if line == "__DONE__":
                    self._launch_btn.config(state="normal")
                    self._stop_btn.config(state="disabled")
                    break
                self._log_line(line, tag)
        except queue.Empty:
            pass
        self.after(80, self._poll_log)

    # ------------------------------------------------------------------
    # Config save / load
    # ------------------------------------------------------------------

    def _collect_config(self) -> dict:
        slides = []
        for sr in self._slide_rows:
            slides.append({
                "slide_id" : sr.slide_id,
                "condition": sr.condition,
                "run_dir"  : sr.path,
            })
        return {
            "slides"         : slides,
            "base_panel_csv" : self._panel_csv_var.get(),
            "output_dir"     : self._output_dir_var.get(),
            "roi_cache_dir"  : self._roi_cache_var.get(),
            "dge_method"     : self._dge_method_var.get(),
            "panel_mode"     : self._panel_mode_var.get(),
            "roi_mode"       : self._roi_mode_var.get(),
            "redraw_roi"     : self._redraw_roi_var.get(),
            "no_roi_gui"     : self._no_roi_gui_var.get(),
            "use_cache"      : self._use_cache_var.get(),
            "leiden_resolution": self._leiden_res_var.get(),
            "n_neighbors"    : self._n_neighbors_var.get(),
            "min_counts"     : self._min_counts_var.get(),
            "log2fc_threshold": self._log2fc_var.get(),
            "pval_threshold"        : self._pval_var.get(),
            "filter_control_probes"  : True,
            "filter_control_codewords": True,
            "normalize_by_cell_area"  : False,
            "min_slides"     : self._min_slides_var.get(),
        }

    def _save_config(self):
        path = filedialog.asksaveasfilename(
            title="Save pipeline config",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            initialfile="xenium_pipeline_config.json",
        )
        if path:
            cfg = self._collect_config()
            Path(path).write_text(json.dumps(cfg, indent=2))
            self._log_line(f"Config saved: {path}", "ok")
            self._status("Config saved", "ok")

    def _load_config(self):
        path = filedialog.askopenfilename(
            title="Load pipeline config",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if not path:
            return
        cfg = json.loads(Path(path).read_text())
        self._apply_config(cfg)
        self._log_line(f"Config loaded: {path}", "ok")
        self._status("Config loaded", "ok")

    def _apply_config(self, cfg: dict):
        for sr, s in zip(self._slide_rows, cfg.get("slides", [])):
            if s.get("run_dir"):
                sr.path_var.set(s["run_dir"])
                sr._path_label.config(fg=ACCENT_DARK)
            if s.get("slide_id"):
                sr.id_var.set(s["slide_id"])

        if "base_panel_csv"  in cfg: self._panel_csv_var.set(cfg["base_panel_csv"])
        if "output_dir"      in cfg: self._output_dir_var.set(cfg["output_dir"])
        if "roi_cache_dir"   in cfg: self._roi_cache_var.set(cfg["roi_cache_dir"])
        if "dge_method"      in cfg: self._dge_method_var.set(cfg["dge_method"])
        if "panel_mode"      in cfg: self._panel_mode_var.set(cfg["panel_mode"])
        if "roi_mode"        in cfg: self._roi_mode_var.set(cfg["roi_mode"])
        if "leiden_resolution" in cfg: self._leiden_res_var.set(cfg["leiden_resolution"])
        if "n_neighbors"     in cfg: self._n_neighbors_var.set(cfg["n_neighbors"])
        if "min_counts"      in cfg: self._min_counts_var.set(cfg["min_counts"])
        if "log2fc_threshold" in cfg: self._log2fc_var.set(cfg["log2fc_threshold"])
        if "pval_threshold"   in cfg: self._pval_var.set(cfg["pval_threshold"])
        if "min_slides"       in cfg: self._min_slides_var.set(cfg["min_slides"])

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def _log_line(self, text: str, tag: str = "info"):
        self._log.config(state="normal")
        self._log.insert("end", text + "\n", tag)
        self._log.see("end")
        self._log.config(state="disabled")

    def _clear_log(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    def _status(self, text: str, level: str = "info"):
        colour = {"ok": SUCCESS, "warn": WARNING, "error": ACCENT_AGED,
                  "info": "#888888"}.get(level, "#888888")
        self._status_var.set(text)
        self._status_lbl.config(fg=colour)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _separator(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=10)


# ---------------------------------------------------------------------------
# Launcher config ingestion (called by run_xenium_mbh.py)
# ---------------------------------------------------------------------------

def load_launcher_config(config_path: str) -> dict:
    """
    Called by run_xenium_mbh.py when --launcher-config is passed.
    Returns a dict that the pipeline can use to override its defaults.
    """
    return json.loads(Path(config_path).read_text())




# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # On macOS, ensure matplotlib uses the right backend before importing it
    os.environ.setdefault("MPLBACKEND", "MacOSX")

    app = XeniumLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
