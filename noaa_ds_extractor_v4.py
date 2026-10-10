import io
import os
import hashlib
import tempfile
from bisect import bisect_right
import re
import json
import ssl
import certifi
import threading
import urllib.parse
import urllib.request
import urllib.error
import tkinter as tk

from tkinter import ttk, messagebox, filedialog
from datetime import datetime

import pandas as pd
import numpy as np
import librosa
import librosa.display
import soundfile as sf

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ============================================================
# NOAA ACOUSTIC DATASET EXPLORER
# ============================================================

BUCKET_NAME = "noaa-passive-bioacoustic"

GCS_API_URL = (
    f"https://storage.googleapis.com/storage/v1/"
    f"b/{BUCKET_NAME}/o"
)

AUDIO_WINDOW_SECONDS = 5

# Fix SSL certificate verification on macOS.
# Do not disable certificate verification.
SSL_CONTEXT = ssl.create_default_context(
    cafile=certifi.where()
)


def open_secure_url(url, timeout=60, headers=None):
    """Open an HTTPS URL using trusted CA certificates."""
    request = urllib.request.Request(
        url,
        headers=headers or {}
    )
    return urllib.request.urlopen(
        request,
        context=SSL_CONTEXT,
        timeout=timeout
    )


def gcs_object_url(blob_name):
    """Public Google Cloud Storage object URL."""
    encoded = urllib.parse.quote(blob_name, safe="")
    return (
        f"https://storage.googleapis.com/"
        f"{BUCKET_NAME}/{encoded}"
    )


def parse_audio_start(filename):
    """Parse NOAA YYYYMMDD-HHMMSS or YYMMDD-HHMMSS filename stamps as UTC."""
    basename = filename.rsplit("/", 1)[-1]
    for pattern, fmt in (
        (r"(?<!\d)(\d{8})[_-]?(\d{6})(?!\d)", "%Y%m%d%H%M%S"),
        (r"(?<!\d)(\d{6})[_-](\d{6})(?!\d)", "%y%m%d%H%M%S"),
    ):
        match = re.search(pattern, basename)
        if match:
            try:
                return pd.to_datetime("".join(match.groups()), format=fmt, utc=True)
            except (ValueError, TypeError):
                pass
    return None


def normalized_stem(name):
    """Compare WAV names from mapping sheets to FLAC objects in GCS."""
    base = os.path.basename(str(name)).rsplit(".", 1)[0]
    return re.sub(r"[^a-z0-9]", "", base.lower())


def build_mapping_index(mapping_df, audio_files):
    """Return verified recording metadata, never inferred two-hour matches."""
    required = {"File", "FileStartTime", "FileDuration(sec)"}
    missing = required.difference(mapping_df.columns)
    if missing:
        raise ValueError("Mapping spreadsheet is missing: " + ", ".join(sorted(missing)))

    by_stem = {}
    by_start = {}
    for blob in audio_files:
        by_stem.setdefault(normalized_stem(blob), []).append(blob)
        timestamp = parse_audio_start(blob)
        if timestamp is not None:
            by_start.setdefault(timestamp, []).append(blob)

    recordings = []
    for _, row in mapping_df.iterrows():
        filename = str(row["File"]).strip()
        if not filename or filename.lower() == "nan":
            continue
        start = pd.to_datetime(row["FileStartTime"], utc=True, errors="coerce")
        duration = pd.to_numeric(row["FileDuration(sec)"], errors="coerce")
        if pd.isna(start) or pd.isna(duration) or duration <= 0:
            continue

        exact = by_stem.get(normalized_stem(filename), [])
        # Only use a timestamp fallback when there is exactly one candidate.
        possible = exact if exact else by_start.get(start, [])
        blob = possible[0] if len(possible) == 1 else None
        recordings.append({
            "source_file": filename,
            "start": start,
            "duration": float(duration),
            "end": start + pd.Timedelta(seconds=float(duration)),
            "file_name": blob,
            "status": "Ready" if blob else ("Ambiguous audio" if len(possible) > 1 else "Audio missing"),
        })

    recordings.sort(key=lambda r: r["start"])
    return recordings


def match_detections(df, recordings):
    """Match absolute UTC detection times to full five-second windows."""
    normalized = {
        re.sub(r"[_\s]", "", str(c).strip().lower()): c for c in df.columns
    }
    time_column = next(
        (normalized[key] for key in ("isostarttime", "starttime", "timestamp", "time") if key in normalized),
        None,
    )
    if time_column is None:
        raise ValueError(
            "This CSV has no absolute detection timestamp. Columns: "
            + ", ".join(map(str, df.columns))
            + "\n\nThe CPOD CSV (File/ChunkEnd/DPH/MinsOn) belongs to a different "
              "dataset and cannot be matched to GA13 using this spreadsheet. "
              "Choose a GA13 detection CSV with ISOStartTime (UTC), or a "
              "verified file-specific timestamp conversion."
        )

    if "detection" in normalized:
        positive = pd.to_numeric(df[normalized["detection"]], errors="coerce")
        df = df.loc[positive == 1].copy()

    starts = [r["start"] for r in recordings]
    max_duration = max((r["duration"] for r in recordings), default=0)
    file_col = normalized.get("file") or normalized.get("audiofile")
    results = []
    for _, row in df.iterrows():
        when = pd.to_datetime(row[time_column], utc=True, errors="coerce")
        if pd.isna(when):
            continue
        idx = bisect_right(starts, when)
        candidates = []
        for i in range(idx - 1, -1, -1):
            rec = recordings[i]
            if (when - rec["start"]).total_seconds() > max_duration:
                break
            offset = (when - rec["start"]).total_seconds()
            if 0 <= offset and offset + AUDIO_WINDOW_SECONDS <= rec["duration"]:
                candidates.append((rec, offset))

        if file_col and pd.notna(row[file_col]) and str(row[file_col]).strip():
            hint = normalized_stem(row[file_col])
            candidates = [
                (r, offset) for r, offset in candidates
                if normalized_stem(r["source_file"]) == hint
                or (r["file_name"] and normalized_stem(r["file_name"]) == hint)
            ]

        ready = [(r, offset) for r, offset in candidates if r["file_name"]]
        if len(ready) == 1 and len(candidates) == 1:
            rec, offset = ready[0]
            status, blob = "Matched", rec["file_name"]
        elif len(candidates) > 1:
            status, blob, offset = "Ambiguous", None, None
        elif len(candidates) == 1:
            status, blob, offset = "Audio missing", None, None
        else:
            status, blob, offset = "Not Found", None, None

        results.append({
            "det_time": str(when), "file_name": blob,
            "offset": offset, "status": status,
        })
    return results


class GCSBucketNavigator(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("NOAA Acoustic Dataset Explorer")
        self.geometry("1200x850")
        self.minsize(850, 600)

        self.current_prefix = ""
        self.items = []

        self.selected_csv = None
        self.csv_is_local = False
        self.mapping_path = None
        self.last_audio_window = None
        self.last_audio_sr = None
        self.last_det_info = None
        self.selected_audio_folder = None

        self.detections = []
        self.audio_files = []

        self._browser_request_id = 0

        self._build_ui()
        self._async_fetch_gcs("")

    # ========================================================
    # THREAD-SAFE UI UPDATES
    # ========================================================

    def _on_ui(self, callback):
        """
        Queue an update for Tkinter's main event loop.

        Worker threads must not directly modify widgets.
        """
        self.after(0, callback)

    def _set_status(self, text, color="black"):
        self.lbl_status.config(
            text=text,
            foreground=color
        )

    # ========================================================
    # MAIN INTERFACE
    # ========================================================

    def _build_ui(self):

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=10)
        top.grid(row=0, column=0, sticky="ew")

        ttk.Label(
            top,
            text="NOAA Passive Acoustic Data Explorer",
            font=("Arial", 16, "bold")
        ).pack(side="left")

        self.lbl_status = ttk.Label(
            top,
            text="Connecting to NOAA..."
        )
        self.lbl_status.pack(side="right", padx=10)

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=10,
            pady=5
        )

        browser_tab = ttk.Frame(self.notebook)
        self.notebook.add(
            browser_tab,
            text="GCS Browser"
        )
        self._build_browser(browser_tab)

        linking_tab = ttk.Frame(self.notebook)
        self.notebook.add(
            linking_tab,
            text="Detection Linking"
        )
        self._build_linking(linking_tab)

        spectrogram_tab = ttk.Frame(self.notebook)
        self.notebook.add(
            spectrogram_tab,
            text="Spectrogram Viewer"
        )
        self._build_spectrogram(spectrogram_tab)

    # ========================================================
    # GCS BROWSER INTERFACE
    # ========================================================

    def _build_browser(self, parent):

        controls = ttk.Frame(parent, padding=5)
        controls.pack(fill="x")

        ttk.Button(
            controls,
            text="Home",
            command=lambda: self._async_fetch_gcs("")
        ).pack(side="left", padx=3)

        ttk.Button(
            controls,
            text="Up",
            command=self._go_up
        ).pack(side="left", padx=3)

        ttk.Button(
            controls,
            text="Refresh",
            command=lambda: self._async_fetch_gcs(
                self.current_prefix
            )
        ).pack(side="left", padx=3)

        self.lbl_path = ttk.Label(
            controls,
            text="/",
            font=("Arial", 10)
        )
        self.lbl_path.pack(side="left", padx=15)

        list_frame = ttk.Frame(parent)
        list_frame.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5
        )

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.lst_files = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("Courier", 11),
            selectmode=tk.BROWSE
        )
        self.lst_files.pack(
            fill="both",
            expand=True
        )

        scrollbar.config(
            command=self.lst_files.yview
        )

        self.lst_files.bind(
            "<Double-Button-1>",
            self._on_double_click
        )

        actions = ttk.Frame(parent, padding=10)
        actions.pack(fill="x")

        ttk.Button(
            actions,
            text="Select as Detection CSV",
            command=self._select_csv
        ).pack(side="left", padx=5)

        ttk.Button(
            actions,
            text="Select as Audio Folder",
            command=self._select_audio_folder
        ).pack(side="left", padx=5)

    # ========================================================
    # DETECTION LINKING INTERFACE
    # ========================================================

    def _build_linking(self, parent):

        info = ttk.Frame(parent, padding=10)
        info.pack(fill="x")

        self.lbl_csv = ttk.Label(
            info,
            text="Detection CSV: Not selected",
            wraplength=1000
        )
        self.lbl_csv.pack(anchor="w", pady=3)

        ttk.Button(
            info, text="Choose Local Detection CSV",
            command=self._choose_local_csv
        ).pack(anchor="w", pady=3)

        self.lbl_mapping = ttk.Label(
            info, text="File Mapping XLSX: Not selected", wraplength=1000
        )
        self.lbl_mapping.pack(anchor="w", pady=3)
        mapping_buttons = ttk.Frame(info)
        mapping_buttons.pack(anchor="w", pady=3)
        ttk.Button(
            mapping_buttons, text="Choose File Mapping XLSX",
            command=self._choose_mapping
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            mapping_buttons, text="Preview Mapping (No CSV Needed)",
            command=self._async_preview_mapping
        ).pack(side="left")

        self.lbl_audio = ttk.Label(
            info,
            text="Audio Folder: Not selected",
            wraplength=1000
        )
        self.lbl_audio.pack(anchor="w", pady=3)

        ttk.Button(
            info,
            text="Load and Match Detections",
            command=self._async_process_linking
        ).pack(anchor="w", pady=10)

        columns = (
            "time",
            "audio",
            "offset",
            "status"
        )

        tree_frame = ttk.Frame(parent)
        tree_frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        self.tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings"
        )

        self.tree.heading(
            "time",
            text="Detection Time"
        )
        self.tree.heading(
            "audio",
            text="Audio File"
        )
        self.tree.heading(
            "offset",
            text="Offset (s)"
        )
        self.tree.heading(
            "status",
            text="Status"
        )

        self.tree.column("time", width=220)
        self.tree.column("audio", width=420)
        self.tree.column("offset", width=100)
        self.tree.column("status", width=130)

        tree_scroll = ttk.Scrollbar(
            tree_frame,
            orient="vertical",
            command=self.tree.yview
        )

        self.tree.configure(
            yscrollcommand=tree_scroll.set
        )

        tree_scroll.pack(side="right", fill="y")
        self.tree.pack(
            side="left",
            fill="both",
            expand=True
        )

        self.tree.bind(
            "<Double-Button-1>",
            self._on_detection_selected
        )

        self.lbl_link_status = ttk.Label(
            parent,
            text="No detections loaded"
        )
        self.lbl_link_status.pack(
            anchor="w",
            padx=10,
            pady=5
        )

    # ========================================================
    # SPECTROGRAM INTERFACE
    # ========================================================

    def _build_spectrogram(self, parent):

        self.lbl_spec_status = ttk.Label(
            parent,
            text="Double-click a matched detection to view audio"
        )
        self.lbl_spec_status.pack(
            anchor="w",
            padx=10,
            pady=10
        )

        self.fig = Figure(
            figsize=(10, 6),
            dpi=100
        )

        self.ax = self.fig.add_subplot(111)

        self.ax.set_title(
            "NOAA Marine Mammal Acoustic Spectrogram"
        )
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Frequency (Hz)")

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=parent
        )

        ttk.Button(
            parent, text="Save Current 5-Second Clip as WAV",
            command=self._save_current_clip
        ).pack(anchor="w", padx=10, pady=5)

        self.canvas.draw()
        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    # ========================================================
    # GOOGLE CLOUD STORAGE API
    # ========================================================

    def _fetch_gcs_items(self, prefix, recursive=False):
        """
        List public GCS objects.

        Uses the public JSON API, which does not require
        Google Cloud credentials for public objects.
        """

        items = []
        page_token = None

        while True:

            params = {
                "prefix": prefix,
                "maxResults": 1000
            }

            if not recursive:
                params["delimiter"] = "/"

            if page_token:
                params["pageToken"] = page_token

            url = (
                GCS_API_URL
                + "?"
                + urllib.parse.urlencode(params)
            )

            with open_secure_url(
                url,
                timeout=60
            ) as response:

                data = json.loads(
                    response.read().decode("utf-8")
                )

            if not recursive:
                for folder in data.get("prefixes", []):
                    items.append({
                        "name": folder,
                        "type": "folder"
                    })

            for obj in data.get("items", []):

                name = obj.get("name", "")

                if not name or name == prefix:
                    continue

                items.append({
                    "name": name,
                    "type": "file",
                    "size": int(obj.get("size", 0))
                })

            page_token = data.get("nextPageToken")

            if not page_token:
                break

        return items

    def _async_fetch_gcs(self, prefix):

        self._browser_request_id += 1
        request_id = self._browser_request_id

        self._set_status(
            "Loading NOAA files...",
            "blue"
        )

        def worker():

            try:
                items = self._fetch_gcs_items(prefix)

                def update():
                    if request_id == self._browser_request_id:
                        self._update_browser(
                            prefix,
                            items
                        )

                self._on_ui(update)

            except Exception as exc:

                error_message = str(exc)

                def show_error():

                    if request_id != self._browser_request_id:
                        return

                    self._set_status(
                        f"GCS Error: {error_message}",
                        "red"
                    )

                    messagebox.showerror(
                        "NOAA Connection Error",
                        "Unable to load NOAA data.\n\n"
                        + error_message
                    )

                self._on_ui(show_error)

        threading.Thread(
            target=worker,
            daemon=True
        ).start()

    def _update_browser(self, prefix, items):

        self.current_prefix = prefix
        self.items = items

        self.lst_files.delete(0, tk.END)

        for item in items:

            name = (
                item["name"]
                .rstrip("/")
                .split("/")[-1]
            )

            if item["type"] == "folder":
                display = f"[DIR] {name}/"
            else:
                display = f"      {name}"

            self.lst_files.insert(
                tk.END,
                display
            )

        self.lbl_path.config(
            text="/" + prefix
        )

        self._set_status(
            f"Connected - {len(items)} items",
            "green"
        )

    def _go_up(self):

        parts = (
            self.current_prefix
            .rstrip("/")
            .split("/")
        )

        if len(parts) <= 1:
            new_prefix = ""
        else:
            new_prefix = (
                "/".join(parts[:-1]) + "/"
            )

        self._async_fetch_gcs(new_prefix)

    def _on_double_click(self, event):

        selection = self.lst_files.curselection()

        if not selection:
            return

        item = self.items[selection[0]]

        if item["type"] == "folder":
            self._async_fetch_gcs(
                item["name"]
            )

    # ========================================================
    # SELECT DATASET FILES
    # ========================================================

    def _select_csv(self):

        selection = self.lst_files.curselection()

        if not selection:
            messagebox.showwarning(
                "No Selection",
                "Select a CSV file first."
            )
            return

        item = self.items[selection[0]]

        if (
            item["type"] != "file"
            or not item["name"].lower().endswith(".csv")
        ):
            messagebox.showwarning(
                "Invalid Selection",
                "Please select a CSV file."
            )
            return

        self.selected_csv = item["name"]
        self.csv_is_local = False

        self.lbl_csv.config(
            text=f"Detection CSV: {self.selected_csv}"
        )

        self._set_status(
            "Detection CSV selected",
            "green"
        )

    def _choose_local_csv(self):
        path = filedialog.askopenfilename(
            title="Choose a GA13 detection CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        if path:
            self.selected_csv = path
            self.csv_is_local = True
            self.lbl_csv.config(text=f"Detection CSV (local): {path}")
            self._set_status("Local detection CSV selected", "green")

    def _choose_mapping(self):
        path = filedialog.askopenfilename(
            title="Choose GA13 file mapping spreadsheet",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if path:
            self.mapping_path = path
            self.lbl_mapping.config(text=f"File Mapping XLSX: {path}")
            self._set_status("Mapping spreadsheet selected", "green")

    def _get_audio_and_mapping(self, audio_folder, mapping_path):
        audio_items = self._fetch_gcs_items(audio_folder, recursive=True)
        audio_files = [
            item["name"] for item in audio_items
            if item["type"] == "file"
            and item["name"].lower().endswith((".wav", ".flac"))
        ]
        mapping_df = pd.read_excel(mapping_path, engine="openpyxl")
        recordings = build_mapping_index(mapping_df, audio_files)
        if not recordings:
            raise ValueError("No valid recording rows in the mapping spreadsheet.")
        return audio_files, recordings

    def _async_preview_mapping(self):
        if not self.mapping_path or not self.selected_audio_folder:
            messagebox.showwarning(
                "Select inputs",
                "Choose the XLSX mapping and select the GA13 audio folder first."
            )
            return
        self.lbl_link_status.config(text="Checking spreadsheet against GCS audio...")
        audio_folder, mapping_path = self.selected_audio_folder, self.mapping_path

        def worker():
            try:
                audio_files, recordings = self._get_audio_and_mapping(audio_folder, mapping_path)
                def update():
                    self.audio_files = audio_files
                    self.detections = []
                    self.tree.heading("time", text="Recording Start (UTC)")
                    self.tree.heading("offset", text="Duration")
                    for item in self.tree.get_children():
                        self.tree.delete(item)
                    for i, rec in enumerate(recordings):
                        self.tree.insert("", "end", iid=str(i), values=(
                            str(rec["start"]), rec["file_name"] or rec["source_file"],
                            f"{rec['duration']:.1f}s", rec["status"]
                        ))
                    ready = sum(r["file_name"] is not None for r in recordings)
                    self.lbl_link_status.config(text=(
                        f"Mapping: {len(recordings)} rows, {ready} audio files matched. "
                        "This is a recording preview, not labeled detections."
                    ))
                    self.notebook.select(1)
                self._on_ui(update)
            except Exception as exc:
                msg = str(exc)
                self._on_ui(lambda: messagebox.showerror("Mapping Preview Error", msg))
        threading.Thread(target=worker, daemon=True).start()

    def _select_audio_folder(self):

        selection = self.lst_files.curselection()

        if selection:

            item = self.items[selection[0]]

            if item["type"] == "folder":
                folder = item["name"]
            else:
                folder = self.current_prefix

        else:
            folder = self.current_prefix

        if not folder:
            messagebox.showwarning(
                "Invalid Audio Folder",
                "Navigate into an audio dataset folder first."
            )
            return

        self.selected_audio_folder = folder

        self.lbl_audio.config(
            text=f"Audio Folder: {folder}"
        )

        self._set_status(
            "Audio folder selected",
            "green"
        )

    # ========================================================
    # DOWNLOAD PUBLIC NOAA FILES
    # ========================================================

    def _download_gcs_bytes(self, blob_name):
        """
        Download a public object into memory.

        Suitable for CSV files. Full audio downloads can
        be large, so audio is handled separately.
        """

        url = gcs_object_url(blob_name)

        with open_secure_url(
            url,
            timeout=180
        ) as response:
            return response.read()

    def _download_audio_to_cache(self, blob_name):
        """Stream a large NOAA FLAC to disk, then reuse the cached copy."""
        cache_dir = os.path.join(tempfile.gettempdir(), "noaa_acoustic_cache")
        os.makedirs(cache_dir, exist_ok=True)
        extension = os.path.splitext(blob_name)[1].lower()
        digest = hashlib.sha256(blob_name.encode("utf-8")).hexdigest()
        path = os.path.join(cache_dir, digest + extension)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
        partial = path + ".partial"
        try:
            with open_secure_url(gcs_object_url(blob_name), timeout=300) as response:
                with open(partial, "wb") as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
            os.replace(partial, path)
        finally:
            if os.path.exists(partial):
                os.remove(partial)
        return path

    # ========================================================
    # DETECTION AND AUDIO MATCHING
    # ========================================================

    def _async_process_linking(self):
        if not self.selected_csv:
            messagebox.showwarning("Missing CSV", "Choose a GA13 detection CSV first.")
            return
        if not self.mapping_path:
            messagebox.showwarning("Missing Mapping", "Choose the GA13 mapping XLSX first.")
            return
        if not self.selected_audio_folder:
            messagebox.showwarning("Missing Audio Folder", "Select the GA13 audio folder first.")
            return

        csv_name, is_local = self.selected_csv, self.csv_is_local
        audio_folder, mapping_path = self.selected_audio_folder, self.mapping_path
        self.lbl_link_status.config(text="Loading detections and verifying recording metadata...")

        def worker():
            try:
                if is_local:
                    df = pd.read_csv(csv_name)
                else:
                    df = pd.read_csv(io.BytesIO(self._download_gcs_bytes(csv_name)))
                if df.empty:
                    raise ValueError("Detection CSV contains no rows.")
                # Validate timestamps before downloading the GCS audio listing.
                if not any(
                    re.sub(r"[_\s]", "", str(c).lower())
                    in {"isostarttime", "starttime", "timestamp", "time"}
                    for c in df.columns
                ):
                    match_detections(df, [])  # raises a helpful format error

                audio_files, recordings = self._get_audio_and_mapping(audio_folder, mapping_path)
                matches = match_detections(df, recordings)
                def update():
                    self.audio_files = audio_files
                    self.tree.heading("time", text="Detection Time (UTC)")
                    self.tree.heading("offset", text="Offset (s)")
                    self._update_linking_results(matches)
                self._on_ui(update)
            except Exception as exc:
                msg = str(exc)
                def show_error():
                    self.lbl_link_status.config(text="Error processing detections")
                    messagebox.showerror("Detection Linking Error", msg)
                self._on_ui(show_error)

        threading.Thread(target=worker, daemon=True).start()

    def _update_linking_results(self, matches):

        self.detections = matches

        for item in self.tree.get_children():
            self.tree.delete(item)

        for index, match in enumerate(matches):

            offset = match["offset"]

            offset_text = (
                f"{offset:.2f}"
                if offset is not None
                else ""
            )

            self.tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    match["det_time"],
                    match["file_name"] or "",
                    offset_text,
                    match["status"]
                )
            )

        matched_count = sum(
            1
            for match in matches
            if match["status"] == "Matched"
        )

        self.lbl_link_status.config(
            text=(
                f"Processed {len(matches)} detections. "
                f"Verified timestamp matches: {matched_count}. "
                f"Audio files found: {len(self.audio_files)}."
            )
        )

        self.notebook.select(1)

    # ========================================================
    # SELECT DETECTION FOR SPECTROGRAM
    # ========================================================

    def _on_detection_selected(self, event):

        selection = self.tree.selection()

        if not selection:
            return

        index = int(selection[0])
        if index >= len(self.detections):
            return  # mapping preview rows are not detections
        detection = self.detections[index]

        if detection["file_name"] is None:

            messagebox.showinfo(
                "No Match",
                "No candidate audio file was found."
            )
            return

        self._async_load_and_plot_spectrogram(
            detection
        )

        self.notebook.select(2)

    # ========================================================
    # LOAD AUDIO AND GENERATE SPECTROGRAM
    # ========================================================

    def _async_load_and_plot_spectrogram(self, det_info):

        self.lbl_spec_status.config(
            text="Downloading and decoding audio...",
            foreground="blue"
        )

        def worker():

            try:

                audio_path = self._download_audio_to_cache(
                    det_info["file_name"]
                )

                with sf.SoundFile(audio_path) as audio:

                    sr = audio.samplerate

                    offset = float(
                        det_info["offset"]
                    )

                    start_frame = int(
                        offset * sr
                    )

                    # Verify that the candidate offset
                    # falls inside the actual recording.
                    if (
                        start_frame < 0
                        or start_frame >= len(audio)
                    ):
                        raise ValueError(
                            "Detection offset falls outside "
                            "the recording duration. "
                            "The filename match is not valid."
                        )

                    audio.seek(start_frame)

                    frames = int(
                        AUDIO_WINDOW_SECONDS * sr
                    )

                    y = audio.read(
                        frames,
                        dtype="float32",
                        always_2d=False
                    )

                if y.ndim > 1:
                    y = y.mean(axis=1)

                if len(y) == 0:
                    raise ValueError(
                        "No audio samples at this offset."
                    )

                # Require a complete training window.
                if len(y) < frames:
                    raise ValueError(
                        "Recording ends before a complete "
                        f"{AUDIO_WINDOW_SECONDS}-second "
                        "window can be extracted."
                    )

                # Compute a mel spectrogram.
                n_fft = min(
                    2048,
                    len(y)
                )

                hop_length = 512

                S = librosa.feature.melspectrogram(
                    y=y,
                    sr=sr,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=128
                )

                S_db = librosa.power_to_db(
                    S,
                    ref=np.max
                )

                duration = len(y) / sr

                self._on_ui(
                    lambda: self._plot_spectrogram(
                        S_db,
                        sr,
                        det_info,
                        duration,
                        y
                    )
                )

            except Exception as exc:

                error_message = str(exc)

                def show_error():

                    self.lbl_spec_status.config(
                        text=(
                            "Audio Error: "
                            + error_message
                        ),
                        foreground="red"
                    )

                    messagebox.showerror(
                        "Spectrogram Error",
                        error_message
                    )

                self._on_ui(show_error)

        threading.Thread(
            target=worker,
            daemon=True
        ).start()

    # ========================================================
    # DISPLAY SPECTROGRAM
    # ========================================================

    def _plot_spectrogram(
        self,
        S_db,
        sr,
        det_info,
        duration,
        y
    ):

        self.last_audio_window = y.copy()
        self.last_audio_sr = sr
        self.last_det_info = det_info
        self.fig.clear()

        self.ax = self.fig.add_subplot(111)

        img = librosa.display.specshow(
            S_db,
            x_axis="time",
            y_axis="mel",
            sr=sr,
            hop_length=512,
            fmax=sr / 2,
            ax=self.ax,
            cmap="magma"
        )

        self.fig.colorbar(
            img,
            ax=self.ax,
            format="%+2.0f dB"
        )

        filename = (
            det_info["file_name"]
            .split("/")[-1]
        )

        self.ax.set_title(
            f"NOAA Audio: {filename}\n"
            f"Offset: {det_info['offset']:.2f}s",
            fontsize=10
        )

        self.ax.set_xlabel(
            "Time (seconds)"
        )

        self.ax.set_ylabel(
            "Frequency (Hz)"
        )

        self.fig.tight_layout()
        self.canvas.draw()

        self.lbl_spec_status.config(
            text=(
                f"Loaded {duration:.1f}s audio window "
                f"from detection {det_info['det_time']}"
            ),
            foreground="green"
        )

    def _save_current_clip(self):
        if self.last_audio_window is None:
            messagebox.showinfo("No audio", "Double-click a matched detection first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save 5-second audio clip", defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
            initialfile="noaa_detection_5s.wav"
        )
        if not path:
            return
        try:
            sf.write(path, self.last_audio_window, self.last_audio_sr)
            self._set_status("Saved 5-second WAV clip", "green")
        except Exception as exc:
            messagebox.showerror("Save Audio Error", str(exc))

    # ========================================================
    # APPLICATION ENTRY POINT
    # ========================================================


if __name__ == "__main__":

    app = GCSBucketNavigator()
    app.mainloop()
