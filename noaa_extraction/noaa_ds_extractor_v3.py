import io
import re
import json
import urllib.parse
import urllib.request
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from datetime import datetime
from google.cloud import storage

# Audio & Plotting imports
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

BUCKET_NAME = "noaa-passive-bioacoustic"
GCS_API_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET_NAME}/o"

class GCSBucketNavigator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NOAA Bioacoustic Navigator & Mel-Spectrogram Inspector")
        self.geometry("1400x900")

        # Anonymous Client for downloading objects
        self.storage_client = storage.Client.create_anonymous_client()
        self.bucket = self.storage_client.bucket(BUCKET_NAME)

        # Separate state tracking for Viewer 1 (CSV) and Viewer 2 (Audio)
        self.v1_prefix = ""
        self.v1_page_token = None
        
        self.v2_prefix = ""
        self.v2_page_token = None

        self.selected_csv_blob = None
        self.selected_audio_folder = None
        
        # Store detection mapping results for interactive selection
        self.detection_results = []

        self._create_widgets()
        # Initialize both viewers at bucket root
        self.load_directory_v1("")
        self.load_directory_v2("")

    def _create_widgets(self):
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ================= LEFT PANEL: DUAL FILE VIEWER =================
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)

        # Vertical paned window to stack the two viewers
        viewer_paned = ttk.PanedWindow(left_frame, orient=tk.VERTICAL)
        viewer_paned.pack(fill=tk.BOTH, expand=True)

        # --- VIEWER 1: CSV SELECTOR ---
        v1_group = ttk.LabelFrame(viewer_paned, text="Viewer 1: CSV Directory Browser")
        viewer_paned.add(v1_group, weight=1)

        v1_nav = ttk.Frame(v1_group)
        v1_nav.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(v1_nav, text="⬆ Up", width=5, command=self._navigate_up_v1).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(v1_nav, text="🏠 Root", width=5, command=lambda: self.load_directory_v1("")).pack(side=tk.LEFT, padx=(0, 2))
        self.lbl_v1_path = ttk.Label(v1_nav, text="/", font=("Consolas", 9, "bold"))
        self.lbl_v1_path.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        v1_list_frame = ttk.Frame(v1_group)
        v1_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        self.v1_list = tk.Listbox(v1_list_frame, font=("Consolas", 9), selectmode=tk.SINGLE, exportselection=False)
        self.v1_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v1_scroll = ttk.Scrollbar(v1_list_frame, orient=tk.VERTICAL, command=self.v1_list.yview)
        v1_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.v1_list.configure(yscrollcommand=v1_scroll.set)
        self.v1_list.bind("<Double-1>", self._on_v1_double_click)

        v1_btn_bar = ttk.Frame(v1_group)
        v1_btn_bar.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(v1_btn_bar, text="Set Selected File as Target CSV", command=self._set_selected_csv).pack(fill=tk.X)

        # --- VIEWER 2: AUDIO DIRECTORY SELECTOR ---
        v2_group = ttk.LabelFrame(viewer_paned, text="Viewer 2: Audio Folder Browser")
        viewer_paned.add(v2_group, weight=1)

        v2_nav = ttk.Frame(v2_group)
        v2_nav.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(v2_nav, text="⬆ Up", width=5, command=self._navigate_up_v2).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(v2_nav, text="🏠 Root", width=5, command=lambda: self.load_directory_v2("")).pack(side=tk.LEFT, padx=(0, 2))
        self.lbl_v2_path = ttk.Label(v2_nav, text="/", font=("Consolas", 9, "bold"))
        self.lbl_v2_path.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        v2_list_frame = ttk.Frame(v2_group)
        v2_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        self.v2_list = tk.Listbox(v2_list_frame, font=("Consolas", 9), selectmode=tk.SINGLE, exportselection=False)
        self.v2_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v2_scroll = ttk.Scrollbar(v2_list_frame, orient=tk.VERTICAL, command=self.v2_list.yview)
        v2_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.v2_list.configure(yscrollcommand=v2_scroll.set)
        self.v2_list.bind("<Double-1>", self._on_v2_double_click)

        v2_btn_bar = ttk.Frame(v2_group)
        v2_btn_bar.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(v2_btn_bar, text="Set Current Path as Audio Folder", command=self._set_current_audio_folder).pack(fill=tk.X)

        # Global Status Footer
        self.lbl_status = ttk.Label(left_frame, text="Ready", foreground="gray")
        self.lbl_status.pack(anchor=tk.W, padx=5, pady=(2, 0))

        # ================= RIGHT PANEL: RESULTS & SPECTROGRAM =================
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        config_group = ttk.LabelFrame(right_frame, text="Target Configuration")
        config_group.pack(fill=tk.X, pady=(0, 5))

        self.lbl_csv = ttk.Label(config_group, text="Selected CSV: None", wraplength=550, foreground="red")
        self.lbl_csv.pack(anchor=tk.W, padx=5, pady=2)

        self.lbl_audio_dir = ttk.Label(config_group, text="Linked Audio Folder: None", wraplength=550, foreground="red")
        self.lbl_audio_dir.pack(anchor=tk.W, padx=5, pady=2)

        ttk.Button(config_group, text="Process Link & Match Detections", command=self.process_linking).pack(padx=5, pady=5, fill=tk.X)

        right_paned = ttk.PanedWindow(right_frame, orient=tk.VERTICAL)
        right_paned.pack(fill=tk.BOTH, expand=True)

        # Detections Table Group
        results_group = ttk.LabelFrame(right_paned, text="Matched Detections (Click a row to load 30s Spectrogram)")
        right_paned.add(results_group, weight=1)

        table_container = ttk.Frame(results_group)
        table_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        columns = ("time", "file", "offset")
        self.det_table = ttk.Treeview(table_container, columns=columns, show="headings", selectmode="browse")
        self.det_table.heading("time", text="Detection Time (UTC)")
        self.det_table.heading("file", text="Audio File")
        self.det_table.heading("offset", text="Offset (sec)")

        self.det_table.column("time", width=180)
        self.det_table.column("file", width=260)
        self.det_table.column("offset", width=90)

        table_scroll = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.det_table.yview)
        self.det_table.configure(yscrollcommand=table_scroll.set)
        
        self.det_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        table_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.det_table.bind("<<TreeviewSelect>>", self._on_detection_selected)

        # Spectrogram Visualizer Group
        spec_group = ttk.LabelFrame(right_paned, text="30-Second Window Mel-Spectrogram")
        right_paned.add(spec_group, weight=1)

        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Select a detection from the table above")
        self.ax.axis("off")

        self.canvas = FigureCanvasTkAgg(self.fig, master=spec_group)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.lbl_spec_status = ttk.Label(spec_group, text="", foreground="blue")
        self.lbl_spec_status.pack(anchor=tk.W, padx=5, pady=2)

    # ---------------- VIEWER 1 LOGIC (CSV) ----------------
    def load_directory_v1(self, prefix, page_token=None):
        self.v1_prefix = prefix
        self.lbl_v1_path.config(text=f"/{prefix}")
        if not page_token:
            self.v1_list.delete(0, tk.END)
        threading.Thread(target=self._async_fetch_gcs, args=(prefix, page_token, 1), daemon=True).start()

    def _navigate_up_v1(self):
        if not self.v1_prefix:
            return
        parts = self.v1_prefix.rstrip("/").split("/")[:-1]
        new_prefix = "/".join(parts) + "/" if parts else ""
        self.load_directory_v1(new_prefix)

    def _on_v1_double_click(self, event):
        selection = self.v1_list.curselection()
        if not selection:
            return
        item_text = self.v1_list.get(selection[0])
        if item_text == "🔄 Load More Items...":
            self.load_directory_v1(self.v1_prefix, page_token=self.v1_page_token)
        elif item_text.startswith("📁 "):
            folder_name = item_text[2:]
            self.load_directory_v1(f"{self.v1_prefix}{folder_name}")

    def _set_selected_csv(self):
        selection = self.v1_list.curselection()
        if not selection:
            return
        item_text = self.v1_list.get(selection[0])
        if item_text.endswith(".csv"):
            file_name = item_text[2:]
            self.selected_csv_blob = f"{self.v1_prefix}{file_name}"
            self.lbl_csv.config(text=f"Selected CSV: {self.selected_csv_blob}", foreground="green")
        else:
            messagebox.showerror("Error", "Please select a .csv file in Viewer 1.")

    # ---------------- VIEWER 2 LOGIC (AUDIO) ----------------
    def load_directory_v2(self, prefix, page_token=None):
        self.v2_prefix = prefix
        self.lbl_v2_path.config(text=f"/{prefix}")
        if not page_token:
            self.v2_list.delete(0, tk.END)
        threading.Thread(target=self._async_fetch_gcs, args=(prefix, page_token, 2), daemon=True).start()

    def _navigate_up_v2(self):
        if not self.v2_prefix:
            return
        parts = self.v2_prefix.rstrip("/").split("/")[:-1]
        new_prefix = "/".join(parts) + "/" if parts else ""
        self.load_directory_v2(new_prefix)

    def _on_v2_double_click(self, event):
        selection = self.v2_list.curselection()
        if not selection:
            return
        item_text = self.v2_list.get(selection[0])
        if item_text == "🔄 Load More Items...":
            self.load_directory_v2(self.v2_prefix, page_token=self.v2_page_token)
        elif item_text.startswith("📁 "):
            folder_name = item_text[2:]
            self.load_directory_v2(f"{self.v2_prefix}{folder_name}")

    def _set_current_audio_folder(self):
        self.selected_audio_folder = self.v2_prefix
        self.lbl_audio_dir.config(text=f"Linked Audio Folder: {self.selected_audio_folder}", foreground="green")

    # ---------------- SHARED GCS ASYNC FETCH ----------------
    def _async_fetch_gcs(self, prefix, page_token, viewer_id):
        try:
            params = {"prefix": prefix, "delimiter": "/", "maxResults": 200}
            if page_token:
                params["pageToken"] = page_token

            url = f"{GCS_API_URL}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url)

            with urllib.request.urlopen(req, timeout=8.0) as response:
                data = json.loads(response.read().decode("utf-8"))

            subfolders = data.get("prefixes", [])
            items = data.get("items", [])
            next_token = data.get("nextPageToken")

            files = [
                item["name"] for item in items 
                if item["name"] != prefix and not item["name"].endswith("/")
            ]

            self.after(0, self._render_items, prefix, subfolders, files, next_token, viewer_id)

        except Exception as e:
            self.after(0, lambda: self.lbl_status.config(text=f"Error fetching V{viewer_id}: {e}", foreground="red"))

    def _render_items(self, prefix, subfolders, files, next_token, viewer_id):
        target_list = self.v1_list if viewer_id == 1 else self.v2_list

        if viewer_id == 1:
            self.v1_page_token = next_token
        else:
            self.v2_page_token = next_token

        if target_list.size() > 0 and target_list.get(tk.END) == "🔄 Load More Items...":
            target_list.delete(tk.END)

        for folder in subfolders:
            folder_name = folder[len(prefix):].rstrip("/")
            target_list.insert(tk.END, f"📁 {folder_name}/")

        for file_path in files:
            file_name = file_path[len(prefix):]
            if file_name:
                icon = "📊" if file_name.endswith(".csv") else "🎵"
                target_list.insert(tk.END, f"{icon} {file_name}")

        if next_token:
            target_list.insert(tk.END, "🔄 Load More Items...")

        self.lbl_status.config(text=f"Viewer {viewer_id}: Loaded {len(subfolders)} folders, {len(files)} files.", foreground="green")

    # ---------------- TIMESTAMP PARSER ----------------
    def parse_timestamp(self, filename):
        patterns = [
            r'(\d{8}T\d{6}Z)',                   # e.g. 20211024T123000Z
            r'(\d{6}-\d{6})',                    # e.g. 211024-123000
            r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', # e.g. 2021-10-24_12-30-00
            r'(\d{12,14})'                        # e.g. 20211024123000
        ]
        for p in patterns:
            m = re.search(p, filename)
            if m:
                t_str = m.group(1)
                try:
                    if 'T' in t_str: return datetime.strptime(t_str, "%Y%m%dT%H%M%SZ")
                    elif '-' in t_str and len(t_str) == 13: return datetime.strptime(t_str, "%y%m%d-%H%M%S")
                    elif '_' in t_str: return datetime.strptime(t_str, "%Y-%m-%d_%H-%M-%S")
                    elif len(t_str) == 14: return datetime.strptime(t_str, "%Y%m%d%H%M%S")
                    elif len(t_str) == 12: return datetime.strptime(t_str, "%y%m%d%H%M%S")
                except ValueError: continue
        return None

    # ---------------- MATCHING ENGINE ----------------
    def process_linking(self):
        if not self.selected_csv_blob or not self.selected_audio_folder:
            messagebox.showwarning("Warning", "Select both a CSV file and an audio directory.")
            return

        for item in self.det_table.get_children():
            self.det_table.delete(item)
        self.detection_results.clear()

        threading.Thread(target=self._async_process_linking, daemon=True).start()

    def _async_process_linking(self):
        try:
            csv_text = self.bucket.blob(self.selected_csv_blob).download_as_text()
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Failed reading CSV: {e}"))
            return

        iso_col = None
        for col_name in df.columns:
            if col_name.lower() in ["isostarttime", "iso_start_time", "start_time", "datetime"]:
                iso_col = col_name
                break

        presence_col = None
        for col_name in df.columns:
            if col_name.lower() in ["presence", "species_presence", "detection"]:
                presence_col = col_name
                break

        if not iso_col:
            self.after(0, lambda: messagebox.showerror("Error", f"Could not find 'ISOStartTime' column in CSV."))
            return

        if presence_col:
            df[presence_col] = pd.to_numeric(df[presence_col], errors='coerce').fillna(0)
            detections_df = df[df[presence_col] == 1].copy()
        else:
            detections_df = df.copy()

        if detections_df.empty:
            self.after(0, lambda: messagebox.showwarning("Warning", "No rows with Presence == 1 were found."))
            return

        audio_items = []
        page_token = None
        while True:
            try:
                params = {"prefix": self.selected_audio_folder, "maxResults": 1000}
                if page_token:
                    params["pageToken"] = page_token
                url = f"{GCS_API_URL}?{urllib.parse.urlencode(params)}"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                audio_items.extend(data.get("items", []))
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"Failed reading audio folder: {e}"))
                return

        audio_list = []
        for item in audio_items:
            name = item["name"]
            if name.endswith(('.flac', '.wav', '.aif', '.mp3')):
                fname = name.split("/")[-1]
                ftime = self.parse_timestamp(fname)
                if ftime:
                    audio_list.append({"name": fname, "full_path": name, "start": ftime})

        audio_df = pd.DataFrame(audio_list)
        if audio_df.empty:
            self.after(0, lambda: messagebox.showwarning("Warning", "No timestamped audio files found in directory."))
            return

        audio_df = audio_df.sort_values("start").reset_index(drop=True)

        results = []
        for _, row in detections_df.iterrows():
            raw_iso = str(row[iso_col])
            try:
                det_time = pd.to_datetime(raw_iso).tz_localize(None).to_pydatetime()
            except Exception:
                continue

            candidates = audio_df[audio_df['start'] <= det_time]
            if not candidates.empty:
                match = candidates.iloc[-1]
                offset = (det_time - match['start']).total_seconds()
                if offset <= 7200:
                    results.append({
                        "det_time": str(det_time),
                        "file_name": match['name'],
                        "full_path": match['full_path'],
                        "offset": offset
                    })

        self.detection_results = results
        self.after(0, self._populate_table_results)

    def _populate_table_results(self):
        for idx, item in enumerate(self.detection_results):
            self.det_table.insert(
                "", "end", iid=str(idx), 
                values=(item['det_time'], item['file_name'], f"{item['offset']:.2f}")
            )

    # ---------------- SPECTROGRAM GENERATOR ----------------
    def _on_detection_selected(self, event):
        selected_item = self.det_table.focus()
        if not selected_item:
            return

        idx = int(selected_item)
        detection_info = self.detection_results[idx]

        self.lbl_spec_status.config(text="Streaming audio snippet from GCS and rendering Mel-Spectrogram...", foreground="blue")
        threading.Thread(target=self._async_load_and_plot_spectrogram, args=(detection_info,), daemon=True).start()

    def _async_load_and_plot_spectrogram(self, det_info):
        blob_path = det_info['full_path']
        offset = det_info['offset']
        duration = 30.0  # 30-second window

        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem(token='anon')
            gcs_file_path = f"{BUCKET_NAME}/{blob_path}"

            with fs.open(gcs_file_path, 'rb') as f:
                with sf.SoundFile(f) as sf_file:
                    sr = sf_file.samplerate
                    channels = sf_file.channels
                    
                    start_frame = int(offset * sr)
                    frames_to_read = int(duration * sr)

                    if start_frame >= len(sf_file):
                        self.after(0, lambda: self.lbl_spec_status.config(
                            text="Error: Selected offset is outside audio duration.", foreground="red"
                        ))
                        return

                    sf_file.seek(start_frame)
                    y = sf_file.read(frames_to_read, dtype='float32', always_2d=True)
                    y = np.mean(y, axis=1) if channels > 1 else y.squeeze()

            if len(y) == 0:
                self.after(0, lambda: self.lbl_spec_status.config(
                    text="Error: Extracted snippet is empty.", foreground="red"
                ))
                return

            S = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmin=10, fmax=sr / 2
            )
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Denoise continuous background noise
            S_dB = S_dB - np.median(S_dB, axis=1, keepdims=True)
            S_dB = np.maximum(S_dB, 0)

            self.after(0, self._update_matplotlib_canvas, S_dB, sr, det_info, duration)

        except Exception as e:
            self.after(0, lambda: self.lbl_spec_status.config(
                text=f"Error rendering audio snippet: {e}", foreground="red"
            ))

    def _update_matplotlib_canvas(self, S_dB, sr, det_info, duration):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)

        img = librosa.display.specshow(
            S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=sr/2, ax=self.ax, cmap='magma'
        )
        self.fig.colorbar(img, ax=self.ax, format='%+2.0f dB')
        
        self.ax.set_title(f"Mel-Spectrogram: {det_info['file_name']} (Offset: {det_info['offset']:.1f}s)", fontsize=9)
        self.ax.set_xlabel("Time (s)", fontsize=8)
        self.ax.set_ylabel("Frequency (Hz)", fontsize=8)
        self.fig.tight_layout()

        self.canvas.draw()
        self.lbl_spec_status.config(text=f"Loaded {duration:.0f}s window from {det_info['det_time']}", foreground="green")


if __name__ == "__main__":
    app = GCSBucketNavigator()
    app.mainloop()
