import os
import queue
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, font, messagebox, scrolledtext, ttk

import pandas as pd

from ann import NeuralNetwork
from build.build_pipeline import build_and_save_model


class TextRedirector:
    """Safely drops stdout/stderr text into a thread-safe bucket (Queue)."""

    def __init__(self, text_queue: queue.Queue):
        self.text_queue = text_queue

    def write(self, text):

        self.text_queue.put(text)

    def flush(self):
        pass

    def isatty(self):
        return False


class PredictorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Surface Roughness")
        self.root.geometry("600x650")

        self.model = None
        self.entries = {}
        self.current_folder = ""

        self.console_queue = queue.Queue()
        self.poll_console_queue()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.predict_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_frame, text="Predictor")
        self.setup_predict_tab()

        self.build_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.build_frame, text="Model Builder")
        self.setup_build_tab()

    def poll_console_queue(self):
        processed = False
        max_per_tick = 10

        for _ in range(max_per_tick):
            if self.console_queue.empty():
                break

            processed = True
            text = self.console_queue.get()

            if text == "<<TRAINING_COMPLETE>>":
                self.btn_build.config(state="normal", text="Start Training")
                if self.current_folder:
                    self.load_pkl_list(self.current_folder)

            elif text == "<<TRAINING_SUCCESS>>":
                self.show_success_popup()

            elif text == "<<TRAINING_ERROR>>":
                self.show_error_popup()

            else:
                self.console_text.insert(tk.END, text)

        if processed:
            self.console_text.see(tk.END)

        delay = 20 if processed else 150
        self.root.after(delay, self.poll_console_queue)

    def setup_predict_tab(self):
        self.btn_folder = ttk.Button(
            self.predict_frame, text="Select Models Folder", command=self.select_folder
        )
        self.btn_folder.pack(pady=(15, 5))

        self.lbl_folder_path = ttk.Label(
            self.predict_frame, text="No folder selected", foreground="gray"
        )
        self.lbl_folder_path.pack(pady=2)

        self.model_combobox = ttk.Combobox(
            self.predict_frame, state="readonly", width=40
        )
        self.model_combobox.pack(pady=10)
        self.model_combobox.set("Select a model...")
        self.model_combobox.bind("<<ComboboxSelected>>", self.on_model_select)

        self.input_frame = ttk.LabelFrame(self.predict_frame, text="Machine Settings")
        self.input_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.btn_predict = ttk.Button(
            self.predict_frame,
            text="Predict",
            command=self.make_prediction,
            state="disabled",
        )
        self.btn_predict.pack(pady=15)

        self.lbl_result = ttk.Label(
            self.predict_frame,
            text="",
            font=("Helvetica", 14, "bold"),
            foreground="blue",
        )
        self.lbl_result.pack(pady=10)

    def select_folder(self):
        folder_path = filedialog.askdirectory(title="Select Models Folder")
        if not folder_path:
            return

        self.current_folder = folder_path
        self.lbl_folder_path.config(
            text=f"Folder: {self.current_folder}", foreground="black"
        )

        self.load_pkl_list(folder_path)

        self.model = None
        self.btn_predict.config(state="disabled")
        for widget in self.input_frame.winfo_children():
            widget.destroy()

    def load_pkl_list(self, folder_path: str):

        current_selection = self.model_combobox.get()

        pkl_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]

        if not pkl_files:
            self.model_combobox["values"] = []
            self.model_combobox.set("No models found")
            return

        self.model_combobox["values"] = pkl_files

        if current_selection in pkl_files:
            self.model_combobox.set(current_selection)
        else:
            self.model_combobox.set("Select a model...")

    def on_model_select(self, event):
        selected_file = self.model_combobox.get()
        if not selected_file or not self.current_folder:
            return

        filepath = os.path.join(self.current_folder, selected_file)

        try:
            self.model = NeuralNetwork.load(filepath)
            self.build_input_fields()
            self.btn_predict.config(state="normal")
            self.lbl_result.config(text="")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load the model:\n{e}")

    def build_input_fields(self):
        if not self.model:
            return
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        self.entries.clear()

        for idx, feature in enumerate(self.model.feature_names):
            lbl = ttk.Label(self.input_frame, text=f"{feature}:")
            lbl.grid(row=idx, column=0, padx=10, pady=10, sticky="e")

            ent = ttk.Entry(self.input_frame, width=20)
            ent.grid(row=idx, column=1, padx=10, pady=10, sticky="w")
            self.entries[feature] = ent

    def make_prediction(self):
        if not self.model:
            return

        user_inputs = []
        for feature, ent in self.entries.items():
            raw_val = ent.get().strip()
            try:
                user_inputs.append(float(raw_val))
            except ValueError:
                messagebox.showwarning(
                    "Invalid Input", f"Please enter a valid number for '{feature}'."
                )
                return

        try:
            raw_machine_settings = pd.DataFrame(
                [user_inputs], columns=self.model.feature_names
            )
            prediction = self.model.predict(raw_machine_settings)

            if isinstance(prediction, (list, tuple, pd.Series)) or (
                hasattr(prediction, "shape") and prediction > 0
            ):
                result = prediction
            else:
                result = prediction

            self.lbl_result.config(text=f"Predicted Ra: {result:.4f}")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred:\n{e}")

    def setup_build_tab(self):
        csv_frame = ttk.Frame(self.build_frame)
        csv_frame.pack(fill="x", padx=20, pady=(20, 5))

        self.lbl_csv = ttk.Label(
            csv_frame, text="No CSV Selected", width=40, anchor="w"
        )
        self.lbl_csv.pack(side="left")

        self.btn_csv = ttk.Button(csv_frame, text="Browse CSV", command=self.select_csv)
        self.btn_csv.pack(side="right")
        self.csv_path = ""

        type_frame = ttk.Frame(self.build_frame)
        type_frame.pack(fill="x", padx=20, pady=5)

        ttk.Label(type_frame, text="Dataset Type:").pack(side="left")
        self.type_combobox = ttk.Combobox(
            type_frame, state="readonly", values=["dataset1-type", "dataset2-type"]
        )
        self.type_combobox.pack(side="right", fill="x", expand=True, padx=(10, 0))
        self.type_combobox.set("dataset1-type")

        name_frame = ttk.Frame(self.build_frame)
        name_frame.pack(fill="x", padx=20, pady=5)

        ttk.Label(name_frame, text="Model Name:").pack(side="left")
        self.ent_model_name = ttk.Entry(name_frame)
        self.ent_model_name.pack(side="right", fill="x", expand=True, padx=(10, 0))
        self.ent_model_name.insert(0, "my_new_model")

        self.btn_build = ttk.Button(
            self.build_frame, text="Start Training", command=self.start_build_thread
        )
        self.btn_build.pack(pady=15)

        ttk.Label(self.build_frame, text="Training Output:").pack(anchor="w", padx=20)
        self.console_text = scrolledtext.ScrolledText(
            self.build_frame,
            height=15,
            bg="black",
            fg="lightgreen",
            font=("Liberation Mono", 11),
        )
        self.console_text.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    def select_csv(self):
        filepath = filedialog.askopenfilename(
            title="Select Dataset", filetypes=[("CSV Files", "*.csv")]
        )
        if filepath:
            self.csv_path = filepath
            filename = os.path.basename(filepath)
            self.lbl_csv.config(text=filename)
            self.ent_model_name.delete(0, tk.END)

            self.ent_model_name.insert(
                0, f"{Path(self.csv_path).stem.replace('-', '_')}_model"
            )

    def start_build_thread(self):
        if not self.csv_path:
            messagebox.showwarning("Missing Data", "Please select a CSV file first.")
            return

        model_name = self.ent_model_name.get().strip()
        if not model_name:
            messagebox.showwarning(
                "Missing Name", "Please provide a name for the model."
            )
            return

        dataset_type = self.type_combobox.get()

        self.btn_build.config(state="disabled", text="Training in progress...")
        self.console_text.delete(1.0, tk.END)

        thread = threading.Thread(
            target=self.run_build_process,
            args=(self.csv_path, model_name, dataset_type),
            daemon=True,
        )
        thread.start()

    def run_build_process(self, csv_path, model_name, dataset_type):
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        redirector = TextRedirector(self.console_queue)
        sys.stdout = redirector
        sys.stderr = redirector

        try:
            build_and_save_model(
                csv_path=csv_path, model_name=model_name, dataset_type=dataset_type
            )
            print("\nTraining Complete!")

            self.latest_model_name = model_name
            self.console_queue.put("<<TRAINING_SUCCESS>>")

        except Exception as e:
            print(f"\n❌ ERROR during training:\n{traceback.format_exc()}")

            self.latest_error = str(e)
            self.console_queue.put("<<TRAINING_ERROR>>")

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.console_queue.put("<<TRAINING_COMPLETE>>")

    def show_success_popup(self, event=None):
        """Triggered by <<TrainingSuccess>>"""
        messagebox.showinfo(
            "Success",
            f"Model '{self.latest_model_name}' trained successfully!",
            parent=self.root,
        )

    def show_error_popup(self, event=None):
        """Triggered by <<TrainingError>>"""
        messagebox.showerror(
            "Training Error",
            f"An error occurred while building the model:\n\n{self.latest_error}\n\nCheck the console output for full details.",
            parent=self.root,
        )


if __name__ == "__main__":
    root = tk.Tk()
    root.tk.call("tk", "scaling", 1.4)
    default_font = font.nametofont("TkDefaultFont")
    default_font.configure(size=11)

    text_font = font.nametofont("TkTextFont")
    text_font.configure(size=11)

    fixed_font = font.nametofont("TkFixedFont")
    fixed_font.configure(size=11)

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
        style.configure(".", font=("DejaVu Sans", 11))

    app = PredictorApp(root)
    root.mainloop()
