import os
import sys
import traceback
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ann import NeuralNetwork
from build.build_pipeline import build_and_save_model


class StreamRedirector(QObject):
    """Safely redirects stdout/stderr to a Qt Signal."""

    text_written = Signal(str)

    def write(self, text):
        self.text_written.emit(text)

    def flush(self):
        pass


class BuildWorker(QThread):
    """Background thread for training the model without freezing the GUI."""

    success = Signal(str)
    error = Signal(str)

    def __init__(self, csv_path, model_name, dataset_type, redirector):
        super().__init__()
        self.csv_path = csv_path
        self.model_name = model_name
        self.dataset_type = dataset_type
        self.redirector = redirector

    def run(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        sys.stdout = self.redirector
        sys.stderr = self.redirector

        try:
            build_and_save_model(
                csv_path=self.csv_path,
                model_name=self.model_name,
                dataset_type=self.dataset_type,
            )
            print("\nTraining Complete!")
            self.success.emit(self.model_name)
        except Exception as e:
            print(f"\n❌ ERROR during training:\n{traceback.format_exc()}")
            self.error.emit(str(e))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class PredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surface Roughness Predictor & Builder")
        self.resize(650, 700)

        self.model = None
        self.entries = {}
        self.current_folder = ""
        self.csv_path = ""

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.setup_predict_tab()
        self.setup_build_tab()

    def setup_predict_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.btn_folder = QPushButton("Select Models Folder")
        self.btn_folder.clicked.connect(self.select_folder)
        layout.addWidget(self.btn_folder)

        self.lbl_folder_path = QLabel("No folder selected")
        self.lbl_folder_path.setStyleSheet("color: gray;")
        layout.addWidget(self.lbl_folder_path)

        self.model_combobox = QComboBox()
        self.model_combobox.addItem("Select a model...")
        self.model_combobox.currentTextChanged.connect(self.on_model_select)
        layout.addWidget(self.model_combobox)

        self.input_group = QGroupBox("Machine Settings")
        self.form_layout = QFormLayout(self.input_group)
        layout.addWidget(self.input_group)

        self.btn_predict = QPushButton("Predict")
        self.btn_predict.setEnabled(False)
        self.btn_predict.clicked.connect(self.make_prediction)
        layout.addWidget(self.btn_predict)

        self.lbl_result = QLabel("")
        self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_result.setStyleSheet(
            "color: #00aaff; font-size: 18px; font-weight: bold;"
        )
        layout.addWidget(self.lbl_result)

        layout.addStretch()
        self.tabs.addTab(tab, "Predictor")

    def setup_build_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        csv_layout = QHBoxLayout()
        self.lbl_csv = QLabel("No CSV Selected")
        self.btn_csv = QPushButton("Browse CSV")
        self.btn_csv.clicked.connect(self.select_csv)
        csv_layout.addWidget(self.lbl_csv, stretch=1)
        csv_layout.addWidget(self.btn_csv)
        layout.addLayout(csv_layout)

        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Dataset Type:"))
        self.type_combobox = QComboBox()
        self.type_combobox.addItems(["dataset1-type", "dataset2-type"])
        type_layout.addWidget(self.type_combobox, stretch=1)
        layout.addLayout(type_layout)

        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Model Name:"))
        self.ent_model_name = QLineEdit("my_new_model")
        name_layout.addWidget(self.ent_model_name, stretch=1)
        layout.addLayout(name_layout)

        self.btn_build = QPushButton("Start Training")
        self.btn_build.clicked.connect(self.start_build_thread)
        layout.addWidget(self.btn_build)

        layout.addWidget(QLabel("Training Output:"))
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setStyleSheet("background-color: #1e1e1e; color: #4af626;")
        self.console_text.setFont(QFont("Monospace", 10))
        layout.addWidget(self.console_text, stretch=1)

        self.tabs.addTab(tab, "Model Builder")

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Models Folder")
        if not folder_path:
            return

        self.current_folder = folder_path
        self.lbl_folder_path.setText(f"Folder: {self.current_folder}")
        self.lbl_folder_path.setStyleSheet("color: white;")

        self.load_pkl_list(folder_path)
        self.model = None
        self.btn_predict.setEnabled(False)
        self.clear_form_layout()

    def load_pkl_list(self, folder_path):
        current_selection = self.model_combobox.currentText()
        pkl_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]

        self.model_combobox.blockSignals(True)
        self.model_combobox.clear()

        if not pkl_files:
            self.model_combobox.addItem("No models found")
        else:
            self.model_combobox.addItems(pkl_files)
            if current_selection in pkl_files:
                self.model_combobox.setCurrentText(current_selection)
            else:
                self.model_combobox.insertItem(0, "Select a model...")
                self.model_combobox.setCurrentIndex(0)

        self.model_combobox.blockSignals(False)

    def on_model_select(self, selected_file):
        if not selected_file or selected_file in [
            "Select a model...",
            "No models found",
        ]:
            return

        filepath = os.path.join(self.current_folder, selected_file)
        try:
            self.model = NeuralNetwork.load(filepath)
            self.build_input_fields()
            self.btn_predict.setEnabled(True)
            self.lbl_result.setText("")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load the model:\n{e}")

    def clear_form_layout(self):
        while self.form_layout.rowCount() > 0:
            self.form_layout.removeRow(0)
        self.entries.clear()

    def build_input_fields(self):
        if not self.model:
            return

        self.clear_form_layout()

        for feature in self.model.feature_names:
            ent = QLineEdit()
            self.form_layout.addRow(f"{feature}:", ent)
            self.entries[feature] = ent

    def make_prediction(self):
        if not self.model:
            return

        user_inputs = []
        for feature, ent in self.entries.items():
            raw_val = ent.text().strip()
            try:
                user_inputs.append(float(raw_val))
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Invalid Input",
                    f"Please enter a valid number for '{feature}'.",
                )
                return

        try:
            raw_machine_settings = pd.DataFrame(
                [user_inputs], columns=self.model.feature_names
            )
            prediction = self.model.predict(raw_machine_settings)

            self.lbl_result.setText(f"Predicted Ra: {prediction:.4f}")

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"An error occurred:\n{e}")

    # --- Builder Logic ---

    def select_csv(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset", "", "CSV Files (*.csv)"
        )
        if filepath:
            self.csv_path = filepath
            self.lbl_csv.setText(os.path.basename(filepath))

            auto_name = f"{Path(self.csv_path).stem.replace('-', '_')}_model"
            self.ent_model_name.setText(auto_name)

    def append_console_text(self, text):
        self.console_text.moveCursor(QTextCursor.MoveOperation.End)
        self.console_text.insertPlainText(text)
        self.console_text.moveCursor(QTextCursor.MoveOperation.End)

    def start_build_thread(self):
        if not self.csv_path:
            QMessageBox.warning(self, "Missing Data", "Please select a CSV file first.")
            return

        model_name = self.ent_model_name.text().strip()
        if not model_name:
            QMessageBox.warning(
                self, "Missing Name", "Please provide a name for the model."
            )
            return

        dataset_type = self.type_combobox.currentText()

        self.btn_build.setEnabled(False)
        self.btn_build.setText("Training in progress...")
        self.console_text.clear()

        self.redirector = StreamRedirector()
        self.redirector.text_written.connect(self.append_console_text)

        self.worker = BuildWorker(
            self.csv_path, model_name, dataset_type, self.redirector
        )
        self.worker.success.connect(self.on_training_success)
        self.worker.error.connect(self.on_training_error)
        self.worker.finished.connect(self.on_training_complete)

        self.worker.start()

    def on_training_success(self, model_name):
        QMessageBox.information(
            self, "Success", f"Model '{model_name}' trained successfully!"
        )
        if self.current_folder:
            self.load_pkl_list(self.current_folder)

    def on_training_error(self, error_msg):
        QMessageBox.critical(
            self,
            "Training Error",
            f"An error occurred while building the model:\n\n{error_msg}\n\nCheck the console.",
        )

    def on_training_complete(self):
        self.btn_build.setEnabled(True)
        self.btn_build.setText("Start Training")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = PredictorApp()
    window.show()
    sys.exit(app.exec())
