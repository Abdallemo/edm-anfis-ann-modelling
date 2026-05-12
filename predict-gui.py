import os
import sys
import traceback
from pathlib import Path

import joblib
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

from anfis import AnfisNet
from ann import NeuralNetwork
from build import build_and_save_model, load_and_train


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

    def __init__(
        self,
        csv_path: str,
        model_name: str,
        dataset_type: str,
        architecture: str,
        run_on: str,
        build_for: str,
        redirector: StreamRedirector,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.model_name = model_name
        self.dataset_type = dataset_type
        self.architecture = architecture
        self.run_on = run_on
        self.build_for = build_for
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
                architecture=self.architecture,
                run_on=self.run_on,
                build_for=self.build_for,
            )
            print("\nTraining Complete!")
            self.success.emit(self.model_name)
        except Exception as e:
            print(f"\nERROR during training:\n{traceback.format_exc()}")
            self.error.emit(str(e))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class RetrainWorker(QThread):
    """Background thread for retraining an existing model."""

    success = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        model_path: str,
        csv_path: str,
        mode_dataset_type: str,
        redirector: StreamRedirector,
    ):
        super().__init__()
        self.model_path = model_path
        self.csv_path = csv_path
        self.mode_dataset_type = mode_dataset_type
        self.redirector = redirector

    def run(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        sys.stdout = self.redirector
        sys.stderr = self.redirector

        try:
            print(f"Loading model: {Path(self.model_path).name}")
            print(f"Loading new data: {Path(self.csv_path).name}")
            load_and_train(self.model_path, self.csv_path, self.mode_dataset_type)
            print("\nRetraining Complete! Model overwritten successfully.")
            self.success.emit(Path(self.model_path).name)
        except Exception as e:
            print(f"\nERROR during retraining:\n{traceback.format_exc()}")
            self.error.emit(str(e))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class PredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surface Roughness Predictor & Builder")
        self.resize(650, 750)

        self.model = None
        self.entries = {}
        self.current_folder = ""
        self.csv_path = ""

        self.retrain_model_path = ""
        self.retrain_csv_path = ""

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.setup_predict_tab()
        self.setup_build_tab()
        self.setup_retrain_tab()

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

        arch_layout = QHBoxLayout()
        arch_layout.addWidget(QLabel("Architecture:"))

        self.arch_combobox = QComboBox()
        self.arch_combobox.addItems(["ANN", "ANFIS"])
        self.arch_combobox.currentTextChanged.connect(self.sync_model_name)
        self.arch_combobox.currentTextChanged.connect(self.device_signal)

        arch_layout.addWidget(self.arch_combobox, stretch=1)
        layout.addLayout(arch_layout)

        hardware_layout = QHBoxLayout()

        hardware_layout.addWidget(QLabel("Run On:"))
        self.combo_run_on = QComboBox()
        self.combo_run_on.addItems(["CPU", "GPU"])
        self.combo_run_on.currentTextChanged.connect(self.sync_model_name)
        hardware_layout.addWidget(self.combo_run_on, stretch=1)

        hardware_layout.addWidget(QLabel("Build For:"))
        self.combo_build_for = QComboBox()
        self.combo_build_for.addItems(["CPU", "GPU", "Both"])
        hardware_layout.addWidget(self.combo_build_for, stretch=1)

        layout.addLayout(hardware_layout)

        self.device_signal()

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

        self.btn_build = QPushButton("Start Grid Search & Build")
        self.btn_build.clicked.connect(self.start_build_thread)
        layout.addWidget(self.btn_build)

        layout.addWidget(QLabel("Training Output:"))
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setStyleSheet("background-color: #1e1e1e; color: #4af626;")
        self.console_text.setFont(QFont("Monospace", 10))
        layout.addWidget(self.console_text, stretch=1)

        self.tabs.addTab(tab, "Model Builder")

    def setup_retrain_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        model_layout = QHBoxLayout()
        self.lbl_retrain_model = QLabel("No Model Selected")
        self.btn_retrain_model = QPushButton("Select Existing Model")
        self.btn_retrain_model.clicked.connect(self.select_retrain_model)
        model_layout.addWidget(self.lbl_retrain_model, stretch=1)
        model_layout.addWidget(self.btn_retrain_model)
        layout.addLayout(model_layout)

        csv_layout = QHBoxLayout()
        self.lbl_retrain_csv = QLabel("No New CSV Selected")
        self.btn_retrain_csv = QPushButton("Select New CSV Data")
        self.btn_retrain_csv.clicked.connect(self.select_retrain_csv)
        csv_layout.addWidget(self.lbl_retrain_csv, stretch=1)
        csv_layout.addWidget(self.btn_retrain_csv)
        layout.addLayout(csv_layout)

        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Dataset Type:"))
        self.model_type_combobox = QComboBox()
        self.model_type_combobox.addItems(["dataset1-type", "dataset2-type"])
        type_layout.addWidget(self.model_type_combobox, stretch=1)
        layout.addLayout(type_layout)

        self.btn_retrain = QPushButton("Retrain and Overwrite")
        self.btn_retrain.clicked.connect(self.start_retrain_thread)
        layout.addWidget(self.btn_retrain)

        layout.addWidget(QLabel("Retraining Output:"))
        self.retrain_console_text = QTextEdit()
        self.retrain_console_text.setReadOnly(True)
        self.retrain_console_text.setStyleSheet(
            "background-color: #1e1e1e; color: #ffaa00;"
        )
        self.retrain_console_text.setFont(QFont("Monospace", 10))
        layout.addWidget(self.retrain_console_text, stretch=1)

        self.tabs.addTab(tab, "Retrainer")

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
        pkl_files = [f for f in os.listdir(folder_path) if f.endswith((".pkl", ".pt"))]

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
            raw_state = joblib.load(filepath)
            engine = raw_state.get("model")
            engine_name = type(engine).__name__

            if engine_name == "SugenoFuzzyCore":
                self.model = AnfisNet.load(filepath)
            elif engine_name == "MLPRegressor":
                self.model = NeuralNetwork.load(filepath)
            else:
                raise ValueError(f"Unrecognized internal engine: {engine_name}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load the model:\n{e}")
            return

        self.build_input_fields()
        self.btn_predict.setEnabled(True)
        self.lbl_result.setText("")

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

            if abs(prediction) < 1e-5:
                self.lbl_result.setText("Predicted Ra: Out of Bounds")
            else:
                self.lbl_result.setText(f"Predicted Ra: {prediction:.4f}")

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"An error occurred:\n{e}")

    def select_csv(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset", "", "CSV Files (*.csv)"
        )
        if filepath:
            self.csv_path = filepath
            self.lbl_csv.setText(os.path.basename(filepath))

            auto_name = f"{Path(self.csv_path).stem.replace('-', '_')}_{self.arch_combobox.currentText()}_model"
            self.ent_model_name.setText(auto_name)

    def sync_model_name(self, _text: str):
        if not self.csv_path:
            return

        auto_name = (
            f"{Path(self.csv_path).stem.replace('-', '_')}_"
            f"{self.arch_combobox.currentText()}_model"
        )
        self.ent_model_name.setText(auto_name)
        self.device_signal()

    def device_signal(self, *_):
        is_anfis = self.arch_combobox.currentText() == "ANFIS"
        self.combo_run_on.setEnabled(is_anfis)

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
        architecture = self.arch_combobox.currentText()
        run_on = self.combo_run_on.currentText().lower()
        build_for = self.combo_build_for.currentText().lower()

        self.btn_build.setEnabled(False)
        self.btn_build.setText("Training in progress...")
        self.console_text.clear()

        self.redirector = StreamRedirector()
        self.redirector.text_written.connect(self.append_console_text)

        self.worker = BuildWorker(
            self.csv_path,
            model_name,
            dataset_type,
            architecture,
            run_on,
            build_for,
            self.redirector,
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
        self.btn_build.setText("Start Grid Search & Build")

    def select_retrain_model(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Existing Model", "", "Model Files (*.pkl *.pt)"
        )
        if filepath:
            self.retrain_model_path = filepath
            self.lbl_retrain_model.setText(os.path.basename(filepath))

    def select_retrain_csv(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select New Dataset", "", "CSV Files (*.csv)"
        )
        if filepath:
            self.retrain_csv_path = filepath
            self.lbl_retrain_csv.setText(os.path.basename(filepath))

    def append_retrain_console_text(self, text):
        self.retrain_console_text.moveCursor(QTextCursor.MoveOperation.End)
        self.retrain_console_text.insertPlainText(text)
        self.retrain_console_text.moveCursor(QTextCursor.MoveOperation.End)

    def start_retrain_thread(self):
        if not self.retrain_model_path or not self.retrain_csv_path:
            QMessageBox.warning(self, "Missing Files", "Select both a model and a CSV.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Overwrite",
            "This will retrain the model and permanently overwrite the original file. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.No:
            return

        self.btn_retrain.setEnabled(False)
        self.btn_retrain.setText("Retraining...")
        self.retrain_console_text.clear()

        self.redirector = StreamRedirector()
        self.redirector.text_written.connect(self.append_retrain_console_text)

        self.retrain_worker = RetrainWorker(
            self.retrain_model_path,
            self.retrain_csv_path,
            self.model_type_combobox.currentText(),
            self.redirector,
        )

        self.retrain_worker.success.connect(self.on_retrain_success)
        self.retrain_worker.error.connect(self.on_retrain_error)
        self.retrain_worker.finished.connect(self.on_retrain_complete)

        self.retrain_worker.start()

    def on_retrain_success(self, model_name):
        QMessageBox.information(
            self, "Success", f"Model '{model_name}' retrained and overwritten!"
        )

    def on_retrain_error(self, error_msg):
        QMessageBox.critical(
            self,
            "Retrain Error",
            f"An error occurred while retraining:\n\n{error_msg}\n\nCheck the console.",
        )

    def on_retrain_complete(self):
        self.btn_retrain.setEnabled(True)
        self.btn_retrain.setText("Retrain and Overwrite")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = PredictorApp()
    window.show()
    sys.exit(app.exec())
