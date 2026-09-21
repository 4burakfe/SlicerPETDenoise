import os
import ast
import gc
import math
import time
import logging
import importlib
import traceback
import configparser

import numpy as np
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *

MODULE_VERSION = "v1.1"
ARCHITECTURES = ("UNET", "SwinUNETR", "SwinUNETR+GCFN")
WINDOW_OVERLAP = 0.25
MIN_VRAM_GB = 1.9
EINOPS_REQUIREMENT = "einops==0.6.1"
SETTINGS_MODEL_FOLDER = "PETDenoise/ModelFolder"
MODEL_ATTRIBUTE = "PETDenoise.Model"
DATE_ATTRIBUTE = "PETDenoise.Date"

# Panel defaults. Applied before every sidecar (.txt), so a key missing from one model's sidecar never
# silently inherits the value left over from the previously selected model.
DEFAULT_PARAMETERS = {
    "dual_channel": False,
    "architecture": "UNET",
    "strides": (2, 2, 2, 2),
    "channels": (128, 256, 512, 1024, 2048),
    "res_units": 2,
    "down_kernel": 3,
    "up_kernel": 3,
    "num_heads": (3, 6, 12, 24),
    "depths": (2, 2, 2, 2),
    "feature_size": 24,
    "do_rate": 0.0,
    "voxel_spacing": (2.0, 2.0, 2.0),
    "block_size": (64, 64, 64),
    "prevent_negative": True,
    "dont_resample": False,
}


# ---------------------------------------------------------------------------
# Helpers (no Slicer / torch needed)
# ---------------------------------------------------------------------------

def parseSequence(text, name, length=None, cast=int):
    """'(2,2,2)' / '[2,2,2]' -> tuple of positive numbers. Uses literal_eval, never eval."""
    try:
        value = ast.literal_eval(str(text).strip())
    except (ValueError, SyntaxError):
        raise ValueError(f"{name}: '{text}' is not a list of numbers, e.g. (2,2,2).")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = (value,)
    try:
        values = tuple(value)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise TypeError
    except TypeError:
        raise ValueError(f"{name}: '{text}' is not a list of numbers.")
    if cast is int and any(not float(v).is_integer() for v in values):
        raise ValueError(f"{name}: whole numbers expected, got '{text}'.")
    values = tuple(cast(v) for v in values)
    if length is not None and len(values) != length:
        raise ValueError(f"{name}: {length} values expected, got {len(values)}.")
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"{name}: all values must be greater than 0.")
    return values


def parseSidecar(text):
    """Model parameters from a <model>.txt sidecar, on top of DEFAULT_PARAMETERS (same keys as before)."""
    params = dict(DEFAULT_PARAMETERS)
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip().lstrip("\ufeff").lower(), value.strip()
        try:
            if key in ("dual_channel", "dont_resample", "prevent_negative"):
                params[key] = value.lower() == "true"
            elif key == "voxel_spacing":
                params[key] = parseSequence(value, key, 3, float)
                params["dont_resample"] = False  # as before: a spacing line switches resampling on
            elif key == "block_size":
                params[key] = parseSequence(value, key, 3, int)
            elif key in ("strides", "channels", "num_heads", "depths"):
                params[key] = parseSequence(value, key)
            elif key in ("res_units", "down_kernel", "up_kernel", "feature_size"):
                params[key] = int(value)
            elif key == "do_rate":
                params[key] = float(value)
            elif key == "architecture":
                # Kept for compatibility: anything other than UNET / SwinUNETR means SwinUNETR+GCFN
                params[key] = value if value in ("UNET", "SwinUNETR") else "SwinUNETR+GCFN"
        except (ValueError, TypeError) as error:
            logging.warning(f"PETDenoise: ignored sidecar line '{line.strip()}': {error}")
    return params


def validateParameters(params):
    """Raises ValueError with a readable message if the network cannot be built / run with these settings."""
    block = params["block_size"]
    if params["architecture"] == "UNET":
        if len(params["channels"]) != len(params["strides"]) + 1:
            raise ValueError(f"UNET: number of channels ({len(params['channels'])}) must be number of strides "
                             f"({len(params['strides'])}) + 1.")
        factor = math.prod(params["strides"])
        if any(b % factor for b in block):
            raise ValueError(f"UNET: block size {block} must be divisible by {factor} (product of the strides).")
    else:
        if len(params["num_heads"]) != len(params["depths"]):
            raise ValueError("SwinUNETR: number of heads and depths must have the same length.")
        if params["feature_size"] % 12:
            raise ValueError("SwinUNETR: feature size must be divisible by 12.")
        if any(b % 32 for b in block):
            raise ValueError(f"SwinUNETR: block size {block} must be divisible by 32.")
    if not 0.0 <= params["do_rate"] < 1.0:
        raise ValueError("Dropout path rate must be between 0 and 1.")


def estimateSlidingWindowCount(imageShape, roiSize, overlap=WINDOW_OVERLAP):
    """Number of windows MONAI's sliding_window_inference evaluates (same arithmetic as MONAI). For progress."""
    total = 1
    for size, roi in zip(imageShape, roiSize):
        roi = int(roi)
        size = max(int(size), roi)
        interval = roi if roi == size else max(int(roi * (1.0 - overlap)), 1)
        count = int(math.ceil(float(size) / interval))
        first = next((d for d in range(count) if d * interval + roi >= size), None)
        total *= first + 1 if first is not None else 1
    return total


def castResult(array, dtype):
    """Back to the input voxel type. Integer types are rounded (not truncated) and clipped to their range."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        return np.clip(np.rint(array), limits.min, limits.max).astype(dtype)
    return np.asarray(array).astype(dtype, copy=False)


def formatSequence(values):
    return "(" + ",".join(f"{v:g}" if isinstance(v, float) else str(v) for v in values) + ")"


def isOutOfMemory(error):
    text = str(error).lower()
    return "out of memory" in text or "can't allocate memory" in text or "not enough memory" in text


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class PETDenoise(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Belenos - PET Denoise"
        parent.categories = ["Nuclear Medicine", "Filtering.Denoising"]
        parent.dependencies = ["PyTorchUtils"]
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = "This module provides automated denoising of PET images with UNET / SwinUNETR neural networks."
        parent.acknowledgementText = "This file was developed by Burak Demir."
        iconPath = os.path.join(os.path.dirname(__file__), "Resources", "logo.png")
        if os.path.exists(iconPath):
            parent.icon = qt.QIcon(iconPath)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class PETDenoiseWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = PETDenoiseLogic()
        self._running = False



        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)
        formLayout = qt.QFormLayout(parametersCollapsibleButton)

        self.architecture = qt.QComboBox()
        self.architecture.addItems(list(ARCHITECTURES))
        formLayout.addRow("Select Model Architecture: ", self.architecture)

        self.inputVolumeSelector = self._volumeSelector("Select the PET image for denoising.")
        formLayout.addRow("Input Volume 1 (PET): ", self.inputVolumeSelector)
        self.inputVolumeSelectorCT = self._volumeSelector("Select the CT image (second channel of dual-input models).")
        formLayout.addRow("Input Volume 2 (CT): ", self.inputVolumeSelectorCT)

        self.dualch_cbox = qt.QCheckBox()
        formLayout.addRow("Dual Volume Input (PET+CT): ", self.dualch_cbox)
        self.neg_val_cbox = qt.QCheckBox()
        self.neg_val_cbox.setChecked(True)
        formLayout.addRow("Prevent Negative Values: ", self.neg_val_cbox)

        self.modelFolderPathEdit = qt.QLineEdit()
        self.modelFolderPathEdit.readOnly = True
        formLayout.addRow("Model Folder:", self.modelFolderPathEdit)
        self.selectFolderButton = qt.QPushButton("Select Model Folder")
        formLayout.addRow(self.selectFolderButton)

        self.modelselector = qt.QComboBox()
        formLayout.addRow("Select Model: ", self.modelselector)
        self.modelInfoBox = qt.QTextEdit()
        self.modelInfoBox.setReadOnly(True)
        self.modelInfoBox.setToolTip("Displays model info from the .txt sidecar file.")
        formLayout.addRow("Model Info:", self.modelInfoBox)

        self.dont_resample_cbox = qt.QCheckBox()
        formLayout.addRow("Do not Resample:", self.dont_resample_cbox)
        self.resample_voxel_size = qt.QLineEdit("(2,2,2)")
        formLayout.addRow("Voxel Spacing for Resample:", self.resample_voxel_size)
        self.denoise_block_size = qt.QLineEdit("(64,64,64)")
        formLayout.addRow("Block Size for denoising:", self.denoise_block_size)

        formLayout.addRow(qt.QLabel("Settings For UNET:"))
        self.strideComboBox = qt.QLineEdit("(2,2,2,2)")
        formLayout.addRow("Strides:", self.strideComboBox)
        self.channels = qt.QLineEdit("(128,256,512,1024,2048)")
        formLayout.addRow("Channels:", self.channels)
        self.resUnitSpinBox = self._spinBox(1, 10, 2)
        formLayout.addRow("Residual Units:", self.resUnitSpinBox)
        self.downkernelSpinBox = self._spinBox(1, 11, 3)
        formLayout.addRow("Down Kernel:", self.downkernelSpinBox)
        self.upkernelSpinBox = self._spinBox(1, 11, 3)
        formLayout.addRow("Up Kernel:", self.upkernelSpinBox)

        formLayout.addRow(qt.QLabel("Settings For SwinUNETR:"))
        self.heads = qt.QLineEdit("(3,6,12,24)")
        formLayout.addRow("Number of Heads:", self.heads)
        self.depths = qt.QLineEdit("(2,2,2,2)")
        formLayout.addRow("Depths:", self.depths)
        self.swinfeaturesize = self._spinBox(12, 252, 24)
        self.swinfeaturesize.setSingleStep(12)
        formLayout.addRow("Feature Size:", self.swinfeaturesize)
        self.dropoutrate = qt.QLineEdit("0.0")
        formLayout.addRow("Dropout Path Rate:", self.dropoutrate)

        self.outputTextBox = qt.QTextEdit()
        self.outputTextBox.setReadOnly(True)
        self.outputTextBox.setToolTip("Displays processing logs and results.")
        formLayout.addRow("Processing Log:", self.outputTextBox)

        self.forceCPUcbox = qt.QCheckBox()
        formLayout.addRow("FORCE CPU: ", self.forceCPUcbox)

        self.calculateButton = qt.QPushButton("Denoise")
        self.calculateButton.toolTip = "Start the denoising process"
        formLayout.addRow(self.calculateButton)

        bannerPath = os.path.join(os.path.dirname(__file__), "Resources", "banner.png")
        if os.path.exists(bannerPath):
            bannerLabel = qt.QLabel()
            bannerLabel.setPixmap(qt.QPixmap(bannerPath).scaledToWidth(600, qt.Qt.SmoothTransformation))
            bannerLabel.setAlignment(qt.Qt.AlignCenter)
            self.layout.addWidget(bannerLabel)


        infoTextBox = qt.QTextEdit()
        infoTextBox.setReadOnly(True)
        infoTextBox.setPlainText(
            "This module provides automatic denoising with ML models on medical images.\n"
            "Select the source volume to denoise.\n"
            "This module is NOT a medical device. Research use only.\n"
            "Developed by: Burak Demir, MD, FEBNM\n"
            "For support, questions and feedback: 4burakfe@gmail.com\n"
            "Demir, B., Atalay, M., Yurtcu, H. et al. Denoising of PET with SwinUNETR neural networks: impact of "
            "tumor oriented loss function, denoising module for 3D slicer. Ann Nucl Med (2026). "
            "https://doi.org/10.1007/s12149-026-02166-4\n"
            f"Version: {MODULE_VERSION}"
        )
        self.layout.addWidget(infoTextBox)

        self._unetWidgets = [self.strideComboBox, self.channels, self.resUnitSpinBox,
                             self.downkernelSpinBox, self.upkernelSpinBox]
        self._swinWidgets = [self.heads, self.depths, self.swinfeaturesize, self.dropoutrate]

        self.loadModelFolderPath()
        self.selectFolderButton.connect("clicked(bool)", self.selectModelFolder)
        self.calculateButton.connect("clicked(bool)", self.onCalculateButtonClicked)
        self.modelselector.connect("currentIndexChanged(int)", self.updateModelInfoBox)
        self.architecture.connect("currentIndexChanged(int)", self.updateEnabledState)
        self.dualch_cbox.connect("toggled(bool)", self.updateEnabledState)
        self.dont_resample_cbox.connect("toggled(bool)", self.updateEnabledState)
        self.updateModelInfoBox()

    @staticmethod
    def _volumeSelector(toolTip):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        selector.selectNodeUponCreation = True
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.noneEnabled = False
        selector.showHidden = False
        selector.showChildNodeTypes = False
        selector.setMRMLScene(slicer.mrmlScene)
        selector.setToolTip(toolTip)
        return selector

    @staticmethod
    def _spinBox(minimum, maximum, value):
        spinBox = qt.QSpinBox()
        spinBox.setRange(minimum, maximum)
        spinBox.setValue(value)
        return spinBox

    def log(self, text):
        self.outputTextBox.append(text)
        slicer.app.processEvents()  # show progress while the run is going

    def updateEnabledState(self, *args):
        isUnet = self.architecture.currentText == "UNET"
        for widget in self._unetWidgets:
            widget.enabled = isUnet
        for widget in self._swinWidgets:
            widget.enabled = not isUnet
        self.inputVolumeSelectorCT.enabled = self.dualch_cbox.checked
        self.resample_voxel_size.enabled = not self.dont_resample_cbox.checked

    # --- Parameters <-> UI -------------------------------------------------

    def setParametersToUi(self, params):
        self.architecture.setCurrentIndex(ARCHITECTURES.index(params["architecture"]))
        self.dualch_cbox.checked = params["dual_channel"]
        self.dont_resample_cbox.checked = params["dont_resample"]
        self.neg_val_cbox.checked = params["prevent_negative"]
        self.resample_voxel_size.text = formatSequence(params["voxel_spacing"])
        self.denoise_block_size.text = formatSequence(params["block_size"])
        self.strideComboBox.text = formatSequence(params["strides"])
        self.channels.text = formatSequence(params["channels"])
        self.resUnitSpinBox.value = params["res_units"]
        self.downkernelSpinBox.value = params["down_kernel"]
        self.upkernelSpinBox.value = params["up_kernel"]
        self.heads.text = formatSequence(params["num_heads"])
        self.depths.text = formatSequence(params["depths"])
        self.swinfeaturesize.value = params["feature_size"]
        self.dropoutrate.text = f"{params['do_rate']:g}"
        self.updateEnabledState()

    def readParametersFromUi(self):
        """Current panel values as a params dict. Raises ValueError with a readable message."""
        params = {
            "architecture": self.architecture.currentText,
            "dual_channel": self.dualch_cbox.checked,
            "dont_resample": self.dont_resample_cbox.checked,
            "prevent_negative": self.neg_val_cbox.checked,
            "block_size": parseSequence(self.denoise_block_size.text, "Block size", 3, int),
            "voxel_spacing": parseSequence(self.resample_voxel_size.text, "Voxel spacing", 3, float),
            "strides": parseSequence(self.strideComboBox.text, "Strides"),
            "channels": parseSequence(self.channels.text, "Channels"),
            "res_units": self.resUnitSpinBox.value,
            "down_kernel": self.downkernelSpinBox.value,
            "up_kernel": self.upkernelSpinBox.value,
            "num_heads": parseSequence(self.heads.text, "Number of heads"),
            "depths": parseSequence(self.depths.text, "Depths"),
            "feature_size": self.swinfeaturesize.value,
        }
        try:
            params["do_rate"] = float(self.dropoutrate.text)
        except ValueError:
            raise ValueError(f"Dropout path rate: '{self.dropoutrate.text}' is not a number.")
        validateParameters(params)
        return params

    def updateModelInfoBox(self, *args):
        modelName = self.modelselector.currentText
        if not modelName:
            self.modelInfoBox.setPlainText("Select a model folder that contains .pth files.")
            return
        sidecarPath = os.path.join(self.modelFolderPathEdit.text, os.path.splitext(modelName)[0] + ".txt")
        text = None
        if os.path.isfile(sidecarPath):
            try:
                with open(sidecarPath, "r", encoding="utf-8-sig", errors="replace") as f:
                    text = f.read()
            except OSError as error:
                logging.warning(f"PETDenoise: could not read {sidecarPath}: {error}")
        # Always start from defaults: no settings leak over from the previously selected model
        self.setParametersToUi(parseSidecar(text))
        if text is not None:
            self.modelInfoBox.setPlainText(text)
        else:
            self.modelInfoBox.setPlainText("ℹ️ No description file (.txt) found. Default parameters were loaded; "
                                           "check that they match how this model was trained.")

    # --- Run ---------------------------------------------------------------

    def checkInputs(self):
        """Error message, or None if the inputs are usable."""
        petNode = self.inputVolumeSelector.currentNode()
        if petNode is None or petNode.GetImageData() is None:
            return "Please select a valid input volume."
        modelName = self.modelselector.currentText
        if not modelName or not os.path.isfile(os.path.join(self.modelFolderPathEdit.text, modelName)):
            return "Please select a model folder and a model (.pth)."
        if self.dualch_cbox.checked:
            ctNode = self.inputVolumeSelectorCT.currentNode()
            if ctNode is None or ctNode.GetImageData() is None:
                return "Dual channel selected, but no CT volume provided."
            if ctNode is petNode:
                return "Dual channel: PET and CT must be two different volumes."
            if ctNode.GetTransformNodeID() != petNode.GetTransformNodeID():
                return ("Dual channel: PET and CT are under different transforms. "
                        "Harden the registration transform (Data module) first.")
        return None

    def onCalculateButtonClicked(self):
        if self._running:
            return
        self.outputTextBox.clear()
        error = self.checkInputs()
        if error:
            self.log(f"❌ {error}")
            return
        try:
            params = self.readParametersFromUi()
        except ValueError as error:
            self.log(f"❌ Invalid parameter. {error}")
            return
        if not self.ensureDependencies():
            self.log("❌ Required Python packages are missing. Stopping.")
            return

        petNode = self.inputVolumeSelector.currentNode()
        ctNode = self.inputVolumeSelectorCT.currentNode() if params["dual_channel"] else None
        modelName = self.modelselector.currentText
        modelPath = os.path.join(self.modelFolderPathEdit.text, modelName)
        outputName = self.logic.uniqueVolumeName(f"{petNode.GetName()}_DN_{os.path.splitext(modelName)[0]}")

        self._running = True
        for widget in (self.calculateButton, self.selectFolderButton, self.modelselector):
            widget.enabled = False
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self.log("🚀 Starting AI denoising process...")
            outputNode = self.logic.run(petNode, ctNode, modelPath, params, outputName,
                                        forceCPU=self.forceCPUcbox.checked, log=self.log)
            self.log(f"✅ Output volume: {outputNode.GetName()} (input volume is unchanged)")
        except Exception as error:
            logging.exception("PETDenoise: denoising failed")
            if isOutOfMemory(error):
                self.log("❌ Out of memory. Try FORCE CPU, a smaller block size, or crop the volume first.")
            else:
                self.log(f"❌ Denoising failed: {error}")
            self.log("Nothing was added to the scene.")
            slicer.util.errorDisplay(f"Denoising failed:\n{error}", detailedText=traceback.format_exc())
        finally:
            qt.QApplication.restoreOverrideCursor()
            for widget in (self.calculateButton, self.selectFolderButton, self.modelselector):
                widget.enabled = True
            self._running = False

    @staticmethod
    def ensureDependencies():
        """PyTorch must come from PyTorch Utils (user picks the CUDA build); offers to install MONAI and einops."""
        try:
            import torch  # noqa: F401
        except ImportError:
            # PyTorch is not installed from here: in the PyTorch Utils module the user can pick the
            # build (CUDA version / CPU) that matches their GPU and driver.
            if slicer.util.confirmOkCancelDisplay(
                    "This module and its models need PyTorch, which is not installed.\n\n"
                    "Install it in the PyTorch Utils module, where you can choose the CUDA version that matches "
                    "your GPU (or CPU only), then restart Slicer and try again.\n\n"
                    "Open PyTorch Utils now?"):
                try:
                    slicer.util.selectModule("PyTorchUtils")
                except Exception:
                    slicer.util.errorDisplay("The PyTorch Utils module was not found. Install the 'PyTorch' "
                                             "extension from the Extensions Manager and restart Slicer.")
            return False
        for moduleName, requirement in (("monai", "monai"), ("einops", EINOPS_REQUIREMENT)):
            try:
                importlib.import_module(moduleName)
                continue
            except ImportError:
                pass
            if not slicer.util.confirmOkCancelDisplay(
                    f"The Python package '{moduleName}' is required but not installed.\n"
                    f"Install it now ({requirement})? You may need to restart Slicer afterwards."):
                return False
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            try:
                slicer.util.pip_install(requirement)
            except Exception as error:
                slicer.util.errorDisplay(f"Could not install {requirement}:\n{error}")
                return False
            finally:
                qt.QApplication.restoreOverrideCursor()
            importlib.invalidate_caches()
            try:
                importlib.import_module(moduleName)
            except ImportError:
                slicer.util.errorDisplay(f"{moduleName} was installed but cannot be imported yet. Restart Slicer.")
                return False
        return True

    # --- Model folder ------------------------------------------------------
    # Stored in the application settings (always writable). model_config.ini next to the module is still
    # written when possible, because Easy_fusion reads the folder from there.

    def selectModelFolder(self):
        folder = qt.QFileDialog.getExistingDirectory(None, "Select Model Folder", self.modelFolderPathEdit.text)
        if folder:
            self.modelFolderPathEdit.setText(folder)
            self.saveModelFolderPath(folder)
            self.refreshModelSelector(folder)

    @staticmethod
    def _iniPath():
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_config.ini")

    def saveModelFolderPath(self, folder):
        qt.QSettings().setValue(SETTINGS_MODEL_FOLDER, folder)
        config = configparser.ConfigParser()
        config["ModelFolder"] = {"path": folder}
        try:
            with open(self._iniPath(), "w") as configfile:
                config.write(configfile)
        except OSError as error:
            logging.info(f"PETDenoise: model_config.ini not written ({error}); folder kept in application settings")

    def loadModelFolderPath(self):
        folder = qt.QSettings().value(SETTINGS_MODEL_FOLDER) or ""
        if not os.path.isdir(folder) and os.path.isfile(self._iniPath()):
            config = configparser.ConfigParser()
            try:
                config.read(self._iniPath())
                folder = config.get("ModelFolder", "path", fallback="")
            except configparser.Error:
                folder = ""
        if folder and os.path.isdir(folder):
            self.modelFolderPathEdit.setText(folder)
            self.refreshModelSelector(folder)

    def refreshModelSelector(self, folder):
        self.modelselector.clear()
        try:
            models = sorted(f for f in os.listdir(folder) if f.lower().endswith(".pth"))
        except OSError as error:
            self.outputTextBox.append(f"❌ Cannot read model folder: {error}")
            return
        self.modelselector.addItems(models)
        if not models:
            self.modelInfoBox.setPlainText("No .pth models found in this folder.")


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

class PETDenoiseLogic(ScriptedLoadableModuleLogic):
    """
    Resample to the model's voxel spacing, sliding-window inference (gaussian, 25% overlap),
    output = input - predicted noise, optional clip of negative values. The input volume is never written to,
    and the output volume is only added to the scene once everything worked.
    """

    @staticmethod
    def uniqueVolumeName(baseName):
        name, number = baseName, 1
        while slicer.mrmlScene.GetFirstNodeByName(name) is not None:
            name = f"{baseName}_{number}"
            number += 1
        return name

    @staticmethod
    def chooseDevice(torch, forceCPU):
        cpu = torch.device("cpu")
        if forceCPU:
            return cpu, "CPU (forced)"
        if not torch.cuda.is_available():
            return cpu, "CPU"
        try:
            properties = torch.cuda.get_device_properties(0)
        except Exception:
            return cpu, "CPU (GPU could not be queried)"
        vramGb = properties.total_memory / 1024.0 ** 3
        if vramGb < MIN_VRAM_GB:
            return cpu, f"CPU (GPU has only {vramGb:.1f} GB VRAM)"
        return torch.device("cuda"), f"GPU {properties.name}, {vramGb:.1f} GB"

    @staticmethod
    def buildNetwork(params, inChannels):
        """Class structure and attribute names (unet / model / gcfn) unchanged, so existing .pth files load."""
        import torch.nn as nn
        import torch.nn.functional as F
        from monai import __version__ as monaiVersion
        from monai.networks.nets import UNet, SwinUNETR
        from packaging import version

        swinExtra = {"img_size": (64, 64, 64)} if version.parse(monaiVersion) < version.parse("1.5") else {}

        class DenoiseUNet(nn.Module):
            def __init__(self, in_channels=1, out_channels=1, channels=(32, 64, 128, 256, 512), num_res_units=2,
                         strides=(2, 2, 2, 2), kernel_size=3, up_kernel_size=3):
                super().__init__()
                self.unet = UNet(strides=strides, num_res_units=num_res_units, kernel_size=kernel_size,
                                 up_kernel_size=up_kernel_size, spatial_dims=3, in_channels=in_channels,
                                 out_channels=out_channels, channels=channels)

            def forward(self, x):
                return self.unet(x)

        class SwinDenoiser(nn.Module):
            def __init__(self, in_channels=1, out_channels=1, feature_size=48, heads=(6, 12, 24, 48),
                         depths=(2, 3, 3, 2), do_rate=0.1):
                super().__init__()
                self.model = SwinUNETR(num_heads=heads, use_v2=True, in_channels=in_channels,
                                       out_channels=out_channels, feature_size=feature_size, depths=depths,
                                       dropout_path_rate=do_rate, use_checkpoint=True, **swinExtra)

            def forward(self, x):
                return self.model(x)

        class GCFN(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.norm = nn.LayerNorm(dim)
                self.fc1 = nn.Linear(dim, dim)
                self.fc2 = nn.Linear(dim, dim)
                self.fc0 = nn.Linear(dim, dim)
                self.conv1 = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim)
                self.conv2 = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim)

            def forward(self, x):
                B, C, D, H, W = x.shape
                x_ = x.permute(0, 2, 3, 4, 1).contiguous().view(B * D * H * W, C)
                x1 = self.fc1(self.norm(x_)).view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
                x2 = self.fc2(self.norm(x_)).view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
                gate = F.gelu(self.conv1(x1)) * self.conv2(x2)
                gate = gate.permute(0, 2, 3, 4, 1).contiguous().view(B * D * H * W, C)
                out = self.fc0(gate).view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
                return out + x

        class SwinGCFN(nn.Module):
            def __init__(self, in_channels=1, out_channels=1, feature_size=48, heads=(6, 12, 24, 48),
                         depths=(2, 3, 3, 2), do_rate=0.1):
                super().__init__()
                self.model = SwinUNETR(num_heads=heads, use_v2=True, in_channels=in_channels,
                                       out_channels=out_channels, feature_size=feature_size, depths=depths,
                                       dropout_path_rate=do_rate, use_checkpoint=True, **swinExtra)
                self.gcfn = GCFN(dim=out_channels)

            def forward(self, x):
                return self.gcfn(self.model(x))

        if params["architecture"] == "UNET":
            return DenoiseUNet(in_channels=inChannels, channels=params["channels"], num_res_units=params["res_units"],
                               strides=params["strides"], kernel_size=params["down_kernel"],
                               up_kernel_size=params["up_kernel"])
        swinClass = SwinDenoiser if params["architecture"] == "SwinUNETR" else SwinGCFN
        return swinClass(in_channels=inChannels, feature_size=params["feature_size"], heads=params["num_heads"],
                         depths=params["depths"], do_rate=params["do_rate"])

    def loadModel(self, torch, modelPath, params, inChannels, device):
        model = self.buildNetwork(params, inChannels).to(device)
        try:
            # weights_only: a .pth file cannot run arbitrary code when it is loaded
            state = torch.load(modelPath, map_location=device, weights_only=True)
        except TypeError:  # PyTorch < 1.13
            state = torch.load(modelPath, map_location=device)
        try:
            model.load_state_dict(state)
        except RuntimeError as error:
            raise RuntimeError(
                f"The weights in {os.path.basename(modelPath)} do not match a {params['architecture']} network with "
                f"{inChannels} input channel(s) and these settings. Check the parameters / the .txt file "
                "(and the Dual Volume Input checkbox).") from error
        model.eval()
        return model

    @staticmethod
    def _addHiddenVolume(name):
        """Scratch volume: hidden from selectors, never saved, removed after the run."""
        node = slicer.vtkMRMLScalarVolumeNode()
        node.SetName(name)
        node.SetHideFromEditors(True)
        node.SetSaveWithScene(False)
        return slicer.mrmlScene.AddNode(node)

    @staticmethod
    def _runCli(module, parameters, label):
        cliNode = slicer.cli.createNode(module)
        try:
            slicer.cli.runSync(module, cliNode, parameters, update_display=False)
            if cliNode.GetStatus() & cliNode.ErrorsMask:
                raise RuntimeError(f"{label} failed: {cliNode.GetErrorText()}")
        finally:
            slicer.mrmlScene.RemoveNode(cliNode)

    @staticmethod
    def _createOutputVolume(inputNode, gridNode, name):
        """New volume with the input's display settings and the model grid's geometry."""
        scene = slicer.mrmlScene
        try:
            outputNode = slicer.modules.volumes.logic().CloneVolumeWithoutImageData(scene, inputNode, name)
        except AttributeError:  # older Slicer
            outputNode = None
        if outputNode is None:
            outputNode = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
            outputNode.CreateDefaultDisplayNodes()
        outputNode.CopyOrientation(gridNode)
        outputNode.SetAndObserveTransformNodeID(inputNode.GetTransformNodeID())
        return outputNode

    def run(self, inputNode, ctNode, modelPath, params, outputName, forceCPU=False, log=None):
        import torch
        from monai.inferers import sliding_window_inference

        log = log or logging.info
        scene = slicer.mrmlScene
        scratchNodes = []
        model = None
        startTime = time.time()
        try:
            device, deviceText = self.chooseDevice(torch, forceCPU)
            log(f"🖥️ Device: {deviceText}")
            inChannels = 2 if params["dual_channel"] else 1
            log(f"📌 Loading {os.path.basename(modelPath)} ({params['architecture']}, {inChannels} channel(s))")
            model = self.loadModel(torch, modelPath, params, inChannels, device)

            if params["dont_resample"]:
                gridNode = inputNode
            else:
                spacing = params["voxel_spacing"]
                log(f"📐 Resampling input to {spacing} mm...")
                gridNode = self._addHiddenVolume("PETDenoiseInput")
                scratchNodes.append(gridNode)
                self._runCli(slicer.modules.resamplescalarvolume,
                             {"InputVolume": inputNode.GetID(), "OutputVolume": gridNode.GetID(),
                              "outputPixelSpacing": ",".join(f"{v:g}" for v in spacing),
                              "interpolationType": "linear"},
                             "Resampling")

            gridArray = slicer.util.arrayFromVolume(gridNode)
            inputDtype = gridArray.dtype
            inputTensor = torch.from_numpy(np.array(gridArray, dtype=np.float32))[None, None]  # (1,1,K,J,I), copy
            del gridArray
            log(f"Input size on model grid: {tuple(inputTensor.shape[2:])}")

            networkInput = inputTensor
            if params["dual_channel"]:
                log("📐 Resampling CT to the PET grid...")
                ctScratch = self._addHiddenVolume("PETDenoiseCT")
                scratchNodes.append(ctScratch)
                self._runCli(slicer.modules.brainsresample,
                             {"inputVolume": ctNode.GetID(), "referenceVolume": gridNode.GetID(),
                              "outputVolume": ctScratch.GetID(), "pixelType": "float",
                              "interpolationMode": "Linear"},
                             "CT resampling")
                ctArray = np.array(slicer.util.arrayFromVolume(ctScratch), dtype=np.float32)
                if ctArray.shape != tuple(inputTensor.shape[2:]):
                    raise RuntimeError(f"Resampled CT {ctArray.shape} does not match PET {tuple(inputTensor.shape[2:])}.")
                networkInput = torch.cat([inputTensor, torch.from_numpy(ctArray)[None, None]], dim=1)
                del ctArray

            roiSize = tuple(params["block_size"])
            total = estimateSlidingWindowCount(inputTensor.shape[2:], roiSize)
            step = max(1, total // 10)
            done = [0]

            def predictor(window, *args, **kwargs):
                prediction = model(window, *args, **kwargs)
                done[0] += 1
                if done[0] % step == 0 or done[0] == total:
                    log(f"   ... window {done[0]} / {total}")
                return prediction

            log(f"🧠 Denoising: {total} windows of {roiSize}")
            # Windows run on the device; the full volume and blending buffers stay in RAM (much less VRAM)
            with torch.no_grad():
                predictedNoise = sliding_window_inference(
                    inputs=networkInput, roi_size=roiSize, sw_batch_size=1, predictor=predictor,
                    overlap=WINDOW_OVERLAP, mode="gaussian", sw_device=device, device=torch.device("cpu"))
            del networkInput

            result = (inputTensor - predictedNoise.float())[0, 0].numpy()
            del predictedNoise, inputTensor
            if params["prevent_negative"]:
                result = np.clip(result, 0, None)
            result = castResult(result, inputDtype)

            outputNode = self._createOutputVolume(inputNode, gridNode, outputName)
            slicer.util.updateVolumeFromArray(outputNode, result)
            outputNode.SetAttribute(MODEL_ATTRIBUTE, os.path.basename(modelPath))
            outputNode.SetAttribute(DATE_ATTRIBUTE, time.strftime("%Y-%m-%d %H:%M:%S"))
            log(f"⏱️ Done in {time.time() - startTime:.1f} s. Output size: {result.shape}")
            return outputNode
        finally:
            for node in scratchNodes:
                if scene.IsNodePresent(node):
                    scene.RemoveNode(node)
            model = None
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
