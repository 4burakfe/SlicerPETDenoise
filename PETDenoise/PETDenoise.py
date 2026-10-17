import os
import ast
import gc
import math
import time
import logging
import importlib
import traceback
import configparser
import json
import shutil
import hashlib
import urllib.request

import numpy as np
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *

MODULE_VERSION = "v1.2"
ARCHITECTURES = ("UNET", "SwinUNETR", "SwinUNETR+GCFN")
WINDOW_OVERLAP = 0.25
MIN_VRAM_GB = 1.9
EINOPS_REQUIREMENT = "einops==0.6.1"
SETTINGS_MODEL_FOLDER = "PETDenoise/ModelFolder"
MODEL_ATTRIBUTE = "PETDenoise.Model"
DATE_ATTRIBUTE = "PETDenoise.Date"

# Published models (GitHub releases of this repository): <name>.pth + <name>.txt. The release lists and the
# download cache folder are the same as in Epona - SPECT/PET Review (PETReviewCompare extension), so a model
# downloaded by one module is found by the other.
MODEL_REPOSITORY = "4burakfe/SlicerPETDenoise"
MODEL_RELEASES = (("Models", "Denoising"), ("SuperRes_Models", "Super-resolution"))  # (release tag, kind)
MODEL_RELEASE_PAGE_URL = "https://github.com/{repository}/releases/tag/{tag}"
MODEL_DOWNLOAD_URL = "https://github.com/{repository}/releases/download/{tag}/{name}"
MODEL_RELEASE_API_URL = "https://api.github.com/repos/{repository}/releases/tags/{tag}"
MODEL_CACHE_FOLDER_NAME = "BelenosModels"  # inside Slicer's cache folder; also used when no model folder is set
MODEL_CATALOG_FILE = "catalog.json"  # release asset lists from the last successful check
MODEL_CATALOG_MAX_AGE_S = 24 * 3600  # GitHub is asked for new / replaced models at most once a day
NETWORK_TIMEOUT_S = 8
DOWNLOAD_CHUNK_BYTES = 1 << 20
# Models published when this version was released: (file, release tag, size in bytes, SHA-256 of the file).
# Used as they are when GitHub cannot be reached; the release lists on GitHub add new and replaced models.
PUBLISHED_MODELS = (
    ("pet_denoiser_std_char.pth", "Models", 80995179,
     "11e4627577c2ed8c0f19616dd004c63ee00c321febb85f54158bb88c485881e8"),
    ("pet_denoiser_x14w3.pth", "Models", 80994666,
     "dbd53df5765e48c61320b39737dbf5b73973b0f2b53a7f8ec3d2541947088a53"),
    ("pet_denoiser_x19w1_5.pth", "Models", 80994666,
     "dd05fb82873bd579ebe8785f50371579e930d2b771e2385947ca1ada46f7c60f"),
    ("PET_superres12.pth", "SuperRes_Models", 26564203,
     "af17bf85515ac4b64404e40fda0290506673fc951a0bfe71c9672a372b6d6d77"),
    ("CT_superres24.pth", "SuperRes_Models", 80994495,
     "51a8ce1b383f3803bdace09f4a6e5e9ee63e8c51b48ef6476b8cac477763bb87"),
)

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


class DownloadCancelled(Exception):
    """Raised by the progress callback of a download when the user pressed Cancel."""


def sidecarName(modelName):
    """<model>.txt for <model>.pth."""
    return os.path.splitext(modelName)[0] + ".txt"


def formatByteSize(numberOfBytes):
    gigabytes = numberOfBytes / 1024.0 ** 3
    return f"{gigabytes:.1f} GB" if gigabytes >= 1.0 else f"{numberOfBytes / 1024.0 ** 2:.0f} MB"


def releaseAssetUrl(tag, name):
    return MODEL_DOWNLOAD_URL.format(repository=MODEL_REPOSITORY, tag=tag, name=name)


def parseReleaseAssets(release):
    """Asset list of a GitHub release (API JSON) as [{"name", "size", "sha256", "url"}]; sha256 None if unknown."""
    assets = []
    for asset in (release or {}).get("assets") or []:
        name = str(asset.get("name") or "")
        if not name:
            continue
        digest = str(asset.get("digest") or "")
        assets.append({
            "name": name,
            "size": int(asset.get("size") or 0) or None,
            "sha256": digest[len("sha256:"):].lower() if digest.lower().startswith("sha256:") else None,
            "url": str(asset.get("browser_download_url") or "") or None,
        })
    return assets


def mergePublishedModelCatalog(releaseAssets=None):
    """
    Published models as {file name: entry}, entry = name, tag, kind, size, sha256, url, sidecarUrl.
    releaseAssets: {release tag: parseReleaseAssets(...)} from GitHub (or its saved copy); None / {} gives the
    built-in list. For every release that was read, its asset list is what counts: added models appear, removed
    ones disappear, and a model whose size changed was replaced, so the built-in checksum is dropped.
    """
    kinds = dict(MODEL_RELEASES)
    models = {}
    for name, tag, size, sha256 in PUBLISHED_MODELS:
        models[name] = {"name": name, "tag": tag, "kind": kinds.get(tag, ""), "size": size, "sha256": sha256,
                        "url": releaseAssetUrl(tag, name), "sidecarUrl": releaseAssetUrl(tag, sidecarName(name))}
    for tag, assets in (releaseAssets or {}).items():
        if tag not in kinds:
            continue
        names = {asset["name"] for asset in assets}
        for name in [name for name, entry in models.items() if entry["tag"] == tag and name not in names]:
            del models[name]
        for asset in assets:
            name = asset["name"]
            if not name.lower().endswith(".pth"):
                continue
            entry = models.get(name)
            if entry is None or entry["tag"] != tag:
                entry = {"name": name, "tag": tag, "kind": kinds[tag], "size": None, "sha256": None}
                models[name] = entry
            elif asset["size"] and asset["size"] != entry["size"]:
                entry["sha256"] = None  # replaced on GitHub: the built-in checksum is for the old file
            entry["size"] = asset["size"] or entry["size"]
            entry["sha256"] = asset["sha256"] or entry["sha256"]
            entry["url"] = asset["url"] or releaseAssetUrl(tag, name)
            sidecar = sidecarName(name)
            entry["sidecarUrl"] = releaseAssetUrl(tag, sidecar) if sidecar in names else None
    tagOrder = {tag: index for index, (tag, _) in enumerate(MODEL_RELEASES)}
    return dict(sorted(models.items(), key=lambda item: (tagOrder.get(item[1]["tag"], 99), item[0].lower())))


def isCompleteModelFile(path, expectedSize):
    """True if the file exists and (when the published size is known) has exactly that size."""
    try:
        return os.path.isfile(path) and (not expectedSize or os.path.getsize(path) == expectedSize)
    except OSError:
        return False


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
        self.setupDownloadSection(formLayout)

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
        # Disabled while a denoising run or a download is going on
        self._busyWidgets = [self.calculateButton, self.selectFolderButton, self.modelselector,
                             self.downloadModelButton, self.checkModelsButton]

        self.loadModelFolderPath()
        self.selectFolderButton.connect("clicked(bool)", self.selectModelFolder)
        self.calculateButton.connect("clicked(bool)", self.onCalculateButtonClicked)
        self.modelselector.connect("currentIndexChanged(int)", self.updateModelInfoBox)
        self.architecture.connect("currentIndexChanged(int)", self.updateEnabledState)
        self.dualch_cbox.connect("toggled(bool)", self.updateEnabledState)
        self.dont_resample_cbox.connect("toggled(bool)", self.updateEnabledState)
        self.downloadCollapsibleButton.connect("contentsCollapsed(bool)", self.onDownloadSectionCollapsed)
        self.publishedModelSelector.connect("currentIndexChanged(int)", self.updateDownloadButton)
        self.downloadModelButton.connect("clicked(bool)", self.onDownloadModelClicked)
        self.checkModelsButton.connect("clicked(bool)", lambda: self.updatePublishedModelCatalog(force=True))
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

        self.setBusy(True)
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
            self.setBusy(False)

    def setBusy(self, busy):
        self._running = busy
        for widget in self._busyWidgets:
            widget.enabled = not busy
        if not busy:
            self.updateDownloadButton()

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
    # written when possible, for older versions of Epona - SPECT/PET Review (PETReviewCompare extension)
    # that read the folder from there; current versions read the application setting.

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
        self.refreshPublishedModelSelector()
        self.modelselector.clear()
        try:
            models = sorted(f for f in os.listdir(folder) if f.lower().endswith(".pth"))
        except OSError as error:
            self.outputTextBox.append(f"❌ Cannot read model folder: {error}")
            return
        self.modelselector.addItems(models)
        if not models:
            self.modelInfoBox.setPlainText("No .pth models found in this folder. Published models can be "
                                           "downloaded in the 'Download Published Models' section.")

    # --- Published models (download from GitHub) ----------------------------
    # Downloaded into the model folder, so they are used like any other model there (and Epona - SPECT/PET
    # Review finds them in that folder too). Without a model folder, Slicer's cache folder shared with Epona is
    # used and becomes the model folder. A model Epona already downloaded there is copied instead of downloaded.

    def setupDownloadSection(self, formLayout):
        self.downloadCollapsibleButton = ctk.ctkCollapsibleButton()
        self.downloadCollapsibleButton.text = "Download Published Models"
        self.downloadCollapsibleButton.collapsed = True
        formLayout.addRow(self.downloadCollapsibleButton)
        downloadLayout = qt.QFormLayout(self.downloadCollapsibleButton)

        self.publishedModelSelector = qt.QComboBox()
        self.publishedModelSelector.setToolTip("Models published on GitHub with this extension.")
        downloadLayout.addRow("Published Model:", self.publishedModelSelector)
        buttonsLayout = qt.QHBoxLayout()
        self.downloadModelButton = qt.QPushButton("Download")
        self.downloadModelButton.setToolTip("Download the model and its .txt file into the model folder, "
                                            "then select it.")
        buttonsLayout.addWidget(self.downloadModelButton)
        self.checkModelsButton = qt.QPushButton("Check for New Models")
        self.checkModelsButton.setToolTip("Read the list of published models from GitHub now.")
        buttonsLayout.addWidget(self.checkModelsButton)
        downloadLayout.addRow(buttonsLayout)
        self.downloadStatusLabel = qt.QLabel("")
        self.downloadStatusLabel.wordWrap = True
        downloadLayout.addRow(self.downloadStatusLabel)

        releases, self._modelCatalogChecked = self.logic.readSavedModelCatalog()
        self._publishedModels = mergePublishedModelCatalog(releases)
        self.refreshPublishedModelSelector()

    def downloadTargetFolder(self):
        folder = self.modelFolderPathEdit.text
        return folder if folder and os.path.isdir(folder) else self.logic.publishedModelFolder()

    def currentPublishedModel(self):
        name = self.publishedModelSelector.itemData(self.publishedModelSelector.currentIndex)
        return self._publishedModels.get(name) if name else None

    def refreshPublishedModelSelector(self):
        folder = self.downloadTargetFolder()
        previousName = self.publishedModelSelector.itemData(self.publishedModelSelector.currentIndex)
        wasBlocked = self.publishedModelSelector.blockSignals(True)
        self.publishedModelSelector.clear()
        for name, entry in self._publishedModels.items():
            present = isCompleteModelFile(os.path.join(folder, name), entry["size"])
            details = [entry["kind"]] if entry["kind"] else []
            if entry["size"]:
                details.append(formatByteSize(entry["size"]))
            if present:
                details.append("in model folder")
            label = os.path.splitext(name)[0] + (f"  ({', '.join(details)})" if details else "")
            self.publishedModelSelector.addItem(label, name)
            tip = (f"{entry['kind']} model published on GitHub ({entry['tag']}), "
                   f"{'already in the model folder' if present else 'not downloaded yet'}.")
            self.publishedModelSelector.setItemData(self.publishedModelSelector.count - 1, tip, qt.Qt.ToolTipRole)
        index = self.publishedModelSelector.findData(previousName) if previousName else -1
        self.publishedModelSelector.setCurrentIndex(index if index >= 0 else (0 if self._publishedModels else -1))
        self.publishedModelSelector.blockSignals(wasBlocked)
        self.updateDownloadButton()

    def updateDownloadButton(self, *args):
        entry = self.currentPublishedModel()
        present = entry is not None and isCompleteModelFile(
            os.path.join(self.downloadTargetFolder(), entry["name"]), entry["size"])
        self.downloadModelButton.text = "Select" if present else "Download"
        self.downloadModelButton.enabled = entry is not None and not self._running

    def onDownloadSectionCollapsed(self, collapsed):
        """Opening the section: check GitHub for new published models (at most once a day)."""
        if not collapsed:
            self.updatePublishedModelCatalog(force=False)

    def updatePublishedModelCatalog(self, force=False):
        """Read the release lists from GitHub (force: now; else if the last check is over a day old)."""
        if self._running:
            return
        failure = None
        if force or time.time() - self._modelCatalogChecked > MODEL_CATALOG_MAX_AGE_S:
            self.downloadStatusLabel.text = "Checking GitHub for published models…"
            slicer.app.processEvents()
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            try:
                self._publishedModels = mergePublishedModelCatalog(self.logic.fetchModelCatalog())
                self.downloadStatusLabel.text = f"{len(self._publishedModels)} published models."
            except Exception as error:
                logging.warning(f"PETDenoise: could not read the list of published models from GitHub: {error}")
                failure = error
                self.downloadStatusLabel.text = "GitHub could not be reached; the models already known are listed."
            finally:
                self._modelCatalogChecked = time.time()  # offline: do not try again on every opening
                qt.QApplication.restoreOverrideCursor()
        self.refreshPublishedModelSelector()
        if failure is not None and force:
            slicer.util.warningDisplay("Could not reach GitHub to check for new models:\n"
                                       f"{failure}\n\nThe models already known are still listed.")

    def onDownloadModelClicked(self):
        entry = self.currentPublishedModel()
        if entry is None or self._running:
            return
        folder = self.downloadTargetFolder()
        if isCompleteModelFile(os.path.join(folder, entry["name"]), entry["size"]):
            if not os.path.isfile(os.path.join(folder, sidecarName(entry["name"]))):
                self.fetchSidecar(entry, folder, cachedFirst=True)
        elif not self.downloadModel(entry, folder):
            return
        if self.modelFolderPathEdit.text != folder:
            self.modelFolderPathEdit.setText(folder)
            self.saveModelFolderPath(folder)
        self.refreshModelSelector(folder)
        index = self.modelselector.findText(entry["name"])
        if index >= 0:
            self.modelselector.setCurrentIndex(index)
        self.updateModelInfoBox()

    def fetchSidecar(self, entry, folder, cachedFirst):
        """The model's .txt into folder (copied from the shared cache folder if cachedFirst). Never fatal."""
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self.logic.fetchSidecar(entry, folder, cachedFirst)
        except Exception as error:
            logging.warning(f"PETDenoise: could not download the description of {entry['name']}: {error}")
            self.downloadStatusLabel.text = (f"The description file (.txt) of {entry['name']} could not be "
                                             "downloaded; check the parameters before denoising.")
        finally:
            qt.QApplication.restoreOverrideCursor()

    def downloadModel(self, entry, folder):
        """Download a published model (and its .txt) behind a cancellable progress dialog. True when done."""
        name = entry["name"]
        self.setBusy(True)
        progress = DownloadProgressDialog(f"Belenos - downloading {name}")
        failure = None
        try:
            copied = self.logic.downloadModel(entry, folder, report=progress)
            self.downloadStatusLabel.text = f"{'Copied' if copied else 'Downloaded'} {name} into {folder}."
        except DownloadCancelled:
            self.downloadStatusLabel.text = "Download cancelled."
            return False
        except Exception as error:
            logging.exception(f"PETDenoise: could not download {name}")
            failure = (error, traceback.format_exc())
        finally:
            progress.close()
            self.setBusy(False)
        if failure is not None:
            releasePage = MODEL_RELEASE_PAGE_URL.format(repository=MODEL_REPOSITORY, tag=entry["tag"])
            self.downloadStatusLabel.text = f"Could not download {name}."
            slicer.util.errorDisplay(
                f"Could not download {name} into\n{folder}:\n{failure[0]}\n\n"
                f"Check the internet connection and that the folder is writable, then try again. You can also "
                f"download the .pth and .txt files yourself from\n{releasePage}\ninto the model folder.",
                detailedText=failure[1])
            return False
        self.fetchSidecar(entry, folder, cachedFirst=copied)
        return True


class DownloadProgressDialog:
    """Modal progress window of a download. Calling it updates the text / bar; after Cancel it raises."""

    def __init__(self, title):
        dialog = qt.QProgressDialog(slicer.util.mainWindow())
        dialog.setWindowTitle(title)
        dialog.setWindowModality(qt.Qt.ApplicationModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumWidth(420)
        dialog.setRange(0, 0)
        dialog.setLabelText("Preparing…")
        dialog.show()
        self.dialog = dialog
        slicer.app.processEvents()

    def __call__(self, text, done=0, total=0, detail=None):
        if self.dialog.wasCanceled:
            raise DownloadCancelled()
        if total > 0:
            self.dialog.setLabelText(f"{text}\n{detail}" if detail else text)
            self.dialog.setRange(0, int(total))
            self.dialog.setValue(min(int(done), int(total)))
        else:
            self.dialog.setLabelText(f"{text}\n{detail}" if detail else text)
            self.dialog.setRange(0, 0)
        slicer.app.processEvents()
        if self.dialog.wasCanceled:
            raise DownloadCancelled()

    def close(self):
        self.dialog.close()
        self.dialog.deleteLater()


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

    # --- Published models --------------------------------------------------

    @staticmethod
    def publishedModelFolder():
        """Download cache shared with Epona - SPECT/PET Review: Slicer's cache folder (kept across updates)."""
        return os.path.join(slicer.app.cachePath, MODEL_CACHE_FOLDER_NAME)

    @classmethod
    def readSavedModelCatalog(cls):
        """(release assets, time of that check) saved by the last successful check of GitHub; ({}, 0) if none."""
        try:
            with open(os.path.join(cls.publishedModelFolder(), MODEL_CATALOG_FILE), "r",
                      encoding="utf-8") as catalogFile:
                saved = json.load(catalogFile)
            releases = {str(tag): [dict(asset) for asset in assets]
                        for tag, assets in (saved.get("releases") or {}).items()}
            for assets in releases.values():
                for asset in assets:
                    if not asset.get("name"):
                        raise ValueError("asset without a name")
                    asset.setdefault("size", None)
                    asset.setdefault("sha256", None)
                    asset.setdefault("url", None)
            return releases, float(saved.get("checked") or 0)
        except FileNotFoundError:
            return {}, 0.0
        except (OSError, ValueError, TypeError, AttributeError):
            logging.warning("PETDenoise: ignored an unreadable saved model list", exc_info=True)
            return {}, 0.0

    @staticmethod
    def openUrl(url, timeout=NETWORK_TIMEOUT_S, accept=None):
        headers = {"User-Agent": "3DSlicer-PETDenoise"}
        if accept:
            headers["Accept"] = accept
        return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)

    @classmethod
    def fetchModelCatalog(cls):
        """Asset lists of the model releases from the GitHub API, saved for later sessions. Raises if offline."""
        releases = {}
        for tag, _ in MODEL_RELEASES:
            url = MODEL_RELEASE_API_URL.format(repository=MODEL_REPOSITORY, tag=tag)
            with cls.openUrl(url, accept="application/vnd.github+json") as response:
                releases[tag] = parseReleaseAssets(json.loads(response.read().decode("utf-8")))
        folder = cls.publishedModelFolder()
        try:
            os.makedirs(folder, exist_ok=True)
            catalogPath = os.path.join(folder, MODEL_CATALOG_FILE)
            with open(catalogPath + ".part", "w", encoding="utf-8") as catalogFile:
                json.dump({"checked": time.time(), "releases": releases}, catalogFile, indent=1)
            os.replace(catalogPath + ".part", catalogPath)
        except OSError:
            logging.warning("PETDenoise: could not save the model list", exc_info=True)
        return releases

    @staticmethod
    def _removePartFile(partPath):
        if os.path.exists(partPath):
            try:
                os.remove(partPath)
            except OSError:
                logging.warning(f"PETDenoise: could not remove {partPath}")

    @classmethod
    def downloadFile(cls, url, destination, expectedSize=None, sha256=None, report=None, text="Downloading…"):
        """
        Download url to destination through destination + ".part", which is renamed only once the size and (when
        known) the SHA-256 checksum match, so an interrupted or corrupted download never looks like a model.
        report(text, done, total, detail) is the progress callback; it raises DownloadCancelled to stop.
        """
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        partPath = destination + ".part"
        hasher = hashlib.sha256()
        done = 0
        try:
            with cls.openUrl(url, timeout=30) as response, open(partPath, "wb") as output:
                total = int(response.headers.get("Content-Length") or 0) or int(expectedSize or 0)
                while True:
                    if report is not None:
                        detail = f"{done / 1e6:.0f} of {total / 1e6:.0f} MB" if total else f"{done / 1e6:.0f} MB"
                        report(text, done, total, detail)
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    output.write(chunk)
                    hasher.update(chunk)
                    done += len(chunk)
            if expectedSize and done != expectedSize:
                raise OSError(f"incomplete download: {done} of {expectedSize} bytes received")
            if sha256 and hasher.hexdigest() != sha256.lower():
                raise OSError("the downloaded file is corrupted (SHA-256 checksum does not match)")
            os.replace(partPath, destination)
        finally:
            cls._removePartFile(partPath)
        return destination

    @classmethod
    def copyFile(cls, source, destination):
        """Copy through destination + ".part", so an interrupted copy never looks like a model."""
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        partPath = destination + ".part"
        try:
            shutil.copyfile(source, partPath)
            os.replace(partPath, destination)
        finally:
            cls._removePartFile(partPath)
        return destination

    @staticmethod
    def _sameFolder(a, b):
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))

    @classmethod
    def downloadModel(cls, entry, folder, report=None):
        """
        Published model's .pth into folder: copied from the shared cache folder when it is complete there,
        otherwise downloaded from GitHub. True if it was copied.
        """
        name = entry["name"]
        destination = os.path.join(folder, name)
        cached = os.path.join(cls.publishedModelFolder(), name)
        if not cls._sameFolder(folder, cls.publishedModelFolder()) and isCompleteModelFile(cached, entry["size"]):
            if report is not None:
                report(f"Copying {name} from the downloaded models…")
            cls.copyFile(cached, destination)
            return True
        cls.downloadFile(entry["url"], destination, expectedSize=entry.get("size"), sha256=entry.get("sha256"),
                         report=report, text=f"Downloading {name} from GitHub…")
        return False

    @classmethod
    def fetchSidecar(cls, entry, folder, cachedFirst=False):
        """Published model's .txt into folder (from the shared cache folder if cachedFirst). Path, or None."""
        name = sidecarName(entry["name"])
        destination = os.path.join(folder, name)
        cached = os.path.join(cls.publishedModelFolder(), name)
        if cachedFirst and not cls._sameFolder(folder, cls.publishedModelFolder()) and os.path.isfile(cached):
            return cls.copyFile(cached, destination)
        if not entry.get("sidecarUrl"):
            return None
        return cls.downloadFile(entry["sidecarUrl"], destination)

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
