import os
import gc
import logging
import importlib
import traceback

import numpy as np
import qt
import ctk
import vtk
import slicer
from slicer.ScriptedLoadableModule import *

MODULE_VERSION = "v1.2"
MIN_VRAM_GB = 1.9
GEOMETRY_TOLERANCE = 1e-3  # mm / direction cosine
CSV_HEADER = "MSE;MAE;SSIM;SSIM_Loss;PSNR;Edge_Loss"
# Resampling is only offered when the grid being resampled onto is covered this much by the other volume
MIN_COVERAGE = 0.5
WARN_COVERAGE = 0.95


class VolumeComparator(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Belenos - Volume Comparator"
        parent.categories = ["Nuclear Medicine", "Informatics", "Filtering.Denoising"]
        parent.dependencies = ["PyTorchUtils"]
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = "This module provides comparison and loss metrics for two volumes."
        parent.acknowledgementText = "This file was developed by Burak Demir."
        iconPath = os.path.join(os.path.dirname(__file__), "Resources", "logo.png")
        if os.path.exists(iconPath):
            parent.icon = qt.QIcon(iconPath)


class VolumeComparatorWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self._running = False


        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)
        formLayout = qt.QFormLayout(parametersCollapsibleButton)

        self.inputVolumeSelector = self._volumeSelector("Volume to evaluate (e.g. denoised image).")
        formLayout.addRow("Input Volume 1 (test): ", self.inputVolumeSelector)
        self.inputVolumeSelector2 = self._volumeSelector("Reference volume (e.g. full-count image). "
                                                         "PSNR uses the maximum of this volume.")
        formLayout.addRow("Input Volume 2 (reference): ", self.inputVolumeSelector2)

        self.ssimloss_range = qt.QLineEdit("10")
        self.ssimloss_range.setToolTip("Without auto-normalization both volumes are clipped to [0, range] "
                                       "and SSIM uses this data range.")
        formLayout.addRow("Range for SSIM:", self.ssimloss_range)

        self.ssimnorm_cbox = qt.QCheckBox()
        self.ssimnorm_cbox.setToolTip("Joint min-max normalization of both volumes to [0, 1] before SSIM.")
        formLayout.addRow("Auto Normalize Before SSIM: ", self.ssimnorm_cbox)
        self.ssimnorm_cbox.connect("toggled(bool)", lambda checked: setattr(self.ssimloss_range, "enabled",
                                                                            not checked))

        self.outputTextBox = qt.QTextEdit()
        self.outputTextBox.setReadOnly(True)
        formLayout.addRow("Processing Log and Results:", self.outputTextBox)

        buttonRow = qt.QHBoxLayout()
        self.calculateButton = qt.QPushButton("Compare")
        self.copyButton = qt.QPushButton("Copy results row")
        self.copyButton.setToolTip("Copies the header and the semicolon-separated result row to the clipboard.")
        self.copyButton.enabled = False
        buttonRow.addWidget(self.calculateButton)
        buttonRow.addWidget(self.copyButton)
        formLayout.addRow(buttonRow)
        self.calculateButton.connect("clicked(bool)", self.onCalculateButtonClicked)
        self.copyButton.connect("clicked(bool)", self.onCopyResults)
        self._lastCsv = ""

        bannerPath = os.path.join(os.path.dirname(__file__), "Resources", "banner.png")
        if os.path.exists(bannerPath):
            bannerLabel = qt.QLabel()
            bannerLabel.setPixmap(qt.QPixmap(bannerPath).scaledToWidth(600, qt.Qt.SmoothTransformation))
            bannerLabel.setAlignment(qt.Qt.AlignCenter)
            self.layout.addWidget(bannerLabel)



        infoTextBox = qt.QTextEdit()
        infoTextBox.setReadOnly(True)
        infoTextBox.setPlainText(
            "This module compares two images. Volumes on different grids can be resampled (BRAINSResample).\n"
            "It calculates MSE, MAE, edge loss, PSNR and SSIM.\n"
            "This module is NOT a medical device. Research use only.\n"
            "Developed by: Burak Demir, MD, FEBNM\n"
            "For support and feedback: 4burakfe@gmail.com\n"
            f"Version: {MODULE_VERSION}"
        )
        self.layout.addWidget(infoTextBox)

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

    def log(self, text):
        self.outputTextBox.append(text)
        slicer.app.processEvents()

    def onCopyResults(self):
        if self._lastCsv:
            qt.QApplication.clipboard().setText(f"{CSV_HEADER}\n{self._lastCsv}")
            slicer.util.showStatusMessage("Results copied to clipboard.", 2000)

    # --- Guardrails ------------------------------------------------------------

    @staticmethod
    def geometryDifferences(node1, node2):
        """Human-readable list of geometry differences (empty if both volumes are on the same grid)."""
        differences = []
        if not np.allclose(node1.GetSpacing(), node2.GetSpacing(), atol=GEOMETRY_TOLERANCE):
            differences.append(f"spacing {tuple(round(v, 4) for v in node1.GetSpacing())} vs "
                               f"{tuple(round(v, 4) for v in node2.GetSpacing())}")
        if not np.allclose(node1.GetOrigin(), node2.GetOrigin(), atol=GEOMETRY_TOLERANCE):
            differences.append("origin differs")
        m1, m2 = vtk.vtkMatrix4x4(), vtk.vtkMatrix4x4()
        node1.GetIJKToRASDirectionMatrix(m1)
        node2.GetIJKToRASDirectionMatrix(m2)
        d1 = np.array([[m1.GetElement(r, c) for c in range(3)] for r in range(3)])
        d2 = np.array([[m2.GetElement(r, c) for c in range(3)] for r in range(3)])
        if not np.allclose(d1, d2, atol=GEOMETRY_TOLERANCE):
            differences.append("axis directions differ")
        if slicer.util.arrayFromVolume(node1).shape != slicer.util.arrayFromVolume(node2).shape:
            differences.append(f"size {slicer.util.arrayFromVolume(node1).shape[::-1]} vs "
                               f"{slicer.util.arrayFromVolume(node2).shape[::-1]}")
        return differences

    @staticmethod
    def gridCoverage(sourceNode, targetNode):
        """Fraction of targetNode's physical box that sourceNode covers (0..1), in world coordinates."""
        s, t = [0.0] * 6, [0.0] * 6
        sourceNode.GetRASBounds(s)
        targetNode.GetRASBounds(t)
        targetVolume = 1.0
        overlap = 1.0
        for axis in range(3):
            lo, hi = max(s[2 * axis], t[2 * axis]), min(s[2 * axis + 1], t[2 * axis + 1])
            overlap *= max(0.0, hi - lo)
            targetVolume *= max(t[2 * axis + 1] - t[2 * axis], 1e-9)
        return overlap / targetVolume

    @staticmethod
    def frameOfReferenceUid(node):
        """DICOM Frame of Reference UID of a volume loaded from DICOM, else None."""
        try:
            uids = (node.GetAttribute("DICOM.instanceUIDs") or "").split()
            if not uids or slicer.dicomDatabase is None:
                return None
            filePath = slicer.dicomDatabase.fileForInstance(uids[0])
            if not filePath:
                return None
            return slicer.dicomDatabase.fileValue(filePath, "0020,0052") or None
        except Exception:
            return None

    @staticmethod
    def _addHiddenVolume(name):
        node = slicer.vtkMRMLScalarVolumeNode()
        node.SetName(name)
        node.SetHideFromEditors(True)
        node.SetSaveWithScene(False)
        return slicer.mrmlScene.AddNode(node)

    def resampleOnto(self, sourceNode, referenceNode):
        """sourceNode resampled onto referenceNode's grid with BRAINSResample (linear). Temporary, hidden node."""
        outputNode = self._addHiddenVolume(f"{sourceNode.GetName()}_resampled")
        cliNode = slicer.cli.createNode(slicer.modules.brainsresample)
        try:
            slicer.cli.runSync(slicer.modules.brainsresample, cliNode,
                               {"inputVolume": sourceNode.GetID(), "referenceVolume": referenceNode.GetID(),
                                "outputVolume": outputNode.GetID(), "pixelType": "float",
                                "interpolationMode": "Linear"},
                               update_display=False)
            if cliNode.GetStatus() & cliNode.ErrorsMask:
                raise RuntimeError(f"BRAINSResample failed: {cliNode.GetErrorText()}")
        except Exception:
            slicer.mrmlScene.RemoveNode(outputNode)
            raise
        finally:
            slicer.mrmlScene.RemoveNode(cliNode)
        return outputNode

    def askGridHandling(self, node1, node2, differences):
        """
        Volumes are not on the same voxel grid. Returns "as_is", "1to2", "2to1", or None (stop).
        Resampling is offered only if both volumes are in the same space and cover the same region.
        """
        if node1.GetTransformNodeID() != node2.GetTransformNodeID():
            self.log("❌ The volumes are under different transforms. Harden the transform (Data module) first.")
            return None
        coverage12 = self.gridCoverage(node1, node2)  # how much of volume 2's box volume 1 fills
        coverage21 = self.gridCoverage(node2, node1)
        if max(coverage12, coverage21) < MIN_COVERAGE:
            self.log(f"❌ The volumes do not cover the same region (overlap {100 * max(coverage12, coverage21):.0f}%). "
                     "Check that they belong to the same study / registration.")
            return None
        sameShape = slicer.util.arrayFromVolume(node1).shape == slicer.util.arrayFromVolume(node2).shape

        esc = lambda text: text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        items = [esc(d) for d in differences]
        for1, for2 = self.frameOfReferenceUid(node1), self.frameOfReferenceUid(node2)
        if for1 and for2 and for1 != for2:
            items.append('<span style="color:#d9534f;"><b>Different DICOM Frame of Reference.</b> The volumes may '
                         "not be registered; resampling will not fix that.</span>")
        items.append("Resampling uses BRAINSResample with linear interpolation. Interpolation smooths the "
                     "resampled volume slightly, which itself changes MSE, PSNR and SSIM.")
        for label, coverage in (("Volume 1 → Volume 2 grid", coverage12), ("Volume 2 → Volume 1 grid", coverage21)):
            if coverage < WARN_COVERAGE:
                items.append(f"{label}: only {100 * coverage:.0f}% of the target grid is covered; the rest is filled "
                             "with 0 and counts in the metrics.")
        if sameShape:
            items.append("<i>Compare as-is</i> compares voxels index by index, ignoring the geometry.")

        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setIcon(qt.QMessageBox.Question)
        box.setWindowTitle("Volume Comparator - different voxel grids")
        box.setTextFormat(qt.Qt.RichText)
        box.setText("<b>The two volumes are not on the same voxel grid.</b><br>"
                    "Resample one onto the other before comparing? The original volumes are not changed.")
        box.setInformativeText("<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>")
        # One distinct role per choice: the clicked button is identified by its role
        roles = {}
        default = None
        if coverage12 >= MIN_COVERAGE:
            default = box.addButton("Resample Volume 1 onto Volume 2", qt.QMessageBox.AcceptRole)
            roles[qt.QMessageBox.AcceptRole] = "1to2"
        if coverage21 >= MIN_COVERAGE:
            button = box.addButton("Resample Volume 2 onto Volume 1", qt.QMessageBox.YesRole)
            roles[qt.QMessageBox.YesRole] = "2to1"
            default = default or button
        if sameShape:
            box.addButton("Compare as-is", qt.QMessageBox.ActionRole)
            roles[qt.QMessageBox.ActionRole] = "as_is"
        cancel = box.addButton(qt.QMessageBox.Cancel)
        box.setDefaultButton(default)
        box.setEscapeButton(cancel)
        box.exec_()
        clicked = box.clickedButton()
        return roles.get(box.buttonRole(clicked)) if clicked is not None else None

    def checkInputs(self):
        """(node1, node2, ssimRange) or raises ValueError with a readable message."""
        node1 = self.inputVolumeSelector.currentNode()
        node2 = self.inputVolumeSelector2.currentNode()
        if node1 is None or node1.GetImageData() is None:
            raise ValueError("Please select a valid first volume.")
        if node2 is None or node2.GetImageData() is None:
            raise ValueError("Please select a valid second volume.")
        if node1 is node2:
            raise ValueError("The same volume is selected twice.")
        ssimRange = None
        if not self.ssimnorm_cbox.checked:
            try:
                ssimRange = float(self.ssimloss_range.text)
            except ValueError:
                raise ValueError(f"Range for SSIM: '{self.ssimloss_range.text}' is not a number.")
            if ssimRange <= 0:
                raise ValueError("Range for SSIM must be greater than 0.")
        return node1, node2, ssimRange

    @staticmethod
    def ensureDependencies():
        """PyTorch must come from PyTorch Utils (user picks the CUDA build); offers to install MONAI."""
        try:
            import torch  # noqa: F401
        except ImportError:
            if slicer.util.confirmOkCancelDisplay(
                    "This module needs PyTorch, which is not installed.\n\n"
                    "Install it in the PyTorch Utils module, where you can choose the CUDA version that matches "
                    "your GPU (or CPU only), then restart Slicer and try again.\n\n"
                    "Open PyTorch Utils now?"):
                try:
                    slicer.util.selectModule("PyTorchUtils")
                except Exception:
                    slicer.util.errorDisplay("The PyTorch Utils module was not found. Install the 'PyTorch' "
                                             "extension from the Extensions Manager and restart Slicer.")
            return False
        try:
            importlib.import_module("monai")
            return True
        except ImportError:
            pass
        if not slicer.util.confirmOkCancelDisplay("The Python package 'monai' is required but not installed.\n"
                                                  "Install it now? You may need to restart Slicer afterwards."):
            return False
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            slicer.util.pip_install("monai")
        except Exception as error:
            slicer.util.errorDisplay(f"Could not install monai:\n{error}")
            return False
        finally:
            qt.QApplication.restoreOverrideCursor()
        importlib.invalidate_caches()
        try:
            importlib.import_module("monai")
        except ImportError:
            slicer.util.errorDisplay("monai was installed but cannot be imported yet. Restart Slicer.")
            return False
        return True

    # --- Run ---------------------------------------------------------------------

    def onCalculateButtonClicked(self):
        if self._running:
            return
        self.outputTextBox.clear()
        try:
            node1, node2, ssimRange = self.checkInputs()
        except ValueError as error:
            self.log(f"❌ {error}")
            return
        choice = "as_is"
        differences = self.geometryDifferences(node1, node2)
        if differences:
            choice = self.askGridHandling(node1, node2, differences)
            if choice is None:
                self.log("Stopped: volumes are not on the same grid.")
                return
            if choice == "as_is":
                self.log("⚠️ Different geometry (" + "; ".join(differences) + "), compared index by index.")
        if not self.ensureDependencies():
            self.log("❌ Required Python packages are missing. Stopping.")
            return

        self._running = True
        self.calculateButton.enabled = False
        self.copyButton.enabled = False
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        scratchNode = None
        try:
            if choice == "1to2":
                self.log(f"📐 Resampling '{node1.GetName()}' onto the grid of '{node2.GetName()}' (BRAINSResample, linear)...")
                scratchNode = node1 = self.resampleOnto(node1, node2)
            elif choice == "2to1":
                self.log(f"📐 Resampling '{node2.GetName()}' onto the grid of '{node1.GetName()}' (BRAINSResample, linear)...")
                scratchNode = node2 = self.resampleOnto(node2, node1)
            self.log("Comparing volumes, please stand by...")
            try:
                self.compare(node1, node2, ssimRange, useGpu=True)
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                self.log("⚠️ GPU ran out of memory, repeating on CPU (this can take a while)...")
                self._freeTorchMemory()
                self.compare(node1, node2, ssimRange, useGpu=False)
            self.copyButton.enabled = True
            self.log("Calculation has been completed, out.")
        except Exception as error:
            logging.exception("VolumeComparator: comparison failed")
            self.log(f"❌ Comparison failed: {error}")
            slicer.util.errorDisplay(f"Comparison failed:\n{error}", detailedText=traceback.format_exc())
        finally:
            if scratchNode is not None and slicer.mrmlScene.IsNodePresent(scratchNode):
                slicer.mrmlScene.RemoveNode(scratchNode)
            qt.QApplication.restoreOverrideCursor()
            self.calculateButton.enabled = True
            self._running = False
            self._freeTorchMemory()

    @staticmethod
    def _freeTorchMemory():
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def chooseDevice(self, torch, useGpu):
        if not useGpu or not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            vramGb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        except Exception as error:
            self.log(f"⚠️ Could not check VRAM: {error}. Using CPU.")
            return torch.device("cpu")
        if vramGb < MIN_VRAM_GB:
            self.log(f"⚠️ CUDA available but only {vramGb:.1f} GB VRAM — using CPU.")
            return torch.device("cpu")
        self.log(f"✅ CUDA available with {vramGb:.1f} GB VRAM — using GPU.")
        return torch.device("cuda")

    def compare(self, node1, node2, ssimRange, useGpu=True):
        """Metric formulas are unchanged from v1.0, so results stay comparable with earlier runs."""
        import torch
        import torch.nn.functional as F
        from monai.losses import SSIMLoss

        device = self.chooseDevice(torch, useGpu)
        a = torch.from_numpy(np.array(slicer.util.arrayFromVolume(node1), dtype=np.float32))[None, None].to(device)
        b = torch.from_numpy(np.array(slicer.util.arrayFromVolume(node2), dtype=np.float32))[None, None].to(device)
        self.log(f"Volume size: {tuple(a.shape[-3:])}")

        def gradient(img):
            dz = F.pad(img[:, :, 1:] - img[:, :, :-1], (0, 0, 0, 0, 0, 1))
            dy = F.pad(img[:, :, :, 1:] - img[:, :, :, :-1], (0, 0, 0, 1, 0, 0))
            dx = F.pad(img[..., 1:] - img[..., :-1], (0, 1, 0, 0, 0, 0))
            return torch.cat([dx, dy, dz], dim=1)

        with torch.no_grad():
            mse = F.mse_loss(a, b).item()
            mae = F.l1_loss(a, b).item()
            edge = F.l1_loss(gradient(a), gradient(b)).item()
            self.log(f"Mean Squared Error is: {mse:.6f}")
            self.log(f"Mean Absolute Error is: {mae:.6f}")
            self.log(f"Edge Loss is: {edge:.6f}")

            maxRef = b.max().item()
            if mse == 0:
                psnr = float("inf")
            elif maxRef <= 0:
                psnr = float("nan")
                self.log("⚠️ PSNR undefined: the reference volume has no positive values.")
            else:
                psnr = 10 * np.log10(maxRef ** 2 / mse)
            self.log(f"Peak SNR is: {psnr:.6f}")

            if ssimRange is None:
                lo = min(a.min().item(), b.min().item())
                hi = max(a.max().item(), b.max().item())
                if hi <= lo:
                    raise ValueError("Both volumes are constant with the same value; SSIM normalization is undefined.")
                a = (a - lo) / (hi - lo)
                b = (b - lo) / (hi - lo)
                self.log(f"SSIM on jointly normalized volumes (range {lo:g} … {hi:g} → 0 … 1).")
                dataRange = 1.0
            else:
                clipped = ((a < 0) | (a > ssimRange) | (b < 0) | (b > ssimRange)).float().mean().item() * 100
                if clipped > 0:
                    self.log(f"⚠️ {clipped:.2f}% of voxels are outside [0, {ssimRange:g}] and are clipped for SSIM.")
                a = torch.clamp(a, 0, ssimRange)
                b = torch.clamp(b, 0, ssimRange)
                dataRange = ssimRange
            ssimLoss = SSIMLoss(spatial_dims=3, data_range=dataRange)(a, b).item()

        self.log(f"Structural Similarity Index is: {1 - ssimLoss:.6f}")
        self.log(f"SSIM Loss (1-SSIM) is: {ssimLoss:.6f}")
        # Same column order as v1.0; Edge_Loss appended at the end
        self._lastCsv = f"{mse:.6f};{mae:.6f};{1 - ssimLoss:.6f};{ssimLoss:.6f};{psnr:.6f};{edge:.6f}"
        self.log(CSV_HEADER)
        self.log(self._lastCsv)
