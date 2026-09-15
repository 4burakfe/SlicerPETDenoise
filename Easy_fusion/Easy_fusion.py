import os
import re
import logging

import numpy as np
import qt
import ctk
import vtk
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


# vtkMRMLViewNode animation modes (Off / Spin)
ANIMATION_OFF = 0
ANIMATION_SPIN = 1

HOT_IRON_NAME = "CustomHotIron"
_HOT_IRON_NAME_PATTERN = re.compile(r"^CustomHotIron([_ ]\d+)?$")

# Bookkeeping stored on MRML nodes so it survives save / reload
ROI_NODE_ATTRIBUTE = "EasyFusion.SUVROIList"
ROI_MODEL_ATTRIBUTE = "EasyFusion.SUVROISpheres"
ROI_RADIUS_ATTRIBUTE = "EasyFusion.RoiRadius_"      # + control point ID
ROI_NUMBER_ATTRIBUTE = "EasyFusion.RoiNumber_"      # + control point ID
ROI_NEXT_NUMBER_ATTRIBUTE = "EasyFusion.RoiNextNumber"
ROI_PET_REFERENCE_ROLE = "EasyFusionPETVolume"
ROI_HANDLES_ATTRIBUTE = "EasyFusion.SUVROIRadiusHandles"
DEFAULT_ROI_RADIUS_MM = 15.0

# Radius handles sit on each sphere along ±R, ±A, ±S, so every slice view through the
# ROI center shows four of them around the circle.
HANDLE_DIRECTIONS = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                     (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
                     (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]

# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------

LAYOUT_FOUR_UP_ID = 3              # Slicer's built-in four-up (vtkMRMLLayoutNode::SlicerLayoutFourUpView)
LAYOUT_AXIAL_FOUR_UP_ID = 7501
LAYOUT_TWO_BY_THREE_ID = 7502
LAYOUT_DUAL_MONITOR_ID = 7503
CUSTOM_LAYOUT_IDS = (LAYOUT_AXIAL_FOUR_UP_ID, LAYOUT_TWO_BY_THREE_ID, LAYOUT_DUAL_MONITOR_ID)
DUAL_MONITOR_WINDOW_TITLE = "EasyFusion - Monitor 2"

# Slice view name -> (orientation, content). Content: fusion = CT + PET overlay,
# ct = CT only, pet = PET only (inverted grey). Red/Yellow/Green keep their usual fusion role.
SLICE_VIEW_ROLES = {
    "Red": ("Axial", "fusion"),
    "Yellow": ("Sagittal", "fusion"),
    "Green": ("Coronal", "fusion"),
    "EFAxialCT": ("Axial", "ct"),
    "EFAxialPET": ("Axial", "pet"),
    "EFSagittalCT": ("Sagittal", "ct"),
    "EFSagittalPET": ("Sagittal", "pet"),
    "EFCoronalCT": ("Coronal", "ct"),
}
_SLICE_VIEW_STYLE = {  # label, color (lighter shades for the extra views, like Slicer's "+" views)
    "Red": ("R", "#F34A33"),
    "Yellow": ("Y", "#EDD54C"),
    "Green": ("G", "#6EB04B"),
    "EFAxialCT": ("A-CT", "#f9a99f"),
    "EFAxialPET": ("A-PET", "#f9a99f"),
    "EFSagittalCT": ("S-CT", "#f6e9a2"),
    "EFSagittalPET": ("S-PET", "#f6e9a2"),
    "EFCoronalCT": ("C-CT", "#c6e0b8"),
}

SETTINGS_NODE_TAG = "EasyFusion"
SETTINGS_PET_ROLE = "EasyFusionPET"
SETTINGS_CT_ROLE = "EasyFusionCT"
PET_ONLY_VOLUME_ATTRIBUTE = "EasyFusion.PETOnlyDisplayVolume"
PET_ONLY_SOURCE_ROLE = "EasyFusionSourcePET"


def _sliceViewItem(name):
    orientation = SLICE_VIEW_ROLES[name][0]
    label, color = _SLICE_VIEW_STYLE[name]
    return (f'<item><view class="vtkMRMLSliceNode" singletontag="{name}">'
            f'<property name="orientation" action="default">{orientation}</property>'
            f'<property name="viewlabel" action="default">{label}</property>'
            f'<property name="viewcolor" action="default">{color}</property>'
            '</view></item>')


_THREED_VIEW_ITEM = ('<item><view class="vtkMRMLViewNode" singletontag="1">'
                     '<property name="viewlabel" action="default">1</property>'
                     '</view></item>')


def _nested(layoutType, items):
    return f'<item><layout type="{layoutType}">' + "".join(items) + '</layout></item>'


def buildLayoutDescriptions():
    """Layout XML for the custom EasyFusion layouts, keyed by layout ID."""
    axialFourUp = (
        '<layout type="vertical">'
        + _nested("horizontal", [_sliceViewItem("Red"), _THREED_VIEW_ITEM])
        + _nested("horizontal", [_sliceViewItem("EFAxialCT"), _sliceViewItem("EFAxialPET")])
        + '</layout>')

    # 3 columns: [axial fusion / sagittal fusion] [axial CT / sagittal CT] [3D spanning both rows]
    twoByThree = (
        '<layout type="horizontal">'
        + _nested("vertical", [_sliceViewItem("Red"), _sliceViewItem("Yellow")])
        + _nested("vertical", [_sliceViewItem("EFAxialCT"), _sliceViewItem("EFSagittalCT")])
        + _THREED_VIEW_ITEM
        + '</layout>')

    # Main window: 3x2 axial/sagittal grid. Second window (drag or auto-placed on monitor 2): 3D + coronal.
    dualMonitor = (
        '<viewports>'
        '<layout type="vertical">'
        + _nested("horizontal", [_sliceViewItem("Red"), _sliceViewItem("EFAxialCT"), _sliceViewItem("EFAxialPET")])
        + _nested("horizontal", [_sliceViewItem("Yellow"), _sliceViewItem("EFSagittalCT"), _sliceViewItem("EFSagittalPET")])
        + '</layout>'
        f'<layout name="EasyFusionMonitor2" type="horizontal" label="{DUAL_MONITOR_WINDOW_TITLE}" dockable="false">'
        + _THREED_VIEW_ITEM + _sliceViewItem("Green") + _sliceViewItem("EFCoronalCT")
        + '</layout>'
        '</viewports>')

    return {
        LAYOUT_AXIAL_FOUR_UP_ID: axialFourUp,
        LAYOUT_TWO_BY_THREE_ID: twoByThree,
        LAYOUT_DUAL_MONITOR_ID: dualMonitor,
    }


def fieldOfViewForTarget(sourceFieldOfView, targetDimensions):
    """Same anatomical width as the source, height following the target view's aspect ratio."""
    width = float(sourceFieldOfView[0])
    if targetDimensions[0] <= 0 or targetDimensions[1] <= 0:
        return None
    return (width, width * float(targetDimensions[1]) / float(targetDimensions[0]), float(sourceFieldOfView[2]))


class Easy_fusion(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Lvgvs - PET/CT Review"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = """
        This module provides easy fusion of SPECT/PET and CT/MR images.
        Spherical ROIs report SUVmax and SUVmean, read directly from the selected PET volume
        (the volume is assumed to already be in SUV units). Press Insert over a slice view to drop
        an ROI at the cursor; drag the yellow edge handles to resize it.
        Layout buttons switch between four-up, axial fusion/CT/PET, 2x3 + 3D and a dual monitor layout
        (the dual monitor layout needs Slicer 5.2 or later).
        """
        parent.acknowledgementText = """
        This file was developed by Burak Demir.
        """
        # **✅ Set the module icon**
        iconPath = os.path.join(os.path.dirname(__file__), "Resources\\Icons\\Easy_fusion.png")
        self.parent.icon = qt.QIcon(iconPath)  # Assign icon to the module
        self.parent = parent

        # Scene-load fixes (Hot Iron repair, no auto-rotation) must work even if the
        # EasyFusion GUI has not been opened yet in this Slicer session.
        slicer.app.connect("startupCompleted()", registerSceneObservers)


# ---------------------------------------------------------------------------
# Scene-level observers (registered once per session)
# ---------------------------------------------------------------------------

_sceneObserverTags = []


def registerSceneObservers():
    """Observe scene loading so saved scenes come back in a consistent state. Safe to call repeatedly."""
    if _sceneObserverTags:
        return
    scene = slicer.mrmlScene
    _sceneObserverTags.append(scene.AddObserver(slicer.vtkMRMLScene.StartImportEvent, _onSceneStartImport))
    _sceneObserverTags.append(scene.AddObserver(slicer.vtkMRMLScene.EndImportEvent, _onSceneEndImport))
    _sceneObserverTags.append(scene.AddObserver(slicer.vtkMRMLScene.EndCloseEvent, _onSceneEndClose))
    _ensureLayoutsRegisteredSafely()


def _ensureLayoutsRegisteredSafely():
    try:
        Easy_fusionLogic.ensureLayoutsRegistered()
    except Exception:
        logging.exception("EasyFusion: could not register layouts")


def _onSceneStartImport(caller, event):
    # A saved scene may use an EasyFusion layout; its description must exist before the layout is restored
    _ensureLayoutsRegisteredSafely()


def _onSceneEndClose(caller, event):
    unbindWindowLevelSync()
    _ensureLayoutsRegisteredSafely()


def _onSceneEndImport(caller, event):
    try:
        logic = Easy_fusionLogic()
        logic.stopAllViewRotations()
        logic.repairHotIronColorNodes()
    except Exception:
        logging.exception("EasyFusion: post-load scene repair failed")
    # Deferred so the restored layout has created its views first
    qt.QTimer.singleShot(0, _restoreViewRolesAfterLoad)


def _restoreViewRolesAfterLoad():
    try:
        Easy_fusionLogic().restoreViewRolesAfterLoad()
    except Exception:
        logging.exception("EasyFusion: could not restore view contents after scene load")


# PET window/level is kept identical between the fusion PET and its PET-only (inverted grey) twin.
_windowLevelSync = {"nodes": None, "tags": [], "busy": False}


def bindWindowLevelSync(displayNodeA, displayNodeB):
    state = _windowLevelSync
    if state["nodes"] is not None and state["nodes"][0] is displayNodeA and state["nodes"][1] is displayNodeB:
        return
    unbindWindowLevelSync()
    state["nodes"] = (displayNodeA, displayNodeB)
    state["tags"] = [
        (displayNodeA, displayNodeA.AddObserver(
            vtk.vtkCommand.ModifiedEvent, lambda caller, event: _syncWindowLevel(displayNodeA, displayNodeB))),
        (displayNodeB, displayNodeB.AddObserver(
            vtk.vtkCommand.ModifiedEvent, lambda caller, event: _syncWindowLevel(displayNodeB, displayNodeA))),
    ]


def unbindWindowLevelSync():
    state = _windowLevelSync
    for node, tag in state["tags"]:
        try:
            node.RemoveObserver(tag)
        except Exception:
            pass
    state["tags"] = []
    state["nodes"] = None


def _syncWindowLevel(source, target):
    state = _windowLevelSync
    if state["busy"]:
        return
    if (abs(source.GetWindow() - target.GetWindow()) < 1e-9
            and abs(source.GetLevel() - target.GetLevel()) < 1e-9):
        return
    state["busy"] = True
    try:
        wasModifying = target.StartModify()
        target.SetAutoWindowLevel(False)
        target.SetWindow(source.GetWindow())
        target.SetLevel(source.GetLevel())
        target.EndModify(wasModifying)
    finally:
        state["busy"] = False


# ---------------------------------------------------------------------------
# Pure helpers (no Slicer dependency, easy to test)
# ---------------------------------------------------------------------------

def hotIronRGB(t):
    """Custom Hot Iron color stops for t in [0, 1]."""
    if t <= 0.5:
        r, g, b = t * 2, 0.0, 0.0
    elif t <= 0.75:
        r, g, b = 1.0, (t - 0.5) * 4, 0.0
    else:
        r, g, b = 1.0, 1.0, (t - 0.75) * 4
    return (min(max(r, 0.0), 1.0), min(max(g, 0.0), 1.0), min(max(b, 0.0), 1.0))


def sphereStatisticsFromArray(voxels, ijkToRas, centerRas, radiusMm):
    """
    Statistics of all voxels whose centers lie inside a sphere.

    voxels:    numpy array indexed [k, j, i] (as returned by slicer.util.arrayFromVolume)
    ijkToRas:  4x4 matrix (numpy) mapping voxel indices to RAS (mm)
    centerRas: sphere center in the volume's RAS coordinate system
    radiusMm:  sphere radius in mm

    Returns dict(max, mean, voxels, volumeMl) or None if the sphere does not touch the volume.
    If the sphere is smaller than a voxel, the voxel containing the center is used.
    """
    voxels = np.asarray(voxels)
    if voxels.ndim > 3:
        voxels = voxels[..., 0]
    matrix = np.asarray(ijkToRas, dtype=float)
    linear, offset = matrix[:3, :3], matrix[:3, 3]
    center = np.asarray(centerRas, dtype=float)
    radius = float(radiusMm)
    dims = np.array(voxels.shape[::-1])  # (i, j, k)

    centerIjk = np.linalg.solve(linear, center - offset)
    # A sphere maps to an axis-aligned ellipsoid in IJK space (orthonormal directions),
    # so a per-axis reach of radius / spacing bounds it exactly.
    voxelSize = np.linalg.norm(linear, axis=0)
    reach = np.ceil(radius / voxelSize).astype(int) + 1
    lo = np.maximum(np.floor(centerIjk).astype(int) - reach, 0)
    hi = np.minimum(np.ceil(centerIjk).astype(int) + reach, dims - 1)

    collected = []
    if np.all(lo <= hi):
        jj, ii = np.meshgrid(np.arange(lo[1], hi[1] + 1), np.arange(lo[0], hi[0] + 1), indexing="ij")
        iFlat, jFlat = ii.ravel(), jj.ravel()
        inPlane = linear[:, 0:1] * iFlat + linear[:, 1:2] * jFlat + offset[:, None]
        radius2 = radius * radius
        for k in range(lo[2], hi[2] + 1):  # slab by slab keeps memory bounded for big spheres
            ras = inPlane + linear[:, 2:3] * k
            inside = np.einsum("ij,ij->j", ras - center[:, None], ras - center[:, None]) <= radius2
            if inside.any():
                collected.append(voxels[k][jFlat[inside], iFlat[inside]])

    if collected:
        values = np.concatenate(collected)
    else:
        nearest = np.round(centerIjk).astype(int)
        if np.all(nearest >= 0) and np.all(nearest < dims):
            values = np.asarray(voxels[nearest[2], nearest[1], nearest[0]]).reshape(1)
        else:
            return None

    voxelVolumeMl = abs(np.linalg.det(linear)) / 1000.0
    return {
        "max": float(values.max()),
        "mean": float(values.mean()),
        "voxels": int(values.size),
        "volumeMl": float(values.size * voxelVolumeMl),
    }


def formatRoiLabel(name, stats, hasPet):
    """Annotation shown next to the ROI in the views (multi-line)."""
    if stats is None:
        return f"{name}\n(outside PET)" if hasPet else f"{name}\n(no PET)"
    return f"{name}\nSUVmax {stats['max']:.2f}\nSUVmean {stats['mean']:.2f}"


def _distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def radiusHandlePosition(center, radius, axis):
    direction = HANDLE_DIRECTIONS[axis]
    return [center[0] + direction[0] * radius,
            center[1] + direction[1] * radius,
            center[2] + direction[2] * radius]


def parseHandleDescription(description):
    """Handle control points store '<roi control point ID>:<axis>' in their description."""
    if not description or ":" not in description:
        return None
    roiID, _, axisText = description.rpartition(":")
    if not roiID or not axisText.isdigit():
        return None
    axis = int(axisText)
    return (roiID, axis) if 0 <= axis < len(HANDLE_DIRECTIONS) else None


def resolveHandleDrag(center, radius, lastGeometry, handles, activeHandleID, tolerance=1e-3):
    """
    Decide whether a radius handle was dragged since the last sync.

    center, radius: current ROI center and stored radius
    lastGeometry:   (center, radius) from the previous sync, or None for a new / reloaded ROI
    handles:        list of (handleID, position, lastPosition or None) for this ROI
    activeHandleID: handle currently grabbed by the mouse, or None

    Returns (radius, draggedHandleID). The returned radius is not clamped.
    If the ROI center moved, handles simply follow it and the radius is unchanged.
    """
    if lastGeometry is None or _distance(lastGeometry[0], center) > tolerance:
        return radius, None
    dragged = None
    for handleID, position, lastPosition in handles:
        if activeHandleID is not None and handleID == activeHandleID:
            dragged = (handleID, position)
            break
        if dragged is None and lastPosition is not None and _distance(lastPosition, position) > tolerance:
            dragged = (handleID, position)
    if dragged is None:
        return radius, None
    return _distance(dragged[1], center), dragged[0]


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class Easy_fusionWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self.observedViewNode = None
        self.roiNode = None
        self.handlesNode = None
        self.placeRoiShortcut = None
        self._updatingRois = False
        self._roiStatsCache = {}
        self._knownRoiIDs = None
        self._lastRoiGeometry = {}       # ROI control point ID -> (center, radius) at last sync
        self._lastHandlePositions = {}   # handle control point ID -> position at last sync
        self._activeHandleID = None      # handle currently being dragged
        self._syncingSlices = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = Easy_fusionLogic()

        # Create collapsible section
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)
        formLayout = qt.QFormLayout(parametersCollapsibleButton)

        # 1️⃣ Input Volume Selector (PET Image)
        self.inputVolumeSelector = slicer.qMRMLNodeComboBox()
        self.inputVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputVolumeSelector.selectNodeUponCreation = True
        self.inputVolumeSelector.addEnabled = False
        self.inputVolumeSelector.removeEnabled = False
        self.inputVolumeSelector.noneEnabled = False
        self.inputVolumeSelector.showHidden = False
        self.inputVolumeSelector.showChildNodeTypes = False
        self.inputVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.inputVolumeSelector.setToolTip("Select the SPECT/PET image for fusion.")
        formLayout.addRow("SPECT/PET: ", self.inputVolumeSelector)

        # 1️⃣ Input Volume Selector (CT Image)
        self.inputVolumeSelectorCT = slicer.qMRMLNodeComboBox()
        self.inputVolumeSelectorCT.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputVolumeSelectorCT.selectNodeUponCreation = True
        self.inputVolumeSelectorCT.addEnabled = False
        self.inputVolumeSelectorCT.removeEnabled = False
        self.inputVolumeSelectorCT.noneEnabled = False
        self.inputVolumeSelectorCT.showHidden = False
        self.inputVolumeSelectorCT.showChildNodeTypes = False
        self.inputVolumeSelectorCT.setMRMLScene(slicer.mrmlScene)
        self.inputVolumeSelectorCT.setToolTip("Select the CT/MR image for fusion.")
        formLayout.addRow("CT/MRI: ", self.inputVolumeSelectorCT)

        # Add dropdown for PET color map
        self.petColorMapSelector = qt.QComboBox()
        self.petColorMapSelector.addItems(["Hot Iron", "Inferno", "Rainbow"])
        formLayout.addRow("PET Color Map:", self.petColorMapSelector)

        # 6️⃣ Fusion Button
        self.FusionButton = qt.QPushButton("Go")
        self.FusionButton.enabled = True
        formLayout.addRow(self.FusionButton)

        # Connect Calculate button to function
        self.FusionButton.connect("clicked(bool)", self.DoFusion)

        # Speed slider
        self.rotationSpeedSlider = ctk.ctkSliderWidget()
        self.rotationSpeedSlider.singleStep = 10
        self.rotationSpeedSlider.minimum = 10
        self.rotationSpeedSlider.maximum = 200
        self.rotationSpeedSlider.value = 50  # Default speed
        self.rotationSpeedSlider.toolTip = "Lower is faster (ms per step)"
        formLayout.addRow("MIP Rotation Speed (ms):", self.rotationSpeedSlider)

        # Toggle button. Its state always mirrors the 3D view node (see updateRotationButton),
        # so it can never get out of sync with what the view is actually doing.
        self.toggleRotationButton = qt.QPushButton("Start MIP Rotation")
        self.toggleRotationButton.checkable = True
        formLayout.addRow(self.toggleRotationButton)

        self.toggleRotationButton.connect('toggled(bool)', self.setRotationEnabled)
        self.rotationSpeedSlider.connect('valueChanged(double)', self.updateRotationSpeed)

        # Orientation buttons layout
        orientationLayout = qt.QHBoxLayout()
        self.orientationAnteriorButton = qt.QPushButton("Anterior")
        self.orientationLeftButton = qt.QPushButton("Left")
        self.orientationRightButton = qt.QPushButton("Right")

        orientationLayout.addWidget(self.orientationAnteriorButton)
        orientationLayout.addWidget(self.orientationLeftButton)
        orientationLayout.addWidget(self.orientationRightButton)
        self.orientationAnteriorButton.connect('clicked()', self.setViewAnterior)
        self.orientationLeftButton.connect('clicked()', self.setViewLeft)
        self.orientationRightButton.connect('clicked()', self.setViewRight)
        formLayout.addRow("Quick View:", orientationLayout)

        # CT windowing buttons
        ctWLLayout = qt.QHBoxLayout()
        self.ctAbdomenBtn = qt.QPushButton("CT: Abdomen")
        self.ctHeadBtn = qt.QPushButton("CT: Head")
        self.ctLungBtn = qt.QPushButton("CT: Lungs")
        self.ctBoneBtn = qt.QPushButton("CT: Bones")

        ctWLLayout.addWidget(self.ctAbdomenBtn)
        ctWLLayout.addWidget(self.ctHeadBtn)
        ctWLLayout.addWidget(self.ctLungBtn)
        ctWLLayout.addWidget(self.ctBoneBtn)
        formLayout.addRow("CT Presets:", ctWLLayout)

        # PET windowing buttons
        petWLLayout = qt.QHBoxLayout()
        self.pet07Btn = qt.QPushButton("PET 0–7")
        self.pet010Btn = qt.QPushButton("PET 0–10")
        self.pet015Btn = qt.QPushButton("PET 0–15")
        self.pet025Btn = qt.QPushButton("PET 0–25")
        petWLLayout.addWidget(self.pet07Btn)
        petWLLayout.addWidget(self.pet010Btn)
        petWLLayout.addWidget(self.pet015Btn)
        petWLLayout.addWidget(self.pet025Btn)
        formLayout.addRow("PET Presets:", petWLLayout)

        self.ctAbdomenBtn.connect('clicked()', lambda: self.setCTWindow(400, 50))
        self.ctHeadBtn.connect('clicked()', lambda: self.setCTWindow(80, 40))
        self.ctLungBtn.connect('clicked()', lambda: self.setCTWindow(1500, -600))
        self.ctBoneBtn.connect('clicked()', lambda: self.setCTWindow(1800, 400))

        self.pet07Btn.connect('clicked()', lambda: self.setPETWindow(7, 3.5))
        self.pet010Btn.connect('clicked()', lambda: self.setPETWindow(10, 5))
        self.pet015Btn.connect('clicked()', lambda: self.setPETWindow(15, 7.5))
        self.pet025Btn.connect('clicked()', lambda: self.setPETWindow(25, 12.5))

        # --- PET Color Map Buttons (Two Rows) ---
        petColorRow1 = qt.QHBoxLayout()
        petColorRow2 = qt.QHBoxLayout()

        # First row
        self.petHotIronBtn = qt.QPushButton("Hot Iron")
        self.petInfernoBtn = qt.QPushButton("Inferno")
        self.petRainbow2Btn = qt.QPushButton("Rainbow-2")

        petColorRow1.addWidget(self.petHotIronBtn)
        petColorRow1.addWidget(self.petInfernoBtn)
        petColorRow1.addWidget(self.petRainbow2Btn)

        # Second row
        self.petRainbow1Btn = qt.QPushButton("PET-DICOM")
        self.petRedBtn = qt.QPushButton("Red")
        self.petHotMetBlue = qt.QPushButton("Hot Metal Blue")

        petColorRow2.addWidget(self.petRainbow1Btn)
        petColorRow2.addWidget(self.petRedBtn)
        petColorRow2.addWidget(self.petHotMetBlue)

        # Add both rows to the form layout
        formLayout.addRow("PET Color Maps:", petColorRow1)
        formLayout.addRow("", petColorRow2)  # no label for second row

        self.petHotIronBtn.connect('clicked()', lambda: self.setPETColorMap(HOT_IRON_NAME))
        self.petInfernoBtn.connect('clicked()', lambda: self.setPETColorMap("Inferno"))
        self.petRainbow2Btn.connect('clicked()', lambda: self.setPETColorMap("PET-Rainbow2"))
        self.petRainbow1Btn.connect('clicked()', lambda: self.setPETColorMap("PET-DICOM"))
        self.petRedBtn.connect('clicked()', lambda: self.setPETColorMap("Red"))
        self.petHotMetBlue.connect('clicked()', lambda: self.setPETColorMap("PET-HotMetalBlue"))

        # 🖥️ Layouts section
        self.setupLayoutSection()

        # 📏 SUV measurement section
        self.setupMeasurementSection()

        self.layout.addStretch(1)

        # **✅ Load the banner image**
        moduleDir = os.path.dirname(__file__)  # Get module directory
        bannerPath = os.path.join(moduleDir, "Resources\\Icons\\fusbanner.jpg")  # Change to your banner file

        if os.path.exists(bannerPath):
            bannerLabel = qt.QLabel()
            bannerPixmap = qt.QPixmap(bannerPath)  # Load image
            bannerLabel.setPixmap(bannerPixmap.scaledToWidth(400, qt.Qt.SmoothTransformation))  # Adjust width

            # **Center the image**
            bannerLabel.setAlignment(qt.Qt.AlignCenter)

            # **Add to layout**
            self.layout.addWidget(bannerLabel)
        else:
            print(f"❌ WARNING: Banner file not found at {bannerPath}")

        # 5️⃣ Info Text Box
        infoTextBox = qt.QTextEdit()
        infoTextBox.setReadOnly(True)  # Make the text box read-only
        infoTextBox.setPlainText(
            "This module provides eased visualization of PET images.\n"
            "This module is NOT a medical device. Research use only.\n"
            "Developed by: Burak Demir, MD, FEBNM \n"
            "For support and feedback: 4burakfe@gmail.com\n"
            "Version: alpha v1.0"
        )
        infoTextBox.setToolTip("Module information and instructions.")  # Add a tooltip for additional help
        self.layout.addWidget(infoTextBox)

        # Observers
        registerSceneObservers()  # in case the module was added after startup
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneEndImport)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAboutToBeRemovedEvent, self.onNodeAboutToBeRemoved)
        self.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onPETVolumeChanged)
        if slicer.app.layoutManager() is not None:
            slicer.app.layoutManager().connect("layoutChanged(int)", self.onLayoutChanged)

        self.observeThreeDViewNode()
        self.restoreSelectorsFromSettings()
        self.connectToExistingRois()
        self.onLayoutChanged()

    def setupLayoutSection(self):
        layoutCollapsibleButton = ctk.ctkCollapsibleButton()
        layoutCollapsibleButton.text = "Layouts"
        self.layout.addWidget(layoutCollapsibleButton)
        layoutFormLayout = qt.QFormLayout(layoutCollapsibleButton)

        buttonGrid = qt.QGridLayout()
        self.layoutButtonGroup = qt.QButtonGroup()
        self.layoutButtonGroup.setExclusive(True)
        self.layoutButtons = {}
        layoutButtonSpecs = [
            (LAYOUT_FOUR_UP_ID, "Four-Up", "Axial, sagittal and coronal fusion + 3D MIP", 0, 0),
            (LAYOUT_AXIAL_FOUR_UP_ID, "Axial Four-Up",
             "Top: axial fusion | 3D MIP\nBottom: axial CT | axial PET (inverted grey)", 0, 1),
            (LAYOUT_TWO_BY_THREE_ID, "2×3 + 3D",
             "Top: axial fusion | axial CT\nBottom: sagittal fusion | sagittal CT\nRight column: 3D MIP", 1, 0),
            (LAYOUT_DUAL_MONITOR_ID, "Dual Monitor",
             "Monitor 1: axial and sagittal fusion | CT | PET (3×2)\n"
             "Monitor 2 (separate window): 3D MIP | coronal fusion | coronal CT\n"
             "Click again to bring the second window back if it was closed.", 1, 1),
        ]
        for layoutID, text, tooltip, row, column in layoutButtonSpecs:
            button = qt.QPushButton(text)
            button.checkable = True
            button.setToolTip(tooltip)
            button.connect('clicked()', lambda layoutID=layoutID: self.setEasyFusionLayout(layoutID))
            self.layoutButtonGroup.addButton(button)
            buttonGrid.addWidget(button, row, column)
            self.layoutButtons[layoutID] = button
        layoutFormLayout.addRow(buttonGrid)

        self.syncSliceViewsCheckBox = qt.QCheckBox("Sync scrolling, panning and zoom of same-orientation views")
        self.syncSliceViewsCheckBox.checked = True
        self.syncSliceViewsCheckBox.setToolTip(
            "Scrolling the axial fusion view also scrolls the axial CT and PET views, and so on.\n"
            "Keep Slicer's own view-link button off in these layouts: it copies volume selections\n"
            "between views and would overwrite the CT-only / PET-only views.")
        layoutFormLayout.addRow(self.syncSliceViewsCheckBox)

    def setupMeasurementSection(self):
        measurementCollapsibleButton = ctk.ctkCollapsibleButton()
        measurementCollapsibleButton.text = "SUV Measurements (spherical ROI)"
        self.layout.addWidget(measurementCollapsibleButton)
        measurementLayout = qt.QFormLayout(measurementCollapsibleButton)

        self.roiRadiusSpinBox = qt.QDoubleSpinBox()
        self.roiRadiusSpinBox.setRange(1.0, 100.0)
        self.roiRadiusSpinBox.setDecimals(1)
        self.roiRadiusSpinBox.setSingleStep(1.0)
        self.roiRadiusSpinBox.setSuffix(" mm")
        self.roiRadiusSpinBox.setValue(DEFAULT_ROI_RADIUS_MM)
        self.roiRadiusSpinBox.setToolTip(
            "Radius used for new ROIs. When an ROI is selected in the table, this changes that ROI's radius.")
        measurementLayout.addRow("ROI radius:", self.roiRadiusSpinBox)

        roiButtonsLayout = qt.QHBoxLayout()
        self.placeRoiButton = qt.QPushButton("Place ROI (Insert)")
        self.placeRoiButton.setToolTip(
            "Click once on a slice view to drop a spherical ROI centered there.\n"
            "Shortcut: hover over a slice view and press Insert to drop an ROI at the mouse cursor.\n"
            "Drag the center point to move the ROI; drag a yellow handle on its edge to resize it.")
        self.deleteRoiButton = qt.QPushButton("Delete Selected")
        self.clearRoisButton = qt.QPushButton("Clear All")
        roiButtonsLayout.addWidget(self.placeRoiButton)
        roiButtonsLayout.addWidget(self.deleteRoiButton)
        roiButtonsLayout.addWidget(self.clearRoisButton)
        measurementLayout.addRow(roiButtonsLayout)

        self.roiPetLabel = qt.QLabel("Measuring on: (no PET selected)")
        measurementLayout.addRow(self.roiPetLabel)

        self.roiTable = qt.QTableWidget()
        self.roiTable.setColumnCount(5)
        self.roiTable.setHorizontalHeaderLabels(["ROI", "Radius (mm)", "SUVmax", "SUVmean", "Volume (mL)"])
        self.roiTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.roiTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.roiTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.roiTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
        self.roiTable.verticalHeader().setVisible(False)
        self.roiTable.setMinimumHeight(140)
        self.roiTable.setToolTip("Select a row to jump the slice views to that ROI.")
        measurementLayout.addRow(self.roiTable)

        self.placeRoiButton.connect('clicked()', self.onPlaceRoi)
        self.deleteRoiButton.connect('clicked()', self.onDeleteSelectedRoi)
        self.clearRoisButton.connect('clicked()', self.onClearRois)
        self.roiRadiusSpinBox.connect('valueChanged(double)', self.onRoiRadiusChanged)
        self.roiTable.connect('itemSelectionChanged()', self.onRoiSelectionChanged)

        # Batch rapid point events (e.g. dragging) into one recomputation
        self.roiUpdateTimer = qt.QTimer()
        self.roiUpdateTimer.setSingleShot(True)
        self.roiUpdateTimer.setInterval(60)
        self.roiUpdateTimer.connect('timeout()', self.updateRois)

        # Insert key: drop an ROI at the mouse cursor (works anywhere in the main window)
        mainWindow = slicer.util.mainWindow()
        if mainWindow is not None:
            self.placeRoiShortcut = qt.QShortcut(qt.QKeySequence(qt.Qt.Key_Insert), mainWindow)
            self.placeRoiShortcut.connect('activated()', self.onPlaceRoiAtCursor)

    def enter(self):
        self.observeThreeDViewNode()
        self.onLayoutChanged()

    def cleanup(self):
        if hasattr(self, "roiUpdateTimer"):
            self.roiUpdateTimer.stop()
        if self.placeRoiShortcut is not None:
            # Otherwise a module reload leaves two Insert shortcuts and Qt fires neither
            self.placeRoiShortcut.setEnabled(False)
            self.placeRoiShortcut.setParent(None)
            self.placeRoiShortcut = None
        if slicer.app.layoutManager() is not None:
            slicer.app.layoutManager().disconnect("layoutChanged(int)", self.onLayoutChanged)
        self.removeObservers()

    # ------------------------------------------------------------------
    # Scene events
    # ------------------------------------------------------------------

    def onSceneEndClose(self, caller=None, event=None):
        self.setRoiNode(None)
        self.setHandlesNode(None)
        self.fillRoiTable([])
        self.observeThreeDViewNode()

    def onSceneEndImport(self, caller=None, event=None):
        self.observeThreeDViewNode()
        self.restoreSelectorsFromSettings()
        self.connectToExistingRois()
        qt.QTimer.singleShot(0, self.onLayoutChanged)  # after the restored layout has built its views

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def onNodeAboutToBeRemoved(self, caller, event, node):
        if node is None:
            return
        if self.roiNode is not None and node.GetID() == self.roiNode.GetID():
            self.setRoiNode(None)
            self.scheduleRoiUpdate()  # removes spheres and handles outside of this scene callback
        elif self.handlesNode is not None and node.GetID() == self.handlesNode.GetID():
            self.setHandlesNode(None)
            self.scheduleRoiUpdate()  # handles get recreated

    def onPETVolumeChanged(self, node=None):
        self.scheduleRoiUpdate()

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def DoFusion(self):
        # Set parameters
        referenceCT = self.inputVolumeSelectorCT.currentNode()
        PETvol = self.inputVolumeSelector.currentNode()
        if PETvol is None or referenceCT is None:
            slicer.util.errorDisplay("Please select both a SPECT/PET and a CT/MRI volume.")
            return
        if PETvol.GetDisplayNode() is None:
            PETvol.CreateDefaultDisplayNodes()
        if referenceCT.GetDisplayNode() is None:
            referenceCT.CreateDefaultDisplayNodes()

        window = 10
        level = 5
        petDisplayNode = PETvol.GetDisplayNode()
        petDisplayNode.SetAutoWindowLevel(False)
        petDisplayNode.SetWindow(window)
        petDisplayNode.SetLevel(level)
        petDisplayNode.SetInterpolate(True)

        # Only one MIP at a time: hide volume rendering of any previously used PET (or any other volume)
        self.logic.showOnlyThisVolumeRendering(PETvol)

        volumeRenderingLogic = slicer.modules.volumerendering.logic()
        MIPdisplayNode = volumeRenderingLogic.CreateDefaultVolumeRenderingNodes(PETvol)
        MIPdisplayNode.SetVisibility(True)
        # Get the associated property node
        propertyNode = MIPdisplayNode.GetVolumePropertyNode()

        if propertyNode:
            scalarOpacity = propertyNode.GetScalarOpacity()
            volumeProperty = propertyNode.GetVolumeProperty()
            colorFunc = volumeProperty.GetRGBTransferFunction(0)
            colorFunc.RemoveAllPoints()
            colorFunc.AddRGBPoint(0, 1.0, 1.0, 1.0)  # Low = white
            colorFunc.AddRGBPoint(10, 0.0, 0.0, 0.0)  # High = black
            # Clear previous function
            scalarOpacity.RemoveAllPoints()

            # Set a flat opacity mapping
            scalarOpacity.AddPoint(0, 1.0)     # intensity 0 → opacity 1.0
            scalarOpacity.AddPoint(10, 1.0)  # intensity max → opacity 1.0

            # Notify Slicer of the update
            propertyNode.Modified()
            MIPdisplayNode.Modified()

        threeDwidg = self.getThreeDWidget()
        if threeDwidg is not None:
            viewNode = threeDwidg.mrmlViewNode()
            wasModifying = viewNode.StartModify()
            viewNode.SetRaycastTechnique(2)   # MIP
            viewNode.SetRenderMode(1)         # orthographic
            viewNode.SetBoxVisible(0)
            viewNode.SetAxisLabelsVisible(0)
            # White background set on the view node itself, so Slicer does not
            # repaint the default gradient every time the view node changes.
            viewNode.SetBackgroundColor(1.0, 1.0, 1.0)
            viewNode.SetBackgroundColor2(1.0, 1.0, 1.0)
            viewNode.SetAnimationMs(int(self.rotationSpeedSlider.value))
            viewNode.SetAnimationMode(ANIMATION_OFF)  # no auto-rotation; the user starts it with the button
            viewNode.EndModify(wasModifying)
            self.observeThreeDViewNode()

            threeDView = threeDwidg.threeDView()
            threeDView.resetFocalPoint()
            threeDView.rotateToViewAxis(3)
            self.fitMIPToView(PETvol)

        referenceCT.GetDisplayNode().SetAndObserveColorNodeID(slicer.util.getNode("Grey").GetID())
        colorMapText = self.petColorMapSelector.currentText
        if colorMapText == "Hot Iron":
            petDisplayNode.SetAndObserveColorNodeID(self.logic.getOrCreateHotIronColorNode().GetID())
        elif colorMapText == "Inferno":
            petDisplayNode.SetAndObserveColorNodeID(slicer.util.getNode("Inferno").GetID())
        elif colorMapText == "Rainbow":
            petDisplayNode.SetAndObserveColorNodeID(slicer.util.getNode("PET-Rainbow2").GetID())

        # **✅ Fill every EasyFusion view: fusion (CT + PET), CT only, PET only (inverted grey)**
        self.logic.rememberVolumes(PETvol, referenceCT)
        changedViews = self.logic.applyViewRoles(PETvol, referenceCT)
        self.afterViewRolesApplied(changedViews)

        self.scheduleRoiUpdate()

    # ------------------------------------------------------------------
    # Layouts
    # ------------------------------------------------------------------

    def setEasyFusionLayout(self, layoutID):
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return
        self.logic.ensureLayoutsRegistered()
        layoutManager.setLayout(layoutID)

        pet = self.inputVolumeSelector.currentNode()
        ct = self.inputVolumeSelectorCT.currentNode()
        changedViews = []
        if pet is not None and ct is not None:
            self.logic.rememberVolumes(pet, ct)
            changedViews = self.logic.applyViewRoles(pet, ct, forceOrientation=True)
        else:
            slicer.util.showStatusMessage("EasyFusion: select the SPECT/PET and CT/MRI volumes to fill the views.", 4000)
        self.afterViewRolesApplied(changedViews)

        if layoutID == LAYOUT_DUAL_MONITOR_ID:
            qt.QTimer.singleShot(300, self.logic.placeSecondaryViewportWindow)
        self.onLayoutChanged()

    def afterViewRolesApplied(self, changedViews):
        self.observeSliceViewsForSync()
        # Views need their final size before fitting / copying zoom, so wait for the layout to settle
        qt.QTimer.singleShot(150, lambda: self.alignSliceViews(changedViews))

    def onLayoutChanged(self, layoutID=None):
        if not hasattr(self, "layoutButtons"):
            return
        layoutManager = slicer.app.layoutManager()
        current = layoutManager.layout if layoutManager is not None else None
        self.layoutButtonGroup.setExclusive(False)
        for buttonLayoutID, button in self.layoutButtons.items():
            button.checked = (buttonLayoutID == current)
        self.layoutButtonGroup.setExclusive(True)
        self.observeSliceViewsForSync()

    def restoreSelectorsFromSettings(self):
        settingsNode = self.logic.getSettingsNode(create=False)
        if settingsNode is None:
            return
        pet = settingsNode.GetNodeReference(SETTINGS_PET_ROLE)
        ct = settingsNode.GetNodeReference(SETTINGS_CT_ROLE)
        if pet is not None:
            self.inputVolumeSelector.setCurrentNode(pet)
        if ct is not None:
            self.inputVolumeSelectorCT.setCurrentNode(ct)

    def _roleSliceWidgets(self, visibleOnly=False):
        layoutManager = slicer.app.layoutManager()
        widgets = []
        if layoutManager is None:
            return widgets
        for name in layoutManager.sliceViewNames():
            if name not in SLICE_VIEW_ROLES:
                continue
            sliceWidget = layoutManager.sliceWidget(name)
            if sliceWidget is None or (visibleOnly and not sliceWidget.visible):
                continue
            widgets.append((name, sliceWidget))
        return widgets

    def observeSliceViewsForSync(self):
        for name, sliceWidget in self._roleSliceWidgets():
            sliceNode = sliceWidget.mrmlSliceNode()
            if not self.hasObserver(sliceNode, vtk.vtkCommand.ModifiedEvent, self.onSliceNodeModified):
                self.addObserver(sliceNode, vtk.vtkCommand.ModifiedEvent, self.onSliceNodeModified)

    def onSliceNodeModified(self, caller, event=None):
        """Mirror user scrolling / panning / zooming to the other visible views with the same orientation."""
        if self._syncingSlices or caller is None or not self.syncSliceViewsCheckBox.checked:
            return
        if not caller.GetInteracting():
            return  # only user interaction; avoids feedback from resizes and programmatic changes
        orientation = self.logic.getSliceOrientation(caller)
        self._syncingSlices = True
        try:
            for name, sliceWidget in self._roleSliceWidgets(visibleOnly=True):
                target = sliceWidget.mrmlSliceNode()
                if target is caller or self.logic.getSliceOrientation(target) != orientation:
                    continue
                self.logic.copySliceGeometry(caller, target)
        finally:
            self._syncingSlices = False

    def alignSliceViews(self, changedViews):
        """Fit views whose CT changed, then give same-orientation views the same position and zoom."""
        groups = {}
        for name, sliceWidget in self._roleSliceWidgets(visibleOnly=True):
            groups.setdefault(self.logic.getSliceOrientation(sliceWidget.mrmlSliceNode()), []).append((name, sliceWidget))
        self._syncingSlices = True
        try:
            for members in groups.values():
                reference = next((m for m in members if SLICE_VIEW_ROLES[m[0]][1] == "fusion"), members[0])
                if any(name in changedViews for name, _ in members):
                    reference[1].sliceLogic().FitSliceToAll()
                for name, sliceWidget in members:
                    if sliceWidget is not reference[1]:
                        self.logic.copySliceGeometry(reference[1].mrmlSliceNode(), sliceWidget.mrmlSliceNode())
        finally:
            self._syncingSlices = False

    def fitMIPToView(self, volumeNode):
        threeDWidget = self.getThreeDWidget()
        if threeDWidget is None or volumeNode is None:
            return
        bounds = [0.0] * 6
        volumeNode.GetRASBounds(bounds)
        height = bounds[5] - bounds[4]
        if height <= 0:
            return
        renderer = threeDWidget.threeDView().renderWindow().GetRenderers().GetFirstRenderer()
        renderer.GetActiveCamera().SetParallelScale(height * 0.6)  # Zoom fit
        threeDWidget.threeDView().forceRender()

    # ------------------------------------------------------------------
    # MIP rotation
    # ------------------------------------------------------------------

    @staticmethod
    def getThreeDWidget():
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None or layoutManager.threeDViewCount < 1:
            return None
        return layoutManager.threeDWidget(0)

    def observeThreeDViewNode(self, updateButton=True):
        """Keep observing the view node of the first 3D view (it can change with layout or scene)."""
        threeDWidget = self.getThreeDWidget()
        viewNode = threeDWidget.mrmlViewNode() if threeDWidget is not None else None
        if viewNode is not self.observedViewNode:
            if self.observedViewNode is not None:
                self.removeObserver(self.observedViewNode, vtk.vtkCommand.ModifiedEvent, self.updateRotationButton)
            self.observedViewNode = viewNode
            if viewNode is not None:
                self.addObserver(viewNode, vtk.vtkCommand.ModifiedEvent, self.updateRotationButton)
        if updateButton:
            self.updateRotationButton()
        return viewNode

    def updateRotationButton(self, caller=None, event=None):
        viewNode = self.observedViewNode
        spinning = viewNode is not None and viewNode.GetAnimationMode() == ANIMATION_SPIN
        wasBlocked = self.toggleRotationButton.blockSignals(True)
        self.toggleRotationButton.checked = spinning
        self.toggleRotationButton.blockSignals(wasBlocked)
        self.toggleRotationButton.text = "Stop MIP Rotation" if spinning else "Start MIP Rotation"

    def setRotationEnabled(self, enabled):
        viewNode = self.observeThreeDViewNode(updateButton=False)
        if viewNode is None:
            self.updateRotationButton()
            return
        wasModifying = viewNode.StartModify()
        if enabled:
            viewNode.SetAnimationMs(int(self.rotationSpeedSlider.value))
            viewNode.SetAnimationMode(ANIMATION_SPIN)
        else:
            viewNode.SetAnimationMode(ANIMATION_OFF)
        viewNode.EndModify(wasModifying)
        self.updateRotationButton()

    def toggleRotation(self):
        viewNode = self.observeThreeDViewNode(updateButton=False)
        spinning = viewNode is not None and viewNode.GetAnimationMode() == ANIMATION_SPIN
        self.setRotationEnabled(not spinning)

    def updateRotationSpeed(self, value):
        viewNode = self.observeThreeDViewNode(updateButton=False)
        if viewNode is not None:
            viewNode.SetAnimationMs(int(value))

    def stopRotationIfActive(self):
        self.setRotationEnabled(False)

    def setViewAnterior(self):
        self.stopRotationIfActive()
        threeDWidget = self.getThreeDWidget()
        if threeDWidget:
            threeDWidget.threeDView().rotateToViewAxis(3)  # 3 = Anterior

    def setViewLeft(self):
        self.stopRotationIfActive()
        threeDWidget = self.getThreeDWidget()
        if threeDWidget:
            threeDWidget.threeDView().rotateToViewAxis(0)  # 0 = Left

    def setViewRight(self):
        self.stopRotationIfActive()
        threeDWidget = self.getThreeDWidget()
        if threeDWidget:
            threeDWidget.threeDView().rotateToViewAxis(1)  # 1 = Right

    # ------------------------------------------------------------------
    # Window / level and color maps
    # ------------------------------------------------------------------

    def setCTWindow(self, window, level):
        ctNode = self.inputVolumeSelectorCT.currentNode()
        if ctNode and ctNode.GetDisplayNode():
            dnode = ctNode.GetDisplayNode()
            dnode.SetAutoWindowLevel(False)
            dnode.SetWindow(window)
            dnode.SetLevel(level)

    def setPETWindow(self, window, level):
        petNode = self.inputVolumeSelector.currentNode()
        if not petNode:
            return
        if petNode.GetDisplayNode():
            dnode = petNode.GetDisplayNode()
            dnode.SetAutoWindowLevel(False)
            dnode.SetWindow(window)
            dnode.SetLevel(level)
        # --- Update Volume Rendering Color Transfer ---
        volumeRenderingLogic = slicer.modules.volumerendering.logic()
        vrDisplayNode = volumeRenderingLogic.GetFirstVolumeRenderingDisplayNode(petNode)
        if vrDisplayNode:
            vrPropNode = vrDisplayNode.GetVolumePropertyNode()
            if vrPropNode:
                volumeProperty = vrPropNode.GetVolumeProperty()
                colorFunc = volumeProperty.GetRGBTransferFunction(0)

                minVal = level - window / 2
                maxVal = level + window / 2

                colorFunc.RemoveAllPoints()
                colorFunc.AddRGBPoint(minVal, 1.0, 1.0, 1.0)  # Low = white
                colorFunc.AddRGBPoint(maxVal, 0.0, 0.0, 0.0)  # High = black

                vrPropNode.Modified()
                vrDisplayNode.Modified()

    def createCustomHotIronColorNode(self):
        # Kept for backward compatibility with any external callers
        return self.logic.getOrCreateHotIronColorNode()

    def setPETColorMap(self, colorNodeName):
        petNode = self.inputVolumeSelector.currentNode()
        if not petNode:
            return

        if colorNodeName == HOT_IRON_NAME:
            colorNode = self.logic.getOrCreateHotIronColorNode()
        else:
            try:
                colorNode = slicer.util.getNode(colorNodeName)
            except Exception:
                slicer.util.errorDisplay(f"Color node '{colorNodeName}' not found.")
                return

        # --- Set 2D display ---
        displayNode = petNode.GetDisplayNode()
        if displayNode:
            displayNode.SetAndObserveColorNodeID(colorNode.GetID())

    # ------------------------------------------------------------------
    # SUV ROI measurement
    # ------------------------------------------------------------------

    @staticmethod
    def _roiNodeEvents():
        eventNames = ["PointAddedEvent", "PointModifiedEvent", "PointRemovedEvent",
                      "PointPositionDefinedEvent", "PointPositionUndefinedEvent"]
        events = []
        for name in eventNames:
            event = getattr(slicer.vtkMRMLMarkupsNode, name, None)
            if event is not None and event not in events:
                events.append(event)
        return events

    def setRoiNode(self, node):
        if node is self.roiNode:
            return
        if self.roiNode is not None:
            for event in self._roiNodeEvents():
                self.removeObserver(self.roiNode, event, self.onRoiNodeModified)
        self.roiNode = node
        self._roiStatsCache = {}
        self._knownRoiIDs = None
        self._lastRoiGeometry = {}
        if node is not None:
            for event in self._roiNodeEvents():
                self.addObserver(node, event, self.onRoiNodeModified)

    def setHandlesNode(self, node):
        if node is self.handlesNode:
            return
        startEvent = getattr(slicer.vtkMRMLMarkupsNode, "PointStartInteractionEvent", None)
        endEvent = getattr(slicer.vtkMRMLMarkupsNode, "PointEndInteractionEvent", None)
        if self.handlesNode is not None:
            for event in self._roiNodeEvents():
                self.removeObserver(self.handlesNode, event, self.onRoiNodeModified)
            if startEvent is not None:
                self.removeObserver(self.handlesNode, startEvent, self.onHandleInteractionStarted)
            if endEvent is not None:
                self.removeObserver(self.handlesNode, endEvent, self.onHandleInteractionEnded)
        self.handlesNode = node
        self._lastHandlePositions = {}
        self._activeHandleID = None
        if node is not None:
            for event in self._roiNodeEvents():
                self.addObserver(node, event, self.onRoiNodeModified)
            if startEvent is not None:
                self.addObserver(node, startEvent, self.onHandleInteractionStarted)
            if endEvent is not None:
                self.addObserver(node, endEvent, self.onHandleInteractionEnded)

    @vtk.calldata_type(vtk.VTK_INT)
    def onHandleInteractionStarted(self, caller, event, index=None):
        node = self.handlesNode
        if node is None:
            return
        if not isinstance(index, int) or index < 0:
            displayNode = node.GetDisplayNode()
            index = displayNode.GetActiveControlPoint() if displayNode is not None else -1
        if 0 <= index < node.GetNumberOfControlPoints():
            self._activeHandleID = node.GetNthControlPointID(index)

    @vtk.calldata_type(vtk.VTK_INT)
    def onHandleInteractionEnded(self, caller, event, index=None):
        self._activeHandleID = None
        self.scheduleRoiUpdate()  # snap the released handle back onto its axis

    def connectToExistingRois(self):
        node = self.logic.findRoiNode()
        if node is not None:
            # Measure on the same PET the ROIs were last measured on (node references survive save/load)
            pet = node.GetNodeReference(ROI_PET_REFERENCE_ROLE)
            if pet is not None and pet.IsA("vtkMRMLScalarVolumeNode"):
                self.inputVolumeSelector.setCurrentNode(pet)
        self.setRoiNode(node)
        self.setHandlesNode(self.logic.findRoiHandlesNode())
        self.scheduleRoiUpdate()

    def onRoiNodeModified(self, caller=None, event=None):
        if not self._updatingRois:
            self.scheduleRoiUpdate()

    def scheduleRoiUpdate(self):
        if hasattr(self, "roiUpdateTimer"):
            self.roiUpdateTimer.start()

    def updateRois(self):
        if self._updatingRois:
            return
        scene = slicer.mrmlScene
        if scene.IsImporting() or scene.IsClosing():
            return  # scene event handlers trigger a refresh when done

        node = self.roiNode
        if node is not None and not scene.IsNodePresent(node):
            self.setRoiNode(None)
            node = None

        pet = self.inputVolumeSelector.currentNode()
        self.roiPetLabel.text = f"Measuring on: {pet.GetName()}" if pet else "Measuring on: (no PET selected)"

        if node is None:
            self.logic.removeRoiSphereModel()
            self.setHandlesNode(None)
            self.logic.removeRoiHandlesNode()
            self.fillRoiTable([])
            return

        selectedID = self.selectedRoiPointID()
        rows, spheres, newCache = [], [], {}
        draggedRoiID = None
        self._updatingRois = True
        try:
            rois = []
            for index in range(node.GetNumberOfControlPoints()):
                if not self.logic.isControlPointDefined(node, index):
                    continue  # e.g. the preview point that follows the mouse in place mode
                pointID = node.GetNthControlPointID(index)
                center = [0.0, 0.0, 0.0]
                node.GetNthControlPointPositionWorld(index, center)
                rois.append((index, pointID, center))

            # May change radii (handle dragged), so it runs before statistics
            draggedRoiID = self.syncRadiusHandles(node, rois)

            for index, pointID, center in rois:
                radius = self.logic.getRoiRadius(node, pointID, self.roiRadiusSpinBox.value)
                number = self.logic.getRoiNumber(node, pointID)

                key = self.logic.statsCacheKey(pet, center, radius)
                if key in self._roiStatsCache:
                    stats = self._roiStatsCache[key]
                else:
                    stats = self.logic.computeSphereStatistics(pet, center, radius)
                newCache[key] = stats

                name = f"ROI-{number}"
                label = formatRoiLabel(name, stats, pet is not None)
                if node.GetNthControlPointLabel(index) != label:
                    node.SetNthControlPointLabel(index, label)

                rows.append((pointID, name, radius, stats))
                spheres.append((center, radius))

            if pet is not None and rows and node.GetNodeReferenceID(ROI_PET_REFERENCE_ROLE) != pet.GetID():
                node.SetNodeReferenceID(ROI_PET_REFERENCE_ROLE, pet.GetID())
        finally:
            self._updatingRois = False

        self._roiStatsCache = newCache

        # Select a freshly placed ROI, or the one being resized, so the radius box follows it
        currentIDs = {row[0] for row in rows}
        newIDs = currentIDs - self._knownRoiIDs if self._knownRoiIDs is not None else set()
        if len(newIDs) == 1:
            selectedID = next(iter(newIDs))
        elif draggedRoiID is not None:
            selectedID = draggedRoiID
        self._knownRoiIDs = currentIDs

        self.logic.updateRoiSphereModel(spheres)
        self.fillRoiTable(rows, selectedID)
        self.syncRadiusSpinBox(rows, selectedID)

    def syncRadiusHandles(self, roiNode, rois):
        """
        Keep six draggable handles on each ROI sphere (see HANDLE_DIRECTIONS).
        Dragging a handle sets the radius to its distance from the ROI center.
        Returns the ID of the ROI resized by a handle in this sync, if any.
        """
        handlesNode = self.handlesNode
        if handlesNode is not None and not slicer.mrmlScene.IsNodePresent(handlesNode):
            self.setHandlesNode(None)
            handlesNode = None
        if handlesNode is None:
            handlesNode = self.logic.findRoiHandlesNode()
            if handlesNode is None and rois:
                handlesNode = self.logic.createRoiHandlesNode()
            self.setHandlesNode(handlesNode)
        if handlesNode is None:
            self._lastRoiGeometry = {}
            return None

        # Drop handles of deleted ROIs, stray points and duplicates
        roiIDs = {pointID for _, pointID, _ in rois}
        seen = set()
        for i in reversed(range(handlesNode.GetNumberOfControlPoints())):
            key = parseHandleDescription(handlesNode.GetNthControlPointDescription(i))
            if key is None or key[0] not in roiIDs or key in seen:
                handlesNode.RemoveNthControlPoint(i)
            else:
                seen.add(key)

        handleIndexByKey = {}
        for i in range(handlesNode.GetNumberOfControlPoints()):
            key = parseHandleDescription(handlesNode.GetNthControlPointDescription(i))
            if key is not None:
                handleIndexByKey[key] = i

        minRadius = float(self.roiRadiusSpinBox.minimum)
        maxRadius = float(self.roiRadiusSpinBox.maximum)
        draggedRoiID = None
        newGeometry, newHandlePositions = {}, {}

        for _, roiID, center in rois:
            radius = self.logic.getRoiRadius(roiNode, roiID, self.roiRadiusSpinBox.value)

            handles = []
            for axis in range(len(HANDLE_DIRECTIONS)):
                index = handleIndexByKey.get((roiID, axis))
                if index is None:
                    continue
                handleID = handlesNode.GetNthControlPointID(index)
                position = [0.0, 0.0, 0.0]
                handlesNode.GetNthControlPointPositionWorld(index, position)
                handles.append((handleID, position, self._lastHandlePositions.get(handleID)))

            draggedRadius, draggedHandleID = resolveHandleDrag(
                center, radius, self._lastRoiGeometry.get(roiID), handles, self._activeHandleID)
            if draggedHandleID is not None:
                draggedRadius = round(min(max(draggedRadius, minRadius), maxRadius), 1)
                if abs(draggedRadius - radius) > 1e-6:
                    self.logic.setRoiRadius(roiNode, roiID, draggedRadius)
                    radius = draggedRadius
                draggedRoiID = roiID

            for axis in range(len(HANDLE_DIRECTIONS)):
                target = radiusHandlePosition(center, radius, axis)
                index = handleIndexByKey.get((roiID, axis))
                if index is None:
                    index = handlesNode.AddControlPoint(target)
                    handlesNode.SetNthControlPointLabel(index, "")
                    handlesNode.SetNthControlPointDescription(index, f"{roiID}:{axis}")
                    handleIndexByKey[(roiID, axis)] = index
                handleID = handlesNode.GetNthControlPointID(index)
                position = [0.0, 0.0, 0.0]
                handlesNode.GetNthControlPointPositionWorld(index, position)
                # Never move the handle under the mouse; it snaps onto its axis when released
                if handleID != self._activeHandleID and _distance(position, target) > 1e-3:
                    handlesNode.SetNthControlPointPositionWorld(index, target[0], target[1], target[2])
                    position = target
                newHandlePositions[handleID] = tuple(position)

            newGeometry[roiID] = (tuple(center), radius)

        self._lastRoiGeometry = newGeometry
        self._lastHandlePositions = newHandlePositions
        return draggedRoiID

    def syncRadiusSpinBox(self, rows, selectedID):
        if selectedID is None:
            return
        for pointID, _, radius, _ in rows:
            if pointID == selectedID:
                if abs(self.roiRadiusSpinBox.value - radius) > 1e-6:
                    wasBlocked = self.roiRadiusSpinBox.blockSignals(True)
                    self.roiRadiusSpinBox.setValue(radius)
                    self.roiRadiusSpinBox.blockSignals(wasBlocked)
                return

    def fillRoiTable(self, rows, selectPointID=None):
        if not hasattr(self, "roiTable"):
            return
        table = self.roiTable
        wasBlocked = table.blockSignals(True)
        try:
            table.setRowCount(len(rows))
            for rowIndex, (pointID, name, radius, stats) in enumerate(rows):
                values = [name, f"{radius:.1f}", "-", "-", "-"]
                if stats is not None:
                    values[2:] = [f"{stats['max']:.2f}", f"{stats['mean']:.2f}", f"{stats['volumeMl']:.2f}"]
                for column, text in enumerate(values):
                    item = qt.QTableWidgetItem(text)
                    if column == 0:
                        item.setData(qt.Qt.UserRole, pointID)
                    table.setItem(rowIndex, column, item)
            table.clearSelection()
            if selectPointID is not None:
                for rowIndex, row in enumerate(rows):
                    if row[0] == selectPointID:
                        table.selectRow(rowIndex)
                        break
        finally:
            table.blockSignals(wasBlocked)

    def selectedRoiPointID(self):
        if not hasattr(self, "roiTable"):
            return None
        selectedRows = self.roiTable.selectionModel().selectedRows()
        if not selectedRows:
            return None
        item = self.roiTable.item(selectedRows[0].row(), 0)
        return item.data(qt.Qt.UserRole) if item is not None else None

    def onRoiSelectionChanged(self):
        node = self.roiNode
        pointID = self.selectedRoiPointID()
        if node is None or pointID is None:
            return
        radius = self.logic.getRoiRadius(node, pointID, self.roiRadiusSpinBox.value)
        wasBlocked = self.roiRadiusSpinBox.blockSignals(True)
        self.roiRadiusSpinBox.setValue(radius)
        self.roiRadiusSpinBox.blockSignals(wasBlocked)
        index = node.GetNthControlPointIndexByID(pointID)
        if index >= 0:
            slicer.modules.markups.logic().JumpSlicesToNthPointInMarkup(node.GetID(), index, True)

    def onRoiRadiusChanged(self, value):
        node = self.roiNode
        pointID = self.selectedRoiPointID()
        if node is None or pointID is None:
            return  # no selection: value only applies to new ROIs
        self.logic.setRoiRadius(node, pointID, value)
        self.scheduleRoiUpdate()

    def prepareRoiNodeForPlacement(self):
        node = self.roiNode
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            node = self.logic.findRoiNode() or self.logic.createRoiNode()
        self.setRoiNode(node)
        node.SetLocked(False)
        if node.GetDisplayNode() is not None:
            node.GetDisplayNode().SetVisibility(True)
        # New ROI takes the radius box value, not the radius of whatever row was selected
        self.roiTable.clearSelection()
        return node

    def onPlaceRoi(self):
        if self.inputVolumeSelector.currentNode() is None:
            slicer.util.warningDisplay("Select a SPECT/PET volume first.")
            return
        node = self.prepareRoiNodeForPlacement()

        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selectionNode.SetActivePlaceNodeID(node.GetID())
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetPlaceModePersistence(0)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

    def onPlaceRoiAtCursor(self):
        """Insert key: drop an ROI centered at the mouse position in the slice view under the cursor."""
        if self.inputVolumeSelector.currentNode() is None:
            slicer.util.showStatusMessage("EasyFusion: select a SPECT/PET volume first.", 3000)
            return
        crosshairNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLCrosshairNode")
        ras = [0.0, 0.0, 0.0]
        xyz = [0.0, 0.0, 0.0]
        insideView = crosshairNode is not None and crosshairNode.GetCursorPositionRAS(ras)
        # GetCursorPositionXYZ returns the slice node only when the cursor is over a slice view
        sliceNode = crosshairNode.GetCursorPositionXYZ(xyz) if insideView else None
        if not insideView or sliceNode is None:
            slicer.util.showStatusMessage("EasyFusion: hover the mouse over a slice view, then press Insert.", 3000)
            return

        node = self.prepareRoiNodeForPlacement()

        # If "Place ROI" was clicked earlier, leave place mode so a second ROI doesn't follow the mouse
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        if (interactionNode.GetCurrentInteractionMode() == interactionNode.Place
                and selectionNode.GetActivePlaceNodeID() == node.GetID()):
            interactionNode.SwitchToViewTransformMode()

        node.AddControlPoint(ras)

    def onDeleteSelectedRoi(self):
        node = self.roiNode
        pointID = self.selectedRoiPointID()
        if node is None or pointID is None:
            return
        index = node.GetNthControlPointIndexByID(pointID)
        if index >= 0:
            node.RemoveNthControlPoint(index)
        self.logic.forgetRoi(node, pointID)
        self.scheduleRoiUpdate()

    def onClearRois(self):
        node = self.roiNode
        if node is None or node.GetNumberOfControlPoints() == 0:
            return
        if not slicer.util.confirmOkCancelDisplay("Remove all SUV ROIs?"):
            return
        node.RemoveAllControlPoints()
        self.logic.forgetAllRois(node)
        self.scheduleRoiUpdate()


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

class Easy_fusionLogic(ScriptedLoadableModuleLogic):

    # --- Views -------------------------------------------------------------

    @staticmethod
    def stopAllViewRotations():
        for viewNode in slicer.util.getNodesByClass("vtkMRMLViewNode"):
            if viewNode.GetAnimationMode() != ANIMATION_OFF:
                viewNode.SetAnimationMode(ANIMATION_OFF)

    @staticmethod
    def showOnlyThisVolumeRendering(volumeNode):
        for vrDisplayNode in slicer.util.getNodesByClass("vtkMRMLVolumeRenderingDisplayNode"):
            if vrDisplayNode.GetVolumeNodeID() != volumeNode.GetID() and vrDisplayNode.GetVisibility():
                vrDisplayNode.SetVisibility(False)

    # --- Layouts and view contents ------------------------------------------

    @staticmethod
    def ensureLayoutsRegistered():
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return
        layoutNode = layoutManager.layoutLogic().GetLayoutNode()
        for layoutID, description in buildLayoutDescriptions().items():
            if layoutNode.IsLayoutDescription(layoutID):
                if layoutNode.GetLayoutDescription(layoutID) != description:
                    layoutNode.SetLayoutDescription(layoutID, description)
            else:
                layoutNode.AddLayoutDescription(layoutID, description)

    @staticmethod
    def getSettingsNode(create=True):
        """Saved with the scene: remembers which PET and CT the views were built from."""
        node = slicer.mrmlScene.GetSingletonNode(SETTINGS_NODE_TAG, "vtkMRMLScriptedModuleNode")
        if node is None and create:
            node = slicer.vtkMRMLScriptedModuleNode()
            node.SetSingletonTag(SETTINGS_NODE_TAG)
            node.SetName("EasyFusion")
            node.SetAttribute("ModuleName", "Easy_fusion")
            node.SetHideFromEditors(True)
            node = slicer.mrmlScene.AddNode(node)
        return node

    def rememberVolumes(self, petNode, ctNode):
        settingsNode = self.getSettingsNode()
        settingsNode.SetNodeReferenceID(SETTINGS_PET_ROLE, petNode.GetID() if petNode else None)
        settingsNode.SetNodeReferenceID(SETTINGS_CT_ROLE, ctNode.GetID() if ctNode else None)

    @staticmethod
    def getSliceOrientation(sliceNode):
        if hasattr(sliceNode, "GetOrientation"):
            return sliceNode.GetOrientation()
        return sliceNode.GetOrientationString()

    @staticmethod
    def setSliceOrientation(sliceNode, orientation):
        if hasattr(sliceNode, "SetOrientation"):
            sliceNode.SetOrientation(orientation)
        else:
            getattr(sliceNode, f"SetOrientationTo{orientation}")()

    @staticmethod
    def copySliceGeometry(source, target):
        """Same slice position, pan and zoom (anatomical width), keeping the target's aspect ratio."""
        wasModifying = target.StartModify()
        target.GetSliceToRAS().DeepCopy(source.GetSliceToRAS())
        target.SetXYZOrigin(*source.GetXYZOrigin())
        fieldOfView = fieldOfViewForTarget(source.GetFieldOfView(), target.GetDimensions())
        if fieldOfView is not None:
            target.SetFieldOfView(*fieldOfView)
        target.UpdateMatrices()
        target.EndModify(wasModifying)

    @staticmethod
    def findPetOnlyVolume():
        for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
            if node.GetAttribute(PET_ONLY_VOLUME_ATTRIBUTE):
                return node
        return None

    def getOrCreatePetOnlyVolume(self, petNode):
        """
        A volume can only have one color map, so PET-only views show a hidden twin volume that
        shares the PET's voxel data (no copy) but has its own inverted-grey display.
        It is not saved; it gets rebuilt after loading a scene.
        """
        if petNode is None or petNode.GetImageData() is None:
            return None
        node = self.findPetOnlyVolume()
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            node.SetAttribute(PET_ONLY_VOLUME_ATTRIBUTE, "1")
            node.SetHideFromEditors(True)
            node.SetSaveWithScene(False)
        node.SetName(f"{petNode.GetName()} (PET only)")
        node.CopyOrientation(petNode)
        if node.GetImageData() is not petNode.GetImageData():
            node.SetAndObserveImageData(petNode.GetImageData())
        if node.GetTransformNodeID() != petNode.GetTransformNodeID():
            node.SetAndObserveTransformNodeID(petNode.GetTransformNodeID())
        node.SetNodeReferenceID(PET_ONLY_SOURCE_ROLE, petNode.GetID())

        if node.GetDisplayNode() is None:
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetSaveWithScene(False)
        displayNode = node.GetDisplayNode()
        petDisplayNode = petNode.GetDisplayNode()
        wasModifying = displayNode.StartModify()
        displayNode.SetAndObserveColorNodeID(slicer.util.getNode("InvertedGrey").GetID())
        if petDisplayNode is not None:
            displayNode.SetAutoWindowLevel(False)
            displayNode.SetWindow(petDisplayNode.GetWindow())
            displayNode.SetLevel(petDisplayNode.GetLevel())
            displayNode.SetInterpolate(petDisplayNode.GetInterpolate())
        displayNode.EndModify(wasModifying)
        if petDisplayNode is not None:
            bindWindowLevelSync(petDisplayNode, displayNode)
        return node

    def applyViewRoles(self, petNode, ctNode, forceOrientation=False):
        """
        Fill every existing EasyFusion slice view according to SLICE_VIEW_ROLES.
        Returns the names of views whose background volume changed (those get re-fitted).
        """
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None or petNode is None or ctNode is None:
            return []
        petOnlyNode = self.getOrCreatePetOnlyVolume(petNode)

        roleWidgets = []
        for name in layoutManager.sliceViewNames():
            if name in SLICE_VIEW_ROLES and layoutManager.sliceWidget(name) is not None:
                roleWidgets.append((name, layoutManager.sliceWidget(name)))

        # Unlink first: linked views copy volume selection to each other and would undo the assignment
        for _, sliceWidget in roleWidgets:
            compositeNode = sliceWidget.mrmlSliceCompositeNode()
            if compositeNode.GetLinkedControl():
                compositeNode.SetLinkedControl(False)

        changedViews = []
        for name, sliceWidget in roleWidgets:
            orientation, content = SLICE_VIEW_ROLES[name]
            sliceNode = sliceWidget.mrmlSliceNode()
            if forceOrientation and self.getSliceOrientation(sliceNode) != orientation:
                self.setSliceOrientation(sliceNode, orientation)
                changedViews.append(name)

            if content == "fusion":
                background, foreground = ctNode, petNode
            elif content == "ct":
                background, foreground = ctNode, None
            else:
                background, foreground = petOnlyNode, None
            if background is None:
                continue

            compositeNode = sliceWidget.mrmlSliceCompositeNode()
            wasModifying = compositeNode.StartModify()
            if compositeNode.GetBackgroundVolumeID() != background.GetID():
                compositeNode.SetBackgroundVolumeID(background.GetID())
                changedViews.append(name)
            foregroundID = foreground.GetID() if foreground is not None else None
            if compositeNode.GetForegroundVolumeID() != foregroundID:
                compositeNode.SetForegroundVolumeID(foregroundID)
                if foreground is not None:
                    # Only when PET is newly placed, so a user-adjusted opacity is kept
                    compositeNode.SetForegroundOpacity(0.5)
            compositeNode.EndModify(wasModifying)
        return changedViews

    def restoreViewRolesAfterLoad(self):
        """PET-only views reference the unsaved twin volume, so rebuild them when a saved EasyFusion layout is loaded."""
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None or layoutManager.layout not in CUSTOM_LAYOUT_IDS:
            return
        settingsNode = self.getSettingsNode(create=False)
        if settingsNode is None:
            return
        pet = settingsNode.GetNodeReference(SETTINGS_PET_ROLE)
        ct = settingsNode.GetNodeReference(SETTINGS_CT_ROLE)
        if pet is not None and ct is not None:
            self.applyViewRoles(pet, ct)

    @staticmethod
    def placeSecondaryViewportWindow():
        """Best effort: show the Monitor 2 window maximized on a screen other than the main window's."""
        try:
            window = None
            for widget in qt.QApplication.topLevelWidgets():
                if widget.windowTitle == DUAL_MONITOR_WINDOW_TITLE:
                    window = widget
                    break
            if window is None:
                return
            window.show()
            mainWindow = slicer.util.mainWindow()
            screens = list(qt.QGuiApplication.screens())
            if mainWindow is None or len(screens) < 2:
                window.raise_()
                return
            mainCenter = mainWindow.frameGeometry.center()
            otherScreen = next((screen for screen in screens if not screen.geometry.contains(mainCenter)), None)
            if otherScreen is not None:
                window.showNormal()
                window.setGeometry(otherScreen.availableGeometry)
                window.showMaximized()
            window.raise_()
        except Exception:
            logging.exception("EasyFusion: could not place the Monitor 2 window; drag it to the second screen manually")

    # --- Custom Hot Iron ---------------------------------------------------

    @staticmethod
    def fillHotIronColorTable(colorNode):
        """(Re)write the table contents. Idempotent, so it also repairs a table that came back broken from disk."""
        wasModifying = colorNode.StartModify()
        try:
            colorNode.SetTypeToUser()
            colorNode.SetNumberOfColors(256)
            if hasattr(colorNode, "SetNamesInitialised"):
                colorNode.SetNamesInitialised(True)
            for i in range(256):
                r, g, b = hotIronRGB(i / 255.0)
                colorNode.SetColor(i, f"Color{i}", r, g, b, 1.0)
            colorNode.GetLookupTable().SetTableRange(0, 255)
        finally:
            colorNode.EndModify(wasModifying)

    def getOrCreateHotIronColorNode(self):
        colorNode = None
        for node in slicer.util.getNodesByClass("vtkMRMLColorTableNode"):
            if node.GetName() == HOT_IRON_NAME:
                colorNode = node
                break
        if colorNode is None:
            colorNode = slicer.vtkMRMLColorTableNode()
            colorNode.SetName(HOT_IRON_NAME)
            self.fillHotIronColorTable(colorNode)
            slicer.mrmlScene.AddNode(colorNode)  # added exactly once
        else:
            self.fillHotIronColorTable(colorNode)
        return colorNode

    def repairHotIronColorNodes(self):
        """After a scene load: rebuild the Hot Iron table, merge duplicates, and refresh every display using it."""
        candidates = [node for node in slicer.util.getNodesByClass("vtkMRMLColorTableNode")
                      if _HOT_IRON_NAME_PATTERN.match(node.GetName() or "")]
        if not candidates:
            return
        canonical = next((node for node in candidates if node.GetName() == HOT_IRON_NAME), candidates[0])
        canonical.SetName(HOT_IRON_NAME)
        self.fillHotIronColorTable(canonical)

        candidateIDs = {node.GetID() for node in candidates}
        for displayNode in slicer.util.getNodesByClass("vtkMRMLDisplayNode"):
            if displayNode.GetColorNodeID() in candidateIDs:
                # Re-assign (not just "same ID") so the display re-fetches the lookup table object
                wasModifying = displayNode.StartModify()
                displayNode.SetAndObserveColorNodeID(None)
                displayNode.SetAndObserveColorNodeID(canonical.GetID())
                displayNode.EndModify(wasModifying)

        for node in candidates:
            if node is not canonical:
                slicer.mrmlScene.RemoveNode(node)

    # --- SUV ROIs ----------------------------------------------------------

    @staticmethod
    def findRoiNode():
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode"):
            if node.GetAttribute(ROI_NODE_ATTRIBUTE):
                return node
        return None

    def createRoiNode(self):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "SUV ROIs")
        node.SetAttribute(ROI_NODE_ATTRIBUTE, "1")
        node.CreateDefaultDisplayNodes()
        self.styleRoiDisplayNode(node.GetDisplayNode())
        return node

    @staticmethod
    def styleRoiDisplayNode(displayNode):
        if displayNode is None:
            return
        displayNode.SetSelectedColor(0.1, 0.9, 0.3)
        displayNode.SetColor(0.1, 0.9, 0.3)
        if hasattr(displayNode, "SetPropertiesLabelVisibility"):
            displayNode.SetPropertiesLabelVisibility(False)
        if hasattr(displayNode, "SetPointLabelsVisibility"):
            displayNode.SetPointLabelsVisibility(True)
        crossDot = getattr(slicer.vtkMRMLMarkupsDisplayNode, "CrossDot2D", None)
        if crossDot is not None:
            displayNode.SetGlyphType(crossDot)
        textProperty = displayNode.GetTextProperty() if hasattr(displayNode, "GetTextProperty") else None
        if textProperty is not None:
            textProperty.SetBackgroundColor(0.0, 0.0, 0.0)
            textProperty.SetBackgroundOpacity(0.5)

    @staticmethod
    def isControlPointDefined(node, index):
        definedStatus = getattr(slicer.vtkMRMLMarkupsNode, "PositionDefined", None)
        if definedStatus is None or not hasattr(node, "GetNthControlPointPositionStatus"):
            return True
        return node.GetNthControlPointPositionStatus(index) == definedStatus

    @staticmethod
    def getRoiRadius(node, pointID, defaultRadius):
        value = node.GetAttribute(ROI_RADIUS_ATTRIBUTE + pointID)
        try:
            radius = float(value)
            if radius > 0:
                return radius
        except (TypeError, ValueError):
            pass
        radius = float(defaultRadius)
        node.SetAttribute(ROI_RADIUS_ATTRIBUTE + pointID, f"{radius:g}")
        return radius

    @staticmethod
    def setRoiRadius(node, pointID, radius):
        node.SetAttribute(ROI_RADIUS_ATTRIBUTE + pointID, f"{float(radius):g}")

    @staticmethod
    def getRoiNumber(node, pointID):
        value = node.GetAttribute(ROI_NUMBER_ATTRIBUTE + pointID)
        if value and value.isdigit():
            return int(value)
        nextValue = node.GetAttribute(ROI_NEXT_NUMBER_ATTRIBUTE)
        number = int(nextValue) if (nextValue and nextValue.isdigit()) else 1
        node.SetAttribute(ROI_NUMBER_ATTRIBUTE + pointID, str(number))
        node.SetAttribute(ROI_NEXT_NUMBER_ATTRIBUTE, str(number + 1))
        return number

    @staticmethod
    def forgetRoi(node, pointID):
        node.RemoveAttribute(ROI_RADIUS_ATTRIBUTE + pointID)
        node.RemoveAttribute(ROI_NUMBER_ATTRIBUTE + pointID)

    @staticmethod
    def forgetAllRois(node):
        for name in list(node.GetAttributeNames()):
            if name.startswith(ROI_RADIUS_ATTRIBUTE) or name.startswith(ROI_NUMBER_ATTRIBUTE):
                node.RemoveAttribute(name)
        node.SetAttribute(ROI_NEXT_NUMBER_ATTRIBUTE, "1")

    @staticmethod
    def statsCacheKey(volumeNode, center, radius):
        roundedCenter = tuple(round(c, 3) for c in center)
        if volumeNode is None:
            return (None, roundedCenter, round(radius, 3))
        imageData = volumeNode.GetImageData()
        transformNode = volumeNode.GetParentTransformNode()
        return (volumeNode.GetID(), volumeNode.GetMTime(),
                imageData.GetMTime() if imageData else 0,
                transformNode.GetMTime() if transformNode else 0,
                roundedCenter, round(radius, 3))

    @staticmethod
    def computeSphereStatistics(volumeNode, centerWorld, radiusMm):
        """SUV statistics of the PET voxels inside a sphere given in world coordinates."""
        if volumeNode is None or volumeNode.GetImageData() is None:
            return None
        center = list(centerWorld)
        transformNode = volumeNode.GetParentTransformNode()
        if transformNode is not None:
            worldToVolume = vtk.vtkGeneralTransform()
            transformNode.GetTransformFromWorld(worldToVolume)
            center = list(worldToVolume.TransformPoint(center))
        ijkToRas = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRas)
        return sphereStatisticsFromArray(
            slicer.util.arrayFromVolume(volumeNode),
            slicer.util.arrayFromVTKMatrix(ijkToRas),
            center, radiusMm)

    @staticmethod
    def findRoiModelNode():
        for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            if node.GetAttribute(ROI_MODEL_ATTRIBUTE):
                return node
        return None

    def removeRoiSphereModel(self):
        modelNode = self.findRoiModelNode()
        if modelNode is not None:
            slicer.mrmlScene.RemoveNode(modelNode)

    @staticmethod
    def findRoiHandlesNode():
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode"):
            if node.GetAttribute(ROI_HANDLES_ATTRIBUTE):
                return node
        return None

    @staticmethod
    def createRoiHandlesNode():
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "SUV ROI radius handles")
        node.SetAttribute(ROI_HANDLES_ATTRIBUTE, "1")
        if hasattr(node, "SetControlPointLabelFormat"):
            node.SetControlPointLabelFormat("")
        node.CreateDefaultDisplayNodes()
        displayNode = node.GetDisplayNode()
        if displayNode is not None:
            displayNode.SetSelectedColor(1.0, 0.85, 0.0)
            displayNode.SetColor(1.0, 0.85, 0.0)
            displayNode.SetGlyphScale(2.0)
            square = getattr(slicer.vtkMRMLMarkupsDisplayNode, "Square2D", None)
            if square is not None:
                displayNode.SetGlyphType(square)
            if hasattr(displayNode, "SetPointLabelsVisibility"):
                displayNode.SetPointLabelsVisibility(False)
            if hasattr(displayNode, "SetPropertiesLabelVisibility"):
                displayNode.SetPropertiesLabelVisibility(False)
            if hasattr(displayNode, "SetVisibility3D"):
                displayNode.SetVisibility3D(False)  # handles are for slice views; keep the MIP clean
        return node

    def removeRoiHandlesNode(self):
        handlesNode = self.findRoiHandlesNode()
        if handlesNode is not None:
            slicer.mrmlScene.RemoveNode(handlesNode)

    def updateRoiSphereModel(self, spheres):
        """One model holding all spheres; its slice intersections draw the ROI circles in 2D views."""
        modelNode = self.findRoiModelNode()
        if not spheres:
            if modelNode is not None:
                modelNode.SetAndObservePolyData(vtk.vtkPolyData())
            return
        if modelNode is None:
            modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "SUV ROI spheres")
            modelNode.SetAttribute(ROI_MODEL_ATTRIBUTE, "1")
            modelNode.CreateDefaultDisplayNodes()
            displayNode = modelNode.GetDisplayNode()
            displayNode.SetColor(0.1, 0.9, 0.3)
            if hasattr(displayNode, "SetVisibility2D"):
                displayNode.SetVisibility2D(True)
                displayNode.SetVisibility3D(False)  # keep the MIP uncluttered; labels still show in 3D
            else:
                displayNode.SetSliceIntersectionVisibility(True)
            displayNode.SetSliceIntersectionThickness(2)

        append = vtk.vtkAppendPolyData()
        for center, radius in spheres:
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(center)
            sphere.SetRadius(radius)
            sphere.SetThetaResolution(48)
            sphere.SetPhiResolution(24)
            sphere.Update()
            append.AddInputData(sphere.GetOutput())
        append.Update()
        polyData = vtk.vtkPolyData()
        polyData.DeepCopy(append.GetOutput())
        modelNode.SetAndObservePolyData(polyData)
