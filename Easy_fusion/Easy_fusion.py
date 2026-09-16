import os
import re
import random
import colorsys
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

# One segment per ROI: voxels inside the sphere at or above the threshold (MTV / TLG)
ROI_SEGMENTATION_ATTRIBUTE = "EasyFusion.SUVROISegmentation"
ROI_SEGMENTATION_PET_ROLE = "EasyFusionSegmentationPET"
ROI_SEGMENT_ID_PREFIX = "EFROI_"
ROI_COLOR_ATTRIBUTE = "EasyFusion.RoiColor_"       # + control point ID, "r g b" (0..1)
ROI_SEGMENT_OUTLINE_PX = 2
THRESHOLD_RELATIVE = "relative"   # % of the ROI's Max
THRESHOLD_ABSOLUTE = "absolute"   # fixed SUV value
DEFAULT_RELATIVE_THRESHOLD = 40.0
DEFAULT_ABSOLUTE_THRESHOLD = 2.5
SETTINGS_THRESHOLD_MODE = "EasyFusion.ThresholdMode"
SETTINGS_RELATIVE_THRESHOLD = "EasyFusion.RelativeThreshold"
SETTINGS_ABSOLUTE_THRESHOLD = "EasyFusion.AbsoluteThreshold"

# ROI appearance
ROI_SPHERE_OUTLINE_PX = 1
ROI_SPHERE_COLOR_ARRAY = "EasyFusionRoiColor"
ROI_HANDLE_GLYPH_SCALE = 1.4
ROI_LABEL_TEXT_SCALE = 2.5

# ROIs projected onto the MIP (3D view). The MIP is fully opaque, so segments and labels are drawn in an
# overlay render layer that shares the 3D camera and is always on top (like a PET workstation MIP overlay).
# 3D segment surfaces follow the voxels exactly (no smoothing), so they match the measured MTV
SEGMENT_SURFACE_SMOOTHING = "0.0"
MIP_OVERLAY_SEGMENT_OPACITY = 0.65
MIP_OVERLAY_TEXT_COLOR = (0.05, 0.05, 0.05)
MIP_OVERLAY_TEXT_BACKGROUND = (1.0, 1.0, 1.0)
MIP_OVERLAY_TEXT_BACKGROUND_OPACITY = 0.75
MIP_OVERLAY_FONT_SIZE = 14
MIP_OVERLAY_TEXT_OFFSET_PX = (14, 10)

# SUV text is drawn by a separate, locked label layer so it can be white and sit one radius
# away from the center. One anchor per ROI lies in only one of the three standard planes, on the
# screen's upper-right diagonal of that view: axial (-R,+A), coronal (-R,+S), sagittal (-A,+S).
ROI_LABELS_ATTRIBUTE = "EasyFusion.SUVROILabels"
LABEL_ANCHOR_DIRECTIONS = [(-1.0, 1.0, 0.0), (-1.0, 0.0, 1.0), (0.0, -1.0, 1.0)]

# Radius handles sit on each sphere along ±R, ±A, ±S, so every slice view through the
# ROI center shows four of them around the circle.
HANDLE_DIRECTIONS = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                     (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
                     (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]

# ---------------------------------------------------------------------------
# Window / level presets
# ---------------------------------------------------------------------------

# (button text, window, level in Hounsfield units, shortcut key over CT-only views)
CT_WINDOW_PRESETS = [
    ("CT: Abdomen", 400, 50, "F5"),
    ("CT: Head", 80, 40, "F6"),
    ("CT: Lungs", 1500, -600, "F7"),
    ("CT: Bones", 1800, 400, "F8"),
]
# Fixed SUV ranges starting at 0 (button text, upper SUV, shortcut key over fusion / PET-only / 3D views)
PET_SUV_PRESETS = [
    ("0–5", 5.0, "F5"),
    ("0–7", 7.0, "F6"),
    ("0–10", 10.0, "F7"),
    ("0–15", 15.0, "F8"),
    ("0–25", 25.0, "F9"),
]
WINDOW_SHORTCUT_KEYS = ["F5", "F6", "F7", "F8", "F9"]
# SPECT (or any uncalibrated image): 0 .. percent of the maximum count in the volume
SPECT_PERCENT_OF_MAX_PRESETS = [10, 25, 50, 75, 100]

# (button text, color node name); PET-DICOM and Hot Metal Blue on the second row
PET_COLOR_MAP_BUTTONS = [
    [("Hot Iron", HOT_IRON_NAME), ("Inferno", "Inferno"), ("Rainbow-2", "PET-Rainbow2")],
    [("PET-DICOM", "PET-DICOM"), ("Red", "Red"), ("Hot Metal Blue", "PET-HotMetalBlue")],
]
# Color map combo box (applied by "Go")
FUSION_COLOR_MAPS = {"Hot Iron": HOT_IRON_NAME, "Inferno": "Inferno", "Rainbow": "PET-Rainbow2"}

# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------

LAYOUT_FOUR_UP_ID = 3              # Slicer's built-in four-up (vtkMRMLLayoutNode::SlicerLayoutFourUpView)
LAYOUT_AXIAL_FOUR_UP_ID = 7501
LAYOUT_TWO_BY_THREE_ID = 7502
LAYOUT_DUAL_MONITOR_ID = 7503
LAYOUT_CT_FUSION_PET_3D_ID = 7504          # full 2x3 (CT | fusion | PET) + 3D
LAYOUT_DUAL_MONITOR_FUSION_MIDDLE_ID = 7505
DUAL_MONITOR_LAYOUT_IDS = (LAYOUT_DUAL_MONITOR_ID, LAYOUT_DUAL_MONITOR_FUSION_MIDDLE_ID)
CUSTOM_LAYOUT_IDS = (LAYOUT_AXIAL_FOUR_UP_ID, LAYOUT_TWO_BY_THREE_ID, LAYOUT_CT_FUSION_PET_3D_ID) + DUAL_MONITOR_LAYOUT_IDS
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
# Fixed IDs for the unsaved twin. An auto-numbered ID (e.g. vtkMRMLScalarVolumeNode3) can be handed to a
# completely different volume in the next scene; these can only ever belong to the twin.
PET_ONLY_VOLUME_ID = "vtkMRMLScalarVolumeNodeEasyFusionPETOnly"
PET_ONLY_DISPLAY_ID = "vtkMRMLScalarVolumeDisplayNodeEasyFusionPETOnly"


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


def _sliceViews(*names):
    return [_sliceViewItem(name) for name in names]


def _dualMonitorLayout(axialRow, sagittalRow):
    """Main window: axial row over sagittal row. Second window (auto-placed on monitor 2): 3D + coronal."""
    return (
        '<viewports>'
        '<layout type="vertical">'
        + _nested("horizontal", _sliceViews(*axialRow))
        + _nested("horizontal", _sliceViews(*sagittalRow))
        + '</layout>'
        f'<layout name="EasyFusionMonitor2" type="horizontal" label="{DUAL_MONITOR_WINDOW_TITLE}" dockable="false">'
        + _THREED_VIEW_ITEM + _sliceViewItem("Green") + _sliceViewItem("EFCoronalCT")
        + '</layout>'
        '</viewports>')


def buildLayoutDescriptions():
    """Layout XML for the custom EasyFusion layouts, keyed by layout ID."""
    axialFourUp = (
        '<layout type="vertical">'
        + _nested("horizontal", [_sliceViewItem("Red"), _THREED_VIEW_ITEM])
        + _nested("horizontal", _sliceViews("EFAxialCT", "EFAxialPET"))
        + '</layout>')

    # 3 columns: [axial fusion / sagittal fusion] [axial CT / sagittal CT] [3D spanning both rows]
    twoByThree = (
        '<layout type="horizontal">'
        + _nested("vertical", _sliceViews("Red", "Yellow"))
        + _nested("vertical", _sliceViews("EFAxialCT", "EFSagittalCT"))
        + _THREED_VIEW_ITEM
        + '</layout>')

    # 4 columns: [axial CT / sagittal CT] [axial fusion / sagittal fusion] [axial PET / sagittal PET] [3D]
    ctFusionPet3D = (
        '<layout type="horizontal">'
        + _nested("vertical", _sliceViews("EFAxialCT", "EFSagittalCT"))
        + _nested("vertical", _sliceViews("Red", "Yellow"))
        + _nested("vertical", _sliceViews("EFAxialPET", "EFSagittalPET"))
        + _THREED_VIEW_ITEM
        + '</layout>')

    return {
        LAYOUT_AXIAL_FOUR_UP_ID: axialFourUp,
        LAYOUT_TWO_BY_THREE_ID: twoByThree,
        LAYOUT_DUAL_MONITOR_ID: _dualMonitorLayout(
            ("Red", "EFAxialCT", "EFAxialPET"), ("Yellow", "EFSagittalCT", "EFSagittalPET")),
        LAYOUT_CT_FUSION_PET_3D_ID: ctFusionPet3D,
        LAYOUT_DUAL_MONITOR_FUSION_MIDDLE_ID: _dualMonitorLayout(
            ("EFAxialCT", "Red", "EFAxialPET"), ("EFSagittalCT", "Yellow", "EFSagittalPET")),
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
        parent.title = "Lvgvs - SPECT/PET Review"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = """
        This module provides easy fusion of SPECT/PET and CT/MR images.
        Spherical ROIs report Max and Mean, plus a thresholded segment inside each ROI
        (default 40% of Max, or an absolute SUV) giving segment Mean, MTV and TLG.
        The label next to each ROI in the views shows Max and the segment Mean. Values are read
        directly from the selected PET volume
        (the volume is assumed to already be in SUV units). Press Insert over a slice view to drop
        an ROI at the cursor; drag the yellow edge handles to resize it.
        PET presets set a fixed SUV range; SPECT presets set 0 to a percentage of the maximum count in the image.
        Keyboard windowing with the mouse over a view: F5-F8 = abdomen, head, lungs, bones on CT views;
        F5-F9 = SUV 0-5, 0-7, 0-10, 0-15, 0-25 on fusion, PET and 3D views.
        Layout buttons switch between four-up, axial fusion/CT/PET, 2x3 + 3D, CT | fusion | PET + 3D and two
        dual monitor layouts (the dual monitor layouts need Slicer 5.2 or later).
        """
        parent.acknowledgementText = """
        This file was developed by Burak Demir.
        """
        parent.icon = qt.QIcon(os.path.join(os.path.dirname(__file__), "Resources", "Icons", "Easy_fusion.png"))

        # Scene-load fixes (Hot Iron repair, no auto-rotation) must work even if the
        # EasyFusion GUI has not been opened yet in this Slicer session.
        slicer.app.connect("startupCompleted()", registerSceneObservers)


# ---------------------------------------------------------------------------
# Scene-level observers (registered once per session)
# ---------------------------------------------------------------------------

_sceneObserverTags = []

# Scene / layout work is never done inside a scene notification. It runs from the event loop once the
# scene has been idle for a short while, so the layout manager has finished rebuilding its views.
SCENE_POLL_MS = 100
SCENE_SETTLE_MS = 300
SCENE_MAX_WAIT_MS = 10 * 60 * 1000


def registerSceneObservers():
    """
    Observe scene loading so saved scenes come back in a consistent state. Safe to call repeatedly.
    Layouts are registered once here (at startup); Slicer keeps them across scene close/load, so
    nothing touches the layout node while a scene is being loaded.
    """
    if _sceneObserverTags:
        return
    scene = slicer.mrmlScene
    for eventName, handler in (("StartCloseEvent", _onSceneStartClose),
                               ("EndImportEvent", _onSceneEndImport),
                               ("StartSaveEvent", _onSceneStartSave),
                               ("EndSaveEvent", _onSceneEndSave)):
        event = getattr(slicer.vtkMRMLScene, eventName, None)
        if event is None:
            logging.warning(f"EasyFusion: this Slicer has no vtkMRMLScene.{eventName}")
            continue
        _sceneObserverTags.append(scene.AddObserver(event, handler))
    try:
        Easy_fusionLogic.ensureLayoutsRegistered()
        Easy_fusionLogic.reapplyRestoredCustomLayout()
    except Exception:
        logging.exception("EasyFusion: could not register layouts")


def sceneIsBusy():
    scene = slicer.mrmlScene
    return scene.IsImporting() or scene.IsClosing() or scene.IsBatchProcessing()


def runWhenSceneSettled(callback, settleMs=SCENE_SETTLE_MS):
    """
    Run callback from the event loop once the scene has been idle for settleMs.
    Never runs while the scene is still loading / closing: if it stays busy too long, the callback is dropped.
    """
    state = {"elapsed": 0, "idleSince": None}

    def poll():
        if sceneIsBusy():
            state["idleSince"] = None
            if state["elapsed"] >= SCENE_MAX_WAIT_MS:
                logging.warning("EasyFusion: scene stayed busy, skipped a deferred update")
                return
        else:
            if state["idleSince"] is None:
                state["idleSince"] = state["elapsed"]
            if state["elapsed"] - state["idleSince"] >= settleMs:
                try:
                    callback()
                except Exception:
                    logging.exception("EasyFusion: deferred scene update failed")
                return
        state["elapsed"] += SCENE_POLL_MS
        qt.QTimer.singleShot(SCENE_POLL_MS, poll)

    qt.QTimer.singleShot(0, poll)


def deferUntilSceneIdle(callback):
    """Leave the current scene/layout notification first, then run callback when the scene is idle."""
    runWhenSceneSettled(callback, settleMs=0)


def _onSceneStartClose(caller, event):
    unbindWindowLevelSync()


# While a scene file is written, PET-only views point at the real PET instead of the unsaved twin,
# so saved scenes never reference a node ID that does not exist in the file.
_saveSwaps = []


def _onSceneStartSave(caller, event):
    try:
        _saveSwaps[:] = Easy_fusionLogic.pointPetOnlyViewsAtSourcePet()
    except Exception:
        logging.exception("EasyFusion: could not prepare PET-only views for saving")


def _onSceneEndSave(caller, event):
    # Restored from the event loop: the MRML file may still be written after this notification returns
    swaps = list(_saveSwaps)
    _saveSwaps[:] = []
    if swaps:
        qt.QTimer.singleShot(0, lambda: _restoreAfterSave(swaps))


def _restoreAfterSave(swaps):
    try:
        Easy_fusionLogic.restorePetOnlyViews(swaps)
    except Exception:
        logging.exception("EasyFusion: could not restore PET-only views after saving")


# Post-load work runs as ONE ordered chain: scene repairs first, then the module panel (if it exists).
# Separate timers per observer used to interleave with each other and with Slicer's own view rebuilding.
_postLoadListeners = []
_postLoadState = {"pending": False}


def addPostLoadListener(callback):
    if callback not in _postLoadListeners:
        _postLoadListeners.append(callback)


def removePostLoadListener(callback):
    if callback in _postLoadListeners:
        _postLoadListeners.remove(callback)


def _onSceneEndImport(caller, event):
    # Nothing is changed inside the import notification itself: the layout manager may still be
    # rebuilding views for the loaded layout. All repairs run afterwards from the event loop.
    if _postLoadState["pending"]:
        return
    _postLoadState["pending"] = True
    runWhenSceneSettled(_afterSceneLoad)


def _afterSceneLoad():
    _postLoadState["pending"] = False
    logic = Easy_fusionLogic()
    # First, before anything creates nodes: scenes saved by earlier versions contain views that refer to
    # the unsaved PET-only twin by ID. A new node given that ID would be picked up half-built.
    steps = [logic.clearDanglingViewReferences,
             logic.stopAllViewRotations,
             logic.repairHotIronColorNodes,
             logic.restoreViewRolesAfterLoad]
    steps += list(_postLoadListeners)
    for step in steps:
        if sceneIsBusy():
            # Another load / close started in the meantime; its own EndImport schedules a new pass
            return
        try:
            step()
        except Exception:
            logging.exception(f"EasyFusion: post-load step {getattr(step, '__name__', step)} failed")


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

def windowPresetForView(viewKind, sliceViewName, key):
    """
    Preset applied by a windowing shortcut key ("F5".."F9").
    viewKind: "slice" or "threeD" for the view under the mouse, None when the mouse is not over a view.
    CT-only slice views use the CT presets; fusion, PET-only and 3D views (and slice views that are not
    EasyFusion views) use the SUV presets.
    Returns ("ct", text, window, level), ("pet", text, window, level), or None if the key does nothing there.
    """
    if viewKind == "slice" and SLICE_VIEW_ROLES.get(sliceViewName, (None, "fusion"))[1] == "ct":
        for text, window, level, presetKey in CT_WINDOW_PRESETS:
            if presetKey == key:
                return ("ct", text, window, level)
        return None
    if viewKind in ("slice", "threeD"):
        for text, upper, presetKey in PET_SUV_PRESETS:
            if presetKey == key:
                return ("pet", text, upper, upper / 2.0)
    return None


def percentOfMaximum(maximum, percent):
    """Upper window limit for the SPECT presets, or None when the image has no positive values."""
    maximum, percent = float(maximum), float(percent)
    if not np.isfinite(maximum) or maximum <= 0 or percent <= 0:
        return None
    return maximum * percent / 100.0


def hotIronRGB(t):
    """Custom Hot Iron color stops for t in [0, 1]."""
    if t <= 0.5:
        r, g, b = t * 2, 0.0, 0.0
    elif t <= 0.75:
        r, g, b = 1.0, (t - 0.5) * 4, 0.0
    else:
        r, g, b = 1.0, 1.0, (t - 0.75) * 4
    return (min(max(r, 0.0), 1.0), min(max(g, 0.0), 1.0), min(max(b, 0.0), 1.0))


def sphereStatisticsFromArray(voxels, ijkToRas, centerRas, radiusMm,
                              thresholdMode=THRESHOLD_RELATIVE, thresholdValue=DEFAULT_RELATIVE_THRESHOLD):
    """
    Statistics of the voxels whose centers lie inside a sphere, plus a thresholded segment inside it.

    voxels:    numpy array indexed [k, j, i] (as returned by slicer.util.arrayFromVolume)
    ijkToRas:  4x4 matrix (numpy) mapping voxel indices to RAS (mm)
    centerRas: sphere center in the volume's RAS coordinate system
    radiusMm:  sphere radius in mm
    thresholdMode / thresholdValue: THRESHOLD_RELATIVE (% of Max in the sphere) or THRESHOLD_ABSOLUTE (SUV)

    Returns None if the sphere does not touch the volume, otherwise a dict with
      max, mean, voxels, volumeMl           (whole sphere)
      threshold, segVoxels, segMean, mtvMl, tlg  (segment; segMean is None if the segment is empty)
      extent (i0, i1, j0, j1, k0, k1) and segMask (bool array [k, j, i] over that extent)
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

    inside = None
    if np.all(lo <= hi):
        ni, nj, nk = (hi - lo + 1)
        jj, ii = np.meshgrid(np.arange(lo[1], hi[1] + 1), np.arange(lo[0], hi[0] + 1), indexing="ij")
        inPlane = linear[:, 0:1] * ii.ravel() + linear[:, 1:2] * jj.ravel() + offset[:, None]
        radius2 = radius * radius
        inside = np.zeros((nk, nj, ni), dtype=bool)
        for kIndex in range(nk):  # slab by slab keeps memory bounded for big spheres
            delta = inPlane + linear[:, 2:3] * (lo[2] + kIndex) - center[:, None]
            inside[kIndex] = (np.einsum("ij,ij->j", delta, delta) <= radius2).reshape(nj, ni)
        if not inside.any():
            inside = None

    if inside is None:
        nearest = np.round(centerIjk).astype(int)
        if not (np.all(nearest >= 0) and np.all(nearest < dims)):
            return None
        lo = hi = nearest
        inside = np.ones((1, 1, 1), dtype=bool)

    sub = voxels[lo[2]:hi[2] + 1, lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
    values = sub[inside]
    voxelVolumeMl = abs(np.linalg.det(linear)) / 1000.0
    suvMax = float(values.max())

    if thresholdMode == THRESHOLD_ABSOLUTE:
        threshold = float(thresholdValue)
    else:
        threshold = suvMax * float(thresholdValue) / 100.0
    segMask = inside & (sub >= threshold)
    segValues = sub[segMask]
    segVoxels = int(segValues.size)
    segMean = float(segValues.mean()) if segVoxels else None
    mtvMl = segVoxels * voxelVolumeMl

    return {
        "max": suvMax,
        "mean": float(values.mean()),
        "voxels": int(values.size),
        "volumeMl": float(values.size * voxelVolumeMl),
        "threshold": threshold,
        "segVoxels": segVoxels,
        "segMean": segMean,
        "mtvMl": float(mtvMl),
        "tlg": float(segMean * mtvMl) if segVoxels else 0.0,
        "extent": (int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2])),
        "segMask": segMask,
    }


def formatRoiLabel(name, stats, hasPet):
    """Annotation shown next to the ROI in the views (multi-line)."""
    if stats is None:
        return f"{name}\n(outside PET)" if hasPet else f"{name}\n(no PET)"
    segMean = stats.get("segMean")
    segText = f"{segMean:.2f}" if segMean is not None else "-"
    return f"{name}\nMax {stats['max']:.2f}\nMean {segText}"  # Mean = thresholded segment mean


def formatRoiTableRow(name, radius, stats):
    """ROI | r (mm) | Max | Mean | Seg Mean | MTV (mL) | TLG"""
    values = [name, f"{radius:.1f}", "-", "-", "-", "-", "-"]
    if stats is not None:
        values[2] = f"{stats['max']:.2f}"
        values[3] = f"{stats['mean']:.2f}"
        if stats.get("segMean") is not None:
            values[4] = f"{stats['segMean']:.2f}"
        values[5] = f"{stats.get('mtvMl', 0.0):.2f}"
        values[6] = f"{stats.get('tlg', 0.0):.2f}"
    return values


def randomRoiColor(rng=random):
    """Random, clearly visible color: any hue, strong saturation and brightness (reads on black PET and white MIP)."""
    hue = rng.random()
    saturation = 0.75 + 0.25 * rng.random()
    value = 0.80 + 0.20 * rng.random()
    return colorsys.hsv_to_rgb(hue, saturation, value)


def parseRoiColor(text):
    """(r, g, b) from "r g b", or None if the text is not a valid color."""
    try:
        values = tuple(float(v) for v in (text or "").split())
    except ValueError:
        return None
    if len(values) != 3 or not all(0.0 <= v <= 1.0 for v in values):
        return None
    return values


def _distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def radiusHandlePosition(center, radius, axis):
    direction = HANDLE_DIRECTIONS[axis]
    return [center[0] + direction[0] * radius,
            center[1] + direction[1] * radius,
            center[2] + direction[2] * radius]


def parseHandleDescription(description, count=len(HANDLE_DIRECTIONS)):
    """Handle / label anchor points store '<roi control point ID>:<index>' in their description."""
    if not description or ":" not in description:
        return None
    roiID, _, indexText = description.rpartition(":")
    if not roiID or not indexText.isdigit():
        return None
    index = int(indexText)
    return (roiID, index) if 0 <= index < count else None


def labelAnchorPosition(center, radius, plane):
    """Label anchor one radius away from the center, on the upper-right diagonal of that plane's view."""
    direction = LABEL_ANCHOR_DIRECTIONS[plane]
    scale = float(radius) / np.sqrt(2.0)
    return [center[i] + direction[i] * scale for i in range(3)]


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
        self._shortcuts = []
        self._updatingRois = False
        self._roiStatsCache = {}
        self._knownRoiIDs = None
        self._lastRoiGeometry = {}       # ROI control point ID -> (center, radius) at last sync
        self._lastHandlePositions = {}   # handle control point ID -> position at last sync
        self._activeHandleID = None      # handle currently being dragged
        self._panelButtons = []
        self.mipOverlay = MipRoiOverlay()

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = Easy_fusionLogic()

        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)
        formLayout = qt.QFormLayout(parametersCollapsibleButton)

        # Input volumes
        self.inputVolumeSelector = self._createVolumeSelector("Select the SPECT/PET image for fusion.")
        formLayout.addRow("SPECT/PET: ", self.inputVolumeSelector)
        self.inputVolumeSelectorCT = self._createVolumeSelector("Select the CT/MR image for fusion.")
        formLayout.addRow("CT/MRI: ", self.inputVolumeSelectorCT)

        self.petColorMapSelector = qt.QComboBox()
        self.petColorMapSelector.addItems(list(FUSION_COLOR_MAPS.keys()))
        formLayout.addRow("PET Color Map:", self.petColorMapSelector)

        self.FusionButton = qt.QPushButton("Go")
        self.FusionButton.connect("clicked(bool)", self.DoFusion)
        formLayout.addRow(self.FusionButton)

        # MIP rotation. The toggle button always mirrors the 3D view node (see updateRotationButton),
        # so it can never get out of sync with what the view is actually doing.
        self.rotationSpeedSlider = ctk.ctkSliderWidget()
        self.rotationSpeedSlider.singleStep = 10
        self.rotationSpeedSlider.minimum = 10
        self.rotationSpeedSlider.maximum = 200
        self.rotationSpeedSlider.value = 50
        self.rotationSpeedSlider.toolTip = "Lower is faster (ms per step)"
        formLayout.addRow("MIP Rotation Speed (ms):", self.rotationSpeedSlider)

        self.toggleRotationButton = qt.QPushButton("Start MIP Rotation")
        self.toggleRotationButton.checkable = True
        formLayout.addRow(self.toggleRotationButton)
        self.toggleRotationButton.connect('toggled(bool)', self.setRotationEnabled)
        self.rotationSpeedSlider.connect('valueChanged(double)', self.updateRotationSpeed)

        formLayout.addRow("Quick View:", self._buttonRow([
            ("Anterior", lambda: self.rotateMIPToViewAxis(3)),
            ("Left", lambda: self.rotateMIPToViewAxis(0)),
            ("Right", lambda: self.rotateMIPToViewAxis(1)),
        ]))

        # Window / level presets
        formLayout.addRow("CT Presets:", self._buttonRow([
            (text, lambda window=window, level=level: self.setCTWindow(window, level),
             f"Shortcut: {key} with the mouse over a CT view")
            for text, window, level, key in CT_WINDOW_PRESETS]))

        formLayout.addRow("PET Presets (SUV):", self._buttonRow([
            (text, lambda upper=upper: self.setPETWindow(upper, upper / 2.0),
             f"Shortcut: {key} with the mouse over a fusion, PET or 3D view")
            for text, upper, key in PET_SUV_PRESETS]))

        formLayout.addRow("SPECT Presets (% max):", self._buttonRow(
            [(f"0–{percent}%", lambda percent=percent: self.setPETWindowPercentOfMax(percent))
             for percent in SPECT_PERCENT_OF_MAX_PRESETS],
            toolTip="Window from 0 to this percentage of the maximum count (voxel value) in the SPECT/PET volume."))

        # PET color maps (two rows, no label on the second)
        for rowIndex, row in enumerate(PET_COLOR_MAP_BUTTONS):
            formLayout.addRow("PET Color Maps:" if rowIndex == 0 else "", self._buttonRow([
                (text, lambda colorNodeName=colorNodeName: self.setPETColorMap(colorNodeName))
                for text, colorNodeName in row]))

        # F5-F9 with the mouse over a view: CT presets on CT views, SUV presets on fusion / PET / 3D views
        for key in WINDOW_SHORTCUT_KEYS:
            self._addApplicationShortcut(getattr(qt.Qt, f"Key_{key}"), lambda key=key: self.onWindowShortcut(key))

        self.setupLayoutSection()
        self.setupMeasurementSection()

        self.layout.addStretch(1)

        bannerPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons", "fusbanner.jpg")
        if os.path.exists(bannerPath):
            bannerLabel = qt.QLabel()
            bannerLabel.setPixmap(qt.QPixmap(bannerPath).scaledToWidth(400, qt.Qt.SmoothTransformation))
            bannerLabel.setAlignment(qt.Qt.AlignCenter)
            self.layout.addWidget(bannerLabel)
        else:
            logging.warning(f"EasyFusion: banner file not found at {bannerPath}")

        infoTextBox = qt.QTextEdit()
        infoTextBox.setReadOnly(True)
        infoTextBox.setPlainText(
            "This module provides eased visualization of PET images.\n"
            "This module is NOT a medical device. Research use only.\n"
            "Developed by: Burak Demir, MD, FEBNM \n"
            "For support and feedback: 4burakfe@gmail.com\n"
            "Version: alpha v1.0"
        )
        infoTextBox.setToolTip("Module information and instructions.")
        self.layout.addWidget(infoTextBox)

        # Observers
        registerSceneObservers()  # in case the module was added after startup
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        addPostLoadListener(self.onSceneLoaded)  # runs after the scene repairs, in the same deferred pass
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAboutToBeRemovedEvent, self.onNodeAboutToBeRemoved)
        self.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onPETVolumeChanged)
        if slicer.app.layoutManager() is not None:
            slicer.app.layoutManager().connect("layoutChanged(int)", self.onLayoutChanged)

        self.observeThreeDViewNode()
        self.restoreFromSettings()
        self.connectToExistingRois()
        self.onLayoutChanged()

    @staticmethod
    def _createVolumeSelector(toolTip):
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

    def _buttonRow(self, buttonSpecs, toolTip=None):
        """Horizontal row of push buttons from (text, callback) or (text, callback, tooltip) tuples."""
        rowLayout = qt.QHBoxLayout()
        for spec in buttonSpecs:
            text, callback = spec[0], spec[1]
            button = qt.QPushButton(text)
            buttonToolTip = spec[2] if len(spec) > 2 else toolTip
            if buttonToolTip:
                button.setToolTip(buttonToolTip)
            button.connect('clicked()', callback)
            rowLayout.addWidget(button)
            # Keep a Python reference: the row layout has no parent widget yet, and PythonQt deletes
            # parentless widgets whose wrapper is garbage collected.
            self._panelButtons.append(button)
        return rowLayout

    def _addApplicationShortcut(self, keyCode, callback):
        """
        Application-wide shortcut: the Monitor 2 viewport is a separate window, and a main-window shortcut
        stops working as soon as that window is active.
        """
        mainWindow = slicer.util.mainWindow()
        if mainWindow is None:
            return
        shortcut = qt.QShortcut(qt.QKeySequence(keyCode), mainWindow)
        shortcut.setContext(qt.Qt.ApplicationShortcut)
        shortcut.connect('activated()', callback)
        self._shortcuts.append(shortcut)

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
            (LAYOUT_CT_FUSION_PET_3D_ID, "CT | Fusion | PET + 3D",
             "Top: axial CT | axial fusion | axial PET (inverted grey)\n"
             "Bottom: sagittal CT | sagittal fusion | sagittal PET (inverted grey)\n"
             "Right column: 3D MIP", 1, 1),
            (LAYOUT_DUAL_MONITOR_ID, "Dual Monitor",
             "Monitor 1: axial and sagittal fusion | CT | PET (3×2)\n"
             "Monitor 2 (separate window): 3D MIP | coronal fusion | coronal CT\n"
             "Click again to bring the second window back if it was closed.", 2, 0),
            (LAYOUT_DUAL_MONITOR_FUSION_MIDDLE_ID, "Dual Monitor (Fusion Middle)",
             "Monitor 1: axial and sagittal CT | fusion | PET (3×2)\n"
             "Monitor 2 (separate window): 3D MIP | coronal fusion | coronal CT\n"
             "Click again to bring the second window back if it was closed.", 2, 1),
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

        thresholdLayout = qt.QHBoxLayout()
        self.thresholdModeComboBox = qt.QComboBox()
        self.thresholdModeComboBox.addItem("% of Max", THRESHOLD_RELATIVE)
        self.thresholdModeComboBox.addItem("Absolute SUV", THRESHOLD_ABSOLUTE)
        self.thresholdModeComboBox.setToolTip(
            "Relative: segment = voxels inside the ROI with SUV >= this % of the ROI's Max.\n"
            "Absolute: segment = voxels inside the ROI with SUV >= this value.")
        self.thresholdValueSpinBox = qt.QDoubleSpinBox()
        self.thresholdValueSpinBox.setDecimals(1)
        thresholdLayout.addWidget(self.thresholdModeComboBox)
        thresholdLayout.addWidget(self.thresholdValueSpinBox)
        measurementLayout.addRow("Segment threshold:", thresholdLayout)
        self._relativeThreshold = DEFAULT_RELATIVE_THRESHOLD
        self._absoluteThreshold = DEFAULT_ABSOLUTE_THRESHOLD
        self._applyThresholdModeToSpinBox(THRESHOLD_RELATIVE)

        self.roiPetLabel = qt.QLabel("Measuring on: (no PET selected)")
        measurementLayout.addRow(self.roiPetLabel)

        self.showRoisOnMipCheckBox = qt.QCheckBox("Show ROI segments and values on the MIP (3D view)")
        self.showRoisOnMipCheckBox.checked = True
        self.showRoisOnMipCheckBox.setToolTip(
            "Draws the thresholded segments and the Max / Mean text on top of the MIP.\n"
            "Display only: nothing is added to the scene or saved.")
        measurementLayout.addRow(self.showRoisOnMipCheckBox)

        self.roiTable = qt.QTableWidget()
        self.roiTable.setColumnCount(7)
        self.roiTable.setHorizontalHeaderLabels(["ROI", "r (mm)", "Max", "Mean", "Seg Mean", "MTV (mL)", "TLG"])
        headerTips = ["", "ROI radius", "Maximum SUV inside the ROI sphere", "Mean SUV of the whole ROI sphere",
                      "Mean SUV of the thresholded segment", "Metabolic tumor volume: volume of the thresholded segment",
                      "Total lesion glycolysis = Seg Mean x MTV"]
        for column, tip in enumerate(headerTips):
            headerItem = self.roiTable.horizontalHeaderItem(column)
            if headerItem is not None and tip:
                headerItem.setToolTip(tip)
        self.roiTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.roiTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.roiTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.roiTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        self.roiTable.horizontalHeader().setStretchLastSection(True)
        self.roiTable.verticalHeader().setVisible(False)
        self.roiTable.setMinimumHeight(140)
        self.roiTable.setToolTip("Select a row to jump the slice views to that ROI.")
        measurementLayout.addRow(self.roiTable)

        self.placeRoiButton.connect('clicked()', self.onPlaceRoi)
        self.deleteRoiButton.connect('clicked()', self.onDeleteSelectedRoi)
        self.clearRoisButton.connect('clicked()', self.onClearRois)
        self.roiRadiusSpinBox.connect('valueChanged(double)', self.onRoiRadiusChanged)
        self.roiTable.connect('itemSelectionChanged()', self.onRoiSelectionChanged)
        self.thresholdModeComboBox.connect('currentIndexChanged(int)', self.onThresholdModeChanged)
        self.thresholdValueSpinBox.connect('valueChanged(double)', self.onThresholdValueChanged)
        self.showRoisOnMipCheckBox.connect('toggled(bool)', self.onShowRoisOnMipToggled)

        # Batch rapid point events (e.g. dragging) into one recomputation
        self.roiUpdateTimer = qt.QTimer()
        self.roiUpdateTimer.setSingleShot(True)
        self.roiUpdateTimer.setInterval(60)
        self.roiUpdateTimer.connect('timeout()', self.updateRois)

        # Insert key: drop an ROI at the mouse cursor
        self._addApplicationShortcut(qt.Qt.Key_Insert, self.onPlaceRoiAtCursor)

    def enter(self):
        self.observeThreeDViewNode()
        self.onLayoutChanged()

    def cleanup(self):
        try:
            self.mipOverlay.clear()
        except Exception:
            logging.exception("EasyFusion: could not remove the MIP overlay")
        if hasattr(self, "roiUpdateTimer"):
            self.roiUpdateTimer.stop()
        # Otherwise a module reload leaves duplicate shortcuts, and Qt fires neither of two identical ones
        for shortcut in self._shortcuts:
            shortcut.setEnabled(False)
            shortcut.setParent(None)
        self._shortcuts = []
        if slicer.app.layoutManager() is not None:
            slicer.app.layoutManager().disconnect("layoutChanged(int)", self.onLayoutChanged)
        removePostLoadListener(self.onSceneLoaded)
        self.removeObservers()

    # ------------------------------------------------------------------
    # Scene events
    # ------------------------------------------------------------------

    def onSceneEndClose(self, caller=None, event=None):
        self.mipOverlay.clear()
        self.setRoiNode(None)
        self.setHandlesNode(None)
        self.fillRoiTable([])
        runWhenSceneSettled(self.refreshViewsAfterSceneChange)

    def onSceneLoaded(self):
        """Called by the post-load chain (see _afterSceneLoad), after the scene repairs."""
        self.restoreFromSettings()
        self.connectToExistingRois()
        self.refreshViewsAfterSceneChange()

    def refreshViewsAfterSceneChange(self):
        self.observeThreeDViewNode()
        self.onLayoutChanged()

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

    def onShowRoisOnMipToggled(self, checked):
        self.mipOverlay.setEnabled(checked)
        self.scheduleRoiUpdate()

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def DoFusion(self):
        ctNode = self.inputVolumeSelectorCT.currentNode()
        petNode = self.inputVolumeSelector.currentNode()
        if petNode is None or ctNode is None:
            slicer.util.errorDisplay("Please select both a SPECT/PET and a CT/MRI volume.")
            return
        if petNode.GetDisplayNode() is None:
            petNode.CreateDefaultDisplayNodes()
        if ctNode.GetDisplayNode() is None:
            ctNode.CreateDefaultDisplayNodes()

        petDisplayNode = petNode.GetDisplayNode()
        petDisplayNode.SetAutoWindowLevel(False)
        petDisplayNode.SetWindow(10)
        petDisplayNode.SetLevel(5)
        petDisplayNode.SetInterpolate(True)

        # Only one MIP at a time: hide volume rendering of any previously used PET (or any other volume)
        self.logic.showOnlyThisVolumeRendering(petNode)
        mipDisplayNode = slicer.modules.volumerendering.logic().CreateDefaultVolumeRenderingNodes(petNode)
        mipDisplayNode.SetVisibility(True)
        self.logic.setMIPRange(mipDisplayNode, 0.0, 10.0, flatOpacity=True)

        threeDWidget = self.getThreeDWidget()
        if threeDWidget is not None:
            viewNode = threeDWidget.mrmlViewNode()
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

            threeDView = threeDWidget.threeDView()
            threeDView.resetFocalPoint()
            threeDView.rotateToViewAxis(3)
            self.fitMIPToView(petNode)

        ctNode.GetDisplayNode().SetAndObserveColorNodeID(slicer.util.getNode("Grey").GetID())
        colorNodeName = FUSION_COLOR_MAPS.get(self.petColorMapSelector.currentText)
        if colorNodeName is not None:
            self.setPETColorMap(colorNodeName)

        # Fill every EasyFusion view: fusion (CT + PET), CT only, PET only (inverted grey)
        self.logic.rememberVolumes(petNode, ctNode)
        changedViews = self.logic.applyViewRoles(petNode, ctNode)
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

        if layoutID in DUAL_MONITOR_LAYOUT_IDS:
            qt.QTimer.singleShot(300, self.logic.placeSecondaryViewportWindow)
        self.onLayoutChanged()

    def afterViewRolesApplied(self, changedViews):
        # Views need their final size before fitting / copying zoom, so wait for the layout to settle
        qt.QTimer.singleShot(150, lambda: self.alignSliceViews(changedViews))

    def onLayoutChanged(self, layoutID=None):
        if not hasattr(self, "layoutButtons"):
            return
        if sceneIsBusy():
            # Layout switches during scene loading: views are still being rebuilt, look at them later
            deferUntilSceneIdle(self.onLayoutChanged)
            return
        layoutManager = slicer.app.layoutManager()
        current = layoutManager.layout if layoutManager is not None else None
        self.layoutButtonGroup.setExclusive(False)
        for buttonLayoutID, button in self.layoutButtons.items():
            button.checked = (buttonLayoutID == current)
        self.layoutButtonGroup.setExclusive(True)
        # A layout switch can create new 3D views (or move them to the Monitor 2 window): re-attach the MIP overlay
        self.scheduleRoiUpdate()

    def restoreFromSettings(self):
        settingsNode = self.logic.getSettingsNode(create=False)
        if settingsNode is None:
            return
        pet = settingsNode.GetNodeReference(SETTINGS_PET_ROLE)
        ct = settingsNode.GetNodeReference(SETTINGS_CT_ROLE)
        if pet is not None:
            self.inputVolumeSelector.setCurrentNode(pet)
        if ct is not None:
            self.inputVolumeSelectorCT.setCurrentNode(ct)

        def readFloat(name, default):
            try:
                return float(settingsNode.GetAttribute(name))
            except (TypeError, ValueError):
                return default
        self._relativeThreshold = readFloat(SETTINGS_RELATIVE_THRESHOLD, DEFAULT_RELATIVE_THRESHOLD)
        self._absoluteThreshold = readFloat(SETTINGS_ABSOLUTE_THRESHOLD, DEFAULT_ABSOLUTE_THRESHOLD)
        mode = settingsNode.GetAttribute(SETTINGS_THRESHOLD_MODE)
        mode = mode if mode in (THRESHOLD_RELATIVE, THRESHOLD_ABSOLUTE) else THRESHOLD_RELATIVE
        wasBlocked = self.thresholdModeComboBox.blockSignals(True)
        self.thresholdModeComboBox.setCurrentIndex(self.thresholdModeComboBox.findData(mode))
        self.thresholdModeComboBox.blockSignals(wasBlocked)
        self._applyThresholdModeToSpinBox(mode)

    # ------------------------------------------------------------------
    # Segment threshold
    # ------------------------------------------------------------------

    def currentThreshold(self):
        mode = self.thresholdModeComboBox.itemData(self.thresholdModeComboBox.currentIndex)
        mode = mode if mode in (THRESHOLD_RELATIVE, THRESHOLD_ABSOLUTE) else THRESHOLD_RELATIVE
        value = self._absoluteThreshold if mode == THRESHOLD_ABSOLUTE else self._relativeThreshold
        return mode, value

    def _applyThresholdModeToSpinBox(self, mode):
        spinBox = self.thresholdValueSpinBox
        wasBlocked = spinBox.blockSignals(True)
        if mode == THRESHOLD_ABSOLUTE:
            spinBox.setRange(0.0, 100.0)
            spinBox.setSingleStep(0.1)
            spinBox.setSuffix(" SUV")
            spinBox.setValue(self._absoluteThreshold)
        else:
            spinBox.setRange(1.0, 100.0)
            spinBox.setSingleStep(1.0)
            spinBox.setSuffix(" %")
            spinBox.setValue(self._relativeThreshold)
        spinBox.blockSignals(wasBlocked)

    def onThresholdModeChanged(self, index=None):
        mode, _ = self.currentThreshold()
        self._applyThresholdModeToSpinBox(mode)
        self.saveThresholdSettings()
        self.scheduleRoiUpdate()

    def onThresholdValueChanged(self, value):
        mode, _ = self.currentThreshold()
        if mode == THRESHOLD_ABSOLUTE:
            self._absoluteThreshold = float(value)
        else:
            self._relativeThreshold = float(value)
        self.saveThresholdSettings()
        self.scheduleRoiUpdate()

    def saveThresholdSettings(self):
        settingsNode = self.logic.getSettingsNode()
        mode, _ = self.currentThreshold()
        settingsNode.SetAttribute(SETTINGS_THRESHOLD_MODE, mode)
        settingsNode.SetAttribute(SETTINGS_RELATIVE_THRESHOLD, f"{self._relativeThreshold:g}")
        settingsNode.SetAttribute(SETTINGS_ABSOLUTE_THRESHOLD, f"{self._absoluteThreshold:g}")

    # ------------------------------------------------------------------
    # Slice view alignment (scroll / pan / zoom sync itself: Slicer's view-link button)
    # ------------------------------------------------------------------

    def alignSliceViews(self, changedViews):
        """Fit views whose CT changed, then give same-orientation views the same position and zoom."""
        if sceneIsBusy():
            return
        groups = {}
        for name, sliceWidget in self.logic.roleSliceWidgets(visibleOnly=True):
            groups.setdefault(self.logic.getSliceOrientation(sliceWidget.mrmlSliceNode()), []).append((name, sliceWidget))
        for members in groups.values():
            referenceWidget = next((w for name, w in members if SLICE_VIEW_ROLES[name][1] == "fusion"), members[0][1])
            if any(name in changedViews for name, _ in members):
                referenceWidget.sliceLogic().FitSliceToAll()
            for name, sliceWidget in members:
                if sliceWidget is not referenceWidget:
                    self.logic.copySliceGeometry(referenceWidget.mrmlSliceNode(), sliceWidget.mrmlSliceNode())

    # ------------------------------------------------------------------
    # MIP (3D view)
    # ------------------------------------------------------------------

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

    def updateRotationSpeed(self, value):
        viewNode = self.observeThreeDViewNode(updateButton=False)
        if viewNode is not None:
            viewNode.SetAnimationMs(int(value))

    def rotateMIPToViewAxis(self, axis):
        """Quick view buttons: 3 = anterior, 0 = left, 1 = right. Stops the rotation first."""
        self.setRotationEnabled(False)
        threeDWidget = self.getThreeDWidget()
        if threeDWidget is not None:
            threeDWidget.threeDView().rotateToViewAxis(axis)

    # ------------------------------------------------------------------
    # Window / level and color maps
    # ------------------------------------------------------------------

    def setCTWindow(self, window, level):
        ctNode = self.inputVolumeSelectorCT.currentNode()
        if ctNode and ctNode.GetDisplayNode():
            displayNode = ctNode.GetDisplayNode()
            displayNode.SetAutoWindowLevel(False)
            displayNode.SetWindow(window)
            displayNode.SetLevel(level)

    def setPETWindow(self, window, level):
        """Window / level of the PET in the slice views (PET-only views follow) and the MIP grey range."""
        petNode = self.inputVolumeSelector.currentNode()
        if not petNode:
            return
        if petNode.GetDisplayNode():
            displayNode = petNode.GetDisplayNode()
            displayNode.SetAutoWindowLevel(False)
            displayNode.SetWindow(window)
            displayNode.SetLevel(level)
        vrDisplayNode = slicer.modules.volumerendering.logic().GetFirstVolumeRenderingDisplayNode(petNode)
        self.logic.setMIPRange(vrDisplayNode, level - window / 2.0, level + window / 2.0)

    def onWindowShortcut(self, key):
        """F5-F9: apply the preset for this key that belongs to the view under the mouse (see windowPresetForView)."""
        view = self.logic.viewUnderCursor()
        preset = windowPresetForView(view[0], view[1], key) if view is not None else None
        if preset is None:
            return
        kind, text, window, level = preset
        if kind == "ct":
            self.setCTWindow(window, level)
            slicer.util.showStatusMessage(f"EasyFusion: {text}", 2000)
        else:
            self.setPETWindow(window, level)
            slicer.util.showStatusMessage(f"EasyFusion: SUV {text}", 2000)

    def setPETWindowPercentOfMax(self, percent):
        """SPECT presets: window from 0 to a percentage of the highest voxel value (count) in the volume."""
        petNode = self.inputVolumeSelector.currentNode()
        if petNode is None or petNode.GetImageData() is None:
            slicer.util.showStatusMessage("EasyFusion: select a SPECT/PET volume first.", 3000)
            return
        upper = percentOfMaximum(self.logic.getVolumeMaximum(petNode), percent)
        if upper is None:
            slicer.util.showStatusMessage("EasyFusion: the SPECT/PET volume has no positive counts.", 3000)
            return
        self.setPETWindow(upper, upper / 2.0)

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
                colorNode = None
            if colorNode is None:
                slicer.util.errorDisplay(f"Color node '{colorNodeName}' not found.")
                return
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
        if node is not None:
            # Scenes saved with an earlier version showed SUV text on the center point itself
            self.logic.styleRoiDisplayNode(node.GetDisplayNode())
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
        if sceneIsBusy():
            return  # scene event handlers trigger a refresh when done

        node = self.roiNode
        if node is not None and not scene.IsNodePresent(node):
            self.setRoiNode(None)
            node = None
        if node is None:
            # A just-loaded scene may already hold ROIs the panel has not connected to yet.
            # Never delete their spheres / handles / labels / segments in that case.
            node = self.logic.findRoiNode()
            if node is not None:
                self.setRoiNode(node)

        pet = self.inputVolumeSelector.currentNode()
        self.roiPetLabel.text = f"Measuring on: {pet.GetName()}" if pet else "Measuring on: (no PET selected)"

        if node is None:
            self.logic.removeRoiSphereModel()
            self.setHandlesNode(None)
            self.logic.removeRoiHandlesNode()
            self.logic.removeRoiLabelsNode()
            self.logic.removeRoiSegmentationNode()
            self.fillRoiTable([])
            self.mipOverlay.clear()
            return

        thresholdMode, thresholdValue = self.currentThreshold()
        selectedID = self.selectedRoiPointID()
        rows, spheres, newCache = [], [], {}
        labelEntries, segmentEntries, mipEntries = [], [], []
        roiColors = {}
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
                color = self.logic.getRoiColor(node, pointID)
                roiColors[pointID] = color

                key = self.logic.statsCacheKey(pet, center, radius, thresholdMode, thresholdValue)
                unchanged = key in self._roiStatsCache
                if unchanged:
                    stats = self._roiStatsCache[key]
                else:
                    stats = self.logic.computeSphereStatistics(pet, center, radius, thresholdMode, thresholdValue)
                newCache[key] = stats

                name = f"ROI-{number}"
                # The center point keeps only the name (shown in the Markups module); the SUV text is
                # drawn by the separate white label layer placed one radius away from the center.
                if node.GetNthControlPointLabel(index) != name:
                    node.SetNthControlPointLabel(index, name)

                rows.append((pointID, name, radius, stats))
                spheres.append((center, radius, color))
                labelEntries.append((pointID, center, radius, formatRoiLabel(name, stats, pet is not None)))
                mipEntries.append((pointID, center, formatRoiLabel(name, stats, pet is not None), color))
                segmentEntries.append((pointID, name, stats, unchanged, color))

            if pet is not None and rows and node.GetNodeReferenceID(ROI_PET_REFERENCE_ROLE) != pet.GetID():
                node.SetNodeReferenceID(ROI_PET_REFERENCE_ROLE, pet.GetID())
        finally:
            self._updatingRois = False

        self._roiStatsCache = newCache

        # Select a freshly placed ROI, or the one being resized, so the radius box follows it
        currentIDs = {row[0] for row in rows}
        newIDs = currentIDs - self._knownRoiIDs if self._knownRoiIDs is not None else set()
        newRoiCenter = None
        if len(newIDs) == 1:
            selectedID = next(iter(newIDs))
            newRoiCenter = next((center for _, pointID, center in rois if pointID == selectedID), None)
        elif draggedRoiID is not None:
            selectedID = draggedRoiID
        self._knownRoiIDs = currentIDs

        self.logic.updateRoiSphereModel(spheres)
        self.logic.updateRoiLabels(labelEntries)
        try:
            self.logic.updateRoiSegments(segmentEntries, pet)
        except Exception:
            logging.exception("EasyFusion: could not update ROI segments")
        try:
            self.mipOverlay.update(mipEntries, self.logic.findRoiSegmentationNode())
        except Exception:
            logging.exception("EasyFusion: could not update the MIP overlay")
        self.fillRoiTable(rows, selectedID, roiColors)
        self.syncRadiusSpinBox(rows, selectedID)
        if newRoiCenter is not None:
            self.jumpSliceViewsTo(newRoiCenter)

    @staticmethod
    def jumpSliceViewsTo(positionWorld):
        """
        Bring every slice view (all orientations, CT-only / PET-only views, both monitors) to a position.
        Offset jump: each view only changes its slice, it is not re-centered or panned, so the view the
        ROI was placed in stays exactly where it is.
        """
        x, y, z = (float(v) for v in positionWorld)
        try:
            slicer.modules.markups.logic().JumpSlicesToLocation(x, y, z, False)
        except Exception:
            slicer.vtkMRMLSliceNode.JumpAllSlices(slicer.mrmlScene, x, y, z, slicer.vtkMRMLSliceNode.OffsetJumpSlice)

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
        self.logic.styleRoiHandlesDisplayNode(handlesNode.GetDisplayNode())

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

    def fillRoiTable(self, rows, selectPointID=None, colors=None):
        if not hasattr(self, "roiTable"):
            return
        table = self.roiTable
        wasBlocked = table.blockSignals(True)
        try:
            table.setRowCount(len(rows))
            for rowIndex, (pointID, name, radius, stats) in enumerate(rows):
                values = formatRoiTableRow(name, radius, stats)
                for column, text in enumerate(values):
                    item = qt.QTableWidgetItem(text)
                    if column == 0:
                        item.setData(qt.Qt.UserRole, pointID)
                        color = (colors or {}).get(pointID)
                        if color is not None:  # color swatch next to the ROI name
                            item.setData(qt.Qt.DecorationRole,
                                         qt.QColor.fromRgbF(float(color[0]), float(color[1]), float(color[2])))
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

class MipRoiOverlay:
    """
    Segment surfaces and ROI text drawn on top of the MIP in every 3D view.
    Pure VTK in an extra render layer: nothing is added to the scene, so nothing is saved.
    Render windows are never kept: each update looks the views up again, so layout changes
    (views re-created, Monitor 2 window) only ever see renderers that belong to live windows.
    """

    def __init__(self):
        self.enabled = True
        self._overlays = []   # [(renderer, props)] currently attached to live 3D views

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    @staticmethod
    def _threeDViews():
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return []
        views = []
        for index in range(layoutManager.threeDViewCount):
            widget = layoutManager.threeDWidget(index)
            if widget is not None and widget.threeDView() is not None:
                views.append(widget.threeDView())
        return views

    def _overlayRenderer(self, threeDView):
        renderWindow = threeDView.renderWindow()
        for renderer, _ in self._overlays:
            if renderWindow.HasRenderer(renderer):
                return renderer
        mainRenderer = renderWindow.GetRenderers().GetFirstRenderer()
        if mainRenderer is None:
            return None
        renderer = vtk.vtkRenderer()
        layer = renderWindow.GetNumberOfLayers()
        renderWindow.SetNumberOfLayers(layer + 1)
        renderer.SetLayer(layer)
        renderer.SetInteractive(False)   # never picks or steals mouse interaction from the 3D view
        renderer.SetActiveCamera(mainRenderer.GetActiveCamera())
        renderWindow.AddRenderer(renderer)
        self._overlays.append((renderer, []))
        return renderer

    def _liveOverlays(self, views):
        """Forget overlays whose render window is gone (only Python references are dropped)."""
        windows = [view.renderWindow() for view in views]
        self._overlays = [(renderer, props) for renderer, props in self._overlays
                          if any(window.HasRenderer(renderer) for window in windows)]

    def clear(self):
        views = self._threeDViews()
        self._liveOverlays(views)
        for renderer, props in self._overlays:
            renderer.RemoveAllViewProps()
            props[:] = []
        for view in views:
            view.scheduleRender()

    @staticmethod
    def _segmentSurfaceWorld(segmentationNode, segmentID):
        polyData = vtk.vtkPolyData()
        if hasattr(segmentationNode, "GetClosedSurfaceRepresentation"):
            segmentationNode.GetClosedSurfaceRepresentation(segmentID, polyData)
        else:
            internal = segmentationNode.GetClosedSurfaceInternalRepresentation(segmentID)
            if internal is not None:
                polyData.DeepCopy(internal)
        if polyData.GetNumberOfPoints() == 0:
            return None
        transformNode = segmentationNode.GetParentTransformNode()
        if transformNode is None:
            return polyData
        toWorld = vtk.vtkGeneralTransform()
        transformNode.GetTransformToWorld(toWorld)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(toWorld)
        transformFilter.SetInputData(polyData)
        transformFilter.Update()
        worldPolyData = vtk.vtkPolyData()
        worldPolyData.DeepCopy(transformFilter.GetOutput())
        return worldPolyData

    @staticmethod
    def _textActor(text, positionWorld):
        actor = vtk.vtkBillboardTextActor3D()
        actor.SetInput(text)
        actor.SetPosition(*positionWorld)
        actor.SetDisplayOffset(*MIP_OVERLAY_TEXT_OFFSET_PX)
        textProperty = actor.GetTextProperty()
        textProperty.SetFontSize(MIP_OVERLAY_FONT_SIZE)
        textProperty.SetColor(*MIP_OVERLAY_TEXT_COLOR)
        textProperty.SetBackgroundColor(*MIP_OVERLAY_TEXT_BACKGROUND)
        textProperty.SetBackgroundOpacity(MIP_OVERLAY_TEXT_BACKGROUND_OPACITY)
        textProperty.SetShadow(False)
        textProperty.SetBold(True)
        textProperty.SetJustificationToLeft()
        textProperty.SetVerticalJustificationToBottom()
        return actor

    def update(self, entries, segmentationNode):
        """
        entries: list of (roiPointID, centerWorld, text, color). Segment surfaces are taken from segmentationNode
        (segment ID = ROI_SEGMENT_ID_PREFIX + roiPointID); a ROI whose segment is empty only shows its text.
        """
        if not self.enabled or not entries or sceneIsBusy():
            self.clear()
            return
        views = self._threeDViews()
        self._liveOverlays(views)
        if not views:
            return

        surfaces = []
        if segmentationNode is not None and slicer.mrmlScene.IsNodePresent(segmentationNode):
            Easy_fusionLogic.ensureUnsmoothedClosedSurface(segmentationNode)
            segmentation = segmentationNode.GetSegmentation()
            for roiID, _, _, color in entries:
                segmentID = ROI_SEGMENT_ID_PREFIX + roiID
                if segmentation.GetSegment(segmentID) is None:
                    continue
                polyData = self._segmentSurfaceWorld(segmentationNode, segmentID)
                if polyData is not None:
                    surfaces.append((polyData, color))

        for view in views:
            renderer = self._overlayRenderer(view)
            if renderer is None:
                continue
            props = next(props for r, props in self._overlays if r is renderer)
            renderer.RemoveAllViewProps()
            props[:] = []
            for polyData, color in surfaces:
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polyData)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*color)
                actor.GetProperty().SetOpacity(MIP_OVERLAY_SEGMENT_OPACITY)
                renderer.AddViewProp(actor)
                props.append(actor)
            for _, center, text, _ in entries:
                actor = self._textActor(text, center)
                renderer.AddViewProp(actor)
                props.append(actor)
            view.scheduleRender()


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

    @staticmethod
    def setMIPRange(vrDisplayNode, lower, upper, flatOpacity=False):
        """MIP shows white at `lower` to black at `upper`. flatOpacity: every intensity fully opaque (set by Go)."""
        propertyNode = vrDisplayNode.GetVolumePropertyNode() if vrDisplayNode is not None else None
        if propertyNode is None:
            return
        colorFunction = propertyNode.GetVolumeProperty().GetRGBTransferFunction(0)
        colorFunction.RemoveAllPoints()
        colorFunction.AddRGBPoint(lower, 1.0, 1.0, 1.0)
        colorFunction.AddRGBPoint(upper, 0.0, 0.0, 0.0)
        if flatOpacity:
            scalarOpacity = propertyNode.GetScalarOpacity()
            scalarOpacity.RemoveAllPoints()
            scalarOpacity.AddPoint(lower, 1.0)
            scalarOpacity.AddPoint(upper, 1.0)
        propertyNode.Modified()
        vrDisplayNode.Modified()

    @staticmethod
    def getVolumeMaximum(volumeNode):
        """Highest voxel value (e.g. SPECT counts) of a volume; 0 if it has no image data."""
        imageData = volumeNode.GetImageData() if volumeNode is not None else None
        if imageData is None or imageData.GetNumberOfPoints() == 0:
            return 0.0
        return float(imageData.GetScalarRange()[1])

    # --- Layouts and view contents ------------------------------------------

    @staticmethod
    def ensureLayoutsRegistered():
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return
        layoutNode = layoutManager.layoutLogic().GetLayoutNode()
        for layoutID, description in buildLayoutDescriptions().items():
            if not layoutNode.IsLayoutDescription(layoutID):
                layoutNode.AddLayoutDescription(layoutID, description)
            elif (layoutNode.GetLayoutDescription(layoutID) != description
                  and layoutNode.GetViewArrangement() != layoutID):
                # Rebuilding the layout that is on screen is never needed and is risky
                layoutNode.SetLayoutDescription(layoutID, description)

    @staticmethod
    def reapplyRestoredCustomLayout():
        """
        Slicer restores the last used layout at startup, before modules can register their layouts. If that
        was an Lvgvs layout, it was applied without a description. Re-applying the arrangement now that the
        description exists is the same call Slicer's own vtkMRMLLayoutLogic makes for this situation.
        """
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return
        layoutNode = layoutManager.layoutLogic().GetLayoutNode()
        arrangement = layoutNode.GetViewArrangement()
        if arrangement in CUSTOM_LAYOUT_IDS:
            layoutNode.SetViewArrangement(arrangement)

    @staticmethod
    def viewUnderCursor():
        """
        ("slice", slice view name) or ("threeD", None) for the view under the mouse, in any window (so the
        Monitor 2 window too); None when the mouse is not over a view.
        """
        widget = qt.QApplication.widgetAt(qt.QCursor.pos())
        while widget is not None:
            if widget.inherits("qMRMLThreeDWidget"):
                return ("threeD", None)
            if widget.inherits("qMRMLSliceWidget"):
                return ("slice", Easy_fusionLogic._sliceViewName(widget))
            widget = widget.parentWidget()
        return None

    @staticmethod
    def _sliceViewName(sliceWidget):
        try:
            return sliceWidget.mrmlSliceNode().GetLayoutName()
        except Exception:
            layoutManager = slicer.app.layoutManager()
            if layoutManager is None:
                return None
            return next((name for name in layoutManager.sliceViewNames()
                         if layoutManager.sliceWidget(name) == sliceWidget), None)

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
    def roleSliceWidgets(visibleOnly=False):
        """(name, slice widget) of every EasyFusion slice view (SLICE_VIEW_ROLES) in the current layout."""
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return []
        widgets = []
        for name in layoutManager.sliceViewNames():
            if name not in SLICE_VIEW_ROLES:
                continue
            sliceWidget = layoutManager.sliceWidget(name)
            if sliceWidget is None or (visibleOnly and not sliceWidget.visible):
                continue
            sliceNode = sliceWidget.mrmlSliceNode()
            if sliceNode is None or not slicer.mrmlScene.IsNodePresent(sliceNode):
                continue  # widget left over from a previous layout / scene
            widgets.append((name, sliceWidget))
        return widgets

    @staticmethod
    def roleSliceNodes():
        """
        (name, slice node, slice composite node) of every EasyFusion view that exists in the scene.
        Works on MRML nodes only, so it is safe while the layout manager is creating or deleting view widgets.
        """
        scene = slicer.mrmlScene
        result = []
        for name in SLICE_VIEW_ROLES:
            sliceNode = scene.GetSingletonNode(name, "vtkMRMLSliceNode")
            compositeNode = scene.GetSingletonNode(name, "vtkMRMLSliceCompositeNode")
            if sliceNode is not None and compositeNode is not None:
                result.append((name, sliceNode, compositeNode))
        return result

    @staticmethod
    def _addWithFixedID(node, nodeID):
        """Add a node under nodeID when that ID is free (otherwise the scene picks one as usual)."""
        if slicer.mrmlScene.GetNodeByID(nodeID) is None:
            try:
                node.SetID(nodeID)
            except Exception:
                pass  # older Slicer: fall back to an automatic ID
        return slicer.mrmlScene.AddNode(node)

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
            # Build the twin completely *before* it enters the scene. Views may already refer to the ID the
            # scene is about to hand out, and must never see a volume without image data or display node.
            node = slicer.vtkMRMLScalarVolumeNode()
            node.SetAttribute(PET_ONLY_VOLUME_ATTRIBUTE, "1")
            node.SetHideFromEditors(True)
            node.SetSaveWithScene(False)
            node.SetName(f"{petNode.GetName()} (PET only)")
            node.CopyOrientation(petNode)
            node.SetAndObserveImageData(petNode.GetImageData())
            node.SetAndObserveTransformNodeID(petNode.GetTransformNodeID())
            node.SetNodeReferenceID(PET_ONLY_SOURCE_ROLE, petNode.GetID())
            displayNode = slicer.vtkMRMLScalarVolumeDisplayNode()
            displayNode.SetSaveWithScene(False)
            displayNode = self._addWithFixedID(displayNode, PET_ONLY_DISPLAY_ID)
            self._configurePetOnlyDisplay(displayNode, petNode)
            node.SetAndObserveDisplayNodeID(displayNode.GetID())
            node = self._addWithFixedID(node, PET_ONLY_VOLUME_ID)
        else:
            node.SetName(f"{petNode.GetName()} (PET only)")
            node.CopyOrientation(petNode)
            if node.GetImageData() is not petNode.GetImageData():
                node.SetAndObserveImageData(petNode.GetImageData())
            if node.GetTransformNodeID() != petNode.GetTransformNodeID():
                node.SetAndObserveTransformNodeID(petNode.GetTransformNodeID())
            node.SetNodeReferenceID(PET_ONLY_SOURCE_ROLE, petNode.GetID())
            if node.GetDisplayNode() is None:
                displayNode = slicer.vtkMRMLScalarVolumeDisplayNode()
                displayNode.SetSaveWithScene(False)
                displayNode = self._addWithFixedID(displayNode, PET_ONLY_DISPLAY_ID)
                node.SetAndObserveDisplayNodeID(displayNode.GetID())
            self._configurePetOnlyDisplay(node.GetDisplayNode(), petNode)

        petDisplayNode = petNode.GetDisplayNode()
        if petDisplayNode is not None:
            bindWindowLevelSync(petDisplayNode, node.GetDisplayNode())
        return node

    @staticmethod
    def _configurePetOnlyDisplay(displayNode, petNode):
        petDisplayNode = petNode.GetDisplayNode()
        wasModifying = displayNode.StartModify()
        displayNode.SetAndObserveColorNodeID(slicer.util.getNode("InvertedGrey").GetID())
        if petDisplayNode is not None:
            displayNode.SetAutoWindowLevel(False)
            displayNode.SetWindow(petDisplayNode.GetWindow())
            displayNode.SetLevel(petDisplayNode.GetLevel())
            displayNode.SetInterpolate(petDisplayNode.GetInterpolate())
        displayNode.EndModify(wasModifying)

    @staticmethod
    def pointPetOnlyViewsAtSourcePet():
        """Called when saving starts. Returns what was changed so it can be undone when saving ends."""
        twin = Easy_fusionLogic.findPetOnlyVolume()
        if twin is None:
            return []
        twinID = twin.GetID()
        sourceID = twin.GetNodeReferenceID(PET_ONLY_SOURCE_ROLE)
        if sourceID is not None and slicer.mrmlScene.GetNodeByID(sourceID) is None:
            sourceID = None
        swaps = []
        for compositeNode in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            if compositeNode.GetBackgroundVolumeID() == twinID:
                compositeNode.SetBackgroundVolumeID(sourceID)
                swaps.append((compositeNode, twinID, sourceID))
        return swaps

    @staticmethod
    def restorePetOnlyViews(swaps):
        for compositeNode, twinID, sourceID in swaps:
            if slicer.mrmlScene.GetNodeByID(twinID) is None:
                continue
            if compositeNode.GetBackgroundVolumeID() == sourceID:
                compositeNode.SetBackgroundVolumeID(twinID)

    @staticmethod
    def clearDanglingViewReferences():
        """Remove slice view volume references to nodes that do not exist (e.g. the unsaved twin in older scenes)."""
        scene = slicer.mrmlScene
        roles = (("GetBackgroundVolumeID", "SetBackgroundVolumeID"),
                 ("GetForegroundVolumeID", "SetForegroundVolumeID"),
                 ("GetLabelVolumeID", "SetLabelVolumeID"))
        cleared = 0
        for compositeNode in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            for getterName, setterName in roles:
                nodeID = getattr(compositeNode, getterName)()
                if nodeID and scene.GetNodeByID(nodeID) is None:
                    getattr(compositeNode, setterName)(None)
                    cleared += 1
        return cleared

    def applyViewRoles(self, petNode, ctNode, forceOrientation=False):
        """
        Fill every EasyFusion slice view according to SLICE_VIEW_ROLES (also views not in the current layout,
        so switching layouts later shows the right content). Only MRML nodes are touched; the views follow.
        Returns the names of views whose background volume or orientation changed (those get re-fitted).
        """
        if petNode is None or ctNode is None or sceneIsBusy():
            return []
        petOnlyNode = self.getOrCreatePetOnlyVolume(petNode)
        roleNodes = self.roleSliceNodes()

        # Unlink while assigning, so linked views cannot copy volume selections to each other. If the views
        # were linked (Slicer's view-link button, used for scroll / pan / zoom sync), link them all again after.
        compositeNodes = [compositeNode for _, _, compositeNode in roleNodes]
        wasLinked = any(compositeNode.GetLinkedControl() for compositeNode in compositeNodes)
        for compositeNode in compositeNodes:
            if compositeNode.GetLinkedControl():
                compositeNode.SetLinkedControl(False)
        try:
            return self._assignViewVolumes(roleNodes, petNode, ctNode, petOnlyNode, forceOrientation)
        finally:
            if wasLinked:
                for compositeNode in compositeNodes:
                    compositeNode.SetLinkedControl(True)

    def _assignViewVolumes(self, roleNodes, petNode, ctNode, petOnlyNode, forceOrientation):
        """Orientation and background / foreground volumes of each EasyFusion view (see applyViewRoles)."""
        changedViews = []
        for name, sliceNode, compositeNode in roleNodes:
            orientation, content = SLICE_VIEW_ROLES[name]
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
        layoutNode = slicer.mrmlScene.GetSingletonNode("vtkMRMLLayoutNode", "vtkMRMLLayoutNode")
        if layoutNode is None or layoutNode.GetViewArrangement() not in CUSTOM_LAYOUT_IDS:
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
                # Re-assign (not just "same ID") so the display re-fetches the lookup table object.
                # Go through a valid table: a display node must never be left without a color node.
                wasModifying = displayNode.StartModify()
                greyNode = slicer.mrmlScene.GetNodeByID("vtkMRMLColorTableNodeGrey")
                if greyNode is not None and displayNode.GetColorNodeID() == canonical.GetID():
                    displayNode.SetAndObserveColorNodeID(greyNode.GetID())
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
        crossDot = getattr(slicer.vtkMRMLMarkupsDisplayNode, "CrossDot2D", None)
        if crossDot is not None:
            displayNode.SetGlyphType(crossDot)
        # Markups draw a point's label in the point's own color and right next to it, so the white,
        # padded SUV text lives in a separate label layer (see updateRoiLabels)
        if hasattr(displayNode, "SetPointLabelsVisibility"):
            displayNode.SetPointLabelsVisibility(False)

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
    def getRoiColor(node, pointID):
        """Random color picked once per ROI and stored on the ROI list node, so it survives save / reload."""
        color = parseRoiColor(node.GetAttribute(ROI_COLOR_ATTRIBUTE + pointID))
        if color is None:
            color = randomRoiColor()
            node.SetAttribute(ROI_COLOR_ATTRIBUTE + pointID, " ".join(f"{c:.4f}" for c in color))
        return color

    @staticmethod
    def forgetRoi(node, pointID):
        node.RemoveAttribute(ROI_RADIUS_ATTRIBUTE + pointID)
        node.RemoveAttribute(ROI_NUMBER_ATTRIBUTE + pointID)
        node.RemoveAttribute(ROI_COLOR_ATTRIBUTE + pointID)

    @staticmethod
    def forgetAllRois(node):
        for name in list(node.GetAttributeNames()):
            if (name.startswith(ROI_RADIUS_ATTRIBUTE) or name.startswith(ROI_NUMBER_ATTRIBUTE)
                    or name.startswith(ROI_COLOR_ATTRIBUTE)):
                node.RemoveAttribute(name)
        node.SetAttribute(ROI_NEXT_NUMBER_ATTRIBUTE, "1")

    @staticmethod
    def statsCacheKey(volumeNode, center, radius, thresholdMode=THRESHOLD_RELATIVE, thresholdValue=DEFAULT_RELATIVE_THRESHOLD):
        roundedCenter = tuple(round(c, 3) for c in center)
        thresholdKey = (thresholdMode, round(float(thresholdValue), 4))
        if volumeNode is None:
            return (None, roundedCenter, round(radius, 3), thresholdKey)
        imageData = volumeNode.GetImageData()
        transformNode = volumeNode.GetParentTransformNode()
        return (volumeNode.GetID(), volumeNode.GetMTime(),
                imageData.GetMTime() if imageData else 0,
                transformNode.GetMTime() if transformNode else 0,
                roundedCenter, round(radius, 3), thresholdKey)

    @staticmethod
    def computeSphereStatistics(volumeNode, centerWorld, radiusMm,
                                thresholdMode=THRESHOLD_RELATIVE, thresholdValue=DEFAULT_RELATIVE_THRESHOLD):
        """SUV statistics (sphere + thresholded segment) of the PET voxels inside a sphere given in world coordinates."""
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
            center, radiusMm, thresholdMode, thresholdValue)

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

    def createRoiHandlesNode(self):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "SUV ROI radius handles")
        node.SetAttribute(ROI_HANDLES_ATTRIBUTE, "1")
        if hasattr(node, "SetControlPointLabelFormat"):
            node.SetControlPointLabelFormat("")
        node.CreateDefaultDisplayNodes()
        self.styleRoiHandlesDisplayNode(node.GetDisplayNode())
        return node

    @staticmethod
    def styleRoiHandlesDisplayNode(displayNode):
        if displayNode is None:
            return
        if displayNode.GetGlyphScale() != ROI_HANDLE_GLYPH_SCALE:
            displayNode.SetGlyphScale(ROI_HANDLE_GLYPH_SCALE)
        if tuple(displayNode.GetSelectedColor()) != (1.0, 0.85, 0.0):
            displayNode.SetSelectedColor(1.0, 0.85, 0.0)
            displayNode.SetColor(1.0, 0.85, 0.0)
        square = getattr(slicer.vtkMRMLMarkupsDisplayNode, "Square2D", None)
        if square is not None and displayNode.GetGlyphType() != square:
            displayNode.SetGlyphType(square)
        if hasattr(displayNode, "SetPointLabelsVisibility") and displayNode.GetPointLabelsVisibility():
            displayNode.SetPointLabelsVisibility(False)
        if hasattr(displayNode, "SetPropertiesLabelVisibility") and displayNode.GetPropertiesLabelVisibility():
            displayNode.SetPropertiesLabelVisibility(False)
        if hasattr(displayNode, "SetVisibility3D") and displayNode.GetVisibility3D():
            displayNode.SetVisibility3D(False)  # handles are for slice views; keep the MIP clean

    def removeRoiHandlesNode(self):
        handlesNode = self.findRoiHandlesNode()
        if handlesNode is not None:
            slicer.mrmlScene.RemoveNode(handlesNode)

    @staticmethod
    def styleRoiSphereDisplayNode(displayNode):
        """Sphere outlines take each ROI's own color from the RGB point scalars (also for older saved scenes)."""
        if displayNode.GetSliceIntersectionThickness() != ROI_SPHERE_OUTLINE_PX:
            displayNode.SetSliceIntersectionThickness(ROI_SPHERE_OUTLINE_PX)
        directMapping = getattr(slicer.vtkMRMLDisplayNode, "UseDirectMapping", None)
        if directMapping is None:
            return  # very old Slicer: single color outlines
        if displayNode.GetActiveScalarName() != ROI_SPHERE_COLOR_ARRAY:
            displayNode.SetActiveScalar(ROI_SPHERE_COLOR_ARRAY, vtk.vtkAssignAttribute.POINT_DATA)
        if displayNode.GetScalarRangeFlag() != directMapping:
            displayNode.SetScalarRangeFlag(directMapping)
        if not displayNode.GetScalarVisibility():
            displayNode.SetScalarVisibility(True)

    def updateRoiSphereModel(self, spheres):
        """
        One model holding all spheres; its slice intersections draw the ROI circles in 2D views.
        spheres: list of (center, radius, color). Each sphere carries its ROI color as RGB point scalars.
        """
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
                displayNode.SetVisibility3D(False)  # keep the MIP uncluttered
            else:
                displayNode.SetSliceIntersectionVisibility(True)
        displayNode = modelNode.GetDisplayNode()
        if displayNode is not None:
            self.styleRoiSphereDisplayNode(displayNode)

        append = vtk.vtkAppendPolyData()
        for center, radius, color in spheres:
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(center)
            sphere.SetRadius(radius)
            sphere.SetThetaResolution(48)
            sphere.SetPhiResolution(24)
            sphere.Update()
            spherePolyData = vtk.vtkPolyData()
            spherePolyData.DeepCopy(sphere.GetOutput())
            colors = vtk.vtkUnsignedCharArray()
            colors.SetName(ROI_SPHERE_COLOR_ARRAY)
            colors.SetNumberOfComponents(3)
            rgb = [int(round(255 * c)) for c in color]
            for _ in range(spherePolyData.GetNumberOfPoints()):
                colors.InsertNextTuple3(*rgb)
            spherePolyData.GetPointData().AddArray(colors)
            append.AddInputData(spherePolyData)
        append.Update()
        polyData = vtk.vtkPolyData()
        polyData.DeepCopy(append.GetOutput())
        modelNode.SetAndObservePolyData(polyData)

    # --- ROI labels (white SUV text, one radius away from the center) -------

    @staticmethod
    def findRoiLabelsNode():
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode"):
            if node.GetAttribute(ROI_LABELS_ATTRIBUTE):
                return node
        return None

    def createRoiLabelsNode(self):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "SUV ROI labels")
        node.SetAttribute(ROI_LABELS_ATTRIBUTE, "1")
        node.SetLocked(True)  # text only: never picked or dragged
        if hasattr(node, "SetControlPointLabelFormat"):
            node.SetControlPointLabelFormat("")
        node.CreateDefaultDisplayNodes()
        self.styleRoiLabelsDisplayNode(node.GetDisplayNode())
        return node

    @staticmethod
    def styleRoiLabelsDisplayNode(displayNode):
        if displayNode is None:
            return
        wasModifying = displayNode.StartModify()
        # Markups use the glyph color for the text, so the whole layer is white
        displayNode.SetColor(1.0, 1.0, 1.0)
        displayNode.SetSelectedColor(1.0, 1.0, 1.0)
        displayNode.SetTextScale(ROI_LABEL_TEXT_SCALE)
        # Practically invisible anchor glyph; also keeps the text right at the anchor
        dash = getattr(slicer.vtkMRMLMarkupsDisplayNode, "Dash2D", None)
        if dash is not None:
            displayNode.SetGlyphType(dash)
        displayNode.SetGlyphScale(0.2)
        if hasattr(displayNode, "SetPointLabelsVisibility"):
            displayNode.SetPointLabelsVisibility(True)
        if hasattr(displayNode, "SetPropertiesLabelVisibility"):
            displayNode.SetPropertiesLabelVisibility(False)
        if hasattr(displayNode, "SetVisibility3D"):
            displayNode.SetVisibility3D(False)  # the MIP background is white
        textProperty = displayNode.GetTextProperty() if hasattr(displayNode, "GetTextProperty") else None
        if textProperty is not None:  # flat: no box, no shadow, no frame
            textProperty.SetBackgroundOpacity(0.0)
            textProperty.SetShadow(False)
            textProperty.SetFrame(False)
        displayNode.EndModify(wasModifying)

    def updateRoiLabels(self, entries):
        """entries: list of (roiPointID, centerWorld, radius, text). Keeps 3 anchors per ROI in sync."""
        labelsNode = self.findRoiLabelsNode()
        if not entries:
            if labelsNode is not None and labelsNode.GetNumberOfControlPoints() > 0:
                labelsNode.RemoveAllControlPoints()
            return
        if labelsNode is None:
            labelsNode = self.createRoiLabelsNode()
        elif not labelsNode.GetAttribute("EasyFusion.StyleVersion") == "1":
            self.styleRoiLabelsDisplayNode(labelsNode.GetDisplayNode())
        labelsNode.SetAttribute("EasyFusion.StyleVersion", "1")

        planeCount = len(LABEL_ANCHOR_DIRECTIONS)
        wanted = {}
        for roiID, center, radius, text in entries:
            for plane in range(planeCount):
                wanted[(roiID, plane)] = (labelAnchorPosition(center, radius, plane), text)

        seen = set()
        for i in reversed(range(labelsNode.GetNumberOfControlPoints())):
            key = parseHandleDescription(labelsNode.GetNthControlPointDescription(i), planeCount)
            if key is None or key not in wanted or key in seen:
                labelsNode.RemoveNthControlPoint(i)
            else:
                seen.add(key)

        indexByKey = {}
        for i in range(labelsNode.GetNumberOfControlPoints()):
            key = parseHandleDescription(labelsNode.GetNthControlPointDescription(i), planeCount)
            if key is not None:
                indexByKey[key] = i

        for key, (position, text) in wanted.items():
            index = indexByKey.get(key)
            if index is None:
                index = labelsNode.AddControlPoint(position)
                labelsNode.SetNthControlPointDescription(index, f"{key[0]}:{key[1]}")
            else:
                current = [0.0, 0.0, 0.0]
                labelsNode.GetNthControlPointPositionWorld(index, current)
                if _distance(current, position) > 1e-3:
                    labelsNode.SetNthControlPointPositionWorld(index, position[0], position[1], position[2])
            if labelsNode.GetNthControlPointLabel(index) != text:
                labelsNode.SetNthControlPointLabel(index, text)

    def removeRoiLabelsNode(self):
        labelsNode = self.findRoiLabelsNode()
        if labelsNode is not None:
            slicer.mrmlScene.RemoveNode(labelsNode)

    # --- ROI segments (threshold inside each sphere: MTV / TLG) -------------

    @staticmethod
    def findRoiSegmentationNode():
        for node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
            if node.GetAttribute(ROI_SEGMENTATION_ATTRIBUTE):
                return node
        return None

    @staticmethod
    def styleRoiSegmentationDisplayNode(displayNode):
        if displayNode is None:
            return
        wasModifying = displayNode.StartModify()
        displayNode.SetVisibility2DFill(False)   # outline only
        displayNode.SetOpacity2DFill(0.0)
        displayNode.SetVisibility2DOutline(True)
        displayNode.SetOpacity2DOutline(1.0)
        displayNode.SetSliceIntersectionThickness(ROI_SEGMENT_OUTLINE_PX)
        displayNode.SetVisibility3D(False)
        displayNode.EndModify(wasModifying)

    @staticmethod
    def ensureUnsmoothedClosedSurface(segmentationNode):
        """Closed surface (3D) without smoothing; rebuilt once if it was made with other settings (older scenes)."""
        segmentation = segmentationNode.GetSegmentation()
        smoothingChanged = segmentation.GetConversionParameter("Smoothing factor") != SEGMENT_SURFACE_SMOOTHING
        if smoothingChanged:
            segmentation.SetConversionParameter("Smoothing factor", SEGMENT_SURFACE_SMOOTHING)
        if not hasattr(segmentationNode, "CreateClosedSurfaceRepresentation"):
            return
        if smoothingChanged and hasattr(segmentationNode, "RemoveClosedSurfaceRepresentation"):
            segmentationNode.RemoveClosedSurfaceRepresentation()
        segmentationNode.CreateClosedSurfaceRepresentation()

    def getOrCreateRoiSegmentationNode(self, petNode):
        segmentationNode = self.findRoiSegmentationNode()
        if segmentationNode is None:
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "SUV ROI segments")
            segmentationNode.SetAttribute(ROI_SEGMENTATION_ATTRIBUTE, "1")
            segmentationNode.CreateDefaultDisplayNodes()
        self.styleRoiSegmentationDisplayNode(segmentationNode.GetDisplayNode())
        # Segment masks are computed on the PET voxel grid, so the segmentation uses the PET geometry
        if segmentationNode.GetNodeReferenceID(ROI_SEGMENTATION_PET_ROLE) != petNode.GetID():
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(petNode)
            segmentationNode.SetNodeReferenceID(ROI_SEGMENTATION_PET_ROLE, petNode.GetID())
        if segmentationNode.GetTransformNodeID() != petNode.GetTransformNodeID():
            segmentationNode.SetAndObserveTransformNodeID(petNode.GetTransformNodeID())
        return segmentationNode

    @staticmethod
    def writeSegmentMask(segmentationNode, segmentID, petNode, extent, mask):
        """Replace a segment with a boolean mask given on the PET voxel grid ([k, j, i] over extent)."""
        from vtk.util import numpy_support
        labelmap = slicer.vtkOrientedImageData()
        ijkToRas = vtk.vtkMatrix4x4()
        petNode.GetIJKToRASMatrix(ijkToRas)
        labelmap.SetImageToWorldMatrix(ijkToRas)
        labelmap.SetExtent(*extent)
        labelmap.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        scalars = labelmap.GetPointData().GetScalars()
        numpy_support.vtk_to_numpy(scalars)[:] = np.asarray(mask, dtype=np.uint8).ravel()
        scalars.Modified()
        mode = getattr(slicer.vtkSlicerSegmentationsModuleLogic, "MODE_REPLACE", 0)
        slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(labelmap, segmentationNode, segmentID, mode)

    def updateRoiSegments(self, entries, petNode):
        """entries: list of (roiPointID, name, stats, unchanged, color). One segment per ROI, removed with the ROI."""
        segmentationNode = self.findRoiSegmentationNode()
        if petNode is None or petNode.GetImageData() is None:
            return
        if not entries and segmentationNode is None:
            return
        segmentationNode = self.getOrCreateRoiSegmentationNode(petNode)
        segmentation = segmentationNode.GetSegmentation()

        wantedIDs = set()
        for roiID, name, stats, unchanged, color in entries:
            segmentID = ROI_SEGMENT_ID_PREFIX + roiID
            wantedIDs.add(segmentID)
            segment = segmentation.GetSegment(segmentID)
            if segment is None:
                segmentation.AddEmptySegment(segmentID, name, list(color))
                unchanged = False
            else:
                if segment.GetName() != name:
                    segment.SetName(name)
                if any(abs(a - b) > 1e-3 for a, b in zip(segment.GetColor(), color)):
                    segment.SetColor(*color)
            if unchanged:
                continue  # same ROI, PET and threshold as last time: segment is already up to date
            if stats is None:
                self.writeSegmentMask(segmentationNode, segmentID, petNode, (0, 0, 0, 0, 0, 0), np.zeros((1, 1, 1), bool))
            else:
                self.writeSegmentMask(segmentationNode, segmentID, petNode, stats["extent"], stats["segMask"])

        existingIDs = vtk.vtkStringArray()
        segmentation.GetSegmentIDs(existingIDs)
        for i in range(existingIDs.GetNumberOfValues()):
            segmentID = existingIDs.GetValue(i)
            if segmentID.startswith(ROI_SEGMENT_ID_PREFIX) and segmentID not in wantedIDs:
                segmentation.RemoveSegment(segmentID)

    def removeRoiSegmentationNode(self):
        segmentationNode = self.findRoiSegmentationNode()
        if segmentationNode is not None:
            slicer.mrmlScene.RemoveNode(segmentationNode)
