import os
import re
import ast
import gc
import html
import math
import time
import random
import colorsys
import logging
import importlib
import traceback
import configparser
import contextlib

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
# Each ROI keeps its own threshold; the panel's threshold is the default given to new ROIs
ROI_THRESHOLD_ATTRIBUTE = "EasyFusion.RoiThreshold_"  # + control point ID, e.g. "relative 40" / "absolute 2.5"
# Who created an ROI (for future AI lesion detection): stored per ROI, saved with the scene
ROI_ORIGIN_ATTRIBUTE = "EasyFusion.RoiOrigin_"        # + control point ID
ROI_ORIGIN_USER = "user"
ROI_ORIGIN_AI = "ai"
# Values that can be shown next to each ROI in the slice views and on the MIP (key, checkbox text).
# A display preference of the user, so it is kept in the application settings, not in the scene.
ROI_LABEL_FIELDS = (("name", "Name"), ("max", "Max"), ("mean", "Mean"), ("mtv", "MTV"), ("tlg", "TLG"),
                    ("radius", "Radius"), ("threshold", "Threshold"))
DEFAULT_ROI_LABEL_FIELDS = ("name", "max", "mean")
SETTINGS_ROI_LABEL_FIELDS = "EasyFusion/RoiLabelFields"
# Folder of the last ROI table export (application setting, like the label fields)
SETTINGS_ROI_EXPORT_FOLDER = "EasyFusion/RoiExportFolder"

# Slice view text. Slicer's own corner annotations (Data Probe) and EasyFusion's window info in the bottom-right corner
DATA_PROBE_ANNOTATIONS_SETTING = "DataProbe/sliceViewAnnotations.enabled"   # Slicer's setting, 1 / 0
# Slicer's per-corner switches ("active corners"): (attribute of the annotations object, its check box, setting)
DATA_PROBE_CORNERS = (
    ("topLeft", "topLeftCheckBox", "DataProbe/sliceViewAnnotations.topLeft"),
    ("topRight", "topRightCheckBox", "DataProbe/sliceViewAnnotations.topRight"),
    ("bottomLeft", "bottomLeftCheckBox", "DataProbe/sliceViewAnnotations.bottomLeft"),
)
DATA_PROBE_CORNER_INDEXES = (2, 3, 0)   # the same corners in the view's vtkCornerAnnotation
# Which corners were on when EasyFusion hid the annotations, e.g. "1,1,0" (restored by showing them again)
SETTINGS_SLICER_ANNOTATION_CORNERS = "EasyFusion/SlicerAnnotationCorners"
DATA_PROBE_FONT_SIZE_SETTING = "DataProbe/sliceViewAnnotations.fontSize"
SETTINGS_SHOW_WINDOW_INFO = "EasyFusion/ShowWindowInfo"
WINDOW_INFO_CORNER = 1               # vtkCornerAnnotation: 0 lower left, 1 lower right, 2 upper left, 3 upper right
WINDOW_INFO_DEFAULT_FONT_SIZE = 14   # same default as Slicer's slice view annotations
WINDOW_INFO_UPDATE_MS = 40           # at most ~25 updates per second while window / level is being dragged
WINDOW_INFO_LIGHT_TEXT = (1.0, 1.0, 1.0)    # on CT / fusion views (dark background)
WINDOW_INFO_DARK_TEXT = (0.1, 0.1, 0.1)     # on PET-only views (inverted grey: white background)

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
# MIP zoom fit: the 3D view always starts showing this much anatomy from top to bottom (mm)
MIP_FIT_HEIGHT_MM = 1200.0
# Fit again once the orthographic switch and the layout change have been applied (after alignSliceViews' 150 ms)
MIP_FIT_DELAY_MS = 300
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
# MRI (arbitrary intensity units, e.g. PET/MRI): window between two percentiles of the tissue voxels
# (button text, lower percentile, upper percentile, shortcut key over CT/MRI views when the volume is an MRI)
MRI_PERCENTILE_PRESETS = [
    ("Standard", 1.0, 99.0, "F5"),
    ("Wide", 0.5, 99.5, "F6"),
    ("Contrast", 5.0, 95.0, "F7"),
    ("Bright", 0.0, 90.0, "F8"),
]
# Voxels below this fraction of the 99.5th percentile count as air / background, not tissue
MRI_BACKGROUND_FRACTION = 0.05
# At most this many voxels are sampled for the percentiles (fast on large volumes)
MRI_PRESET_SAMPLE_SIZE = 2000000
# A CT has air around -1000 HU; an MRI (almost) never goes this far below 0
CT_MINIMUM_BELOW = -500.0
# Wait for both selectors before updating the views (e.g. one change of each in a row)
INPUT_UPDATE_DELAY_MS = 150

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

# ---------------------------------------------------------------------------
# AI post-processing filters (models of the Belenos PET Denoise module: <name>.pth + <name>.txt sidecar)
# ---------------------------------------------------------------------------

# A filter never changes its input: the result is always a new volume, tagged with where it came from
FILTER_MODEL_ATTRIBUTE = "EasyFusion.FilterModel"
FILTER_DATE_ATTRIBUTE = "EasyFusion.FilterDate"
FILTER_SOURCE_ROLE = "EasyFusionFilterSource"
FILTER_TARGET_PET = "pet"
FILTER_TARGET_CT = "ct"
SETTINGS_FILTER_MODEL_FOLDER = "EasyFusion/FilterModelFolder"  # application setting (qt.QSettings)
# Inference settings of the PETDenoise module, so a model gives the same result in both modules
FILTER_WINDOW_OVERLAP = 0.25
FILTER_MIN_VRAM_GB = 1.9
FILTER_EINOPS_REQUIREMENT = "einops==0.6.1"
# Rough peak memory of one run, in float32 copies of the volume on the model's voxel grid
FILTER_MEMORY_COPIES = 6
# A run counts as large (the user is advised to crop) above this memory estimate or amount of network work
# (number of windows x voxels per window; 5e8 is about 1900 windows of 64x64x64)
FILTER_LARGE_MEMORY_BYTES = 4 * 1024 ** 3
FILTER_LARGE_WORK_VOXELS = 5e8
# Optional crop box ("Limit to ROI"): temporary, never saved, removed after a successful run
FILTER_CROP_ROI_ATTRIBUTE = "EasyFusion.FilterCropROI"
FILTER_CROP_ROI_NAME = "Filter crop ROI"
FILTER_CROP_ROI_COLOR = (0.0, 0.85, 1.0)
FILTER_CROPPED_ATTRIBUTE = "EasyFusion.FilterCroppedToROI"
# Voxel-based crop of a volume under a non-linear (e.g. deformable registration) transform, which Crop Volume
# refuses: the ROI surface is sampled on this many points per edge and mapped into the volume's own coordinates
FILTER_CROP_SAMPLES_PER_EDGE = 9
# Used for keys missing from a sidecar (or models without one): the defaults of the PETDenoise panel
FILTER_DEFAULT_PARAMETERS = {
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
        parent.title = "Epona - SPECT/PET Review"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        iconUrl = qt.QUrl.fromLocalFile(os.path.join(os.path.dirname(__file__), "Resources", "Icons", "Easy_fusion.png")).toString()        
        parent.helpText = f"""
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
        Below them, "Slicer Annotations" shows or hides Slicer's own slice view annotations, and "Window Info"
        shows the CT window / level and the SPECT/PET range in the bottom-right corner of each slice view.
        Post-processing filters run the AI denoising / super-resolution models of the Belenos PET Denoise module
        (a .pth file with its .txt parameter file) on the SPECT/PET or CT/MRI volume. The result is always a new
        volume (the original is kept unchanged) and the views, MIP and SUV ROIs switch to it. SUVs measured on
        a filtered volume differ from those of the original.
        <p align="center"><img src="{iconUrl}" width="300"></p>
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


def tissueSample(values, backgroundFraction=MRI_BACKGROUND_FRACTION):
    """
    Voxel values that belong to the imaged body: padding (the minimum value) and air (below
    backgroundFraction of the 99.5th percentile) are left out. Falls back to all finite values when
    almost nothing would be left.
    """
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    minimum = values.min()
    upper = np.percentile(values, 99.5)
    cutoff = max(minimum, minimum + backgroundFraction * (upper - minimum))
    tissue = values[values > cutoff]
    return tissue if tissue.size >= max(10, values.size // 100) else values


def percentileWindow(values, lowerPercentile, upperPercentile):
    """(window, level) spanning two percentiles of values, or None when they are equal / there is no data."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lower, upper = np.percentile(values, [float(lowerPercentile), float(upperPercentile)])
    if not upper > lower:
        return None
    return float(upper - lower), float((upper + lower) / 2.0)


def looksLikeCT(scalarMinimum):
    """True for a CT (air / padding far below 0 HU), False for an MRI or other positive-valued image."""
    return scalarMinimum is not None and np.isfinite(scalarMinimum) and float(scalarMinimum) <= CT_MINIMUM_BELOW


def formatWindowNumber(value):
    """Compact number for the window info: 400, -600, 2.5, 0.35, 1.2e+04 -> 12000."""
    value = float(value)
    if not np.isfinite(value):
        return "?"
    if abs(value) < 1e-9:
        return "0"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def formatWindowInfoLine(kind, window, level, unit=""):
    """
    One line of the window info in the slice views.
    kind "CT" / "MRI": window and level ("CT  W 400  L 50").
    Anything else (PET, SPECT/PET, other volume names): the displayed range, which is how nuclear medicine
    windows are usually read ("PET  SUV 0–10", "SPECT/PET  0–1250").
    """
    if kind in ("CT", "MRI"):
        return f"{kind}  W {formatWindowNumber(window)}  L {formatWindowNumber(level)}"
    lower, upper = level - window / 2.0, level + window / 2.0
    unitText = f"{unit} " if unit else ""
    return f"{kind}  {unitText}{formatWindowNumber(lower)}–{formatWindowNumber(upper)}"


def voxelUnitLabel(volumeNode):
    """ "SUV" for SUV images (from the voxel value units, else the name), else the units' short text or "". """
    if volumeNode is None:
        return ""
    texts = []
    try:
        units = volumeNode.GetVoxelValueUnits()
        if units is not None:
            texts = [units.GetCodeValue() or "", units.GetCodeMeaning() or ""]
    except Exception:
        pass
    if any("SUV" in text.upper() for text in texts) or "SUV" in (volumeNode.GetName() or "").upper():
        return "SUV"
    meaning = texts[1] if len(texts) > 1 else ""
    return meaning if meaning and len(meaning) <= 12 else ""


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


def formatRoiThreshold(mode, value):
    """Stored form of an ROI's threshold: "relative 40" / "absolute 2.5"."""
    mode = mode if mode in (THRESHOLD_RELATIVE, THRESHOLD_ABSOLUTE) else THRESHOLD_RELATIVE
    return f"{mode} {float(value):g}"


def parseRoiThreshold(text):
    """(mode, value) from formatRoiThreshold's text, or None if the text is not a valid threshold."""
    parts = (text or "").split()
    if len(parts) != 2 or parts[0] not in (THRESHOLD_RELATIVE, THRESHOLD_ABSOLUTE):
        return None
    try:
        value = float(parts[1])
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return parts[0], value


def describeRoiThreshold(mode, value):
    """Short text for the table and the view labels: "40%" / "2.5 SUV"."""
    return f"{float(value):g} SUV" if mode == THRESHOLD_ABSOLUTE else f"{float(value):g}%"


def parseRoiLabelFields(text):
    """Label fields from their stored form ("name,max,mean"); None (never set) gives the defaults."""
    if text is None:
        return tuple(DEFAULT_ROI_LABEL_FIELDS)
    wanted = {part.strip() for part in str(text).split(",")}
    return tuple(key for key, _ in ROI_LABEL_FIELDS if key in wanted)


def formatRoiLabel(name, stats, hasPet, fields=DEFAULT_ROI_LABEL_FIELDS, radius=None, threshold=None):
    """
    Text shown next to the ROI in the views: the chosen fields, one per line ("" when none is chosen).
    Mean is the mean of the thresholded segment. threshold: (mode, value) of this ROI.
    """
    fields = set(fields)
    lines = [name] if "name" in fields else []
    if stats is None:
        lines.append("(outside PET)" if hasPet else "(no PET)")
    else:
        segMean = stats.get("segMean")
        if "max" in fields:
            lines.append(f"Max {stats['max']:.2f}")
        if "mean" in fields:
            lines.append(f"Mean {segMean:.2f}" if segMean is not None else "Mean -")
        if "mtv" in fields:
            lines.append(f"MTV {stats.get('mtvMl', 0.0):.2f} mL")
        if "tlg" in fields:
            lines.append(f"TLG {stats.get('tlg', 0.0):.2f}")
    if "radius" in fields and radius is not None:
        lines.append(f"r {float(radius):.1f} mm")
    if "threshold" in fields and threshold is not None:
        lines.append(f"Thr {describeRoiThreshold(*threshold)}")
    return "\n".join(lines)


def formatRoiTableRow(name, radius, stats, threshold=None):
    """ROI | r (mm) | Thr. | Max | Mean | MTV (mL) | TLG   (Mean = mean of the thresholded segment)"""
    values = [name, f"{radius:.1f}", describeRoiThreshold(*threshold) if threshold else "-", "-", "-", "-", "-"]
    if stats is not None:
        values[3] = f"{stats['max']:.2f}"
        if stats.get("segMean") is not None:
            values[4] = f"{stats['segMean']:.2f}"
        values[5] = f"{stats.get('mtvMl', 0.0):.2f}"
        values[6] = f"{stats.get('tlg', 0.0):.2f}"
    return values


ROI_TSV_COLUMNS = [
    "ROI", "Origin", "Center R (mm)", "Center A (mm)", "Center S (mm)", "Radius (mm)",
    "Threshold mode", "Threshold setting", "Threshold (SUV)",
    "SUVmax", "SUVmean sphere", "Sphere volume (mL)",
    "SUVmean segment", "MTV (mL)", "TLG", "Segment voxels",
    "PET volume", "PET filter",
]


def _tsvCell(value, digits=4):
    """One TSV cell: numbers at fixed precision, None / NaN empty, tabs and line breaks removed from text."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}" if math.isfinite(value) else ""
    return re.sub(r"[\t\r\n]+", " ", str(value)).strip()


def formatRoiTableTsv(rows, centers=None, petName="", petFilter=""):
    """
    ROI table as tab-separated text (header + one line per ROI), at full precision rather than the
    rounded values shown in the panel.
    rows:    (pointID, name, radius, stats, threshold, origin) as in the ROI table
    centers: {pointID: (R, A, S)} ROI centers in world coordinates (mm)
    """
    centers = centers or {}
    lines = ["\t".join(ROI_TSV_COLUMNS)]
    for pointID, name, radius, stats, threshold, origin in rows:
        center = centers.get(pointID) or (None, None, None)
        mode, setting = threshold if threshold else (None, None)
        stats = stats or {}
        values = [
            name, origin or ROI_ORIGIN_USER,
            _tsvCell(center[0], 2), _tsvCell(center[1], 2), _tsvCell(center[2], 2),
            _tsvCell(radius, 2),
            mode or "",
            (describeRoiThreshold(mode, setting) if mode is not None else ""),
            _tsvCell(stats.get("threshold")),
            _tsvCell(stats.get("max")), _tsvCell(stats.get("mean")), _tsvCell(stats.get("volumeMl")),
            _tsvCell(stats.get("segMean")), _tsvCell(stats.get("mtvMl")),
            _tsvCell(stats.get("tlg") if stats else None), _tsvCell(stats.get("segVoxels")),
            petName or "", petFilter or "",
        ]
        lines.append("\t".join(_tsvCell(value) for value in values))
    return "\n".join(lines) + "\n"


def defaultRoiExportFileName(petName, timestamp=None):
    """e.g. "PT_Patient1_SUV_ROIs_20260920-143000.tsv" (characters unsafe in file names replaced)."""
    stamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    base = re.sub(r"[^\w.\-]+", "_", petName or "").strip("._")
    return f"{base + '_' if base else ''}SUV_ROIs_{stamp}.tsv"


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


class FilterCancelled(Exception):
    """Raised by the progress callback of a filter run when the user pressed Cancel."""


def parseFilterMetadata(text):
    """
    Model parameters from a PETDenoise sidecar file (<model>.txt next to <model>.pth), read the way the PETDenoise
    module reads them: same keys, same defaults, and (as there) a voxel_spacing line switches resampling back on.

    Returns (params, notes). params has every key of FILTER_DEFAULT_PARAMETERS (+ "modality" if the file has one).
    notes maps every other "key: value" line (lower-case key) to (key as written, value), e.g. the SUV biases.
    """
    params = dict(FILTER_DEFAULT_PARAMETERS)
    notes = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        rawKey, value = line.split(":", 1)
        rawKey, value = rawKey.strip().lstrip("\ufeff"), value.strip()
        key = rawKey.lower()
        if not key:
            continue
        try:
            if key in ("dual_channel", "dont_resample", "prevent_negative"):
                params[key] = value.lower() == "true"
            elif key in ("voxel_spacing", "block_size"):
                values = tuple(ast.literal_eval(value))
                if len(values) != 3:
                    raise ValueError("three values expected")
                if key == "voxel_spacing":
                    params[key] = tuple(float(v) for v in values)
                    params["dont_resample"] = False
                else:
                    params[key] = tuple(int(v) for v in values)
            elif key in ("strides", "channels", "num_heads", "depths"):
                params[key] = tuple(int(v) for v in ast.literal_eval(value))
            elif key in ("res_units", "down_kernel", "up_kernel", "feature_size"):
                params[key] = int(value)
            elif key == "do_rate":
                params[key] = float(value)
            elif key == "architecture":
                # As in PETDenoise: anything other than UNET / SwinUNETR selects SwinUNETR+GCFN
                params[key] = value if value in ("UNET", "SwinUNETR") else "SwinUNETR+GCFN"
            elif key == "modality":
                params[key] = value
            else:
                notes[key] = (rawKey, value)
        except (ValueError, SyntaxError, TypeError):
            logging.warning(f"EasyFusion: ignored model parameter line '{rawKey}: {value}'")
    return params, notes


def filterSuvNotes(notes):
    """Sidecar lines whose key starts with "SUV" (e.g. "SUVmax Bias: -0.28 g/mL"), as written, for the warning."""
    return [f"{rawKey}: {value}" for key, (rawKey, value) in notes.items() if key.startswith("suv")]


def guessFilterTarget(modelName, params):
    """
    Volume a model is meant for. A "modality" line in the sidecar decides (CT... / MR... -> CT/MRI, else SPECT/PET).
    Without one, a file name containing a separate word CT, MR or MRI (e.g. CT_superres24.pth) means CT/MRI.
    """
    modality = str(params.get("modality") or "").strip().lower()
    if modality:
        return FILTER_TARGET_CT if modality.startswith(("ct", "mr")) else FILTER_TARGET_PET
    stem = os.path.splitext(os.path.basename(modelName or ""))[0].lower()
    return FILTER_TARGET_CT if re.search(r"(^|[^a-z])(ct|mr|mri)([^a-z]|$)", stem) else FILTER_TARGET_PET


def resampledGridShape(dimensionsIjk, spacing, targetSpacing):
    """Approximate array shape (k, j, i) of a volume after resampling to targetSpacing (None: not resampled)."""
    if targetSpacing is None:
        dims = [int(d) for d in dimensionsIjk]
    else:
        dims = [max(1, int(round(d * s / t))) for d, s, t in zip(dimensionsIjk, spacing, targetSpacing)]
    return tuple(dims[::-1])


def estimateSlidingWindowCount(imageShape, roiSize, overlap=FILTER_WINDOW_OVERLAP):
    """
    Number of windows MONAI's sliding_window_inference evaluates (= predictor calls with sw_batch_size=1).
    Same arithmetic as monai.inferers.utils (_get_scan_interval, dense_patch_slices). Drives the progress bar.
    """
    total = 1
    for size, roi in zip(imageShape, roiSize):
        roi = int(roi)
        size = max(int(size), roi)  # MONAI pads the image up to the window size
        interval = roi if roi == size else max(int(roi * (1.0 - overlap)), 1)
        count = int(math.ceil(float(size) / interval))
        first = next((d for d in range(count) if d * interval + roi >= size), None)
        total *= first + 1 if first is not None else 1
    return total


def filterJobSize(dimensionsIjk, spacing, params, channels=1):
    """
    Size of a filter run on a volume with these dimensions (i, j, k) and spacing: grid shape (k, j, i) on the
    model's voxel grid, voxel count, rough peak memory, number of windows, and whether it counts as large.
    """
    targetSpacing = None if params["dont_resample"] else params["voxel_spacing"]
    shape = resampledGridShape(dimensionsIjk, spacing, targetSpacing)
    voxels = math.prod(shape)  # Python ints: no overflow for huge grids
    memoryBytes = float(voxels) * 4 * (FILTER_MEMORY_COPIES + channels - 1)
    windows = estimateSlidingWindowCount(shape, params["block_size"])
    work = float(windows) * math.prod(int(v) for v in params["block_size"])
    return {"shape": shape, "voxels": voxels, "memoryBytes": memoryBytes, "windows": windows,
            "large": memoryBytes >= FILTER_LARGE_MEMORY_BYTES or work >= FILTER_LARGE_WORK_VOXELS}


def voxelCopyRegion(sourceIjkToRas, sourceShape, croppedIjkToRas, croppedShape, tolerance=1e-3):
    """
    Where a voxel-based crop lies in its source volume. Shapes are array shapes [k, j, i] (as arrayFromVolume);
    matrices are IJK -> RAS (4x4).

    Returns (sourceSlices, croppedSlices, firstSourceIjk): the block both volumes share, as [k, j, i] slice tuples,
    and the source IJK index of its first voxel. Voxels of the crop outside the source (padding) are left out.
    Raises ValueError("resampled") if the crop is not on the source's voxel grid (other voxel size or axes, or
    shifted by a fraction of a voxel, i.e. interpolated), ValueError("outside") if it shares no voxel with it.
    """
    source = np.asarray(sourceIjkToRas, dtype=float)
    cropped = np.asarray(croppedIjkToRas, dtype=float)
    scale = max(float(np.abs(source[:3, :3]).max()), 1e-12)
    if not np.allclose(source[:3, :3], cropped[:3, :3], rtol=0.0, atol=1e-6 * scale):
        raise ValueError("resampled")
    offset = np.linalg.solve(source[:3, :3], cropped[:3, 3] - source[:3, 3])  # crop voxel (0,0,0) in source IJK
    rounded = np.round(offset)
    if np.any(np.abs(offset - rounded) > tolerance):
        raise ValueError("resampled")
    offset = rounded.astype(int)
    sourceDims = np.array(sourceShape[:3][::-1])    # (i, j, k)
    croppedDims = np.array(croppedShape[:3][::-1])
    lo = np.maximum(offset, 0)
    hi = np.minimum(offset + croppedDims, sourceDims)
    if np.any(hi <= lo):
        raise ValueError("outside")
    sourceSlices = tuple(slice(int(lo[a]), int(hi[a])) for a in (2, 1, 0))
    croppedSlices = tuple(slice(int(lo[a] - offset[a]), int(hi[a] - offset[a])) for a in (2, 1, 0))
    return sourceSlices, croppedSlices, tuple(int(v) for v in lo)


def boxSurfacePoints(size, samplesPerEdge=FILTER_CROP_SAMPLES_PER_EDGE):
    """
    Points on the surface of a box of the given size (x, y, z) centered at the origin (an ROI's object coordinates):
    a samplesPerEdge x samplesPerEdge grid on each face, corners and edges included. Under a smooth deformation the
    image of the surface encloses the image of the whole box, so these points are enough to bound it.
    """
    half = np.asarray(size, dtype=float) / 2.0
    n = max(int(samplesPerEdge), 2)
    t = np.linspace(-1.0, 1.0, n)
    u, v = np.meshgrid(t, t, indexing="ij")
    faces = []
    for axis in range(3):
        a, b = [x for x in range(3) if x != axis]
        for side in (-1.0, 1.0):
            face = np.empty((u.size, 3))
            face[:, axis] = side
            face[:, a] = u.ravel()
            face[:, b] = v.ravel()
            faces.append(face)
    return np.unique(np.vstack(faces), axis=0) * half


def voxelBlockFromIjkPoints(ijkPoints, dimensions, margin=0):
    """
    Smallest block of voxels touched by the given continuous IJK points (voxel centers at integer IJK), grown by
    margin voxels and clipped to the volume. dimensions: (I, J, K) as vtkImageData.GetDimensions().
    Returns (lo, hi) as (i, j, k) integer tuples, hi exclusive, or None if the block misses the volume.
    Non-finite points (e.g. where an inverse transform did not converge) are ignored.
    """
    points = np.asarray(ijkPoints, dtype=float).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.size == 0:
        return None
    dims = np.asarray(dimensions[:3], dtype=int)
    lo = np.floor(points.min(axis=0) + 0.5).astype(int) - int(margin)   # voxel containing the lowest point
    hi = np.floor(points.max(axis=0) + 0.5).astype(int) + 1 + int(margin)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, dims)
    if np.any(hi <= lo):
        return None
    return tuple(int(x) for x in lo), tuple(int(x) for x in hi)

def castFilterResult(array, dtype):
    """Filtered voxels in the input's voxel type. Integer types are rounded (not truncated) and kept in range."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        return np.clip(np.rint(array), limits.min, limits.max).astype(dtype)
    return np.asarray(array).astype(dtype, copy=False)


def formatByteSize(numberOfBytes):
    gigabytes = numberOfBytes / 1024.0 ** 3
    return f"{gigabytes:.1f} GB" if gigabytes >= 1.0 else f"{numberOfBytes / 1024.0 ** 2:.0f} MB"


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
        self._suppressInputUpdate = 0    # > 0 while the panel itself sets the volume selectors
        self._mriSample = (None, None)   # ((node ID, image MTime), tissue sample) for the MRI presets
        self._roiTableRows = []          # rows currently shown in the ROI table (for the TSV export)
        self._roiCenters = {}            # ROI control point ID -> center (world, mm) at last update
        self._lastRoiGeometry = {}       # ROI control point ID -> (center, radius) at last sync
        self._lastHandlePositions = {}   # handle control point ID -> position at last sync
        self._activeHandleID = None      # handle currently being dragged
        self._panelButtons = []
        self.mipOverlay = MipRoiOverlay()
        self.windowInfoOverlay = SliceWindowInfoOverlay(
            lambda: (self.inputVolumeSelector.currentNode(), self.inputVolumeSelectorCT.currentNode()))
        self.filterLogic = None
        self._filterRunning = False
        self._filterParams = None     # parameters of the selected model (from its .txt sidecar)
        self._filterNotes = {}        # other sidecar lines (training data, reported SUV bias, ...)
        self.filterCropRoiNode = None  # adjustable box of "Limit to ROI"

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = Easy_fusionLogic()
        self.filterLogic = Easy_fusionFilterLogic()

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

        formLayout.addRow("MRI Presets (relative):", self._buttonRow([
            (text, lambda low=low, high=high, text=text: self.setMRIWindowPercentile(low, high, text),
             f"Window from percentile {low:g} to {high:g} of the tissue intensities in the CT/MRI volume\n"
             f"(air and background left out), for MRI or any image without absolute units.\n"
             f"Shortcut: {key} with the mouse over a CT/MRI view when that volume is an MRI.")
            for text, low, high, key in MRI_PERCENTILE_PRESETS]))

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
        self.setupFilterSection()

        self.layout.addStretch(1)

        bannerPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons", "fusbanner.jpg")
        if os.path.exists(bannerPath):
            bannerLabel = qt.QLabel()
            bannerLabel.setPixmap(qt.QPixmap(bannerPath).scaledToWidth(600, qt.Qt.SmoothTransformation))
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
        # After the first "Go": a new SPECT/PET or CT/MRI selection updates the views right away
        self.inputUpdateTimer = qt.QTimer()
        self.inputUpdateTimer.setSingleShot(True)
        self.inputUpdateTimer.setInterval(INPUT_UPDATE_DELAY_MS)
        self.inputUpdateTimer.connect('timeout()', self.applyChangedInputVolumes)
        self.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputVolumeSelectionChanged)
        self.inputVolumeSelectorCT.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputVolumeSelectionChanged)
        if slicer.app.layoutManager() is not None:
            slicer.app.layoutManager().connect("layoutChanged(int)", self.onLayoutChanged)

        self.observeThreeDViewNode()
        with self.selectorsSetByPanel():
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

        # Text in the slice views
        textRow = qt.QHBoxLayout()
        self.sliceAnnotationsButton = qt.QPushButton("Slicer Annotations")
        self.sliceAnnotationsButton.checkable = True
        self.sliceAnnotationsButton.setToolTip(
            "Show or hide Slicer's built-in slice view annotations (Data Probe): patient and study\n"
            "information and the names of the shown volumes in the corners of the slice views.\n"
            "It switches Slicer's active corners (top left, top right, bottom left), as Slicer's own\n"
            "settings do, so the choice is remembered after a restart. Showing them again restores the\n"
            "corners that were on before.")
        self.sliceAnnotationsButton.connect("toggled(bool)", self.onSlicerAnnotationsToggled)
        textRow.addWidget(self.sliceAnnotationsButton)
        self.windowInfoButton = qt.QPushButton("Window Info (bottom right)")
        self.windowInfoButton.checkable = True
        self.windowInfoButton.setToolTip(
            "Show the current windowing in the bottom-right corner of every slice view:\n"
            "CT (or MRI) window / level, and the SPECT/PET display range (in SUV for PET).\n"
            "Fusion views show both. It follows presets, F5-F9 and mouse window / level drags.")
        self.windowInfoButton.connect("toggled(bool)", self.onWindowInfoToggled)
        textRow.addWidget(self.windowInfoButton)
        layoutFormLayout.addRow("Slice view text:", textRow)

        showWindowInfo = str(qt.QSettings().value(SETTINGS_SHOW_WINDOW_INFO, "true")).lower() in ("true", "1")
        self.windowInfoOverlay.enabled = showWindowInfo
        wasBlocked = self.windowInfoButton.blockSignals(True)
        self.windowInfoButton.checked = showWindowInfo
        self.windowInfoButton.blockSignals(wasBlocked)
        self.syncSlicerAnnotationsButton()

    # --- Slice view text ------------------------------------------------------

    @staticmethod
    def slicerSliceAnnotations():
        """Slicer's slice view annotations object (Data Probe module), or None when it is not available."""
        try:
            return slicer.modules.DataProbeInstance.infoWidget.sliceAnnotations
        except AttributeError:
            return None

    @staticmethod
    def _slicerAnnotationCorners(annotations):
        """[0/1 per corner of DATA_PROBE_CORNERS], or None if this Slicer version has no per-corner switches."""
        if not all(hasattr(annotations, attribute) for attribute, _, _ in DATA_PROBE_CORNERS):
            return None
        return [1 if getattr(annotations, attribute) else 0 for attribute, _, _ in DATA_PROBE_CORNERS]

    def slicerAnnotationsShown(self):
        annotations = self.slicerSliceAnnotations()
        if annotations is None:
            return False
        corners = self._slicerAnnotationCorners(annotations)
        return bool(annotations.sliceViewAnnotationsEnabled) and (corners is None or any(corners))

    def syncSlicerAnnotationsButton(self):
        """The button shows Slicer's current state (it may have been changed in Slicer's settings)."""
        if not hasattr(self, "sliceAnnotationsButton"):
            return
        annotations = self.slicerSliceAnnotations()
        self.sliceAnnotationsButton.enabled = annotations is not None
        if annotations is None:
            return
        wasBlocked = self.sliceAnnotationsButton.blockSignals(True)
        self.sliceAnnotationsButton.checked = self.slicerAnnotationsShown()
        self.sliceAnnotationsButton.blockSignals(wasBlocked)

    @staticmethod
    def _setCheckBox(annotations, checkBoxName, checked):
        """Mirror a change in Slicer's own settings panel (if it was created), without triggering it."""
        checkBox = getattr(annotations, checkBoxName, None)
        if checkBox is None:
            return
        try:
            wasBlocked = checkBox.blockSignals(True)
            checkBox.checked = bool(checked)
            checkBox.blockSignals(wasBlocked)
        except Exception:
            logging.debug(f"EasyFusion: could not update Slicer's {checkBoxName}", exc_info=True)

    def _clearSlicerCornerTexts(self, cornerIndexes):
        """Blank Slicer's corner texts right away (they are otherwise only rewritten on the next view change)."""
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return
        for name in layoutManager.sliceViewNames():
            sliceWidget = layoutManager.sliceWidget(name)
            sliceView = sliceWidget.sliceView() if sliceWidget is not None else None
            if sliceView is None:
                continue
            cornerAnnotation = sliceView.cornerAnnotation()
            for index in cornerIndexes:
                cornerAnnotation.SetText(index, "")
            sliceView.scheduleRender()

    def onSlicerAnnotationsToggled(self, enabled):
        """
        Show / hide Slicer's slice view annotations. Turning off only Slicer's master switch leaves the texts
        already drawn in the views, so this switches the active corners (top left, top right, bottom left),
        as Slicer's settings panel does, and remembers which were on so that showing them restores them.
        """
        annotations = self.slicerSliceAnnotations()
        if annotations is None:
            slicer.util.showStatusMessage("EasyFusion: Slicer's slice view annotations (Data Probe) are not available.",
                                          3000)
            self.syncSlicerAnnotationsButton()
            return
        settings = qt.QSettings()
        try:
            corners = self._slicerAnnotationCorners(annotations)
            if corners is None:
                # Older Slicer without per-corner switches: master switch, then blank the views ourselves
                annotations.sliceViewAnnotationsEnabled = 1 if enabled else 0
                self._setCheckBox(annotations, "sliceViewAnnotationsCheckBox", enabled)
                settings.setValue(DATA_PROBE_ANNOTATIONS_SETTING, 1 if enabled else 0)
                annotations.updateSliceViewFromGUI()
                if not enabled:
                    self._clearSlicerCornerTexts(DATA_PROBE_CORNER_INDEXES)
                return

            if enabled:
                saved = str(settings.value(SETTINGS_SLICER_ANNOTATION_CORNERS, "") or "").split(",")
                wanted = [1 if value.strip() == "1" else 0 for value in saved] if len(saved) == len(corners) else []
                if not any(wanted):
                    wanted = [1] * len(corners)  # nothing remembered (or all were off): show every corner
            else:
                if any(corners):
                    settings.setValue(SETTINGS_SLICER_ANNOTATION_CORNERS, ",".join(str(c) for c in corners))
                wanted = [0] * len(corners)

            # The master switch stays on: the annotations must keep updating to draw (or blank) the corners
            annotations.sliceViewAnnotationsEnabled = 1
            self._setCheckBox(annotations, "sliceViewAnnotationsCheckBox", True)
            settings.setValue(DATA_PROBE_ANNOTATIONS_SETTING, 1)
            for (attribute, checkBoxName, settingName), value in zip(DATA_PROBE_CORNERS, wanted):
                setattr(annotations, attribute, value)
                self._setCheckBox(annotations, checkBoxName, value)
                settings.setValue(settingName, value)
            if hasattr(annotations, "updateEnabledButtons"):
                annotations.updateEnabledButtons()
            annotations.updateSliceViewFromGUI()
            if not enabled:
                self._clearSlicerCornerTexts(DATA_PROBE_CORNER_INDEXES)
        except Exception:
            logging.exception("EasyFusion: could not switch Slicer's slice view annotations")
        finally:
            self.syncSlicerAnnotationsButton()

    def onWindowInfoToggled(self, enabled):
        qt.QSettings().setValue(SETTINGS_SHOW_WINDOW_INFO, "true" if enabled else "false")
        self.windowInfoOverlay.setEnabled(enabled)


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
            "Radius of the ROI selected in the table. With no ROI selected: the radius given to new ROIs.")
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
            "Absolute: segment = voxels inside the ROI with SUV >= this value.\n"
            "Each ROI keeps its own threshold: this changes only the ROI selected in the table.\n"
            "With no ROI selected, it sets the threshold given to new ROIs.")
        self.thresholdValueSpinBox = qt.QDoubleSpinBox()
        self.thresholdValueSpinBox.setDecimals(1)
        thresholdLayout.addWidget(self.thresholdModeComboBox)
        thresholdLayout.addWidget(self.thresholdValueSpinBox)
        measurementLayout.addRow("Segment threshold:", thresholdLayout)
        self._relativeThreshold = DEFAULT_RELATIVE_THRESHOLD
        self._absoluteThreshold = DEFAULT_ABSOLUTE_THRESHOLD
        self._applyThresholdModeToSpinBox(THRESHOLD_RELATIVE)

        # Which ROI the radius / threshold controls edit right now
        editTargetLayout = qt.QHBoxLayout()
        self.roiEditTargetLabel = qt.QLabel()
        self.roiEditTargetLabel.setStyleSheet("color: gray;")
        self.roiDeselectButton = qt.QPushButton("Deselect")
        self.roiDeselectButton.setToolTip("Deselect the ROI, so radius and threshold set the values for new ROIs.")
        editTargetLayout.addWidget(self.roiEditTargetLabel, 1)
        editTargetLayout.addWidget(self.roiDeselectButton)
        measurementLayout.addRow(editTargetLayout)

        self.roiPetLabel = qt.QLabel("Measuring on: (no PET selected)")
        self.roiPetLabel.wordWrap = True
        measurementLayout.addRow(self.roiPetLabel)

        self.showRoisOnMipCheckBox = qt.QCheckBox("Show ROI segments and values on the MIP (3D view)")
        self.showRoisOnMipCheckBox.checked = True
        self.showRoisOnMipCheckBox.setToolTip(
            "Draws the thresholded segments and the Max / Mean text on top of the MIP.\n"
            "Display only: nothing is added to the scene or saved.")
        measurementLayout.addRow(self.showRoisOnMipCheckBox)

        # Values drawn next to each ROI in the slice views and on the MIP
        labelFieldsLayout = qt.QGridLayout()
        shownFields = parseRoiLabelFields(qt.QSettings().value(SETTINGS_ROI_LABEL_FIELDS))
        self.roiLabelFieldCheckBoxes = {}
        for position, (key, text) in enumerate(ROI_LABEL_FIELDS):
            checkBox = qt.QCheckBox(text)
            checkBox.checked = key in shownFields
            checkBox.connect('toggled(bool)', self.onRoiLabelFieldsChanged)
            labelFieldsLayout.addWidget(checkBox, position // 4, position % 4)
            self.roiLabelFieldCheckBoxes[key] = checkBox
        measurementLayout.addRow("Show near ROI:", labelFieldsLayout)

        self.roiTable = qt.QTableWidget()
        self.roiTable.setColumnCount(7)
        self.roiTable.setHorizontalHeaderLabels(["ROI", "r (mm)", "Thr.", "Max", "Mean", "MTV (mL)", "TLG"])
        headerTips = ["", "ROI radius", "Segment threshold of this ROI (% of its Max, or absolute SUV)",
                      "Maximum SUV inside the ROI sphere", "Mean SUV of the thresholded segment",
                      "Metabolic tumor volume: volume of the thresholded segment",
                      "Total lesion glycolysis = Mean x MTV"]
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
        self.roiTable.setToolTip("Select a row to jump to that ROI and edit its radius and threshold.")
        measurementLayout.addRow(self.roiTable)

        exportLayout = qt.QHBoxLayout()
        exportLayout.addStretch(1)
        self.exportRoiTableButton = qt.QPushButton("Export Table (.tsv)…")
        self.exportRoiTableButton.setToolTip(
            "Save all ROIs to a tab-separated file (opens in Excel, LibreOffice, R, Python ...).\n"
            "Values are written at full precision, with ROI centers (RAS, mm), sphere and segment\n"
            "statistics, threshold in SUV and the PET volume they were measured on.")
        self.exportRoiTableButton.enabled = False
        exportLayout.addWidget(self.exportRoiTableButton)
        measurementLayout.addRow(exportLayout)

        self.placeRoiButton.connect('clicked()', self.onPlaceRoi)
        self.deleteRoiButton.connect('clicked()', self.onDeleteSelectedRoi)
        self.clearRoisButton.connect('clicked()', self.onClearRois)
        self.roiDeselectButton.connect('clicked()', self.onDeselectRoi)
        self.exportRoiTableButton.connect('clicked()', self.onExportRoiTable)
        self.updateRoiEditTarget()
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

    def setupFilterSection(self):
        filterCollapsibleButton = ctk.ctkCollapsibleButton()
        filterCollapsibleButton.text = "Post-processing Filters (AI)"
        filterCollapsibleButton.collapsed = True
        self.layout.addWidget(filterCollapsibleButton)
        filterLayout = qt.QFormLayout(filterCollapsibleButton)

        folderLayout = qt.QHBoxLayout()
        self.filterModelFolderEdit = qt.QLineEdit()
        self.filterModelFolderEdit.readOnly = True
        self.filterModelFolderEdit.placeholderText = "Folder with the .pth models and their .txt files"
        self.filterModelFolderButton = qt.QPushButton("Browse…")
        folderLayout.addWidget(self.filterModelFolderEdit)
        folderLayout.addWidget(self.filterModelFolderButton)
        filterLayout.addRow("Model folder:", folderLayout)

        self.filterModelSelector = qt.QComboBox()
        self.filterModelSelector.setToolTip("Denoising / super-resolution model (PETDenoise .pth file).")
        filterLayout.addRow("Model:", self.filterModelSelector)

        self.filterInfoBox = qt.QPlainTextEdit()
        self.filterInfoBox.readOnly = True
        self.filterInfoBox.setMaximumHeight(120)
        self.filterInfoBox.setToolTip("Contents of the model's .txt file (parameters, training data, validation).")
        filterLayout.addRow("Model info:", self.filterInfoBox)

        self.filterTargetSelector = qt.QComboBox()
        self.filterTargetSelector.addItem("SPECT/PET", FILTER_TARGET_PET)
        self.filterTargetSelector.addItem("CT/MRI", FILTER_TARGET_CT)
        self.filterTargetSelector.setToolTip(
            "Volume the filter is applied to. Set automatically from the model's .txt file ('modality: CT') or its\n"
            "file name (a CT / MR model such as CT_superres24.pth); change it if the guess is wrong.")
        filterLayout.addRow("Apply to:", self.filterTargetSelector)

        self.filterLimitToRoiCheckBox = qt.QCheckBox("Limit to ROI (crop before filtering)")
        self.filterLimitToRoiCheckBox.setToolTip(
            "Places an adjustable box (Slicer's Crop Volume ROI). Only the voxels inside it are filtered, which is\n"
            "much faster and needs far less memory. The original volume is never cropped or changed.\n"
            "The box is removed after filtering.")
        filterLayout.addRow(self.filterLimitToRoiCheckBox)

        self.filterForceCpuCheckBox = qt.QCheckBox("Force CPU")
        self.filterForceCpuCheckBox.setToolTip(
            f"Run on the CPU even when a GPU with at least {FILTER_MIN_VRAM_GB:g} GB of memory is available.")
        filterLayout.addRow(self.filterForceCpuCheckBox)

        self.applyFilterButton = qt.QPushButton("Apply Filter")
        self.applyFilterButton.setToolTip(
            "Creates a NEW filtered volume; the original volume is not changed.\n"
            "The views, the MIP and the SUV ROIs then switch to the filtered volume.")
        filterLayout.addRow(self.applyFilterButton)

        self.filterStatusLabel = qt.QLabel("")
        self.filterStatusLabel.wordWrap = True
        filterLayout.addRow(self.filterStatusLabel)

        self.filterModelFolderButton.connect("clicked()", self.onBrowseFilterModelFolder)
        self.filterModelSelector.connect("currentIndexChanged(int)", self.onFilterModelChanged)
        self.applyFilterButton.connect("clicked()", self.onApplyFilter)
        self.filterLimitToRoiCheckBox.connect("toggled(bool)", self.onFilterLimitToRoiToggled)
        self.restoreFilterModelFolder()

    def enter(self):
        self.observeThreeDViewNode()
        self.onLayoutChanged()
        self.syncSlicerAnnotationsButton()
        if hasattr(self, "filterModelSelector") and not self._filterRunning:
            self.refreshFilterModels()  # models may have been added to the folder in the meantime

    def cleanup(self):
        try:
            self.mipOverlay.clear()
        except Exception:
            logging.exception("EasyFusion: could not remove the MIP overlay")
        try:
            self.windowInfoOverlay.enabled = False
            self.windowInfoOverlay.clear()
        except Exception:
            logging.exception("EasyFusion: could not remove the slice view window info")
        if hasattr(self, "roiUpdateTimer"):
            self.roiUpdateTimer.stop()
        if hasattr(self, "inputUpdateTimer"):
            self.inputUpdateTimer.stop()
        try:
            self.removeFilterCropRoi()
        except Exception:
            logging.exception("EasyFusion: could not remove the filter crop ROI")
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
        self.windowInfoOverlay.scheduleUpdate()  # views are emptied: their text follows
        self.filterCropRoiNode = None
        self._setLimitToRoiChecked(False)
        self.setRoiNode(None)
        self.setHandlesNode(None)
        self.fillRoiTable([])
        runWhenSceneSettled(self.refreshViewsAfterSceneChange)

    def onSceneLoaded(self):
        """Called by the post-load chain (see _afterSceneLoad), after the scene repairs."""
        with self.selectorsSetByPanel():
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
        if node in (self.inputVolumeSelector.currentNode(), self.inputVolumeSelectorCT.currentNode()):
            # The selector jumps to another volume on removal: that is not a choice of the user, so the
            # views are not updated to it. Released once the removal (and the selector's reaction) is done.
            self._suppressInputUpdate += 1
            qt.QTimer.singleShot(0, self._releaseInputUpdate)
        if self.roiNode is not None and node.GetID() == self.roiNode.GetID():
            self.setRoiNode(None)
            self.scheduleRoiUpdate()  # removes spheres and handles outside of this scene callback
        elif self.handlesNode is not None and node.GetID() == self.handlesNode.GetID():
            self.setHandlesNode(None)
            self.scheduleRoiUpdate()  # handles get recreated
        elif self.filterCropRoiNode is not None and node.GetID() == self.filterCropRoiNode.GetID():
            # The crop box was deleted elsewhere (e.g. Data module): "Limit to ROI" follows
            self.filterCropRoiNode = None
            self._setLimitToRoiChecked(False)

    # ------------------------------------------------------------------
    # Changing the input volumes after "Go"
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def selectorsSetByPanel(self):
        """Selector changes made by the panel itself (scene load, filters) do not trigger a view update."""
        self._suppressInputUpdate += 1
        try:
            yield
        finally:
            self._suppressInputUpdate -= 1

    def _releaseInputUpdate(self):
        self._suppressInputUpdate = max(0, self._suppressInputUpdate - 1)

    def onInputVolumeSelectionChanged(self, node=None):
        self.windowInfoOverlay.scheduleUpdate()  # the CT / PET labels follow the selectors
        if self._suppressInputUpdate or sceneIsBusy():
            return
        self.inputUpdateTimer.start()  # (re)started: a PET and a CT change in a row give one update

    def applyChangedInputVolumes(self):
        """
        Once the views have been built with "Go", show a newly selected SPECT/PET or CT/MRI in them right away.
        Slice positions, pan and zoom, the 3D camera and the fusion opacity stay as they are.
        """
        if sceneIsBusy() or self._filterRunning:
            return
        settingsNode = self.logic.getSettingsNode(create=False)
        if settingsNode is None:
            return  # "Go" has not been pressed in this scene yet
        oldPet = settingsNode.GetNodeReference(SETTINGS_PET_ROLE)
        oldCt = settingsNode.GetNodeReference(SETTINGS_CT_ROLE)
        pet = self.inputVolumeSelector.currentNode()
        ct = self.inputVolumeSelectorCT.currentNode()
        if pet is None or ct is None or (pet is oldPet and ct is oldCt):
            return
        if pet is ct:
            return  # e.g. a newly loaded volume selected in both boxes: wait for the user to pick the other one
        try:
            self.updateFusionVolumes(oldPet, oldCt, pet, ct)
        except Exception:
            logging.exception("EasyFusion: could not update the views to the new volumes")
            return
        slicer.util.showStatusMessage(f"EasyFusion: showing {pet.GetName()} on {ct.GetName()}", 3000)

    def updateFusionVolumes(self, oldPet, oldCt, pet, ct):
        """Like "Go", but without touching slice positions, zoom or the 3D camera."""
        for node in (pet, ct):
            if node.GetDisplayNode() is None:
                node.CreateDefaultDisplayNodes()
        opacities = self.logic.foregroundOpacities(oldPet)  # keep the user's fusion opacity

        if pet is not oldPet:
            if oldPet is not None and oldPet.GetDisplayNode() is not None:
                # Same color map and window as before (switch e.g. original <-> filtered, or another time point)
                self.logic.copyScalarDisplaySettings(oldPet, pet)
            else:
                displayNode = pet.GetDisplayNode()
                displayNode.SetAutoWindowLevel(False)
                displayNode.SetWindow(10)
                displayNode.SetLevel(5)
                displayNode.SetInterpolate(True)
                colorNodeName = FUSION_COLOR_MAPS.get(self.petColorMapSelector.currentText)
                if colorNodeName is not None:
                    self.setPETColorMap(colorNodeName)
            if oldPet is None or not self.logic.moveMipToVolume(oldPet, pet):
                if oldPet is None:  # no MIP to move: show one, as "Go" does (a MIP the user hid stays hidden)
                    displayNode = pet.GetDisplayNode()
                    self.logic.showOnlyThisVolumeRendering(pet)
                    vrDisplayNode = slicer.modules.volumerendering.logic().CreateDefaultVolumeRenderingNodes(pet)
                    vrDisplayNode.SetVisibility(True)
                    self.logic.setMIPRange(vrDisplayNode, displayNode.GetLevel() - displayNode.GetWindow() / 2.0,
                                           displayNode.GetLevel() + displayNode.GetWindow() / 2.0, flatOpacity=True)

        if ct is not oldCt:
            # Window left to the volume itself (Slicer's auto window for a new one); presets set it afterwards
            ct.GetDisplayNode().SetAndObserveColorNodeID(slicer.util.getNode("Grey").GetID())

        self.logic.rememberVolumes(pet, ct)
        self.logic.applyViewRoles(pet, ct)  # composite nodes only: no fit, no orientation change
        self.logic.restoreForegroundOpacities(opacities)
        self.scheduleRoiUpdate()

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
            # The first time, Slicer applies the orthographic switch and the new layout size only after this
            # function returns, which replaces the zoom above; fit again once everything has settled.
            qt.QTimer.singleShot(MIP_FIT_DELAY_MS, lambda: self.fitMIPToView(petNode))

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
        if layoutManager.layout == layoutID:
            self.scheduleMIPRefit(layoutID)  # same layout clicked again: no layoutChanged signal, refit anyway
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
        # ... and new slice views: give them their window info
        self.windowInfoOverlay.scheduleUpdate()
        if layoutID is not None:  # a real layout switch (the signal), not a refresh on module enter / scene load
            self.scheduleMIPRefit(current)

    def scheduleMIPRefit(self, layoutID=None):
        """Fit the MIP again once the new layout has its final size (Monitor 2 window placed after 300 ms)."""
        qt.QTimer.singleShot(MIP_FIT_DELAY_MS, self.refitMIP)
        if layoutID in DUAL_MONITOR_LAYOUT_IDS:
            qt.QTimer.singleShot(MIP_FIT_DELAY_MS + 400, self.refitMIP)

    def refitMIP(self):
        """Re-fit only when the MIP of the selected PET is shown; otherwise leave the 3D camera alone."""
        if sceneIsBusy():
            return
        pet = self.inputVolumeSelector.currentNode()
        if pet is None or not slicer.mrmlScene.IsNodePresent(pet):
            return
        displayNode = slicer.modules.volumerendering.logic().GetFirstVolumeRenderingDisplayNode(pet)
        if displayNode is None or not displayNode.GetVisibility():
            return
        self.fitMIPToView(pet)

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
        self.applyThresholdControls()

    def onThresholdValueChanged(self, value):
        mode, _ = self.currentThreshold()
        if mode == THRESHOLD_ABSOLUTE:
            self._absoluteThreshold = float(value)
        else:
            self._relativeThreshold = float(value)
        self.applyThresholdControls()

    def applyThresholdControls(self):
        """
        A threshold change goes to the ROI selected in the table only. What the controls show is also saved as
        the threshold for new ROIs. With no ROI selected, existing ROIs keep their own thresholds.
        """
        self.saveThresholdSettings()
        node = self.roiNode
        pointID = self.selectedRoiPointID()
        if node is not None and pointID is not None:
            self.logic.setRoiThreshold(node, pointID, *self.currentThreshold())
            self.scheduleRoiUpdate()

    def setThresholdControls(self, mode, value):
        """Show a threshold in the controls without applying it to anything (e.g. the selected ROI's own)."""
        mode = mode if mode in (THRESHOLD_RELATIVE, THRESHOLD_ABSOLUTE) else THRESHOLD_RELATIVE
        if mode == THRESHOLD_ABSOLUTE:
            self._absoluteThreshold = float(value)
        else:
            self._relativeThreshold = float(value)
        wasBlocked = self.thresholdModeComboBox.blockSignals(True)
        self.thresholdModeComboBox.setCurrentIndex(self.thresholdModeComboBox.findData(mode))
        self.thresholdModeComboBox.blockSignals(wasBlocked)
        self._applyThresholdModeToSpinBox(mode)
        self.saveThresholdSettings()  # the controls' values are what new ROIs get

    def syncThresholdControls(self, rows, selectedID):
        """After a programmatic selection (new or dragged ROI), show that ROI's threshold in the controls."""
        if selectedID is None:
            return
        for pointID, _, _, _, (mode, value), _ in rows:
            if pointID == selectedID:
                currentMode, currentValue = self.currentThreshold()
                if mode != currentMode or abs(value - currentValue) > 1e-6:
                    self.setThresholdControls(mode, value)
                return

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
        """Center the MIP on the volume and zoom so the 3D view shows MIP_FIT_HEIGHT_MM from top to bottom."""
        threeDWidget = self.getThreeDWidget()
        if threeDWidget is None or volumeNode is None or not slicer.mrmlScene.IsNodePresent(volumeNode):
            return
        threeDView = threeDWidget.threeDView()
        threeDView.forceRender()  # applies pending view node changes (orthographic mode) before the fit
        bounds = [0.0] * 6
        volumeNode.GetRASBounds(bounds)
        if bounds[1] < bounds[0]:
            return  # empty volume
        renderer = threeDView.renderWindow().GetRenderers().GetFirstRenderer()
        camera = renderer.GetActiveCamera()
        # Move the camera sideways to the volume center, keeping the viewing direction and distance
        center = [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0]
        focalPoint, position = camera.GetFocalPoint(), camera.GetPosition()
        camera.SetFocalPoint(*center)
        camera.SetPosition(*[p + c - f for p, c, f in zip(position, center, focalPoint)])
        camera.SetParallelScale(MIP_FIT_HEIGHT_MM / 2.0)  # parallel scale = half of the visible height
        renderer.ResetCameraClippingRange()
        threeDView.forceRender()

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
        if kind == "ct" and not self.ctMriLooksLikeCT():
            mriPreset = next((p for p in MRI_PERCENTILE_PRESETS if p[3] == key), None)
            if mriPreset is not None:
                self.setMRIWindowPercentile(mriPreset[1], mriPreset[2], mriPreset[0])
            return
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

    def mriTissueSample(self, volumeNode):
        """Sampled tissue intensities of the CT/MRI volume (cached until the volume or its voxels change)."""
        imageData = volumeNode.GetImageData() if volumeNode is not None else None
        if imageData is None or imageData.GetNumberOfPoints() == 0:
            return None
        key = (volumeNode.GetID(), imageData.GetMTime())
        if self._mriSample[0] == key:
            return self._mriSample[1]
        voxels = slicer.util.arrayFromVolume(volumeNode)
        if voxels.ndim > 3:
            voxels = voxels[..., 0]
        step = max(1, int(math.ceil((voxels.size / float(MRI_PRESET_SAMPLE_SIZE)) ** (1.0 / 3.0))))
        sample = tissueSample(voxels[::step, ::step, ::step])
        self._mriSample = (key, sample)
        return sample

    def setMRIWindowPercentile(self, lowerPercentile, upperPercentile, text=""):
        """MRI presets: window between two percentiles of the tissue intensities of the CT/MRI volume."""
        volumeNode = self.inputVolumeSelectorCT.currentNode()
        if volumeNode is None or volumeNode.GetImageData() is None:
            slicer.util.showStatusMessage("EasyFusion: select a CT/MRI volume first.", 3000)
            return
        sample = self.mriTissueSample(volumeNode)
        windowLevel = percentileWindow(sample, lowerPercentile, upperPercentile) if sample is not None else None
        if windowLevel is None:
            slicer.util.showStatusMessage("EasyFusion: the CT/MRI volume has no intensity range.", 3000)
            return
        self.setCTWindow(*windowLevel)
        label = f"{text} " if text else ""
        slicer.util.showStatusMessage(
            f"EasyFusion: MRI {label}(percentile {lowerPercentile:g}–{upperPercentile:g})", 2000)

    def ctMriLooksLikeCT(self):
        volumeNode = self.inputVolumeSelectorCT.currentNode()
        imageData = volumeNode.GetImageData() if volumeNode is not None else None
        if imageData is None or imageData.GetNumberOfPoints() == 0:
            return True  # nothing to decide on: keep the CT presets
        return looksLikeCT(imageData.GetScalarRange()[0])

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
        self.updateMeasuringLabel(pet)

        if node is None:
            self.logic.removeRoiSphereModel()
            self.setHandlesNode(None)
            self.logic.removeRoiHandlesNode()
            self.logic.removeRoiLabelsNode()
            self.logic.removeRoiSegmentationNode()
            self.fillRoiTable([])
            self.mipOverlay.clear()
            return

        defaultThreshold = self.currentThreshold()  # given to ROIs that have none yet (new / older scenes)
        labelFields = self.roiLabelFields()
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
            self._roiCenters = {pointID: tuple(center) for _, pointID, center in rois}

            for index, pointID, center in rois:
                radius = self.logic.getRoiRadius(node, pointID, self.roiRadiusSpinBox.value)
                number = self.logic.getRoiNumber(node, pointID)
                color = self.logic.getRoiColor(node, pointID)
                roiColors[pointID] = color
                threshold = self.logic.getRoiThreshold(node, pointID, defaultThreshold)
                origin = self.logic.getRoiOrigin(node, pointID)

                key = self.logic.statsCacheKey(pet, center, radius, *threshold)
                unchanged = key in self._roiStatsCache
                if unchanged:
                    stats = self._roiStatsCache[key]
                else:
                    stats = self.logic.computeSphereStatistics(pet, center, radius, *threshold)
                newCache[key] = stats

                name = f"ROI-{number}"
                # The center point keeps only the name (shown in the Markups module); the SUV text is
                # drawn by the separate white label layer placed one radius away from the center.
                if node.GetNthControlPointLabel(index) != name:
                    node.SetNthControlPointLabel(index, name)

                rows.append((pointID, name, radius, stats, threshold, origin))
                spheres.append((center, radius, color))
                labelText = formatRoiLabel(name, stats, pet is not None, labelFields, radius, threshold)
                if labelText:  # nothing chosen under "Show near ROI": no label at all
                    labelEntries.append((pointID, center, radius, labelText))
                mipEntries.append((pointID, center, labelText, color))
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
        self.syncThresholdControls(rows, selectedID)
        self.updateRoiEditTarget()
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
        for pointID, _, radius, *_ in rows:
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
        self._roiTableRows = list(rows)
        if not rows:
            self._roiCenters = {}
        if hasattr(self, "exportRoiTableButton"):
            self.exportRoiTableButton.enabled = bool(rows)
        wasBlocked = table.blockSignals(True)
        try:
            table.setRowCount(len(rows))
            for rowIndex, (pointID, name, radius, stats, threshold, origin) in enumerate(rows):
                values = formatRoiTableRow(name, radius, stats, threshold)
                for column, text in enumerate(values):
                    item = qt.QTableWidgetItem(text)
                    if column == 0:
                        item.setData(qt.Qt.UserRole, pointID)
                        item.setToolTip("Placed by AI" if origin == ROI_ORIGIN_AI else "Placed by the user")
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
        self.updateRoiEditTarget()
        node = self.roiNode
        pointID = self.selectedRoiPointID()
        if node is None or pointID is None:
            return
        radius = self.logic.getRoiRadius(node, pointID, self.roiRadiusSpinBox.value)
        wasBlocked = self.roiRadiusSpinBox.blockSignals(True)
        self.roiRadiusSpinBox.setValue(radius)
        self.roiRadiusSpinBox.blockSignals(wasBlocked)
        # The ROI's own threshold is loaded into the controls (they now edit this ROI only)
        self.setThresholdControls(*self.logic.getRoiThreshold(node, pointID, self.currentThreshold()))
        index = node.GetNthControlPointIndexByID(pointID)
        if index >= 0:
            slicer.modules.markups.logic().JumpSlicesToNthPointInMarkup(node.GetID(), index, True)

    def onDeselectRoi(self):
        self.roiTable.clearSelection()  # -> onRoiSelectionChanged: the controls now set values for new ROIs

    def updateRoiEditTarget(self):
        """Tell which ROI the radius / threshold controls edit: the selected one, or new ROIs."""
        if not hasattr(self, "roiEditTargetLabel"):
            return
        selectedRows = self.roiTable.selectionModel().selectedRows()
        item = self.roiTable.item(selectedRows[0].row(), 0) if selectedRows else None
        if item is not None:
            self.roiEditTargetLabel.text = f"Radius and threshold: editing {item.text()} only"
            self.roiDeselectButton.enabled = True
        else:
            self.roiEditTargetLabel.text = "Radius and threshold: values for new ROIs"
            self.roiDeselectButton.enabled = False

    def roiLabelFields(self):
        if not hasattr(self, "roiLabelFieldCheckBoxes"):
            return tuple(DEFAULT_ROI_LABEL_FIELDS)
        return tuple(key for key, _ in ROI_LABEL_FIELDS if self.roiLabelFieldCheckBoxes[key].checked)

    def onRoiLabelFieldsChanged(self, checked=None):
        qt.QSettings().setValue(SETTINGS_ROI_LABEL_FIELDS, ",".join(self.roiLabelFields()))
        self.scheduleRoiUpdate()

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

    def onExportRoiTable(self):
        """Write the ROI table to a .tsv file chosen by the user."""
        # A pending recomputation (e.g. right after a drag) would make the file lag behind the views
        if self.roiUpdateTimer.isActive():
            self.roiUpdateTimer.stop()
            self.updateRois()
        rows = list(self._roiTableRows)
        if not rows:
            slicer.util.infoDisplay("There are no ROIs to export.", windowTitle="Export ROI table")
            return

        pet = self.inputVolumeSelector.currentNode()
        petName = pet.GetName() if pet is not None else ""
        petFilter = ""
        if pet is not None and pet.GetAttribute(FILTER_MODEL_ATTRIBUTE):
            petFilter = pet.GetAttribute(FILTER_MODEL_ATTRIBUTE)
            if pet.GetAttribute(FILTER_CROPPED_ATTRIBUTE):
                petFilter += " (ROI only)"

        settings = qt.QSettings()
        folder = settings.value(SETTINGS_ROI_EXPORT_FOLDER) or ""
        if not os.path.isdir(folder):
            folder = qt.QStandardPaths.writableLocation(qt.QStandardPaths.DocumentsLocation) or ""
        path = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(), "Export ROI table",
            os.path.join(folder, defaultRoiExportFileName(petName)),
            "Tab-separated values (*.tsv);;All files (*)")
        if isinstance(path, (tuple, list)):  # some Qt bindings return (fileName, selectedFilter)
            path = path[0] if path else ""
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".tsv"

        text = formatRoiTableTsv(rows, self._roiCenters, petName, petFilter)
        try:
            with open(path, "w", encoding="utf-8", newline="") as tsvFile:
                tsvFile.write(text)
        except OSError as error:
            slicer.util.errorDisplay(f"Could not write the ROI table:\n{path}\n\n{error}",
                                     windowTitle="Export ROI table")
            return
        settings.setValue(SETTINGS_ROI_EXPORT_FOLDER, os.path.dirname(path))
        slicer.util.showStatusMessage(f"Exported {len(rows)} ROI(s) to {path}", 5000)
        logging.info(f"EasyFusion: exported {len(rows)} ROI(s) to {path}")

    def updateMeasuringLabel(self, pet):
        """Which PET the ROIs measure on; a filtered volume gets a permanent reminder that its SUVs differ."""
        label = self.roiPetLabel
        if pet is None:
            label.text, toolTip, style = "Measuring on: (no PET selected)", "", ""
        elif pet.GetAttribute(FILTER_MODEL_ATTRIBUTE):
            source = pet.GetNodeReference(FILTER_SOURCE_ROLE)
            region = ", ROI only" if pet.GetAttribute(FILTER_CROPPED_ATTRIBUTE) else ""
            label.text = (f"Measuring on: {pet.GetName()}\n"
                          f"⚠ AI-filtered ({pet.GetAttribute(FILTER_MODEL_ATTRIBUTE)}{region}): "
                          "SUVs differ from the original")
            toolTip = f"Original volume: {source.GetName()}" if source is not None else ""
            style = "color: #d9822b;"
        else:
            label.text, toolTip, style = f"Measuring on: {pet.GetName()}", "", ""
        label.setToolTip(toolTip)
        label.setStyleSheet(style)

    # ------------------------------------------------------------------
    # AI post-processing filters
    # ------------------------------------------------------------------

    def restoreFilterModelFolder(self):
        """This module's last folder; otherwise the folder last used in the PETDenoise module."""
        folder = qt.QSettings().value(SETTINGS_FILTER_MODEL_FOLDER) or ""
        if not os.path.isdir(folder):
            folder = self.filterLogic.petDenoiseModelFolder() or ""
        self.filterModelFolderEdit.text = folder
        self.refreshFilterModels()

    def onBrowseFilterModelFolder(self):
        folder = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(), "Select the model folder", self.filterModelFolderEdit.text or "")
        if not folder:
            return
        self.filterModelFolderEdit.text = folder
        qt.QSettings().setValue(SETTINGS_FILTER_MODEL_FOLDER, folder)
        self.refreshFilterModels()

    def refreshFilterModels(self):
        folder = self.filterModelFolderEdit.text
        previous = self.filterModelSelector.currentText
        try:
            models = sorted((name for name in os.listdir(folder) if name.lower().endswith(".pth")), key=str.lower)
        except OSError:
            models = []
        existing = [self.filterModelSelector.itemText(i) for i in range(self.filterModelSelector.count)]
        if models == existing and self._filterParams is not None:
            return  # nothing new: keep the selection and a manually chosen "Apply to"
        wasBlocked = self.filterModelSelector.blockSignals(True)
        self.filterModelSelector.clear()
        self.filterModelSelector.addItems(models)
        if previous in models:
            self.filterModelSelector.setCurrentIndex(models.index(previous))
        self.filterModelSelector.blockSignals(wasBlocked)
        self.onFilterModelChanged()

    def onFilterModelChanged(self, index=None):
        modelName = self.filterModelSelector.currentText
        self.applyFilterButton.enabled = bool(modelName) and not self._filterRunning
        if not modelName:
            self._filterParams, self._filterNotes = None, {}
            self.filterInfoBox.setPlainText(
                "No .pth models in this folder." if self.filterModelFolderEdit.text else "Select a model folder.")
            return
        sidecarPath = os.path.join(self.filterModelFolderEdit.text, os.path.splitext(modelName)[0] + ".txt")
        text = None
        if os.path.isfile(sidecarPath):
            try:
                with open(sidecarPath, "r", encoding="utf-8-sig", errors="replace") as sidecar:
                    text = sidecar.read()
            except OSError:
                logging.exception(f"EasyFusion: could not read {sidecarPath}")
        self._filterParams, self._filterNotes = parseFilterMetadata(text)
        if text is None:
            self.filterInfoBox.setPlainText(
                "No description file (.txt) found for this model. The default parameters of the PETDenoise "
                "module are used, which may not match the model.")
        else:
            self.filterInfoBox.setPlainText(text.strip())
        target = guessFilterTarget(modelName, self._filterParams)
        self.filterTargetSelector.setCurrentIndex(self.filterTargetSelector.findData(target))

    def currentFilterTarget(self):
        target = self.filterTargetSelector.itemData(self.filterTargetSelector.currentIndex)
        return target if target in (FILTER_TARGET_PET, FILTER_TARGET_CT) else FILTER_TARGET_PET

    def _setLimitToRoiChecked(self, checked):
        """Change the "Limit to ROI" box without creating / removing the crop ROI."""
        if not hasattr(self, "filterLimitToRoiCheckBox"):
            return
        wasBlocked = self.filterLimitToRoiCheckBox.blockSignals(True)
        self.filterLimitToRoiCheckBox.checked = checked
        self.filterLimitToRoiCheckBox.blockSignals(wasBlocked)

    def filterTargetVolume(self):
        """(volume the filter applies to, its kind for messages), following "Apply to"."""
        if self.currentFilterTarget() == FILTER_TARGET_PET:
            return self.inputVolumeSelector.currentNode(), "SPECT/PET"
        return self.inputVolumeSelectorCT.currentNode(), "CT/MRI"

    def onFilterLimitToRoiToggled(self, checked):
        if not checked:
            self.removeFilterCropRoi()
            if self.filterStatusLabel.text.startswith("Crop ROI placed"):
                self.filterStatusLabel.text = ""
            return
        volumeNode, kind = self.filterTargetVolume()
        if volumeNode is None or volumeNode.GetImageData() is None:
            slicer.util.warningDisplay(f"Select a {kind} volume first: the crop ROI is fitted to it.")
            self._setLimitToRoiChecked(False)
            return
        roiNode = self.filterCropRoiNode
        if roiNode is None or not slicer.mrmlScene.IsNodePresent(roiNode):
            self.removeFilterCropRoi()  # leftovers, e.g. from a module reload
            try:
                roiNode = self.filterLogic.createCropRoi(volumeNode)
            except Exception:
                logging.exception("EasyFusion: could not create the filter crop ROI")
                self.removeFilterCropRoi()
                roiNode = None
        if roiNode is None:
            slicer.util.errorDisplay("Could not create the crop ROI.")
            self._setLimitToRoiChecked(False)
            return
        self.filterCropRoiNode = roiNode
        self.filterStatusLabel.text = (
            f"Crop ROI placed around '{volumeNode.GetName()}'. Drag the handles of the cyan box in the slice or 3D "
            "views to the region you need, then press Apply Filter. Only that region is filtered; the original "
            "volume is not changed.")

    def removeFilterCropRoi(self):
        self.filterCropRoiNode = None  # first, so the node-removal observer has nothing left to react to
        for node in self.filterLogic.findCropRois() if self.filterLogic is not None else []:
            slicer.mrmlScene.RemoveNode(node)

    def resetFilterCropRoi(self):
        """After a successful run: remove the box and untick "Limit to ROI"."""
        self._setLimitToRoiChecked(False)
        self.removeFilterCropRoi()

    def onApplyFilter(self):
        if self._filterRunning:
            return
        modelName = self.filterModelSelector.currentText
        modelPath = os.path.join(self.filterModelFolderEdit.text, modelName) if modelName else ""
        if not modelName or not os.path.isfile(modelPath):
            slicer.util.warningDisplay("Select a model folder and a model (.pth) first.")
            return
        params = self._filterParams or dict(FILTER_DEFAULT_PARAMETERS)
        target = self.currentFilterTarget()
        petNode = self.inputVolumeSelector.currentNode()
        ctNode = self.inputVolumeSelectorCT.currentNode()
        if target == FILTER_TARGET_PET:
            sourceNode, otherNode, sourceKind, otherKind = petNode, ctNode, "SPECT/PET", "CT/MRI"
        else:
            sourceNode, otherNode, sourceKind, otherKind = ctNode, petNode, "CT/MRI", "SPECT/PET"
        if sourceNode is None or sourceNode.GetImageData() is None:
            slicer.util.warningDisplay(f"Select a {sourceKind} volume first.")
            return

        secondNode = None
        if params["dual_channel"]:
            if otherNode is None or otherNode.GetImageData() is None:
                slicer.util.warningDisplay(
                    f"{modelName} is a dual-channel model: select the {otherKind} volume too (second input).")
                return
            if otherNode.GetTransformNodeID() != sourceNode.GetTransformNodeID():
                slicer.util.warningDisplay(
                    "Dual-channel models need both volumes under the same transform. "
                    "Harden the registration transform (Data module) first.")
                return
            secondNode = otherNode

        roiNode = None
        if self.filterLimitToRoiCheckBox.checked:
            roiNode = self.filterCropRoiNode
            if roiNode is None or not slicer.mrmlScene.IsNodePresent(roiNode):
                self.resetFilterCropRoi()
                slicer.util.warningDisplay("The crop ROI no longer exists. Tick 'Limit to ROI' again to place a new one.")
                return

        croppedNode = None
        try:
            # With "Limit to ROI" the filter reads Crop Volume's output. It is removed on every path below
            # (dialog cancelled, progress cancelled, failure, success); the original volume stays whole.
            inputNode = sourceNode
            if roiNode is not None:
                try:
                    croppedNode = self.filterLogic.cropToRoi(sourceNode, roiNode)
                except Exception as error:
                    logging.exception("EasyFusion: cropping to the filter ROI failed")
                    slicer.util.errorDisplay(f"Cropping to the ROI failed:\n{error}",
                                             detailedText=traceback.format_exc())
                    return
                if croppedNode is None or croppedNode.GetImageData() is None:
                    slicer.util.errorDisplay("Crop Volume did not produce a cropped volume.")
                    return
                inputNode = croppedNode

            suffix = "_crop" if croppedNode is not None else ""
            outputName = self.filterLogic.uniqueVolumeName(
                f"{sourceNode.GetName()}_{os.path.splitext(modelName)[0]}{suffix}")
            decision = self.confirmFilter(sourceNode, inputNode, secondNode, modelName, params, target, outputName)
            if decision == "crop":
                self.filterLimitToRoiCheckBox.checked = True  # places the adjustable box (onFilterLimitToRoiToggled)
                return
            if decision != "apply" or not self._prepareFilterDependencies():
                return

            result = self._runFilterWithProgress(inputNode, secondNode, modelPath, params, outputName, sourceNode)
            if result is None:
                if roiNode is not None:
                    self.filterStatusLabel.text += " The crop ROI is kept so you can adjust it and try again."
                return
            outputNode, deviceText, seconds = result
        finally:
            if croppedNode is not None and slicer.mrmlScene.IsNodePresent(croppedNode):
                slicer.mrmlScene.RemoveNode(croppedNode)

        if roiNode is not None:
            self.resetFilterCropRoi()  # the box has done its job
        try:
            self.showFilteredVolume(sourceNode, outputNode, otherNode, target)
        except Exception:
            logging.exception("EasyFusion: could not switch the views to the filtered volume")
            slicer.util.warningDisplay(
                f"'{outputNode.GetName()}' was created but could not be shown automatically. "
                f"Select it as {sourceKind} and press Go.")
        followers = "views, MIP and SUV ROIs now use it" if target == FILTER_TARGET_PET else "views now use it"
        region = " (ROI region only)" if roiNode is not None else ""
        message = (f"Created '{outputNode.GetName()}'{region} in {seconds:.0f} s ({deviceText}). The {followers}; "
                   f"'{sourceNode.GetName()}' is unchanged.")
        self.filterStatusLabel.text = message
        slicer.util.showStatusMessage(f"EasyFusion: {message}", 6000)
        logging.info(f"EasyFusion: {message} Model: {modelPath}")

    def _prepareFilterDependencies(self):
        try:
            return self.ensureFilterDependencies()
        except Exception as error:
            logging.exception("EasyFusion: could not prepare the AI filter dependencies")
            slicer.util.errorDisplay(f"Could not install the AI filter dependencies:\n{error}",
                                     detailedText=traceback.format_exc())
            return False

    def _runFilterWithProgress(self, inputNode, secondNode, modelPath, params, outputName, sourceNode):
        """Filter behind a modal, cancellable progress dialog. (outputNode, device, seconds), or None if not done."""
        self._filterRunning = True
        self.applyFilterButton.enabled = False
        self.filterStatusLabel.text = ""
        progress = FilterProgressDialog(f"EasyFusion - {os.path.basename(modelPath)}")
        try:
            return self.filterLogic.run(inputNode, secondNode, modelPath, params, outputName,
                                        forceCPU=self.filterForceCpuCheckBox.checked, report=progress,
                                        originalNode=sourceNode)
        except FilterCancelled:
            self.filterStatusLabel.text = "Cancelled. Nothing was added; the original volume is unchanged."
        except Exception as error:
            logging.exception("EasyFusion: AI filter failed")
            self.filterStatusLabel.text = "Filtering failed. Nothing was added; the original volume is unchanged."
            slicer.util.errorDisplay(f"Filtering with {os.path.basename(modelPath)} failed:\n{error}",
                                     detailedText=traceback.format_exc())
        finally:
            progress.close()
            self._filterRunning = False
            self.applyFilterButton.enabled = True
        return None

    def confirmFilter(self, sourceNode, inputNode, secondNode, modelName, params, target, outputName):
        """
        Warn about what the filter changes (SUVs above all) and about large runs, before anything is computed.
        inputNode: what will be filtered (sourceNode, or its cropped copy with "Limit to ROI").
        Returns "apply", "crop" (the user wants to crop first) or None (cancelled).
        """
        esc = html.escape
        sourceName = sourceNode.GetName()
        cropped = inputNode is not sourceNode
        spacingText = " × ".join(f"{v:g}" for v in sourceNode.GetSpacing())
        targetSpacing = None if params["dont_resample"] else params["voxel_spacing"]
        job = filterJobSize(inputNode.GetImageData().GetDimensions(), inputNode.GetSpacing(), params,
                            channels=2 if secondNode is not None else 1)
        gridText = " × ".join(str(d) for d in job["shape"][::-1])
        offerCrop = job["large"] and not cropped
        items = []

        if job["large"]:
            advice = ("Consider filtering only the region you need: <b>Crop with ROI first</b> places an adjustable "
                      "box." if not cropped else "Consider making the crop ROI smaller.")
            items.append(f'<span style="color:#d9534f;"><b>Large volume:</b> about {job["voxels"] / 1e6:.0f} million '
                         f'voxels on the model\'s grid ({gridText}), roughly {formatByteSize(job["memoryBytes"])} of '
                         f'memory and {job["windows"]} windows to process. This can take a long time (especially on '
                         f'CPU) or run out of memory. {advice}</span>')

        if target == FILTER_TARGET_PET:
            headline = "Filtering changes SUV values"
            items.append("SUVmax, SUVmean, MTV and TLG measured on the filtered volume <b>will differ</b> from the "
                         "original. Denoising removes voxel noise, which usually lowers SUVmax, and it can change "
                         "the contrast and apparent size of small lesions.")
            if targetSpacing is not None:
                items.append(f"The image is resampled from {spacingText} mm to "
                             f"{' × '.join(f'{v:g}' for v in targetSpacing)} mm voxels and the result stays on that "
                             "grid. Resampling alone already changes SUVmax.")
            suvNotes = filterSuvNotes(self._filterNotes)
            if suvNotes:
                items.append("Bias reported for this model on its validation data: <b>"
                             + esc("; ".join(suvNotes)) + "</b>. Other scanners, reconstructions and tracers "
                             "can behave differently (see Model info).")
            if params["prevent_negative"]:
                items.append("Negative voxel values are set to 0.")
            if cropped:
                items.append("Only the region inside the crop ROI is filtered. The new volume covers only that "
                             "region, so SUV ROIs outside it will show '(outside PET)'.")
            roiCount = 0
            if self.roiNode is not None:
                roiCount = sum(1 for i in range(self.roiNode.GetNumberOfControlPoints())
                               if self.logic.isControlPointDefined(self.roiNode, i))
            reMeasured = f" ({roiCount} ROI{'s' if roiCount != 1 else ''} will be re-measured)" if roiCount else ""
            items.append(f"The views, the MIP and the SUV ROIs switch to the new volume{reMeasured}. To go back, "
                         f"select <b>{esc(sourceName)}</b> as SPECT/PET and press Go.")
        else:
            headline = "Filtering changes CT/MR voxel values"
            items.append("Voxel values (e.g. Hounsfield units) of the filtered volume will differ from the original. "
                         "SUV measurements are not affected: they are read from the SPECT/PET volume.")
            if targetSpacing is not None:
                items.append(f"The image is resampled from {spacingText} mm to "
                             f"{' × '.join(f'{v:g}' for v in targetSpacing)} mm voxels and the result stays on that grid.")
            if params["prevent_negative"]:
                items.append("<b>This model sets negative values to 0</b>, which removes negative Hounsfield units "
                             "(air, lung, fat). Check that the model is really meant for CT/MR.")
            if cropped:
                items.append("Only the region inside the crop ROI is filtered; the new volume covers only that region.")
            items.append(f"The views switch to the new volume. To go back, select <b>{esc(sourceName)}</b> "
                         "as CT/MRI and press Go.")

        if cropped:
            items.append(f"The original <b>{esc(sourceName)}</b> itself is not cropped. The crop ROI is removed "
                         "after filtering.")
        if secondNode is not None:
            items.append(f"Dual-channel model: <b>{esc(secondNode.GetName())}</b> is used as the second input.")
        if not job["large"]:
            items.append(f"Working grid about {gridText} voxels; roughly {formatByteSize(job['memoryBytes'])} "
                         "of memory needed.")
        items.append("Research use only: filtered values must not replace measurements on the original images "
                     "for clinical reporting.")

        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setIcon(qt.QMessageBox.Warning)
        box.setWindowTitle("EasyFusion - AI post-processing filter")
        box.setTextFormat(qt.Qt.RichText)
        box.setText(f"<b>{headline}</b><br>Model: {esc(modelName)}<br>"
                    f"The original <b>{esc(sourceName)}</b> is not modified. "
                    f"The result is a new volume: <b>{esc(outputName)}</b>")
        box.setInformativeText("<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>")
        box.addButton("Apply Filter", qt.QMessageBox.AcceptRole)
        cropButton = box.addButton("Crop with ROI first", qt.QMessageBox.ActionRole) if offerCrop else None
        cancelButton = box.addButton(qt.QMessageBox.Cancel)
        box.setDefaultButton(cropButton if cropButton is not None else cancelButton)
        box.setEscapeButton(cancelButton)
        box.exec_()
        clicked = box.clickedButton()
        role = box.buttonRole(clicked) if clicked is not None else None
        if role == qt.QMessageBox.AcceptRole:
            return "apply"
        if role == qt.QMessageBox.ActionRole:
            return "crop"
        return None

    @staticmethod
    def ensureFilterDependencies():
        """PyTorch, MONAI and einops; offers to install what is missing. True when all of them can be imported."""
        try:
            import torch  # noqa: F401
        except ImportError:
            try:
                import PyTorchUtils
            except ImportError:
                slicer.util.errorDisplay(
                    "AI filters need PyTorch. Install the 'PyTorch' extension from the Extensions Manager, "
                    "restart Slicer and try again.")
                return False
            if PyTorchUtils.PyTorchUtilsLogic().installTorch(askConfirmation=True) is None:
                return False
        for moduleName, requirement in (("monai", "monai"), ("einops", FILTER_EINOPS_REQUIREMENT)):
            try:
                importlib.import_module(moduleName)
                continue
            except ImportError:
                pass
            if not slicer.util.confirmOkCancelDisplay(
                    f"AI filters need the Python package '{moduleName}', which is not installed.\n"
                    f"Install it now ({requirement})? You may need to restart Slicer afterwards."):
                return False
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            try:
                slicer.util.pip_install(requirement)
            finally:
                qt.QApplication.restoreOverrideCursor()
            importlib.invalidate_caches()
            importlib.import_module(moduleName)  # raises if the installation did not work
        return True

    def showFilteredVolume(self, sourceNode, filteredNode, otherNode, target):
        """
        Show a freshly filtered volume in place of its source: same color map and window, same slice positions
        and zoom, the MIP (PET) and the SUV ROIs follow. The source volume stays in the scene, untouched.
        """
        self.logic.copyScalarDisplaySettings(sourceNode, filteredNode)
        if otherNode is not None and not slicer.mrmlScene.IsNodePresent(otherNode):
            otherNode = None
        if target == FILTER_TARGET_PET:
            petNode, ctNode = filteredNode, otherNode
        else:
            petNode, ctNode = otherNode, filteredNode
        # Both selectors set explicitly: the other one must still show the volume it showed before the run
        with self.selectorsSetByPanel():
            if petNode is not None:
                self.inputVolumeSelector.setCurrentNode(petNode)
            if ctNode is not None:
                self.inputVolumeSelectorCT.setCurrentNode(ctNode)

        if petNode is not None and ctNode is not None:
            opacities = self.logic.foregroundOpacities(sourceNode)  # keep the user's fusion opacity
            self.logic.rememberVolumes(petNode, ctNode)
            self.logic.applyViewRoles(petNode, ctNode)
            self.logic.restoreForegroundOpacities(opacities)
            # Same anatomy, so nothing is re-fitted: every view keeps its slice, pan and zoom
        else:
            slicer.util.setSliceViewerLayers(background=filteredNode)
        if target == FILTER_TARGET_PET:
            self.logic.moveMipToVolume(sourceNode, filteredNode)
        self.scheduleRoiUpdate()


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

class SliceWindowInfoOverlay:
    """
    Window / level of the CT and display range of the SPECT/PET in the bottom-right corner of every slice view.
    Each view lists the volumes it shows (foreground above background), so fusion views have two lines,
    CT-only and PET-only views one. Pure VTK text on the views: nothing is added to the scene or saved.

    It follows window / level changes from anywhere (presets, F5-F9, mouse drags, Volumes module) by
    observing the display nodes of the shown volumes, and view contents by observing the slice composite nodes.
    Slicer's own annotations (Data Probe) rewrite all four corners of the view's built-in corner annotation,
    so this uses a separate text actor of its own.
    """

    def __init__(self, volumesCallback):
        """volumesCallback() -> (SPECT/PET node, CT/MRI node) currently selected in the panel (either may be None)."""
        self.enabled = True
        self._volumes = volumesCallback
        self._actors = {}         # slice view name -> (renderer, vtkCornerAnnotation)
        self._lastText = {}       # slice view name -> (text, color) last drawn
        self._observations = {}   # MRML node ID -> (node, observer tag)
        self._timer = qt.QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(WINDOW_INFO_UPDATE_MS)
        self._timer.connect("timeout()", self.update)

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)
        if self.enabled:
            self.scheduleUpdate()
        else:
            self.clear()

    def scheduleUpdate(self, caller=None, event=None):
        if not self._timer.isActive():  # not restarted: keeps updating during a continuous drag
            self._timer.start()

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _sliceWidgets():
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return []
        widgets = []
        for name in layoutManager.sliceViewNames():
            sliceWidget = layoutManager.sliceWidget(name)
            if sliceWidget is None or sliceWidget.sliceView() is None:
                continue
            compositeNode = sliceWidget.mrmlSliceCompositeNode()
            if compositeNode is None or not slicer.mrmlScene.IsNodePresent(compositeNode):
                continue  # widget left over from a previous layout / scene
            widgets.append((name, sliceWidget, compositeNode))
        return widgets

    @staticmethod
    def _fontSize():
        try:
            return int(qt.QSettings().value(DATA_PROBE_FONT_SIZE_SETTING, WINDOW_INFO_DEFAULT_FONT_SIZE))
        except (TypeError, ValueError):
            return WINDOW_INFO_DEFAULT_FONT_SIZE

    def _describe(self, volumeNode, petNode, ctNode):
        """Window info line of one shown volume, or None when it has no scalar display."""
        displayNode = volumeNode.GetDisplayNode() if volumeNode is not None else None
        if displayNode is None or not hasattr(displayNode, "GetWindow"):
            return None
        window, level = displayNode.GetWindow(), displayNode.GetLevel()
        if volumeNode.GetAttribute(PET_ONLY_VOLUME_ATTRIBUTE):
            # Inverted-grey twin of the PET: same window as the PET, named and labelled after it
            volumeNode = volumeNode.GetNodeReference(PET_ONLY_SOURCE_ROLE) or petNode or volumeNode
        imageData = volumeNode.GetImageData()
        scalarMinimum = imageData.GetScalarRange()[0] if imageData is not None and imageData.GetNumberOfPoints() else None
        unit = voxelUnitLabel(volumeNode)
        nodeID = volumeNode.GetID()
        if ctNode is not None and nodeID == ctNode.GetID():
            kind = "CT" if scalarMinimum is None or looksLikeCT(scalarMinimum) else "MRI"
        elif (petNode is not None and nodeID == petNode.GetID()) or unit == "SUV":
            kind = "PET" if unit == "SUV" else "SPECT/PET"
        elif scalarMinimum is not None and looksLikeCT(scalarMinimum):
            kind = "CT"
        else:
            name = volumeNode.GetName() or "Volume"
            kind = name if len(name) <= 20 else name[:19] + "…"
        return formatWindowInfoLine(kind, window, level, unit)

    def _viewContent(self, compositeNode, petNode, ctNode):
        """(text, text color) for one slice view."""
        scene = slicer.mrmlScene
        background = scene.GetNodeByID(compositeNode.GetBackgroundVolumeID() or "")
        foreground = scene.GetNodeByID(compositeNode.GetForegroundVolumeID() or "")
        lines = []
        if foreground is not None and compositeNode.GetForegroundOpacity() > 0.0:
            lines.append(self._describe(foreground, petNode, ctNode))
        if background is not None:
            lines.append(self._describe(background, petNode, ctNode))
        darkBackground = not (background is not None and background.GetAttribute(PET_ONLY_VOLUME_ATTRIBUTE))
        color = WINDOW_INFO_LIGHT_TEXT if darkBackground else WINDOW_INFO_DARK_TEXT
        return "\n".join(line for line in lines if line), color

    def _actorFor(self, name, sliceView):
        renderWindow = sliceView.renderWindow()
        entry = self._actors.get(name)
        if entry is not None and renderWindow.HasRenderer(entry[0]):
            return entry[1]
        renderer = renderWindow.GetRenderers().GetFirstRenderer()
        if renderer is None:
            return None
        actor = vtk.vtkCornerAnnotation()
        actor.SetPickable(False)
        size = self._fontSize()
        actor.SetMaximumFontSize(size)
        actor.SetMinimumFontSize(size)
        actor.SetNonlinearFontScaleFactor(1)
        actor.GetTextProperty().SetFontFamilyToArial()
        renderer.AddViewProp(actor)
        self._actors[name] = (renderer, actor)
        self._lastText.pop(name, None)
        return actor

    def _syncObservations(self, nodes):
        """Observe exactly these MRML nodes (Modified event)."""
        wanted = {node.GetID(): node for node in nodes if node is not None and node.GetID()}
        for nodeID in list(self._observations):
            node, tag = self._observations[nodeID]
            if nodeID not in wanted or wanted[nodeID] is not node:
                node.RemoveObserver(tag)
                del self._observations[nodeID]
        for nodeID, node in wanted.items():
            if nodeID not in self._observations:
                tag = node.AddObserver(vtk.vtkCommand.ModifiedEvent, self.scheduleUpdate)
                self._observations[nodeID] = (node, tag)

    # --- update / clear ----------------------------------------------------

    def update(self):
        if not self.enabled:
            return
        if sceneIsBusy():
            self._timer.start()  # try again once the scene is idle
            return
        try:
            petNode, ctNode = self._volumes()
        except Exception:
            petNode, ctNode = None, None
        observed = []
        liveNames = set()
        for name, sliceWidget, compositeNode in self._sliceWidgets():
            liveNames.add(name)
            observed.append(compositeNode)
            for volumeID in (compositeNode.GetBackgroundVolumeID(), compositeNode.GetForegroundVolumeID()):
                volumeNode = slicer.mrmlScene.GetNodeByID(volumeID or "")
                if volumeNode is not None:
                    observed.append(volumeNode.GetDisplayNode())
            try:
                text, color = self._viewContent(compositeNode, petNode, ctNode)
            except Exception:
                logging.debug(f"EasyFusion: no window info for slice view {name}", exc_info=True)
                text, color = "", WINDOW_INFO_LIGHT_TEXT
            sliceView = sliceWidget.sliceView()
            actor = self._actorFor(name, sliceView)
            if actor is None or self._lastText.get(name) == (text, color):
                continue
            actor.SetText(WINDOW_INFO_CORNER, text)
            textProperty = actor.GetTextProperty()
            textProperty.SetColor(*color)
            textProperty.SetShadow(color == WINDOW_INFO_LIGHT_TEXT)  # a dark shadow only helps light text
            self._lastText[name] = (text, color)
            sliceView.scheduleRender()
        # Views that no longer exist: forget them (only Python references)
        for name in [name for name in self._actors if name not in liveNames]:
            renderer, actor = self._actors.pop(name)
            renderer.RemoveViewProp(actor)
            self._lastText.pop(name, None)
        self._syncObservations(observed)

    def clear(self):
        """Remove the text from every view and stop observing."""
        self._timer.stop()
        self._syncObservations([])
        widgets = {name: sliceWidget for name, sliceWidget, _ in self._sliceWidgets()}
        for name, (renderer, actor) in self._actors.items():
            renderer.RemoveViewProp(actor)
            if name in widgets:
                widgets[name].sliceView().scheduleRender()
        self._actors = {}
        self._lastText = {}


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
                if not text:
                    continue
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
        was an Epona layout, it was applied without a description. Re-applying the arrangement now that the
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

    # --- Filtered volumes --------------------------------------------------

    @staticmethod
    def copyScalarDisplaySettings(sourceNode, targetNode):
        """Same color map, window / level and interpolation, so a filtered volume is shown like its source."""
        sourceDisplay = sourceNode.GetDisplayNode() if sourceNode is not None else None
        if sourceDisplay is None or targetNode is None:
            return
        if targetNode.GetDisplayNode() is None:
            targetNode.CreateDefaultDisplayNodes()
        targetDisplay = targetNode.GetDisplayNode()
        wasModifying = targetDisplay.StartModify()
        if sourceDisplay.GetColorNodeID():
            targetDisplay.SetAndObserveColorNodeID(sourceDisplay.GetColorNodeID())
        targetDisplay.SetAutoWindowLevel(False)
        targetDisplay.SetWindow(sourceDisplay.GetWindow())
        targetDisplay.SetLevel(sourceDisplay.GetLevel())
        targetDisplay.SetInterpolate(sourceDisplay.GetInterpolate())
        targetDisplay.EndModify(wasModifying)

    @staticmethod
    def foregroundOpacities(volumeNode):
        """{composite node: opacity} of the EasyFusion views that show volumeNode as foreground (fusion views)."""
        if volumeNode is None:
            return {}
        return {compositeNode: compositeNode.GetForegroundOpacity()
                for _, _, compositeNode in Easy_fusionLogic.roleSliceNodes()
                if compositeNode.GetForegroundVolumeID() == volumeNode.GetID()}

    @staticmethod
    def restoreForegroundOpacities(opacities):
        for compositeNode, opacity in opacities.items():
            if abs(compositeNode.GetForegroundOpacity() - opacity) > 1e-6:
                compositeNode.SetForegroundOpacity(opacity)

    def moveMipToVolume(self, sourceNode, targetNode):
        """If sourceNode is the MIP in the 3D view, show targetNode there instead, with the same grey range."""
        vrLogic = slicer.modules.volumerendering.logic()
        sourceVr = vrLogic.GetFirstVolumeRenderingDisplayNode(sourceNode)
        if sourceVr is None or not sourceVr.GetVisibility():
            return False
        lower = upper = None
        propertyNode = sourceVr.GetVolumePropertyNode()
        if propertyNode is not None:
            lower, upper = propertyNode.GetVolumeProperty().GetRGBTransferFunction(0).GetRange()
        if lower is None or not upper > lower:
            displayNode = targetNode.GetDisplayNode()
            lower = displayNode.GetLevel() - displayNode.GetWindow() / 2.0
            upper = displayNode.GetLevel() + displayNode.GetWindow() / 2.0
        targetVr = vrLogic.CreateDefaultVolumeRenderingNodes(targetNode)
        self.showOnlyThisVolumeRendering(targetNode)  # only one MIP at a time
        targetVr.SetVisibility(True)
        self.setMIPRange(targetVr, lower, upper, flatOpacity=True)
        return True

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
    def getRoiThreshold(node, pointID, defaultThreshold):
        """
        (mode, value) of this ROI's segment threshold. An ROI without one (just placed, or from a scene saved
        before thresholds were per ROI) gets defaultThreshold, the panel's current setting, and keeps it.
        """
        threshold = parseRoiThreshold(node.GetAttribute(ROI_THRESHOLD_ATTRIBUTE + pointID))
        if threshold is None:
            threshold = (defaultThreshold[0], float(defaultThreshold[1]))
            node.SetAttribute(ROI_THRESHOLD_ATTRIBUTE + pointID, formatRoiThreshold(*threshold))
        return threshold

    @staticmethod
    def setRoiThreshold(node, pointID, mode, value):
        text = formatRoiThreshold(mode, value)
        if node.GetAttribute(ROI_THRESHOLD_ATTRIBUTE + pointID) != text:
            node.SetAttribute(ROI_THRESHOLD_ATTRIBUTE + pointID, text)

    @staticmethod
    def getRoiOrigin(node, pointID):
        """ROI_ORIGIN_USER or ROI_ORIGIN_AI. ROIs without the flag were placed by hand, so they become "user"."""
        origin = node.GetAttribute(ROI_ORIGIN_ATTRIBUTE + pointID)
        if origin not in (ROI_ORIGIN_USER, ROI_ORIGIN_AI):
            origin = ROI_ORIGIN_USER
            node.SetAttribute(ROI_ORIGIN_ATTRIBUTE + pointID, origin)
        return origin

    @staticmethod
    def setRoiOrigin(node, pointID, origin):
        if origin not in (ROI_ORIGIN_USER, ROI_ORIGIN_AI):
            raise ValueError(f"ROI origin must be '{ROI_ORIGIN_USER}' or '{ROI_ORIGIN_AI}', not {origin!r}")
        node.SetAttribute(ROI_ORIGIN_ATTRIBUTE + pointID, origin)

    def addRoi(self, node, centerWorld, radius, thresholdMode, thresholdValue, origin=ROI_ORIGIN_AI):
        """
        Add an ROI from code, e.g. from an AI lesion detector, with its own radius, threshold and origin flag.
        The panel picks it up on its next update like a hand-placed ROI. Returns the ROI's control point ID.
        """
        index = node.AddControlPoint([float(c) for c in centerWorld])
        pointID = node.GetNthControlPointID(index)
        self.setRoiRadius(node, pointID, radius)
        self.setRoiThreshold(node, pointID, thresholdMode, thresholdValue)
        self.setRoiOrigin(node, pointID, origin)
        return pointID

    PER_ROI_ATTRIBUTES = (ROI_RADIUS_ATTRIBUTE, ROI_NUMBER_ATTRIBUTE, ROI_COLOR_ATTRIBUTE,
                          ROI_THRESHOLD_ATTRIBUTE, ROI_ORIGIN_ATTRIBUTE)

    @staticmethod
    def forgetRoi(node, pointID):
        for prefix in Easy_fusionLogic.PER_ROI_ATTRIBUTES:
            node.RemoveAttribute(prefix + pointID)

    @staticmethod
    def forgetAllRois(node):
        for name in list(node.GetAttributeNames()):
            if name.startswith(Easy_fusionLogic.PER_ROI_ATTRIBUTES):
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


# ---------------------------------------------------------------------------
# AI post-processing filters
# ---------------------------------------------------------------------------

class FilterProgressDialog:
    """
    Modal progress window of a filter run. Calling it updates the text / bar and lets Qt repaint;
    once Cancel was pressed, the next call raises FilterCancelled (the run then cleans up after itself).
    """

    def __init__(self, title):
        dialog = qt.QProgressDialog(slicer.util.mainWindow())
        dialog.setWindowTitle(title)
        dialog.setWindowModality(qt.Qt.ApplicationModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumWidth(420)
        dialog.setRange(0, 0)  # busy indicator until the number of windows is known
        dialog.setLabelText("Preparing…")
        dialog.show()
        self.dialog = dialog
        slicer.app.processEvents()

    def __call__(self, text, done=0, total=0):
        if self.dialog.wasCanceled:
            raise FilterCancelled()
        if total > 0:
            done = min(int(done), int(total))
            self.dialog.setLabelText(f"{text}\nWindow {done} of {total}")
            self.dialog.setRange(0, int(total))
            self.dialog.setValue(done)
        else:
            self.dialog.setLabelText(text)
            self.dialog.setRange(0, 0)
        slicer.app.processEvents()
        if self.dialog.wasCanceled:
            raise FilterCancelled()

    def close(self):
        self.dialog.close()
        self.dialog.deleteLater()


class Easy_fusionFilterLogic:
    """
    AI post-processing (denoising / super-resolution) with the models of the Belenos PET Denoise module.
    Same pipeline as PETDenoise: linear resampling to the model's voxel spacing, sliding-window inference
    (gaussian blending, 25% overlap), result = input - predicted noise, optional clipping of negative values.
    The source volume is never written to: the result is always a new volume.
    """

    @staticmethod
    def petDenoiseModelFolder():
        """Model folder last used in the PETDenoise module (its model_config.ini), if that module is installed."""
        directories = []
        try:
            directories.append(os.path.dirname(slicer.modules.petdenoise.path))
        except AttributeError:
            pass
        # Usual extension layout: <extension>/Easy_fusion/Easy_fusion.py next to <extension>/PETDenoise/
        directories.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PETDenoise"))
        for directory in directories:
            iniPath = os.path.join(directory, "model_config.ini")
            if not os.path.isfile(iniPath):
                continue
            config = configparser.ConfigParser()
            try:
                config.read(iniPath)
            except configparser.Error:
                continue
            folder = config.get("ModelFolder", "path", fallback="")
            if folder and os.path.isdir(folder):
                return folder
        return None

    @staticmethod
    def uniqueVolumeName(baseName):
        name, number = baseName, 1
        while slicer.mrmlScene.GetFirstNodeByName(name) is not None:
            name = f"{baseName}_{number}"
            number += 1
        return name

    @staticmethod
    def chooseDevice(torch, forceCPU):
        """GPU when available with at least FILTER_MIN_VRAM_GB of memory (as in PETDenoise), otherwise CPU."""
        cpu = torch.device("cpu")
        if forceCPU:
            return cpu, "CPU, forced"
        if not torch.cuda.is_available():
            return cpu, "CPU"
        try:
            properties = torch.cuda.get_device_properties(0)
        except Exception:
            logging.exception("EasyFusion: could not query the GPU")
            return cpu, "CPU, GPU could not be queried"
        vramGb = properties.total_memory / 1024.0 ** 3
        if vramGb < FILTER_MIN_VRAM_GB:
            return cpu, f"CPU, GPU has only {vramGb:.1f} GB"
        return torch.device("cuda"), f"GPU {properties.name}, {vramGb:.1f} GB"

    @staticmethod
    def buildNetwork(params, inChannels):
        """
        Network of a PETDenoise model. Class structure and attribute names (unet / model / gcfn) are exactly those
        of the PETDenoise module, so its .pth state dicts load unchanged.
        """
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

        architecture = params["architecture"]
        if architecture == "UNET":
            return DenoiseUNet(in_channels=inChannels, channels=params["channels"], num_res_units=params["res_units"],
                               strides=params["strides"], kernel_size=params["down_kernel"],
                               up_kernel_size=params["up_kernel"])
        swinClass = SwinDenoiser if architecture == "SwinUNETR" else SwinGCFN
        return swinClass(in_channels=inChannels, feature_size=params["feature_size"], heads=params["num_heads"],
                         depths=params["depths"], do_rate=params["do_rate"])

    def loadModel(self, torch, modelPath, params, inChannels, device):
        model = self.buildNetwork(params, inChannels).to(device)
        try:
            state = torch.load(modelPath, map_location=device, weights_only=True)
        except TypeError:  # PyTorch older than 1.13 has no weights_only
            state = torch.load(modelPath, map_location=device)
        try:
            model.load_state_dict(state)
        except RuntimeError as error:
            raise RuntimeError(
                f"The weights in {os.path.basename(modelPath)} do not match the {params['architecture']} network "
                "described by its .txt file. Check the parameters in the .txt file.") from error
        model.eval()
        return model

    @staticmethod
    def _addHiddenVolume(name):
        """Scratch volume: hidden from selectors *before* it enters the scene, never saved."""
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

    def resampleToGrid(self, volumeNode, referenceNode):
        """Voxels of volumeNode on the voxel grid of referenceNode, float32 [k, j, i] (second input channel)."""
        scratchNode = self._addHiddenVolume("EasyFusionFilterChannel2")
        try:
            self._runCli(slicer.modules.brainsresample,
                         {"inputVolume": volumeNode.GetID(), "referenceVolume": referenceNode.GetID(),
                          "outputVolume": scratchNode.GetID(), "pixelType": "float", "interpolationMode": "Linear"},
                         "Resampling the second input")
            return np.array(slicer.util.arrayFromVolume(scratchNode), dtype=np.float32)
        finally:
            slicer.mrmlScene.RemoveNode(scratchNode)

    # --- Crop ("Limit to ROI"), done with Slicer's Crop Volume module ---------

    @staticmethod
    def findCropRois():
        return [node for node in slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
                if node.GetAttribute(FILTER_CROP_ROI_ATTRIBUTE)]

    @staticmethod
    def styleCropRoiDisplayNode(displayNode):
        if displayNode is None:
            return
        displayNode.SetSaveWithScene(False)
        wasModifying = displayNode.StartModify()
        displayNode.SetSelectedColor(*FILTER_CROP_ROI_COLOR)
        displayNode.SetColor(*FILTER_CROP_ROI_COLOR)
        # Resize / move handles only: the box stays aligned with the volume axes, as voxel-based cropping expects
        for methodName, value in (("SetHandlesInteractive", True), ("SetScaleHandleVisibility", True),
                                  ("SetTranslationHandleVisibility", True), ("SetRotationHandleVisibility", False),
                                  ("SetFillOpacity", 0.05), ("SetOutlineOpacity", 1.0)):
            method = getattr(displayNode, methodName, None)
            if method is not None:
                method(value)
        displayNode.EndModify(wasModifying)

    def createCropRoi(self, volumeNode):
        """Adjustable crop box fitted to volumeNode with Crop Volume's "Fit to volume". Never saved with the scene."""
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", FILTER_CROP_ROI_NAME)
        roiNode.SetAttribute(FILTER_CROP_ROI_ATTRIBUTE, "1")
        roiNode.SetSaveWithScene(False)
        roiNode.CreateDefaultDisplayNodes()
        self.styleCropRoiDisplayNode(roiNode.GetDisplayNode())
        parametersNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLCropVolumeParametersNode")
        try:
            parametersNode.SetInputVolumeNodeID(volumeNode.GetID())
            parametersNode.SetROINodeID(roiNode.GetID())
            slicer.modules.cropvolume.logic().FitROIToInputVolume(parametersNode)
        finally:
            slicer.mrmlScene.RemoveNode(parametersNode)
        return roiNode

    @staticmethod
    def _hasNonLinearTransform(node):
        transformNode = node.GetParentTransformNode() if node is not None else None
        return transformNode is not None and not transformNode.IsTransformToWorldLinear()

    def cropToRoi(self, volumeNode, roiNode):
        """
        Voxel-based crop of volumeNode to roiNode: the original voxels inside the ROI are copied, nothing is
        interpolated or resampled. Returns a new volume (volumeNode is not changed); the caller removes it when done.

        Slicer's Crop Volume is used when it can do the job. It refuses volumes under a non-linear transform (e.g. a CT
        deformably registered to the PET: "voxel-based cropping of non-linearly transformed input volume is not
        supported"), and Apply() reports that only through its return code (0 = success). In that case, or if it fails
        for any other reason, the crop is done here instead (cropVoxelsUnderTransform).
        """
        if self._hasNonLinearTransform(volumeNode) or self._hasNonLinearTransform(roiNode):
            logging.info(f"EasyFusion: '{volumeNode.GetName()}' is under a non-linear transform; "
                         "cropping it without Crop Volume")
            return self.cropVoxelsUnderTransform(volumeNode, roiNode)

        scene = slicer.mrmlScene
        parametersNode = scene.AddNewNodeByClass("vtkMRMLCropVolumeParametersNode")
        outputNode = None
        try:
            parametersNode.SetInputVolumeNodeID(volumeNode.GetID())
            parametersNode.SetROINodeID(roiNode.GetID())
            parametersNode.SetVoxelBased(True)
            parametersNode.SetIsotropicResampling(False)
            errorCode = slicer.modules.cropvolume.logic().Apply(parametersNode)
            outputID = parametersNode.GetOutputVolumeNodeID()
            outputNode = scene.GetNodeByID(outputID) if outputID else None
        finally:
            scene.RemoveNode(parametersNode)
        if errorCode == 0 and outputNode is not None and outputNode.GetImageData() is not None:
            return outputNode

        logging.warning(f"EasyFusion: Crop Volume failed on '{volumeNode.GetName()}' (error code {errorCode}); "
                        "cropping it without Crop Volume")
        if outputNode is not None and outputNode is not volumeNode and scene.IsNodePresent(outputNode):
            scene.RemoveNode(outputNode)  # empty / half-made output of the failed Crop Volume run
        return self.cropVoxelsUnderTransform(volumeNode, roiNode)

    def cropVoxelsUnderTransform(self, volumeNode, roiNode):
        """
        Voxel-based crop that works whatever transforms volumeNode and roiNode are under, linear or not.
        The ROI box is mapped into the volume's own (untransformed) coordinates through the transforms between the two
        nodes, and the block of original voxels covering it is copied into a new volume that keeps volumeNode's voxel
        grid and parent transform. Under a deformable transform the box is warped in the volume's coordinates, so the
        block is its bounding box there: it covers the whole ROI and, near the edges, slightly more.
        """
        imageData = volumeNode.GetImageData()
        if imageData is None:
            raise RuntimeError(f"'{volumeNode.GetName()}' has no image data.")

        # ROI surface: object coordinates -> ROI node coordinates -> volume node coordinates -> volume IJK
        objectToNode = roiNode.GetObjectToNodeMatrix()
        roiToVolume = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            roiNode.GetParentTransformNode(), volumeNode.GetParentTransformNode(), roiToVolume)
        rasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIjk)
        ijkPoints = []
        for point in boxSurfacePoints(roiNode.GetSize()):
            inRoiNode = objectToNode.MultiplyPoint([float(point[0]), float(point[1]), float(point[2]), 1.0])[:3]
            inVolumeNode = roiToVolume.TransformPoint(inRoiNode)
            ijkPoints.append(rasToIjk.MultiplyPoint(list(inVolumeNode) + [1.0])[:3])

        # One extra voxel all round: the warped box between the sampled points may bulge a little further
        margin = 1 if self._hasNonLinearTransform(volumeNode) or self._hasNonLinearTransform(roiNode) else 0
        block = voxelBlockFromIjkPoints(ijkPoints, imageData.GetDimensions(), margin=margin)
        if block is None:
            raise RuntimeError(f"The crop ROI does not overlap '{volumeNode.GetName()}'. Move the box onto the volume.")
        (i0, j0, k0), (i1, j1, k1) = block
        croppedArray = np.array(slicer.util.arrayFromVolume(volumeNode)[k0:k1, j0:j1, i0:i1])  # copy

        croppedNode = self._addHiddenVolume(f"{volumeNode.GetName()}_cropped")
        try:
            ijkToRas = vtk.vtkMatrix4x4()
            volumeNode.GetIJKToRASMatrix(ijkToRas)
            croppedNode.SetIJKToRASMatrix(ijkToRas)  # same spacing and axes as the original
            croppedNode.SetOrigin(ijkToRas.MultiplyPoint([float(i0), float(j0), float(k0), 1.0])[:3])
            slicer.util.updateVolumeFromArray(croppedNode, croppedArray)
            # Same local coordinates as the original, so it belongs under the same transform (as Crop Volume does)
            croppedNode.SetAndObserveTransformNodeID(volumeNode.GetTransformNodeID())
        except Exception:
            slicer.mrmlScene.RemoveNode(croppedNode)
            raise
        return croppedNode

    @staticmethod
    def _recordProvenance(outputNode, sourceNode, modelPath):
        outputNode.SetAttribute(FILTER_MODEL_ATTRIBUTE, os.path.basename(modelPath))
        outputNode.SetAttribute(FILTER_DATE_ATTRIBUTE, time.strftime("%Y-%m-%d %H:%M:%S"))
        outputNode.SetNodeReferenceID(FILTER_SOURCE_ROLE, sourceNode.GetID())
        # Same quantity and units as the source (e.g. SUVbw, g/ml), so the Data Probe labels values the same way
        for getterName, setterName in (("GetVoxelValueQuantity", "SetVoxelValueQuantity"),
                                       ("GetVoxelValueUnits", "SetVoxelValueUnits")):
            try:
                entry = getattr(sourceNode, getterName)()
                if entry is not None:
                    copied = slicer.vtkCodedEntry()
                    copied.Copy(entry)
                    getattr(outputNode, setterName)(copied)
            except Exception:
                logging.debug(f"EasyFusion: could not copy {getterName} to the filtered volume", exc_info=True)
        # Next to the source in the Data module tree (same patient / study)
        try:
            shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
            parentItem = shNode.GetItemParent(shNode.GetItemByDataNode(sourceNode))
            outputItem = shNode.GetItemByDataNode(outputNode)
            if parentItem and outputItem:
                shNode.SetItemParent(outputItem, parentItem)
        except Exception:
            logging.debug("EasyFusion: could not place the filtered volume in the subject hierarchy", exc_info=True)

    def run(self, sourceNode, secondNode, modelPath, params, outputName, forceCPU=False, report=None,
            originalNode=None):
        """
        Filter sourceNode into a NEW volume called outputName. sourceNode is only read, never modified.
        originalNode: the volume the user selected, when sourceNode is a cropped copy of it ("Limit to ROI");
        the result is tagged with it and placed under its transform.
        secondNode: second input channel of dual-channel models (else None).
        report(text, done=0, total=0): progress callback; it may raise FilterCancelled to stop the run.
        The output volume is added to the scene only once everything worked, so a failed or cancelled run
        leaves nothing behind. Returns (outputNode, device description, seconds).
        """
        import torch
        from monai.inferers import sliding_window_inference

        report = report or (lambda text, done=0, total=0: None)
        originalNode = originalNode or sourceNode
        startTime = time.time()
        scene = slicer.mrmlScene
        scratchNode = None
        model = None
        try:
            report("Loading the model…")
            device, deviceText = self.chooseDevice(torch, forceCPU)
            model = self.loadModel(torch, modelPath, params, 2 if params["dual_channel"] else 1, device)

            # Input on the model's voxel grid. The volume in the scene is never written to: resampling goes into a
            # scratch volume, and without resampling the voxels are copied out before anything else happens.
            if params["dont_resample"]:
                gridNode = sourceNode
            else:
                spacing = params["voxel_spacing"]
                report(f"Resampling to {' × '.join(f'{v:g}' for v in spacing)} mm voxels…")
                scratchNode = self._addHiddenVolume("EasyFusionFilterInput")
                self._runCli(slicer.modules.resamplescalarvolume,
                             {"InputVolume": sourceNode.GetID(), "OutputVolume": scratchNode.GetID(),
                              "outputPixelSpacing": ",".join(f"{float(v):g}" for v in spacing),
                              "interpolationType": "linear"},
                             "Resampling")
                gridNode = scratchNode
            gridArray = slicer.util.arrayFromVolume(gridNode)
            inputDtype = gridArray.dtype
            inputTensor = torch.from_numpy(np.array(gridArray, dtype=np.float32))[None, None]  # (1, 1, K, J, I)
            del gridArray
            networkInput = inputTensor
            if params["dual_channel"]:
                report("Resampling the second input…")
                secondTensor = torch.from_numpy(self.resampleToGrid(secondNode, gridNode))[None, None]
                networkInput = torch.cat([inputTensor, secondTensor], dim=1)
                del secondTensor

            # Windows run on the chosen device; the whole volume and the blending buffers stay in RAM
            roiSize = tuple(params["block_size"])
            total = estimateSlidingWindowCount(inputTensor.shape[2:], roiSize)
            label = f"Filtering on {deviceText}…"
            done = [0]

            def predictor(window, *args, **kwargs):
                report(label, done[0], total)
                prediction = model(window, *args, **kwargs)
                done[0] += 1
                return prediction

            report(label, 0, total)
            with torch.no_grad():
                predictedNoise = sliding_window_inference(
                    inputs=networkInput, roi_size=roiSize, sw_batch_size=1, predictor=predictor,
                    overlap=FILTER_WINDOW_OVERLAP, mode="gaussian", sw_device=device, device=torch.device("cpu"))
            report(label, total, total)
            del networkInput

            # The networks predict the noise: filtered image = input - predicted noise (as in PETDenoise)
            result = (inputTensor - predictedNoise.to(inputTensor.dtype))[0, 0].cpu().numpy()
            del predictedNoise, inputTensor
            if params["prevent_negative"]:
                result = np.clip(result, 0, None)
            result = castFilterResult(result, inputDtype)

            report("Creating the filtered volume…")
            outputNode = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", outputName)
            outputNode.CopyOrientation(gridNode)
            slicer.util.updateVolumeFromArray(outputNode, result)
            # Same local coordinates as the original, so it belongs under the same (e.g. registration) transform
            outputNode.SetAndObserveTransformNodeID(originalNode.GetTransformNodeID())
            outputNode.CreateDefaultDisplayNodes()
            self._recordProvenance(outputNode, originalNode, modelPath)
            if originalNode is not sourceNode:
                outputNode.SetAttribute(FILTER_CROPPED_ATTRIBUTE, "1")
            return outputNode, deviceText, time.time() - startTime
        finally:
            if scratchNode is not None and scene.IsNodePresent(scratchNode):
                scene.RemoveNode(scratchNode)
            model = None
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
