import os
import numpy as np
import slicer
from slicer.ScriptedLoadableModule import *
import logging
import qt
import ctk
import vtk
  

class Easy_fusion(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "EasyFusion"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = """
        This module provides easy fusion of SPECT/PET and CT/MR images.
        """
        parent.acknowledgementText = """
        This file was developed by Burak Demir.
        """
        # **✅ Set the module icon**
        iconPath = os.path.join(os.path.dirname(__file__), "Resources\\Icons\\Easy_fusion.png")
        self.parent.icon = qt.QIcon(iconPath)  # Assign icon to the module
        self.parent = parent

class Easy_fusionWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)




        # Create collapsible section
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)
        formLayout = qt.QFormLayout(parametersCollapsibleButton)
        
        
       
        # 1️⃣ Input Volume Selector (CT Image)
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

        rotationLayout = qt.QHBoxLayout()
  
        # Speed slider
        self.rotationSpeedSlider = ctk.ctkSliderWidget()
        self.rotationSpeedSlider.singleStep = 10
        self.rotationSpeedSlider.minimum = 10
        self.rotationSpeedSlider.maximum = 200
        self.rotationSpeedSlider.value = 50  # Default speed
        self.rotationSpeedSlider.toolTip = "Lower is faster (ms per step)"
        formLayout.addRow("MIP Rotation Speed (ms):", self.rotationSpeedSlider)

        # Toggle button
        self.toggleRotationButton = qt.QPushButton("MIP Stop Rotation")
        formLayout.addRow(self.toggleRotationButton)
        # State tracking
        self.rotationActive = True

        self.toggleRotationButton.connect('clicked()', self.toggleRotation)
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

        self.petHotIronBtn.connect('clicked()', lambda: self.setPETColorMap("CustomHotIron"))
        self.petInfernoBtn.connect('clicked()', lambda: self.setPETColorMap("Inferno"))
        self.petRainbow2Btn.connect('clicked()', lambda: self.setPETColorMap("PET-Rainbow2"))
        self.petRainbow1Btn.connect('clicked()', lambda: self.setPETColorMap("PET-DICOM"))
        self.petRedBtn.connect('clicked()', lambda: self.setPETColorMap("Red"))
        self.petHotMetBlue.connect('clicked()', lambda: self.setPETColorMap("PET-HotMetalBlue"))

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


    def DoFusion(self):
        # Set parameters
        referenceCT = self.inputVolumeSelectorCT.currentNode()
        PETvol = self.inputVolumeSelector.currentNode()

        window = 10
        level = 5
        PETvol.GetDisplayNode().SetAutoWindowLevel(False)
        PETvol.GetDisplayNode().SetWindow(window)
        PETvol.GetDisplayNode().SetLevel(level)
        PETvol.GetDisplayNode().SetInterpolate(True)
        PETvol.GetDisplayNode().SetAndObserveColorNodeID(slicer.util.getNode("InvertedGrey").GetID())


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
            scalarOpacity.AddPoint(10, 1.0) # intensity max → opacity 1.0

            # Notify Slicer of the update
            propertyNode.Modified()
            MIPdisplayNode.Modified()        

        
        threeDwidg = slicer.app.layoutManager().threeDWidget(0)
        viewNode = threeDwidg.mrmlViewNode()
        viewNode.SetRaycastTechnique(2)
        viewNode.SetRenderMode(1)
        viewNode.SetBoxVisible(0)
        viewNode.SetAxisLabelsVisible(0)
        threeDwidg.threeDView().resetFocalPoint()        
        renderer = threeDwidg.threeDView().renderWindow().GetRenderers().GetFirstRenderer()
    


        # Force the view to refresh
        threeDwidg.threeDView().forceRender()
        threeDwidg.threeDView().rotateToViewAxis(3)
        viewNode.SetAnimationMode(1)
        viewNode.SetAnimationMs(50)  # 10 degrees/sec
        bounds = [0]*6
        PETvol.GetRASBounds(bounds)
        height = bounds[5] - bounds[4]
        camera = renderer.GetActiveCamera()        
        camera.SetParallelScale(height * 0.6)  # Zoom fit 
        renderer.GradientBackgroundOff()  # This is still valid and required        
        # Set solid background color (white)
        renderer.SetBackground(1.0, 1.0, 1.0)  # R, G, B
        renderer.SetBackground2(1.0, 1.0, 1.0)  # R, G, B

        # **✅ Set Background to Reference CT (Grayscale)**
        sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode()
        sliceCompositeNode.SetBackgroundVolumeID(referenceCT.GetID())
        sliceCompositeNode.SetForegroundVolumeID(PETvol.GetID())
        sliceCompositeNode.SetForegroundOpacity(0.5)
        referenceCT.GetDisplayNode().SetAndObserveColorNodeID(slicer.util.getNode("Grey").GetID())
        if  self.petColorMapSelector.currentText == "Hot Iron":
            customColorNode = self.createCustomHotIronColorNode()
            PETvol.GetDisplayNode().SetAndObserveColorNodeID(customColorNode.GetID())
        elif self.petColorMapSelector.currentText == "Inferno":
            PETvol.GetDisplayNode().SetAndObserveColorNodeID(slicer.util.getNode("Inferno").GetID())
        elif self.petColorMapSelector.currentText == "Rainbow":
            PETvol.GetDisplayNode().SetAndObserveColorNodeID(slicer.util.getNode("PET-Rainbow2").GetID())




        sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Green").mrmlSliceCompositeNode()
        sliceCompositeNode.SetBackgroundVolumeID(referenceCT.GetID())
        sliceCompositeNode.SetForegroundVolumeID(PETvol.GetID())
        sliceCompositeNode.SetForegroundOpacity(0.5)        
        
        sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Yellow").mrmlSliceCompositeNode()
        sliceCompositeNode.SetBackgroundVolumeID(referenceCT.GetID())
        sliceCompositeNode.SetForegroundVolumeID(PETvol.GetID())
        sliceCompositeNode.SetForegroundOpacity(0.5)                
        

    def createCustomHotIronColorNode(self):

        customColorNode = None

        # Check if it already exists
        if slicer.util.getNodes('CustomHotIron*'):
            customColorNode = slicer.util.getNode('CustomHotIron')
        else:
            customColorNode = slicer.vtkMRMLColorTableNode()


        # Create a new color table node
        customColorNode.SetName("CustomHotIron")
        customColorNode.SetTypeToUser()
        customColorNode.SetNumberOfColors(256)
        customColorNode.SetNamesInitialised(True)

        # Fill the table using the defined color stops
        for i in range(256):
            t = i / 255.0
            if t <= 0.5:
                r = t * 2
                g = 0
                b = 0
            elif t <= 0.75:
                r = 1
                g = (t - 0.5) * 4
                b = 0
            else:
                r = 1
                g = 1
                b = (t - 0.75) * 4

            # Clamp to [0,1]
            r = min(max(r, 0), 1)
            g = min(max(g, 0), 1)
            b = min(max(b, 0), 1)

            customColorNode.SetColor(i, f"Color{i}", r, g, b, 1)

        # Add to scene
        slicer.mrmlScene.AddNode(customColorNode)
        return customColorNode
        
        
        
    def toggleRotation(self):
        viewNode = slicer.app.layoutManager().threeDWidget(0).mrmlViewNode()
        renderer = slicer.app.layoutManager().threeDWidget(0).threeDView().renderWindow().GetRenderers().GetFirstRenderer()      

        if self.rotationActive:
            viewNode.SetAnimationMode(0)
            renderer.GradientBackgroundOff()  # This is still valid and required        
            # Set solid background color (white)
            renderer.SetBackground(1.0, 1.0, 1.0)  # R, G, B
            renderer.SetBackground2(1.0, 1.0, 1.0)  # R, G, B
            self.toggleRotationButton.setText("Start Rotation")
        else:
            ms = int(self.rotationSpeedSlider.value)
            viewNode.SetAnimationMs(ms)
            viewNode.SetAnimationMode(1)
            renderer.GradientBackgroundOff()  # This is still valid and required        
            # Set solid background color (white)
            renderer.SetBackground(1.0, 1.0, 1.0)  # R, G, B
            renderer.SetBackground2(1.0, 1.0, 1.0)  # R, G, B            
            self.toggleRotationButton.setText("Stop Rotation")
        bounds = [0]*6
        self.inputVolumeSelector.currentNode().GetRASBounds(bounds)
        height = bounds[5] - bounds[4]
        camera = renderer.GetActiveCamera()        
        camera.SetParallelScale(height * 0.6)  # Zoom fit 
        renderer.GradientBackgroundOff()  # This is still valid and required   
        self.rotationActive = not self.rotationActive

    def updateRotationSpeed(self, value):
        if self.rotationActive:
            viewNode = slicer.app.layoutManager().threeDWidget(0).mrmlViewNode()
            viewNode.SetAnimationMs(int(value)) 
            renderer = slicer.app.layoutManager().threeDWidget(0).threeDView().renderWindow().GetRenderers().GetFirstRenderer()            
            bounds = [0]*6
            self.inputVolumeSelector.currentNode().GetRASBounds(bounds)
            height = bounds[5] - bounds[4]
            camera = renderer.GetActiveCamera()        
            camera.SetParallelScale(height * 0.6)  # Zoom fit 
            renderer.GradientBackgroundOff()  # This is still valid and required   
            # Set solid background color (white)
            renderer.SetBackground(1.0, 1.0, 1.0)  # R, G, B
            renderer.SetBackground2(1.0, 1.0, 1.0)  # R, G, B             

    def stopRotationIfActive(self):
        if self.rotationActive:
            viewNode = slicer.app.layoutManager().threeDWidget(0).mrmlViewNode()
            viewNode.SetAnimationMode(0)
            renderer = slicer.app.layoutManager().threeDWidget(0).threeDView().renderWindow().GetRenderers().GetFirstRenderer()      
            renderer.GradientBackgroundOff()  # This is still valid and required        
            # Set solid background color (white)
            renderer.SetBackground(1.0, 1.0, 1.0)  # R, G, B
            renderer.SetBackground2(1.0, 1.0, 1.0)  # R, G, B            
            self.rotationActive = False
            self.toggleRotationButton.setText("Start Rotation")
            # Set solid background color (white)
            renderer.SetBackground(1.0, 1.0, 1.0)  # R, G, B
            renderer.SetBackground2(1.0, 1.0, 1.0)  # R, G, B                 

    def setViewAnterior(self):
        self.stopRotationIfActive()
        threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
        threeDView.rotateToViewAxis(3)  # 3 = Anterior

    def setViewLeft(self):
        self.stopRotationIfActive()
        threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
        threeDView.rotateToViewAxis(0)  # 0 = Left

    def setViewRight(self):
        self.stopRotationIfActive()
        threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
        threeDView.rotateToViewAxis(1)  # 1 = Right

    def setCTWindow(self, window, level):
        ctNode = self.inputVolumeSelectorCT.currentNode()
        if ctNode and ctNode.GetDisplayNode():
            dnode = ctNode.GetDisplayNode()
            dnode.SetAutoWindowLevel(False)
            dnode.SetWindow(window)
            dnode.SetLevel(level)

    def setPETWindow(self, window, level):
        petNode = self.inputVolumeSelector.currentNode()
        if petNode and petNode.GetDisplayNode():
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


    def setPETColorMap(self, colorNodeName):
        petNode = self.inputVolumeSelector.currentNode()
        if not petNode:
            return

        if colorNodeName == "CustomHotIron":
            colorNode = self.createCustomHotIronColorNode()
        else:
            try:
                colorNode = slicer.util.getNode(colorNodeName)
            except:
                slicer.util.errorDisplay(f"Color node '{colorNodeName}' not found.")
                return

        # --- Set 2D display ---
        displayNode = petNode.GetDisplayNode()
        if displayNode:
            displayNode.SetAndObserveColorNodeID(colorNode.GetID())



