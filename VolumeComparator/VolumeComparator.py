import os
import slicer
from slicer.ScriptedLoadableModule import *
import qt
import ctk
import vtk
import time  
import numpy as np
import configparser

  

class VolumeComparator(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Belenos - Volume Comparator"
        parent.categories = ["Nuclear Medicine","Informatics","Filtering.Denoising"]
        parent.dependencies = ["PyTorchUtils"]
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = """
        This module provides comparison and loss metrics for two volumes.
        """
        parent.acknowledgementText = """
        This file was developed by Burak Demir.
        """
        # **✅ Set the module icon**
        iconPath = os.path.join(os.path.dirname(__file__), "Resources\\logo.png")
        self.parent.icon = qt.QIcon(iconPath)  # Assign icon to the module
        self.parent = parent

class VolumeComparatorWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        # Create collapsible section
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)
        formLayout = qt.QFormLayout(parametersCollapsibleButton)
        
        
        # **✅ Load the banner image**
        moduleDir = os.path.dirname(__file__)  # Get module directory
        bannerPath = os.path.join(moduleDir, "Resources\\banner.png")  # Change to your banner file

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
        self.inputVolumeSelector.setToolTip("Select first volume.")
        formLayout.addRow("Input Volume 1: ", self.inputVolumeSelector)


        # 1️⃣ Input Volume Selector (PET Image)
        self.inputVolumeSelector2 = slicer.qMRMLNodeComboBox()
        self.inputVolumeSelector2.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputVolumeSelector2.selectNodeUponCreation = True
        self.inputVolumeSelector2.addEnabled = False
        self.inputVolumeSelector2.removeEnabled = False
        self.inputVolumeSelector2.noneEnabled = False
        self.inputVolumeSelector2.showHidden = False
        self.inputVolumeSelector2.showChildNodeTypes = False
        self.inputVolumeSelector2.setMRMLScene(slicer.mrmlScene)
        self.inputVolumeSelector2.setToolTip("Select second volume.")
        formLayout.addRow("Input Volume 2: ", self.inputVolumeSelector2)

        # 📁 Select Model Folder Button and Display
        self.ssimloss_range = qt.QLineEdit()
        self.ssimloss_range.readOnly = False
        self.ssimloss_range.setText("10")
        formLayout.addRow("Range for SSIM:", self.ssimloss_range)

        self.ssimnorm_cbox = qt.QCheckBox()
        formLayout.addRow("Auto Normalize Before SSIM: ", self.ssimnorm_cbox)



        # 5️⃣ Output Log Text Box
        self.outputTextBox = qt.QTextEdit()
        self.outputTextBox.setReadOnly(True)
        self.outputTextBox.setToolTip("Displays processing logs and results.")
        formLayout.addRow("Processing Log and Results:", self.outputTextBox)
        
       
        # 6️⃣ Calculate Button
        self.calculateButton = qt.QPushButton("Compare")
        self.calculateButton.enabled = True
        formLayout.addRow(self.calculateButton)

        # Connect Calculate button to function
        self.calculateButton.connect("clicked(bool)", self.onCalculateButtonClicked)
        
        
        # 5️⃣ Info Text Box
        infoTextBox = qt.QTextEdit()
        infoTextBox.setReadOnly(True)  # Make the text box read-only
        infoTextBox.setPlainText(
            "This module compares two images with identical sizes.\n"
            "It calculates various loss metrics.\n"
            "This module is NOT a medical device. Research use only.\n"
            "Developed by: Burak Demir, MD, FEBNM \n"
            "For support and feedback: 4burakfe@gmail.com\n"
            "Version: v1.0"
        )
        infoTextBox.setToolTip("Module information and instructions.")  # Add a tooltip for additional help
        self.layout.addWidget(infoTextBox)


    def onCalculateButtonClicked(self):
        self.outputTextBox.clear()
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            self.outputTextBox.append("❌ PyTorch is not installed. Please install it before running this module.")
            return None

        self.outputTextBox.append("Now will compare two volumes. Please standby.")
        try:
            from monai.losses import SSIMLoss
        except ImportError:
            msgBox = qt.QMessageBox()
            msgBox.setIcon(qt.QMessageBox.Warning)
            msgBox.setWindowTitle("MONAI Not Installed")
            msgBox.setText("The MONAI library is required but not installed.\nI can install it. Would you like me to install it now? You may want to restart Slicer after installation.")
            msgBox.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
            response = msgBox.exec_()

            if response == qt.QMessageBox.Yes:
                import subprocess, sys
                try:
                    self.outputTextBox.append("📦 Installing MONAI, please wait...")
                    subprocess.check_call([sys.executable,  "-m","pip", "install", "monai"])
                    self.outputTextBox.append("✅ MONAI installed successfully.")
                    
                except Exception as e:
                    self.outputTextBox.append(f"❌ Failed to install MONAI: {e}")
                    return None
            else:
                self.outputTextBox.append("❌ MONAI installation canceled by user.")
                return None
        inputVolumeNode = self.inputVolumeSelector.currentNode()
        if inputVolumeNode is None:
            self.outputTextBox.append("❌ Process stopped. Please select a valid volume...")
            return None
        inputVolumeNode2 = self.inputVolumeSelector2.currentNode()
        if inputVolumeNode2 is None:
            self.outputTextBox.append("❌ Process stopped. Please select a valid secondary volume...")
            return None

        def gradient(img):
            dz = img[:, :, 1:, :, :] - img[:, :, :-1, :, :]
            dy = img[:, :, :, 1:, :] - img[:, :, :, :-1, :]
            dx = img[:, :, :, :, 1:] - img[:, :, :, :, :-1]

            # Pad to match original size
            dz = F.pad(dz, (0,0,0,0,0,1))
            dy = F.pad(dy, (0,0,0,1,0,0))
            dx = F.pad(dx, (0,1,0,0,0,0))

            return torch.cat([dx, dy, dz], dim=1)

        volumesLogic = slicer.modules.volumes.logic()
        inputImage = slicer.util.arrayFromVolume(inputVolumeNode)  # shape: [slices, height, width]
        inputImage2 = slicer.util.arrayFromVolume(inputVolumeNode2)  # shape: [slices, height, width]

        use_cuda = False
        if torch.cuda.is_available():
            try:
                total_vram = torch.cuda.get_device_properties(0).total_memory
                vram_gb = total_vram / (1024 ** 3)  # Convert bytes to GB
                if vram_gb >= 1.9:
                    use_cuda = True
                    self.outputTextBox.append(f"✅ CUDA available with {vram_gb:.1f} GB VRAM — using GPU.")
                else:
                    self.outputTextBox.append(f"⚠️ CUDA available but only {vram_gb:.1f} GB VRAM — using CPU fallback.")
            except Exception as e:
                self.outputTextBox.append(f"⚠️ Could not check VRAM: {e}. Using CPU.")
        device = torch.device("cuda" if use_cuda else "cpu")






        inputTensor = torch.tensor(inputImage).float().unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, D, H, W)
        self.outputTextBox.append(f"Input image size is: {inputTensor.shape[-3:]}")
        inputTensor2 = torch.tensor(inputImage2).float().unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, D, H, W)
        self.outputTextBox.append(f"Secondary Input image size is: {inputTensor2.shape[-3:]}")
        del inputImage
        del inputImage2
        mse = F.mse_loss(inputTensor, inputTensor2)
        self.outputTextBox.append(f"Mean Squared Error is: {mse.item():.6f}")
        del mse




        l1 = F.l1_loss(inputTensor, inputTensor2)
        self.outputTextBox.append(f"Mean Absolute Error is: {l1.item():.6f}")
        del l1



        edge = torch.nn.functional.l1_loss(gradient(inputTensor), gradient(inputTensor2))
        self.outputTextBox.append(f"Edge Loss is: {edge.item():.6f}")
        del edge

        def compute_psnr_dynamic(pred, target):
            mse = F.mse_loss(pred, target, reduction='mean').item()
            max_val = torch.max(target).item()
            if mse == 0:
                return float('inf')
            psnr = 10 * torch.log10(torch.tensor(max_val ** 2 / mse))
            return psnr.item()

        psnr = compute_psnr_dynamic(inputTensor, inputTensor2)
        self.outputTextBox.append(f"Peak SNR is: {psnr:.6f}")        



        if self.ssimnorm_cbox.checkState():

            # 2. --- SSIM-Specific Normalization ---
            # Define the clinical HU range you care about (e.g., Air to Dense Bone)
            hu_min = inputTensor.min()
            hu_max = inputTensor.max()
            hu_min2 = inputTensor2.min()
            hu_max2 = inputTensor2.max()
            
            if hu_min2 < hu_min:
                hu_min = hu_min2

            if hu_max2 > hu_max:
                hu_max = hu_max2

            # Clamp values to remove extreme outliers (like metal artifacts)
            # This prevents the normalization range from being skewed.
            inputTensor = torch.clamp(inputTensor, min=hu_min, max=hu_max)
            inputTensor2 = torch.clamp(inputTensor2, min=hu_min, max=hu_max)
        
            # Min-Max normalize to a strict [0, 1] scale
            inputTensor = (inputTensor - hu_min) / (hu_max - hu_min)
            inputTensor2 = (inputTensor2 - hu_min) / (hu_max - hu_min)

            # Calculate SSIM using the normalized tensors and a fixed data_range of 1.0
            # (Note: We bypass self.ssim_datarange.get() here since we forced the range to [0, 1])
            ssim = SSIMLoss(spatial_dims=3, data_range=1.0)(inputTensor, inputTensor2)

        else:
            inputTensor = torch.clamp(inputTensor, 0, float(self.ssimloss_range.text))
            inputTensor2 = torch.clamp(inputTensor2, 0, float(self.ssimloss_range.text))
            ssim = SSIMLoss(spatial_dims=3, data_range=float(self.ssimloss_range.text))(inputTensor, inputTensor2)



        self.outputTextBox.append(f"Structural Similarity Index is: {(1-ssim.item()):.6f}")
        self.outputTextBox.append(f"SSIM Loss (1-SSIM) is: {ssim.item():.6f}")
        del ssim



        self.outputTextBox.append("Calculation has been completed, out.")





        del inputTensor
        del inputTensor2

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        import gc
        gc.collect()





 

