import os
import slicer
from slicer.ScriptedLoadableModule import *
import qt
import ctk
import vtk
import time  
import numpy as np
import configparser

  

class PETDenoise(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Belenos - PET Denoise"
        parent.categories = ["Nuclear Medicine","Filtering.Denoising"]
        parent.dependencies = ["PyTorchUtils"]
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = """
        This module provides automated denoising of PET images withh UNET neural networks.
        """
        parent.acknowledgementText = """
        This file was developed by Burak Demir.
        """
        # **✅ Set the module icon**
        iconPath = os.path.join(os.path.dirname(__file__), "Resources\\logo.png")
        self.parent.icon = qt.QIcon(iconPath)  # Assign icon to the module
        self.parent = parent

class PETDenoiseWidget(ScriptedLoadableModuleWidget):

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

        
        self.architecture = qt.QComboBox()
        self.architecture.addItem("UNET")
        self.architecture.addItem("SwinUNETR")
        self.architecture.addItem("SwinUNETR+GCFN")
        formLayout.addRow("Select Model Architecture: ", self.architecture)



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
        self.inputVolumeSelector.setToolTip("Select the PET image for denoising.")
        formLayout.addRow("Input PET Volume: ", self.inputVolumeSelector)


        # 1️⃣ Input Volume Selector (PET Image)
        self.inputVolumeSelectorCT = slicer.qMRMLNodeComboBox()
        self.inputVolumeSelectorCT.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputVolumeSelectorCT.selectNodeUponCreation = True
        self.inputVolumeSelectorCT.addEnabled = False
        self.inputVolumeSelectorCT.removeEnabled = False
        self.inputVolumeSelectorCT.noneEnabled = False
        self.inputVolumeSelectorCT.showHidden = False
        self.inputVolumeSelectorCT.showChildNodeTypes = False
        self.inputVolumeSelectorCT.setMRMLScene(slicer.mrmlScene)
        self.inputVolumeSelectorCT.setToolTip("Select the CT image for denoising.")
        formLayout.addRow("Input CT Volume: ", self.inputVolumeSelectorCT)

        self.dualch_cbox = qt.QCheckBox()
        formLayout.addRow("Dual Volume Input (PET+CT): ", self.dualch_cbox)


        # 📁 Select Model Folder Button and Display
        self.modelFolderPathEdit = qt.QLineEdit()
        self.modelFolderPathEdit.readOnly = True
        formLayout.addRow("Model Folder:", self.modelFolderPathEdit)

        self.selectFolderButton = qt.QPushButton("Select Model Folder")
        formLayout.addRow(self.selectFolderButton)

        self.selectFolderButton.connect("clicked(bool)", self.selectModelFolder)


        self.modelInfoBox = qt.QTextEdit()
        self.modelInfoBox.setReadOnly(True)
        self.modelInfoBox.setToolTip("Displays model info from .txt sidecar file.")

        self.modelselector = qt.QComboBox()

        models = []
        for dirpath, dirnames, filenames in os.walk(os.path.dirname(self.modelselector.currentText)):
           for filename in filenames:
              if filename.endswith(".pth"):
                 models.append(filename)
        self.modelselector.addItems(models)
        self.loadModelFolderPath()

        formLayout.addRow("Select Model: ", self.modelselector)


        formLayout.addRow("Model Info:", self.modelInfoBox)
        self.modelInfoBox.setPlainText("Select a model to display info.")


        self.resample_voxel_size = qt.QLineEdit()
        self.resample_voxel_size.setText("[2,2,2]")
        formLayout.addRow("Voxel Spacing for Resample:", self.resample_voxel_size)

        self.denoise_block_size = qt.QLineEdit()
        self.denoise_block_size.setText("(64,64,64)")
        formLayout.addRow("Block Size for denoising:", self.denoise_block_size)







        self.labelUNET = qt.QLabel()
        self.labelUNET.setText("Settings For UNET:")
        formLayout.addRow(self.labelUNET)

        # Strides ComboBox
        self.strideComboBox = qt.QLineEdit()
        self.strideComboBox.setText("(2,2,2,2)")
        formLayout.addRow("Strides:", self.strideComboBox)

        # Channels Entry
        self.channels = qt.QLineEdit()
        self.channels.setText("(128,256,512,1024,2048)")
        formLayout.addRow("Channels:", self.channels)



        # Res Units
        self.resUnitSpinBox = qt.QSpinBox()
        self.resUnitSpinBox.setMinimum(1)
        self.resUnitSpinBox.setMaximum(10)
        self.resUnitSpinBox.setValue(2)
        formLayout.addRow("Residual Units:", self.resUnitSpinBox)

        # Res Units
        self.downkernelSpinBox = qt.QSpinBox()
        self.downkernelSpinBox.setMinimum(1)
        self.downkernelSpinBox.setMaximum(11)
        self.downkernelSpinBox.setValue(3)
        formLayout.addRow("Down Kernel:", self.downkernelSpinBox)

        self.upkernelSpinBox = qt.QSpinBox()
        self.upkernelSpinBox.setMinimum(1)
        self.upkernelSpinBox.setMaximum(11)
        self.upkernelSpinBox.setValue(3)
        formLayout.addRow("Up Kernel:", self.upkernelSpinBox)

        self.labelSWIN = qt.QLabel()
        self.labelSWIN.setText("Settings For SwinUNETR:")
        formLayout.addRow(self.labelSWIN)

        self.heads = qt.QLineEdit()
        self.heads.setText("(3,6,12,24)")
        formLayout.addRow("Number of Heads:", self.heads)

        self.depths = qt.QLineEdit()
        self.depths.setText("(2,2,2,2)")
        formLayout.addRow("Depths:", self.depths)

        # Channels Entry
        self.swinfeaturesize = qt.QSpinBox()
        self.swinfeaturesize.setMinimum(4)
        self.swinfeaturesize.setMaximum(256)
        self.swinfeaturesize.setValue(24)
        formLayout.addRow("Feature Size:", self.swinfeaturesize)

        # Channels Entry
        self.dropoutrate = qt.QLineEdit()
        self.dropoutrate.setText("0.0")
        formLayout.addRow("Dropout Path Rate:", self.dropoutrate)


        # 5️⃣ Output Log Text Box
        self.outputTextBox = qt.QTextEdit()
        self.outputTextBox.setReadOnly(True)
        self.outputTextBox.setToolTip("Displays processing logs and results.")
        formLayout.addRow("Processing Log:", self.outputTextBox)
        
        self.forceCPUcbox = qt.QCheckBox()
        formLayout.addRow("FORCE CPU: ", self.forceCPUcbox)
        
        # 6️⃣ Calculate Button
        self.calculateButton = qt.QPushButton("Denoise")
        self.calculateButton.toolTip = "Start the denoising process"
        self.calculateButton.enabled = True
        formLayout.addRow(self.calculateButton)

        # Connect Calculate button to function
        self.calculateButton.connect("clicked(bool)", self.onCalculateButtonClicked)
        
        
        # Info Text Box
        infoTextBox = qt.QTextEdit()
        infoTextBox.setReadOnly(True)  # Make the text box read-only
        infoTextBox.setPlainText(
            "This module provides automatic denoising with ML models on medical images.\n"
            "Select the source volume to denoise.\n"
            "This module is NOT a medical device. Research use only.\n"
            "Developed by: Burak Demir, MD, FEBNM \n"
            "For support, questions and feedback: 4burakfe@gmail.com\n"
            "Demir, B., Atalay, M., Yurtcu, H. et al. Denoising of PET with SwinUNETR neural networks: impact of tumor oriented loss function, denoising module for 3D slicer. Ann Nucl Med (2026). https://doi.org/10.1007/s12149-026-02166-4"
            "Version: v1.0"
        )
        infoTextBox.setToolTip("Module information and instructions.")  # Add a tooltip for additional help
        self.layout.addWidget(infoTextBox)
        self.modelselector.connect("currentIndexChanged(int)", self.updateModelInfoBox)
        self.updateModelInfoBox()

    def updateModelInfoBox(self):
        modelName = self.modelselector.currentText
        fullPath = self.modelFolderPathEdit.text+"/" + os.path.splitext(modelName)[0] + ".txt"
        if os.path.exists(fullPath):
            with open(fullPath, "r", encoding="utf-8") as f:
                info_lines = f.readlines()

            # Populate text box
            self.modelInfoBox.setPlainText("".join(info_lines))

            # Try to extract parameters
            for line in info_lines:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    key = str(key.strip().lower())
                    value = str(value.strip())
                    try:
                        if key == "strides":
                            self.strideComboBox.setText(value)
                        elif key == "dual_channel":
                            if value == "true":
                                self.dualch_cbox.setChecked(True)
                            else:
                                self.dualch_cbox.setChecked(False)
                        elif key == "channels":
                            self.channels.setText(value)
                        elif key == "res_units":
                            self.resUnitSpinBox.setValue(int(value))
                        elif key == "down_kernel":
                            self.downkernelSpinBox.setValue(int(value))
                        elif key == "up_kernel":
                            self.upkernelSpinBox.setValue(int(value))
                        elif key == "depths":
                            self.depths.setText(value)                        
                        elif key == "num_heads":
                            self.heads.setText(value)
                        elif key == "feature_size":
                            self.swinfeaturesize.setValue(int(value))
                        elif key == "do_rate":
                            self.dropoutrate.setText(value)
                        elif key == "block_size":
                            self.denoise_block_size.setText(value)
                        elif key == "voxel_spacing":
                            self.resample_voxel_size.setText(value)
                        elif key == "architecture":
                            if value == "SwinUNETR":
                                self.architecture.setCurrentIndex(1)
                            elif value == "UNET":
                                self.architecture.setCurrentIndex(0)
                            else:
                                self.architecture.setCurrentIndex(2)


                    except Exception as e:
                        print(f"⚠️ Error parsing parameter '{key}': {e}")
        else:
            self.modelInfoBox.setPlainText("ℹ️ No description file found.")



    def onCalculateButtonClicked(self):
        self.outputTextBox.clear()
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            self.outputTextBox.append("❌ PyTorch is not installed. Please install it before running this module.")
            return None

        self.outputTextBox.append("🚀 Starting AI denoising process...")
        try:
            from monai.networks.nets import UNet
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
        try:
            import einops
        except ImportError:
            msgBox = qt.QMessageBox()
            msgBox.setIcon(qt.QMessageBox.Warning)
            msgBox.setWindowTitle("einops is Not Installed")
            msgBox.setText("The einops library is required but not installed.\nI can install it. Would you like me to install it now? You may want to restart Slicer after installation.")
            msgBox.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
            response = msgBox.exec_()

            if response == qt.QMessageBox.Yes:
                import subprocess, sys
                try:
                    self.outputTextBox.append("📦 Installing MONAI, please wait...")
                    subprocess.check_call([sys.executable,  "-m","pip", "install", "einops==0.6.1"])

                    self.outputTextBox.append("✅ MONAI installed successfully.")
                    
                except Exception as e:
                    self.outputTextBox.append(f"❌ Failed to install MONAI: {e}")
                    return None
            else:
                self.outputTextBox.append("❌ MONAI installation canceled by user.")
                return None

        #Load AI Model
        model = self.load_model()
        if model is None:
            self.outputTextBox.append("❌ AI model loading failed. Stopping process.")
            return

        volumesLogic = slicer.modules.volumes.logic()
        outputVolumeNode = volumesLogic.CloneVolume(slicer.mrmlScene, inputVolumeNode, f"{inputVolumeNode.GetName()}_DN_{self.modelselector.currentText}")
        # Dual channel mode: preprocess CT
        ctImage = None

        def center_pad_to_shape(img_tensor, target_shape):
            _, _, d, h, w = img_tensor.shape
            td, th, tw = target_shape
            pd = td - d
            ph = th - h
            pw = tw - w
            pad = (
                pd // 2, pd - pd // 2,
                ph // 2, ph - ph // 2,
                pw // 2, pw - pw // 2

            )
            return F.pad(img_tensor, pad, mode="constant", value=0)

        def get_max_shape(shape1, shape2):
            """Return the element-wise maximum shape from two 3D volume shapes."""
            return tuple(max(s1, s2) for s1, s2 in zip(shape1, shape2))



        
        resample_parameters = {
            "InputVolume": inputVolumeNode.GetID(),  # ✅ Corrected to use GetID()
            "OutputVolume": outputVolumeNode.GetID(),  # ✅ Corrected to use GetID()
            "outputPixelSpacing": eval(self.resample_voxel_size.text),  # ✅ Ensured correct format
            "interpolationType": "linear"
        }
        resampleSuccess = slicer.cli.runSync(slicer.modules.resamplescalarvolume, None, resample_parameters)
        inputImage = slicer.util.arrayFromVolume(outputVolumeNode)  # shape: [slices, height, width]



        inputTensor = torch.tensor(inputImage).float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)
        self.outputTextBox.append(f"Input image size is: {inputTensor.shape[-3:]}")
        current_shape = inputTensor.shape[-3:]
        target_shape = inputImage.shape

        if self.dualch_cbox.checkState():
            inputVolumeNodeCT = self.inputVolumeSelectorCT.currentNode()
            if inputVolumeNodeCT is None:
                self.outputTextBox.append("❌ Dual channel selected, but CT volume not provided.")
                return None

            self.outputTextBox.append("📐 Resampling CT volume to match PET...")
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ResampledCT")
            resampledCTNode = slicer.mrmlScene.GetFirstNodeByName("ResampledCT")


            # Set parameters explicitly
            parameters = {
                "InputVolume": inputVolumeNodeCT.GetID(),
                "OutputVolume": resampledCTNode.GetID(),
                "interpolationType": "linear",
                "outputPixelSpacing": outputVolumeNode.GetSpacing()
            }
            resampleSuccess = slicer.cli.runSync(slicer.modules.resamplescalarvolume, None, parameters)

            if(resampleSuccess):
                self.outputTextBox.append("✅ Resampling completed successfully.")
            else:
                self.outputTextBox.append("❌ Resampling failed.")
            ctImage = slicer.util.arrayFromVolume(resampledCTNode)
            ctImage = np.clip(ctImage, -1000, 1000)
            ctImage = (ctImage + 1000) * 0.005  # Normalize to 0–10
            pet_shape = inputImage.shape  # e.g., (128, 192, 192)
            ct_shape = ctImage.shape    # e.g., (126, 190, 190)
            target_shape = get_max_shape(pet_shape, ct_shape)

        # Apply zero padding


        self.outputTextBox.append(f"Input image size is resized to: {target_shape}")
        self.outputTextBox.append("Don't worry, will crop again to original size.")
        start_time = time.time()  # ⏱️ Start timer
        from monai.inferers import sliding_window_inference

        if self.dualch_cbox.checkState():
            ctTensor = torch.tensor(ctImage).float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)
            self.outputTextBox.append(f"CT size: {ctTensor.shape[-3:]}")
            slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName("ResampledCT"))
            paddedCT = center_pad_to_shape(ctTensor, target_shape)
            paddedPET = center_pad_to_shape(inputTensor, target_shape)
            self.outputTextBox.append(f"CT size after padding to match PET: {paddedCT.shape[-3:]}")

            combinedTensor=torch.cat([paddedPET, paddedCT], dim=1)
            with torch.no_grad():
                device = next(model.parameters()).device
                combinedTensor = combinedTensor.to(device)
                predicted_noise = sliding_window_inference(
                    inputs=combinedTensor,
                    roi_size=eval(self.denoise_block_size.text),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.25,
                    mode="gaussian"
                )
            paddedPET = paddedPET.to(device)
            outputTensor = paddedPET - predicted_noise
            self.outputTextBox.append(f"Denoising is done! Output image size: {outputTensor.shape[-3:]}")
            outputTensor = outputTensor[:, :, :current_shape[0], :current_shape[1], :current_shape[2]]
            elapsed = time.time() - start_time  # ⏱️ End timer
            self.outputTextBox.append(f"⏱️ Inference time: {elapsed:.2f} seconds")  # ✅ Log time
            outputArray = outputTensor.squeeze().cpu().numpy()
            outputArray = np.clip(outputArray, 0, None)  # remove negative values
            outputArray = outputArray.astype(inputImage.dtype)  # match original type
            self.outputTextBox.append(f"Output image size is resized to: {outputArray.shape}")

            slicer.util.updateVolumeFromArray(outputVolumeNode, outputArray)
            self.outputTextBox.append(f"Output image name is named as: {outputVolumeNode.GetName()}")
            del model
            del paddedPET
            del paddedCT
            del outputArray
            del outputTensor
            del ctTensor
            del inputTensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            import gc
            gc.collect()

        else:
            with torch.no_grad():
                device = next(model.parameters()).device
                inputTensor = inputTensor.to(device)
                predicted_noise = sliding_window_inference(
                    inputs=inputTensor,
                    roi_size=(64, 64, 64),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.25,
                    mode="gaussian"
                )
            outputTensor = inputTensor - predicted_noise
            self.outputTextBox.append(f"Denoising is done! Output image size: {outputTensor.shape[-3:]}")
            outputTensor = outputTensor[:, :, :current_shape[0], :current_shape[1], :current_shape[2]]
            elapsed = time.time() - start_time  # ⏱️ End timer
            self.outputTextBox.append(f"⏱️ Inference time: {elapsed:.2f} seconds")  # ✅ Log time
            outputArray = outputTensor.squeeze().cpu().numpy()
            outputArray = np.clip(outputArray, 0, None)  # remove negative values
            outputArray = outputArray.astype(inputImage.dtype)  # match original type
            self.outputTextBox.append(f"Output image size is resized to: {outputArray.shape}")

            slicer.util.updateVolumeFromArray(outputVolumeNode, outputArray)
            self.outputTextBox.append(f"Output image name is named as: {outputVolumeNode.GetName()}")
            del model
            del outputArray
            del outputTensor
            del inputTensor

            for var in ["inputImage", "ctImage", "resampledCTNode", "predicted_noise", "combinedTensor",
                        "current_shape", "target_shape", "pet_shape", "ct_shape", "start_time", "elapsed"]:
                if var in locals():
                    del locals()[var]

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()









    def load_model(self):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            self.outputTextBox.append("❌ PyTorch is not installed. Please install it before running this module.")
            return None

        try:
            from monai.networks.nets import UNet,SwinUNETR
        except ImportError:
            self.outputTextBox.append("❌ MONAI is not installed. Please install it before running this module.")
            return None
        """
        Load the trained PyTorch model from model.pth.
        """
        model_path = self.modelFolderPathEdit.text + "/" + self.modelselector.currentText
          
        if not os.path.exists(model_path):
            self.outputTextBox.append(f"❌ Model file not found: {model_path}")
            return None

        self.outputTextBox.append(f"📌 Loading AI model from: {model_path}")
        
        
        # Parse user inputs
        strides = eval(self.strideComboBox.text)
        res_units = self.resUnitSpinBox.value
        num_heads = eval(self.heads.text)
        depths = eval(self.depths.text)
        do_rate= float(self.dropoutrate.text)
        channels = eval(self.channels.text)
        from monai import __version__ as monai_version
        from packaging import version
        class DenoiseUNet(nn.Module):
            def __init__(self, in_channels=1, out_channels=1,  channels=(32, 64, 128, 256,512), num_res_units=2,strides=(2, 2, 2, 2),kernel_size=3,up_kernel_size=3):
                super(DenoiseUNet, self).__init__()
        
                # Use MONAI's 3D U-Net as the denoising backbone
                self.unet = UNet(
                    strides=strides,
                    num_res_units=num_res_units,
                    kernel_size=kernel_size,
                    up_kernel_size=up_kernel_size,           
                    spatial_dims=3,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    channels=channels
                )

            def forward(self, x):
                return self.unet(x)  # ✅ Predict noise only

        class SwinDenoiser(nn.Module):
            def __init__(self,  in_channels=1, out_channels=1, feature_size=48,heads=(6,12,24,48),depths=(2,3,3,2),do_rate=0.1):
                super(SwinDenoiser, self).__init__()
                self.model = SwinUNETR(
                    num_heads = heads,
                    use_v2=True,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    feature_size=feature_size,
                    depths = depths,
                    dropout_path_rate=do_rate,
                    **({"img_size": (64, 64, 64)} if version.parse(monai_version) < version.parse("1.5") else {}),
                    use_checkpoint=True
                )

            def forward(self, x):
                return self.model(x)
        class GCFN(nn.Module):
            def __init__(self, dim):
                super(GCFN, self).__init__()
                self.norm = nn.LayerNorm(dim)

                self.fc1 = nn.Linear(dim, dim)
                self.fc2 = nn.Linear(dim, dim)
                self.fc0 = nn.Linear(dim, dim)

                self.conv1 = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim)
                self.conv2 = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim)

            def forward(self, x):
                # x: (B, C, D, H, W)
                B, C, D, H, W = x.shape
                x_ = x.permute(0, 2, 3, 4, 1).contiguous().view(B * D * H * W, C)

                x1 = self.fc1(self.norm(x_))
                x2 = self.fc2(self.norm(x_))

                x1 = x1.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
                x2 = x2.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)

                gate = F.gelu(self.conv1(x1)) * self.conv2(x2)

                gate = gate.permute(0, 2, 3, 4, 1).contiguous().view(B * D * H * W, C)
                out = self.fc0(gate).view(B, D, H, W, C).permute(0, 4, 1, 2, 3)

                return out + x
        class SwinGCFN(nn.Module):
            def __init__(self,  in_channels=1, out_channels=1, feature_size=48,heads=(6,12,24,48),depths=(2,3,3,2),do_rate=0.1):
                super(SwinGCFN, self).__init__()
                self.model = SwinUNETR(
                    num_heads = heads,
                    use_v2=True,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    feature_size=feature_size,
                    depths = depths,
                    dropout_path_rate=do_rate,
                    **({"img_size": (64, 64, 64)} if version.parse(monai_version) < version.parse("1.5") else {}),
                    use_checkpoint=True
                )
                self.gcfn = GCFN(dim=out_channels)

            def forward(self, x):
                out = self.model(x)      # (B, 1, D, H, W)
                out = self.gcfn(out)     # Apply GCFN enhancement
                return out


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


        if self.forceCPUcbox.checkState():
            use_cuda = False
            self.outputTextBox.append(f"⚠️ Force CPU is selected. Using CPU.")


        # Instantiate model on correct device
        device = torch.device("cuda" if use_cuda else "cpu")


        if self.architecture.currentText=="UNET":
            if self.dualch_cbox.checkState()==False:
                model = DenoiseUNet(in_channels = 1, channels=channels, num_res_units=res_units,strides=strides,kernel_size=self.downkernelSpinBox.value,up_kernel_size=self.upkernelSpinBox.value).to(device)
            else:
                model = DenoiseUNet(in_channels = 2, channels=channels, num_res_units=res_units,strides=strides,kernel_size=self.downkernelSpinBox.value,up_kernel_size=self.upkernelSpinBox.value).to(device)
        elif self.architecture.currentText=="SwinUNETR":
            if self.dualch_cbox.checkState()==False:
                model = SwinDenoiser(in_channels = 1, feature_size=self.swinfeaturesize.value ,heads=num_heads,depths=depths,do_rate=do_rate).to(device)
            else:
                model = SwinDenoiser(in_channels = 2, feature_size=self.swinfeaturesize.value ,heads=num_heads,depths=depths,do_rate=do_rate).to(device)
        elif self.architecture.currentText=="SwinUNETR+GCFN":
            if self.dualch_cbox.checkState()==False:
                model = SwinGCFN(in_channels = 1, feature_size=self.swinfeaturesize.value ,heads=num_heads,depths=depths,do_rate=do_rate).to(device)
            else:
                model = SwinGCFN(in_channels = 2, feature_size=self.swinfeaturesize.value ,heads=num_heads,depths=depths,do_rate=do_rate).to(device)


        model.load_state_dict(torch.load(model_path, device))
        model.eval()  # Set to evaluation mode (no gradients needed)

        self.outputTextBox.append("✅ AI model loaded successfully!")
        return model
        
    def selectModelFolder(self):
        folder = qt.QFileDialog.getExistingDirectory()
        if folder:
            self.modelFolderPathEdit.setText(folder)
            self.saveModelFolderPath(folder)
            self.refreshModelSelector(folder)

    def saveModelFolderPath(self, folder):
        ini_path = os.path.join(os.path.dirname(__file__), "model_config.ini")
        config = configparser.ConfigParser()
        config["ModelFolder"] = {"path": folder}
        with open(ini_path, "w") as configfile:
            config.write(configfile)

    def loadModelFolderPath(self):
        ini_path = os.path.join(os.path.dirname(__file__), "model_config.ini")
        config = configparser.ConfigParser()
        if os.path.exists(ini_path):
            config.read(ini_path)
            folder = config.get("ModelFolder", "path", fallback="")
            if os.path.exists(folder):
                self.modelFolderPathEdit.setText(folder)
                self.refreshModelSelector(folder)

    def refreshModelSelector(self, folder):
        self.modelselector.clear()
        models = [f for f in os.listdir(folder) if f.endswith(".pth")]
        self.modelselector.addItems(models)