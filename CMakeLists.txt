cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(PETDenoise)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/4burakfe/SlicerPETDenoise/")
set(EXTENSION_CATEGORY "Nuclear Medicine")
set(EXTENSION_CONTRIBUTORS "Burak Demir")
set(EXTENSION_DESCRIPTION "SlicerPETDenoise is a 3D Slicer extension that includes three modules intended for medical image research, particularly PET and CT image workflows. These modules assist in denoising, comparing, and visually fusing volumetric medical data. The tools are developed with research utility in mind and are not intended for clinical use.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/4burakfe/SlicerPETDenoise/ed0f24892556718d029948076f1e4e8fae46c261/SlicerPETDenoise.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/4burakfe/SlicerPETDenoise/blob/ed0f24892556718d029948076f1e4e8fae46c261/scr1.jpg?raw=true")
set(EXTENSION_DEPENDS "PyTorch") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(PETDenoise)
add_subdirectory(Easy_fusion)
add_subdirectory(VolumeComparator)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
