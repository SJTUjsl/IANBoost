import logging
import os
from typing import Annotated, Optional

import vtk
import numpy as np

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
import slicer.logic

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode
try:
    from PIL import Image
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import UNet
    from monai.transforms import (
        Activations,
        EnsureChannelFirstd,
        AsDiscrete,
        Compose,
        LoadImaged,
        EnsureTyped,
        ScaleIntensityRanged,
    )
    import onnxruntime as ort
except:
    try:
        import PyTorchUtils
    except ModuleNotFoundError as e:
        pass
    else:
        torchLogic = PyTorchUtils.PyTorchUtilsLogic()
    try:
        torch = torchLogic.installTorch(askConfirmation=True)
    except:
        pass   
    slicer.util.pip_install('monai torch torchvision torchaudio')
    slicer.util.pip_install('nibabel')
    slicer.util.pip_install('onnxruntime')
    from PIL import Image
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import UNet
    from monai.transforms import (
        Activations,
        EnsureChannelFirstd,
        AsDiscrete,
        Compose,
        LoadImaged,
        EnsureTyped,
        ScaleIntensityRanged,
    )
    import onnxruntime as ort

#
# IANBoost
#


class IANBoost(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("IANBoost")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#IANBoost">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)



#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # IANBoost1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="IANBoost",
        sampleName="IANBoost1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "IANBoost1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="IANBoost1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="IANBoost1",
    )

    # IANBoost2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="IANBoost",
        sampleName="IANBoost2",
        thumbnailFileName=os.path.join(iconsPath, "IANBoost2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="IANBoost2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="IANBoost2",
    )


#
# IANBoostParameterNode
#


@parameterNodeWrapper
class IANBoostParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    outputSegmentation: vtkMRMLSegmentationNode


#
# IANBoostWidget
#


class IANBoostWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/IANBoost.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = IANBoostLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.applyButton.enabled = True

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("inputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID('inputVolume', firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode: Optional[IANBoostParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        # if self._parameterNode:
        #     self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        #     self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        # self._parameterNode = inputParameterNode
        # if self._parameterNode:
        #     # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
        #     # ui element that needs connection.
        #     self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
        #     self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            # self._checkCanApply()

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self.updateGUIFromParameterNode()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.outputSegmentation:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """This method is called when the parameter node is changed."""
        # Update each GUI element from parameter node
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True
        # Update node selectors 
        self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("inputVolume"))
        self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("outputSegmentation"))

        if self._parameterNode.GetNodeReference("inputVolume") and self._parameterNode.GetNodeReference("outputSegmentation"):  # Both input and output are selected
            self.ui.applyButton.enabled = True

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False
    
    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """This method is called when the user makes any change in the GUI."""
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        # developer area
        self._parameterNode.SetNodeReferenceID("inputVolume", self.ui.inputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("outputSegmentation", self.ui.outputSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

        slicer.util.setSliceViewerLayers(background=self.ui.inputSelector.currentNodeID)
        slicer.util.resetSliceViews()
        self.ui.applyButton.enabled = True

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            # self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
            #                    self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)
            # if self.ui.invertedOutputSelector.currentNode():
            #     # If additional output volume is selected then result with inverted threshold is written there
            #     self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
            #                        self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)
            print("onApplyButton")
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode())

#
# IANBoostLogic
#


class IANBoostLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode) -> None:
        """Set default parameters to parameter node."""
        if not parameterNode.GetParameter("inputVolume"):
            parameterNode.SetParameter("inputVolume", "")

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputSeg: vtkMRMLSegmentationNode,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputSeg: segmentation result
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputSeg:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        print("Processing started")

        image = slicer.util.arrayFromVolume(inputVolume)
        mandible_seg = self.infer_mandible(image)
        print("predicted type: ", type(mandible_seg))
        mandible_seg = np.array(mandible_seg)
        slicer.util.updateSegmentBinaryLabelmapFromArray(mandible_seg, outputSeg, segmentId="mandible", referenceVolumeNode=inputVolume)

        stopTime = time.time()
        print(f"Processing completed in {stopTime-startTime:.2f} seconds")



    def infer_mandible(self, image):
        model_path = os.path.join(os.path.dirname(__file__), "Resources/mandible.onnx")

        # Define transforms for image and segmentation
        transforms = Compose(
            [
                # LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"], strict_check=False),  # Ensure the channel dimension is first
                EnsureTyped(keys=["image"]),
                ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1500, b_min=0.0, b_max=1.0, clip=True),
            ]
        )
        data = {"image": image}
        transformed_data = transforms(data)
        # Add batch dimension
        img = transformed_data["image"] # (H, W, D)
        # meta_data = transformed_data["image_meta_dict"]

        post_trans = Compose([Activations(softmax=True), AsDiscrete(threshold=0.5, argmax=True)])
        pred = sliding_window_infer(img, model_path, window_size=(64, 64, 64), overlap=0.25)
        pred = post_trans(pred).squeeze()
        print(pred.shape)
        return pred
    
def sliding_window_infer(image, model_path, window_size=(64, 64, 64), overlap=0.5):
    """
    Perform sliding window inference on a large image using an ONNX model.

    :param image: The input image as a numpy array (shape [H, W, D]).
    :param model_path: Path to the ONNX model file.
    :param window_size: The size of the sliding window (default is (64, 64, 64)).
    :param overlap: The fraction of overlap between windows (default is 0.5).
    :return: The combined prediction as a numpy array.
    """
    # Load the ONNX model
    session = ort.InferenceSession(model_path)

    # Calculate step size based on overlap
    step_size = [int(window_size[i] * (1 - overlap)) for i in range(3)]

    # Determine model output shape with a dummy run
    output_channels = 2

    # Prepare output array
    output_shape = (output_channels,) + image.shape
    combined_output = np.zeros(output_shape, dtype=np.float32)
    counts = np.zeros(image.shape, dtype=np.float32)

    # Sliding window inference
    for z in range(0, image.shape[2] - window_size[2] + 1, step_size[2]):
        for y in range(0, image.shape[1] - window_size[1] + 1, step_size[1]):
            for x in range(0, image.shape[0] - window_size[0] + 1, step_size[0]):
                window = image[x:x + window_size[0], y:y + window_size[1], z:z + window_size[2]]
                input_data = np.expand_dims(window, axis=(0, 1))  # Shape: [1, 1, D, H, W]

                output = session.run(None, {session.get_inputs()[0].name: input_data})[0]
                combined_output[:, x:x + window_size[0], y:y + window_size[1], z:z + window_size[2]] += output.squeeze()
                counts[x:x + window_size[0], y:y + window_size[1], z:z + window_size[2]] += 1

    # Normalize by counts
    combined_output /= np.maximum(counts, 1)

    return combined_output
#
# IANBoostTest
#


class IANBoostTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_IANBoost1()

    def test_IANBoost1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("IANBoost1")
        print(type(inputVolume))
        self.delayDisplay("Loaded test data set")
        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

        # Test the module logic

        logic = IANBoostLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, True)
        print("Done")
        self.delayDisplay("Test passed")

