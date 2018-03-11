import Recording.OpenBCI_Python.open_bci_v3 as open_bci
import Recording.Conversion.InputConversion as input_conversion

import Inference.Master.inference as inference

_inference = inference.Inference()
_input_conversion = input_conversion.InputConversion(_inference)
open_bci.start(_input_conversion.handle_sample)

#Calling inferences works now
#TODO: Fix the the byte skipping problem
#TODO: Include some way to measure impedance (Optional. You can always calibrate impedance with the GUI)
#TODO: Test whether or not meditation is detected.