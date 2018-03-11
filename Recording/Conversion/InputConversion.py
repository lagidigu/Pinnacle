import Recording.OpenBCI_Python.open_bci_v3 as open_bci
import numpy as np
import Inference.Master.inference as inference

class InputConversion:
    def __init__(self, inference):
        self.inference = inference                                                                                      # Reference to the master inference class
        self.input = []                                                                                                 # Input placeholder that will be used for inference.
        self.time_step_position = 0                                                                                     # Position in time on the sample
        self.current_crop = np.zeros((6, 500), dtype=np.float32)                                                        # Current crop being parsed into input
        self.batch_size = 128                                                                                           # Size of the input
        self.batch_position = 0

    def handle_sample(self, sample):  # TODO: Make a new crop every tick, and treat the batch
        print(sample.id, sample.channel_data)

    def parse_to_input_buffer(self, sample):
        if (self.batch_position < self.batch_size):                                                                     # Go through batches
            if (self.time_step_position < self.current_crop.shape[1]):                                                  # Go through time steps (500)
                self.current_crop[self.time_step_position] = self.get_shortened_channel(1, 7,
                                                                                        sample)                         # Assign shortened channel (2-7) to current batch
                self.time_step_position += 1                                                                            # Increase the time step position
            else:
                self.time_step_position = 0                                                                             # If crop is full, move to next crop of batch
                self.batch_position += 1
        else:
            self.send_input()                                                                                           # If batch is full, send the input and reset values for next input
            self.batch_position = 0

    def get_shortened_channel(self, start, end, sample):
        return sample.channel_data[start:end]

    def send_input(self):
        self.inference.perform_inference(self.current_crop)





