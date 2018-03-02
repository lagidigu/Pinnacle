#import Recording.OpenBCI_Python.open_bci_v3 as open_bci
import numpy as np
import Inference.Master.inference as inference

class InputConversion:

    def __init__(self):
        self.input = []
        self.batch_size = 128
        self.batch_position = 0
        self.current_crop = np.zeros((6, 500), dtype = np.float32) #TODO: Invert?
        self.crop_position = 0

    def handle_sample(self, sample): #TODO: Make a new crop every tick, and treat the batch
      print(sample.id, sample.channel_data)

    def parse_to_input_buffer(self, sample):
        if (self.batch_position < self.batch_size):
            if (self.crop_position < self.current_crop.shape[0]):
                self.current_crop[self.crop_position] = self.get_shortened_channel(1, 7, sample)
                self.crop_position += 1
            else:
                self.crop_position = 0
                self.batch_position += 1
        else:
            self.send_input()
            self.batch_position = 0

    def get_shortened_channel(self, start, end, sample):
        return sample.channel_data[start:end]

    def send_input(self):
        return None
        #TODO: Send the current crop


#TODO: Move everything to Main
Input = InputConversion()
Inference = inference.Inference()
for i in range (0, 128):
    Input.input.append(Input.current_crop)
print(Inference.perform_inference(Input.input))

#board = open_bci.OpenBCIBoard()
#board.print_register_settings()
#board.start_streaming(handle_sample)

