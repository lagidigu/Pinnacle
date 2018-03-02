from Inference.Control_To_Meditation_Classification import ShallowConvNet as shallow_conv


class Inference:

    def perform_inference(self, input_sample):
        print(shallow_conv.inference(input_sample))
