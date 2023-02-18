'''
A collection of all the models, for the convenience of internal calling
'''


# Lower level encoder decoders
try:
    from .voice_encoders import V2F1DCNN, TransEncoder
except:
    from voice_encoders import V2F1DCNN, TransEncoder

try:
    from .face_decoders import V2FDecoder, AttnV2FDecoder, MRDecoder, \
        FaceGanDecoder, FaceGanDecoder_v2
except:
    from face_decoders import V2FDecoder, AttnV2FDecoder, MRDecoder, \
        FaceGanDecoder, FaceGanDecoder_v2

try:
    from .attention import Attention
except:
    from .attention import Attention


class ModelCollection():
    def __init__(self):
        self.V2F1DCNN = V2F1DCNN
        self.V2FDecoder = V2FDecoder
        self.AttnV2FDecoder = AttnV2FDecoder
        self.TransEncoder = TransEncoder
        self.MRDecoder = MRDecoder
        self.FaceGanDecoder = FaceGanDecoder
        self.FaceGanDecoder_v2 = FaceGanDecoder_v2
        # self.PganDecoder = PganDecoder
        self.Attention = Attention


model_collection = ModelCollection()
