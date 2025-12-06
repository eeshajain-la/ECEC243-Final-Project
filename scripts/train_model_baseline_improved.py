modelName = "speechGRU_improved"

args = {}
args["outputDir"] = (
    "/content/drive/MyDrive/C243A_Final_Project/neural_seq_decoder/logs/speech_logs/"
    + modelName
)
args["datasetPath"] = (
    "/content/drive/MyDrive/C243A_Final_Project/neural_seq_decoder/data/ptDecoder_ctc"
)
args["seqLen"] = 150
args["maxTimeSeriesLen"] = 1200
args["batchSize"] = 64
args["lrStart"] = 0.02
args["lrEnd"] = 0.002  # ← CHANGED: Add LR decay
args["nUnits"] = 1024
args["nBatch"] = 15000  # ← CHANGED: Train longer
args["nLayers"] = 3  # ← CHANGED: Fewer layers
args["seed"] = 0
args["nClasses"] = 40
args["nInputFeatures"] = 256
args["dropout"] = 0.3  # ← CHANGED: Less dropout
args["whiteNoiseSD"] = 0.8
args["constantOffsetSD"] = 0.2
args["gaussianSmoothWidth"] = 2.0
args["strideLen"] = 4
args["kernelLen"] = 32
args["bidirectional"] = True  # ← CHANGED: Bidirectional
args["l2_decay"] = 1e-5

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
