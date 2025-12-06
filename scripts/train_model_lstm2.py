modelName = "speechLSTM"

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
args["batchSize"] = 64  # 64 is a bit high for LSTM, 32 safer to avoid OOM
args["lrStart"] = 0.02  # 0.02 is too fast for LSTM, could cause exploding gradients
args["lrEnd"] = 0.02  # 0.02 originally
args["nUnits"] = 1024
args["nBatch"] = 3000  # 3000
args["nLayers"] = 5  # 5, Use 3 layers unless layer norm or residuals added
args["seed"] = 0
args["nClasses"] = 40
args["nInputFeatures"] = 256
args["dropout"] = 0.4
args["whiteNoiseSD"] = 0.8  # Was 0.8
args["constantOffsetSD"] = 0.2  # Was 0.2
args["gaussianSmoothWidth"] = 2.0
args["strideLen"] = 4
args["kernelLen"] = 32  # 32 originally
args["bidirectional"] = False
args["l2_decay"] = 1e-5

from neural_decoder.neural_decoder_trainer_lstm import trainModel

trainModel(args)
