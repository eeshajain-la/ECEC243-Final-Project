import torch
from torch import nn

from .augmentations import GaussianSmoothing


# Decoder that processes neural time-series using day-specific transforms,
# smoothing, unfolding (sliding windows), and an LSTM to produce class logits.
class LSTMDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(LSTMDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        # Nonlinearity applied after day-specific transform
        self.inputLayerNonlinearity = torch.nn.Softsign()
        # Unfold operator to extract sliding windows from the neural sequence
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        # Temporal Gaussian smoothing to reduce high-frequency noise
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        # Day-specific linear transformation parameters
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        # Initialize each day's weight matrix as identity
        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # Main LSTM (sequence model)
        self.lstm_decoder = nn.LSTM(
            (neural_dim) * self.kernelLen,  # Each LSTM step sees a flattened window
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        # Initialize LSTM weights
        for name, param in self.lstm_decoder.named_parameters():
            if "weight_hh" in name:  # recurrent weights
                nn.init.orthogonal_(param)
            if "weight_ih" in name:  # input weights
                nn.init.xavier_uniform_(param)

        # Build one small input layer per day (identity-initialized)
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        # Initialize each day’s input layer as identity + small noise
        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # lstm outputs
        # Final projection to class logits (+1 for CTC blank token)
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(
                hidden_dim, n_classes + 1
            )  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        # Apply Gaussian smoothing (requires channel-first layout)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        # Select the appropriate day-specific transform
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        # Convert sequence into sliding windows using Unfold
        # Output shape: (B, new_T, neural_dim * kernelLen)
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # Determine number of directions from bidirectionality
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
            # LSTM needs cell state too
            c0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
            # LSTM needs cell state too
            c0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        # Run the LSTM over the windowed sequence
        hid, _ = self.lstm_decoder(stridedInputs, (h0.detach(), c0.detach()))

        # Project each timestep's hidden state into class logits
        seq_out = self.fc_decoder_out(hid)
        return seq_out
