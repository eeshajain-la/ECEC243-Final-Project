import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model_layernorm_residual_two_block import GRUDecoder
from .dataset import SpeechDataset


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def time_mask(X, max_width=20, num_masks=2):
    # X: [B, T, C]
    B, T, C = X.shape
    for _ in range(num_masks):
        width = torch.randint(0, max_width + 1, (1,)).item()
        if width == 0 or width >= T:
            continue
        start = torch.randint(0, T - width, (1,)).item()
        X[:, start:start+width, :] = 0.0
    return X

def trainModel(args):
    """
    Train GRU baseline model.

    Supports:
      - clean run (default)
      - resume if args['resume'] = True and optional args['resumeModelDir']

    For resume, it expects:
      - <resumeModelDir>/modelWeights      (state_dict saved by this script)
      - <resumeModelDir>/trainingStats     (dict with 'testLoss' and 'testCER')
    """

    # Handle both DictConfig (Hydra) and plain dict
    args_dict_for_save = dict(args)

    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    # Resume flags (optional)
    resume = bool(args.get("resume", False))
    resumeDir = args.get("resumeModelDir", args["outputDir"])

    # Save current args for reference
    with open(os.path.join(args["outputDir"], "args"), "wb") as file:
        pickle.dump(args_dict_for_save, file)

    # --- Data loaders ---
    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    # --- Model ---
    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    # If resuming, load model weights (if present)
    if resume:
        weight_path = os.path.join(resumeDir, "modelWeights")
        if os.path.exists(weight_path):
            print(f"[Resume] Loading model weights from: {weight_path}")
            state_dict = torch.load(weight_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            print(f"[Resume] modelWeights not found in {resumeDir}; starting from scratch")

    # --- Loss / Optimizer / Scheduler ---
    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nBatch"],
    )

    # --- Training bookkeeping ---
    testLoss = []
    testCER = []
    start_batch = 0

    # If resuming, load previous training stats to continue indexing
    if resume:
        stats_path = os.path.join(resumeDir, "trainingStats")
        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                tStats = pickle.load(f)
            # these were saved as numpy arrays; convert to lists so we can append
            prevLoss = list(tStats.get("testLoss", []))
            prevCER = list(tStats.get("testCER", []))
            testLoss.extend(prevLoss)
            testCER.extend(prevCER)

            if len(testCER) > 0:
                # each entry corresponds to an eval every 100 batches
                start_batch = len(testCER) * 100
                print(f"[Resume] Found {len(testCER)} eval points; resuming from batch {start_batch}")
            else:
                print("[Resume] trainingStats has no CER entries; starting from batch 0")
        else:
            print(f"[Resume] trainingStats not found in {resumeDir}; starting from batch 0")

    # --- Train loop ---
    startTime = time.time()
    train_iter = iter(trainLoader)

    for batch in range(start_batch, args["nBatch"]):
        model.train()

        try:
            X, y, X_len, y_len, dayIdx = next(train_iter)
        except StopIteration:
            # restart the epoch when we exhaust the loader
            train_iter = iter(trainLoader)
            X, y, X_len, y_len, dayIdx = next(train_iter)

        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        # Forward + loss
        pred = model.forward(X, dayIdx)
        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Eval every 100 batches
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0

                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            # Save best model (same behavior as before)
            if len(testCER) == 0 or cer < np.min(testCER):
                torch.save(model.state_dict(), os.path.join(args["outputDir"], "modelWeights"))

            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {
                "testLoss": np.array(testLoss),
                "testCER": np.array(testCER),
            }

            with open(os.path.join(args["outputDir"], "trainingStats"), "wb") as file:
                pickle.dump(tStats, file)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()