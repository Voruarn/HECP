import os
import json
from tqdm import tqdm

import numpy as np
import fire

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

def pad_to_max_len_label(batch_data):
    """
    Pad a batch of variable-length sequences to the max length in the batch.
    
    Args:
        batch_data (List[List[List[float]]]): Input data of shape [B, ?, 2]
    
    Returns:
        torch.Tensor: Padded tensor of shape [B, max_len, 2]
    """
    # Calculate the maximum length in the batch
    max_len = max(len(seq) for seq in batch_data)
    
    # Create a padded tensor filled with zeros
    padded_tensor = torch.zeros((len(batch_data), max_len), dtype=torch.float32)
    
    # Fill the padded tensor with the values from batch_data
    for i, seq in enumerate(batch_data):
        seq_len = len(seq)
        padded_tensor[i, :seq_len] = torch.tensor(seq, dtype=torch.float32)
    
    return padded_tensor

def pad_to_max_len_logit(batch_data):
    """
    Pad a batch of variable-length sequences to the max length in the batch.
    
    Args:
        batch_data (List[List[List[float]]]): Input data of shape [B, ?, 2]
    
    Returns:
        torch.Tensor: Padded tensor of shape [B, max_len, 2]
    """
    # Calculate the maximum length in the batch
    max_len = max(len(seq) for seq in batch_data)
    
    # Create a padded tensor filled with zeros
    padded_tensor = torch.zeros((len(batch_data), max_len, 92553), dtype=torch.float32)
    
    # Fill the padded tensor with the values from batch_data
    for i, seq in enumerate(batch_data):
        seq_len = len(seq)
        padded_tensor[i, :seq_len, :] = torch.tensor(seq, dtype=torch.float32)
    
    return padded_tensor

class ModelWithTemperature(nn.Module):
    def __init__(self):
        super(ModelWithTemperature, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, test_loader):
        res = {
            "temperature": None,
            "val": {
                "before_temperature": {
                    "nll": None,
                    "ece": None
                },
                "before_temperature_positive": {
                    "nll": None,
                    "ece": None
                },
                "before_temperature_negative": {
                    "nll": None,
                    "ece": None
                },
                "after_temperature": {
                    "nll": None,
                    "ece": None
                },
                "after_temperature_positive": {
                    "nll": None,
                    "ece": None
                },
                "after_temperature_negative": {
                    "nll": None,
                    "ece": None
                },
            },
            "test": {
                "before_temperature": {
                    "nll": None,
                    "ece": None
                },
                "before_temperature_positive": {
                    "nll": None,
                    "ece": None
                },
                "before_temperature_negative": {
                    "nll": None,
                    "ece": None
                },
                "after_temperature": {
                    "nll": None,
                    "ece": None
                },
                "after_temperature_positive": {
                    "nll": None,
                    "ece": None
                },
                "after_temperature_negative": {
                    "nll": None,
                    "ece": None,
                },
            }
        }
        # self.cuda()
        nll_criterion = nn.CrossEntropyLoss() # .cuda()
        ece_criterion = _ECELoss() # .cuda()

        # First: collect all the logits and labels for the validation set
        val_logits_list = []
        val_labels_list = []
        with torch.no_grad():
            for logits, labels in valid_loader:
                val_logits_list.append(logits)
                val_labels_list.append(labels)
            val_logits = torch.cat(val_logits_list) # .cuda()
            val_labels = torch.cat(val_labels_list) # .cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(val_logits, val_labels).item()
        before_temperature_ece = ece_criterion(val_logits, val_labels).item()
        print('Before temperature(VAL) - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        res["val"]["before_temperature"]["nll"] = before_temperature_nll
        res["val"]["before_temperature"]["ece"] = before_temperature_ece

        # Calculate NLL and ECE before temperature scaling for val_labels = 1
        mask = val_labels == 1
        before_temperature_nll = nll_criterion(val_logits[mask], val_labels[mask]).item()
        before_temperature_ece = ece_criterion(val_logits[mask], val_labels[mask]).item()
        print('Before temperature(VAL) Positive - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        res["val"]["before_temperature_positive"]["nll"] = before_temperature_nll
        res["val"]["before_temperature_positive"]["ece"] = before_temperature_ece

        # Calculate NLL and ECE before temperature scaling for val_labels = 0
        mask = val_labels == 0
        before_temperature_nll = nll_criterion(val_logits[mask], val_labels[mask]).item()
        before_temperature_ece = ece_criterion(val_logits[mask], val_labels[mask]).item()
        print('Before temperature(VAL) Negative - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        res["val"]["before_temperature_negative"]["nll"] = before_temperature_nll
        res["val"]["before_temperature_negative"]["ece"] = before_temperature_ece

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(val_logits), val_labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(val_logits), val_labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        res["temperature"] = self.temperature.item()
        res["val"]["after_temperature"]["nll"] = after_temperature_nll
        res["val"]["after_temperature"]["ece"] = after_temperature_ece

        # Calculate NLL and ECE after temperature scaling for val_labels = 1
        mask = val_labels == 1
        after_temperature_nll = nll_criterion(self.temperature_scale(val_logits[mask]), val_labels[mask]).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(val_logits[mask]), val_labels[mask]).item()
        print('After temperature(VAL) Positive - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        res["val"]["after_temperature_positive"]["nll"] = after_temperature_nll
        res["val"]["after_temperature_positive"]["ece"] = after_temperature_ece

        # Calculate NLL and ECE after temperature scaling for val_labels = 0
        mask = val_labels == 0
        after_temperature_nll = nll_criterion(self.temperature_scale(val_logits[mask]), val_labels[mask]).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(val_logits[mask]), val_labels[mask]).item()
        print('After temperature(VAL) Negative - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        res["val"]["after_temperature_negative"]["nll"] = after_temperature_nll
        res["val"]["after_temperature_negative"]["ece"] = after_temperature_ece

        # TEST
        test_logits_list = []
        test_labels_list = []
        with torch.no_grad():
            for logits, labels in test_loader:
                test_logits_list.append(logits)
                test_labels_list.append(labels)
            test_logits = torch.cat(test_logits_list) # .cuda()
            test_labels = torch.cat(test_labels_list) # .cuda()

        before_temperature_nll = nll_criterion(test_logits, test_labels).item()
        before_temperature_ece = ece_criterion(test_logits, test_labels).item()
        print('Before temperature(TEST) - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        res["test"]["before_temperature"]["nll"] = before_temperature_nll
        res["test"]["before_temperature"]["ece"] = before_temperature_ece

        mask = test_labels == 1
        before_temperature_nll = nll_criterion(test_logits[mask], test_labels[mask]).item()
        before_temperature_ece = ece_criterion(test_logits[mask], test_labels[mask]).item()
        print('Before temperature(TEST) Positive - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        res["test"]["before_temperature_positive"]["nll"] = before_temperature_nll
        res["test"]["before_temperature_positive"]["ece"] = before_temperature_ece

        mask = test_labels == 0
        before_temperature_nll = nll_criterion(test_logits[mask], test_labels[mask]).item()
        before_temperature_ece = ece_criterion(test_logits[mask], test_labels[mask]).item()
        print('Before temperature(TEST) Negative - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        res["test"]["before_temperature_negative"]["nll"] = before_temperature_nll
        res["test"]["before_temperature_negative"]["ece"] = before_temperature_ece

        after_temperature_nll = nll_criterion(self.temperature_scale(test_logits), test_labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(test_logits), test_labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        res["test"]["after_temperature"]["nll"] = after_temperature_nll
        res["test"]["after_temperature"]["ece"] = after_temperature_ece

        mask = test_labels == 1
        after_temperature_nll = nll_criterion(self.temperature_scale(test_logits[mask]), test_labels[mask]).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(test_logits[mask]), test_labels[mask]).item()
        print('After temperature(TEST) Positive - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        res["test"]["after_temperature_positive"]["nll"] = after_temperature_nll
        res["test"]["after_temperature_positive"]["ece"] = after_temperature_ece

        mask = test_labels == 0
        after_temperature_nll = nll_criterion(self.temperature_scale(test_logits[mask]), test_labels[mask]).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(test_logits[mask]), test_labels[mask]).item()
        print('After temperature(TEST) Negative - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        res["test"]["after_temperature_negative"]["nll"] = after_temperature_nll
        res["test"]["after_temperature_negative"]["ece"] = after_temperature_ece

        return res


# class _ECELoss(nn.Module):
#     def __init__(self, n_bins=15):
#         """
#         n_bins (int): number of confidence interval bins
#         """
#         super(_ECELoss, self).__init__()
#         bin_boundaries = torch.linspace(0, 1, n_bins + 1)
#         self.bin_lowers = bin_boundaries[:-1]
#         self.bin_uppers = bin_boundaries[1:]

#     def forward(self, logits, labels):
#         softmaxes = F.softmax(logits, dim=1)
#         confidences, predictions = torch.max(softmaxes, 1)
#         accuracies = predictions.eq(labels)

#         ece = torch.zeros(1, device=logits.device)
#         for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
#             # Calculated |confidence - accuracy| in each bin
#             in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
#             prop_in_bin = in_bin.float().mean()
#             if prop_in_bin.item() > 0:
#                 accuracy_in_bin = accuracies[in_bin].float().mean()
#                 avg_confidence_in_bin = confidences[in_bin].mean()
#                 ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

#         return ece


class _ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        self.n_bins = n_bins

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        percentiles = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)
        bin_boundaries = torch.quantile(confidences, percentiles)

        self.bin_lowers = bin_boundaries[:-1]  # Lower bounds of bins
        self.bin_uppers = bin_boundaries[1:]

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
    

class MyDataset(Dataset):
    def __init__(self, labels, logits, attention_masks):
        labels = torch.tensor(labels, dtype=torch.long)
        logits = torch.tensor(logits, dtype=torch.float32)
        attention_masks = torch.tensor(attention_masks, dtype=torch.float32)

        labels = labels.view(-1)
        logits = logits.view(-1, logits.size(-1))
        attention_masks = attention_masks.view(-1)

        mask = attention_masks == 1
        self.labels = labels[mask]
        self.logits = logits[mask]

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, idx):
        return self.logits[idx], self.labels[idx]
    
    def get_data_loader(self, batch_size=64):
        return DataLoader(self, batch_size=batch_size, shuffle=False)

def main(
        subset: str
):
    test_dir = f"logprobs/{subset}/test/internvl"
    val_dir = f"logprobs/{subset}/val/internvl"

    test_label_fnames = sorted([fname for fname in os.listdir(test_dir) if fname.startswith("all_labels")], key=lambda x: int(x.split(".npy")[0].split("_batch")[-1]))
    test_attention_masks_fnames = sorted([fname for fname in os.listdir(test_dir) if fname.startswith("label_masks")], key=lambda x: int(x.split(".npy")[0].split("_batch")[-1]))
    test_logit_fnames = sorted([fname for fname in os.listdir(test_dir) if fname.startswith("logits")], key=lambda x: int(x.split(".npy")[0].split("_batch")[-1]))

    test_labels = []

    for fname in test_label_fnames[25:50] + test_label_fnames[-50:-25]:
        labels = np.load(os.path.join(test_dir, fname))
        test_labels.extend(labels)

    test_labels = 1.0 - pad_to_max_len_label(test_labels) # flip the labels, value closer to 0 is positive
    print(test_labels.shape)

    test_attention_masks = []

    for fname in test_attention_masks_fnames[25:50] + test_attention_masks_fnames[-50:-25]:
        attention_masks = np.load(os.path.join(test_dir, fname))
        test_attention_masks.extend(attention_masks)

    test_attention_masks = pad_to_max_len_label(test_attention_masks)
    print(test_attention_masks.shape)

    test_logits = []
    for fname in tqdm(test_logit_fnames[25:50] + test_logit_fnames[-50:-25]):
        logits = np.load(os.path.join(test_dir, fname))
        test_logits.extend(logits)

    test_logits = pad_to_max_len_logit(test_logits)
    print(test_logits.shape)

    val_label_fnames = sorted([fname for fname in os.listdir(val_dir) if fname.startswith("all_labels")], key=lambda x: int(x.split(".npy")[0].split("_batch")[-1]))
    val_attention_masks_fnames = sorted([fname for fname in os.listdir(val_dir) if fname.startswith("label_masks")], key=lambda x: int(x.split(".npy")[0].split("_batch")[-1]))
    val_logit_fnames = sorted([fname for fname in os.listdir(val_dir) if fname.startswith("logits")], key=lambda x: int(x.split(".npy")[0].split("_batch")[-1]))

    assert len(val_label_fnames) == len(val_attention_masks_fnames) == len(val_logit_fnames)

    val_labels = []

    for fname in val_label_fnames[25:50] + val_label_fnames[-50:-25]:
        labels = np.load(os.path.join(val_dir, fname))
        val_labels.extend(labels)
    
    val_labels = 1.0 - pad_to_max_len_label(val_labels) # flip the labels, value closer to 0 is positive
    print(val_labels.shape)

    val_attention_masks = []

    for fname in val_attention_masks_fnames[25:50] + val_attention_masks_fnames[-50:-25]:
        attention_masks = np.load(os.path.join(val_dir, fname))
        val_attention_masks.extend(attention_masks)
    
    val_attention_masks = pad_to_max_len_label(val_attention_masks)
    print(val_attention_masks.shape)

    val_logits = []

    for fname in tqdm(val_logit_fnames[25:50] + val_logit_fnames[-50:-25]):
        logits = np.load(os.path.join(val_dir, fname))
        val_logits.extend(logits)
    
    val_logits = pad_to_max_len_logit(val_logits)
    print(val_logits.shape)

    dataset = MyDataset(val_labels, val_logits, val_attention_masks)
    valid_loader = dataset.get_data_loader()

    dataset = MyDataset(test_labels, test_logits, test_attention_masks)
    test_loader = dataset.get_data_loader()

    scaled_model = ModelWithTemperature()
    res = scaled_model.set_temperature(valid_loader, test_loader)

    save_dir = os.path.join('evaluation/calibration', f"{subset}/internvl_only")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"internvl_calibration.json")
    with open(save_path, "w") as f:
        json.dump(res, f, indent=4)


if __name__ == '__main__':
    fire.Fire(main)