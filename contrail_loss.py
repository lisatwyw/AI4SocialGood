import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from functools import partial
 
import ssl
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import albumentations as albu
import segmentation_models_pytorch as smp

# %%
# avoid error for downloading pre-trained model
ssl._create_default_https_context = ssl._create_unverified_context

# %%
default_encoder = "resnet50"


# %%
# helper function for data visualization
def visualize(**images):
    n = len(images)
    plt.figure(figsize=(n * 5, 5))
    for i, (name, image) in enumerate(images.items()):
        image = (image - image.min()) / (image.max() - image.min())
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        if image.ndim == 3:
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
        plt.imshow(image.squeeze())
    plt.tight_layout()
    plt.show()


# %%
# classes for data loading and preprocessing
class Dataset:
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder, Default: None
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, perspective, gamma, etc.)
        preprocessing (albumentations.Compose): data preprocessing from pre-trained model
            (e.g. normalization, shape manipulation, etc.)
    """

    def __init__(
        self, image_paths, mask_paths=None, augmentation=None, preprocessing=None
    ):
        self.ids = image_paths
        self.image_paths = image_paths

        self.has_mask = True if mask_paths is not None else False

        if self.has_mask:
            self.mask_paths = mask_paths

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read image in color
        image = cv2.imread(self.image_paths[i], cv2.IMREAD_COLOR)

        if self.has_mask:
            # read mask (png) and convert to grayscale
            mask = np.amax(cv2.imread(self.mask_paths[i], cv2.IMREAD_UNCHANGED), axis=2)
            mask = mask / mask.max()

            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]

            return image, mask

        else:
            if self.augmentation:
                sample = self.augmentation(image=image)
                image = sample["image"]

            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image = sample["image"]

        return image

    def __len__(self):
        return len(self.ids)


# %%


def get_train_augmentation():
    """Define augmentation for contrail training images."""

    transform = [
        albu.ShiftScaleRotate(
            scale_limit=0.2,
            rotate_limit=180,
            shift_limit=0.3,
            border_mode=0,
            p=1,
        ),
        albu.PadIfNeeded(
            min_height=320, min_width=320, always_apply=True, border_mode=0
        ),
        albu.Resize(320, 320),
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.3, p=1
                ),
                albu.RandomGamma(gamma_limit=(20, 100), p=1),
            ],
            p=0.5,
        ),
    ]
    return albu.Compose(transform)


def get_val_augmentation():
    """Define augmentation for contrail validation images."""

    transform = [
        albu.ShiftScaleRotate(
            scale_limit=0.2,
            rotate_limit=180,
            shift_limit=0.3,
            border_mode=0,
            p=1,
        ),
        albu.PadIfNeeded(
            min_height=320, min_width=320, always_apply=True, border_mode=0
        ),
        albu.Resize(320, 320),
    ]
    return albu.Compose(transform)


def get_test_augmentation():
    """Define augmentation for contrail testing images (pad and resize only)."""
    transform = [
        albu.PadIfNeeded(
            min_height=320, min_width=320, always_apply=True, border_mode=0
        ),
        albu.Resize(320, 320),
    ]
    return albu.Compose(transform)


def get_preprocessing():
    """Construct preprocessing transform

    Return:
        transform: albumentations.Compose

    """

    def train_to_tensor(image, **kwargs):
        return image.transpose(2, 0, 1).astype("float32")

    def mask_to_tensor(mask, **kwargs):
        return np.expand_dims(mask, 0).astype("float32")

    _transform = [
        albu.Lambda(image=smp.encoders.get_preprocessing_fn(default_encoder)),
        albu.Lambda(image=train_to_tensor, mask=mask_to_tensor),
    ]
    return albu.Compose(_transform)


# %%


import pytorch_lightning as pl


#class LightningModule(pl.LightningModule):

# pytorch model based on lighting
class ContrailModel(pl.LightningModule):
    def __init__(self, arch, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_weights=None,
            encoder_name=default_encoder,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # preprocessing parameters for image
        params = smp.encoders.get_preprocessing_params(default_encoder)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        loss_name = kwargs.get("loss", "dice").lower()
        assert loss_name in ["dice", "focal", "sr"]

        if loss_name == "dice":
            self.loss_fn = loss.DiceLoss(from_logits=True)
        if loss_name == "focal":
            self.loss_fn = loss.FocalLoss(normalized=True)
        if loss_name == "sr":
            self.loss_fn = loss.SRLoss(from_logits=True)

        self.outputs = {"train": [], "valid": []}

    def forward(self, image):
        # normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch[0]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # for grayscale images, expand channels dim to be [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32);
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Compute metrics for some threshold
        # first convert mask values to probabilities, then apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # Compute true positive, false positive, false negative and true negative
        # of 'pixels' for each image and class. Aggregate them at epoch end.
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )

        stats = {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

        self.outputs[stage].append(stats)

        return stats

    def shared_epoch_end(self, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in self.outputs[stage]])
        fp = torch.cat([x["fp"] for x in self.outputs[stage]])
        fn = torch.cat([x["fn"] for x in self.outputs[stage]])
        tn = torch.cat([x["tn"] for x in self.outputs[stage]])

        # First calculate IoU score for each image, then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # aggregate intersection and union over whole dataset and compute IoU score.
        # For dataset with "empty" images (images without target class), a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        self.shared_epoch_end("train")
        self.outputs["train"].clear()

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        self.shared_epoch_end("valid")
        self.outputs["valid"].clear()

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.0001)
        return torch.optim.Adam(self.parameters())


class DiceLoss(_Loss):
    def __init__(
        self,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7,
        dh_weight=0.6,
    ):
        """Compute Dice loss"""
        super().__init__()
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.dh_weight = dh_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            y_pred = F.logsigmoid(y_pred).exp()

        # dimention 0 are images in a batch
        dims = (1, 2, 3)

        predict, target = y_pred, y_true.type_as(y_pred)

        smooth = self.smooth
        eps = self.eps

        intersection = torch.sum(predict * target, dims)
        cardinality = torch.sum(predict + target, dims)
        dice = (2 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
        if self.log_loss:
            loss = -torch.log(dice)
        else:
            loss = 1 - dice

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        agg_loss = loss.mean()

        return agg_loss


class FocalLoss(_Loss):
    def __init__(
        self,
        alpha=None,
        gamma=2.0,
        ignore_index=None,
        reduction="mean",
        normalized: bool = False,
        reduced_threshold=None,
        eps=1e-6,
    ):
        """Compute Focal loss"""
        super().__init__()

        assert self.reduction in ["mean", "sum", "batchwise_mean"]

        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma
        self.reduced_threshold = reduced_threshold
        self.reduction = reduction
        self.normalized = normalized
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        batch_size = y_pred.size(0)

        predict, target = y_pred, y_true.type_as(y_pred)

        # reduce to two dimensions, dim-0 for the images in the batch
        predict = predict.view(batch_size, -1)
        target = target.view(batch_size, -1)

        logpt = F.binary_cross_entropy_with_logits(predict, target, reduction="none")
        pt = torch.exp(-logpt)

        assert pt.ndim == 2

        # compute the loss
        if self.reduced_threshold is None:
            focal_term = (1.0 - pt).pow(self.gamma)
        else:
            focal_term = ((1.0 - pt) / self.reduced_threshold).pow(self.gamma)
            focal_term[pt < self.reduced_threshold] = 1

        loss = focal_term * logpt

        assert loss.ndim == 2

        if self.alpha is not None:
            loss *= self.alpha * target + (1 - self.alpha) * (1 - target)

        if self.normalized:
            norm_factor = focal_term.sum(1).clamp_min(self.eps).view(batch_size, -1)
            loss /= norm_factor

        if self.reduction == "mean":
            agg_loss = loss.mean()
        if self.reduction == "sum":
            agg_loss = loss.sum()
        if self.reduction == "batchwise_mean":
            agg_loss = loss.sum(0)

        return agg_loss


def hough_transform(batch, threshold=50, return_coordinates=False):
    height, width = batch[0].squeeze().shape

    thetas = torch.arange(0, 180, 0.5)
    d = torch.sqrt(torch.tensor(width) ** 2 + torch.tensor(height) ** 2)
    rhos = torch.arange(-d, d, 3)

    cos_thetas = torch.cos(torch.deg2rad(thetas))
    sin_thetas = torch.sin(torch.deg2rad(thetas))

    hough_matrices = torch.Tensor(
        batch.shape[0], rhos.shape[0] - 1, thetas.shape[0] - 1
    )

    for i, img in enumerate(batch):
        img = img.squeeze()
        points = torch.argwhere(img > 0.5).type_as(cos_thetas)
        rho_values = torch.matmul(points, torch.stack((sin_thetas, cos_thetas)))

        accumulator, (theta_vals, rho_vals) = torch.histogramdd(
            torch.stack(
                (
                    torch.tile(thetas, (rho_values.shape[0],)),
                    rho_values.ravel(),
                )
            ).T,
            bins=[thetas, rhos],
        )

        accumulator = torch.transpose(accumulator, 0, 1)

        if return_coordinates:
            hough_lines = torch.argwhere(accumulator > threshold)
            rho_idxs, theta_idxs = hough_lines[:, 0], hough_lines[:, 1]
            hough_rhos, hough_thetas = rhos[rho_idxs], thetas[theta_idxs]
            hough_coordinates = torch.stack((hough_rhos, hough_thetas))
            return hough_coordinates
        else:
            hough_matrix = torch.where(accumulator > threshold, 1, 0)
            hough_matrices[i] = hough_matrix
            return hough_matrices


class SRLoss(_Loss):
    def __init__(
        self,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7,
        dh_weight=0.5,
    ):
        """Dice loss for image segmentation task with Hough Transform constraint."""
        super(SRLoss, self).__init__()
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.dh_weight = dh_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            y_pred = F.logsigmoid(y_pred).exp()

        # bs = y_true.size(0)
        # dims = (1, 2, 3)

        # y_true = y_true.view(bs, 1, -1)
        # y_pred = y_pred.view(bs, 1, -1)

        predict, target = y_pred, y_true.type_as(y_pred)

        smooth = self.smooth
        eps = self.eps

        intersection = torch.sum(predict * target)
        cardinality = torch.sum(predict + target)
        dice = (2 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
        if self.log_loss:
            dice_loss = -torch.log(dice.clamp_min(self.eps))
        else:
            dice_loss = 1 - dice

        # compute customized hough loss
        hough_predict = hough_transform(predict)
        hough_target = hough_transform(target)

        h_intersection = torch.sum(hough_predict * hough_target)
        h_cardinality = torch.sum(hough_predict + hough_target)
        dice = (2 * h_intersection + smooth) / (h_cardinality + smooth).clamp_min(eps)

        if self.log_loss:
            hough_dice_loss = -torch.log(dice.clamp_min(self.eps))
        else:
            hough_dice_loss = 1 - dice

        loss = self.dh_weight * dice_loss + (1 - self.dh_weight) * hough_dice_loss

        return loss
