from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode


# Keep each augmentation operating on an isolated copy so later transforms
# never mutate the original sample in place.
def clone_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        # image: uint8 tensor with shape [C, H, W]
        "image": sample["image"].clone(),
        # boxes: float tensor with shape [N, 4] in xyxy format
        "boxes": sample["boxes"].clone(),
        # labels: long tensor with shape [N]
        "labels": sample["labels"].clone(),
        "sample_id": sample.get("sample_id"),
        "meta": dict(sample.get("meta", {})),
    }


def image_size(image: torch.Tensor) -> tuple[int, int]:
    # All image tensors in this project use channel-first layout: [C, H, W].
    return int(image.shape[-2]), int(image.shape[-1])


def sanitize_boxes(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Clip boxes to the valid image region and drop invalid boxes after transform.
    # Input boxes shape: [N, 4], labels shape: [N].
    if boxes.numel() == 0:
        return boxes.reshape(0, 4).to(dtype=torch.float32), labels.reshape(0).to(dtype=torch.long)

    boxes = boxes.to(dtype=torch.float32)
    labels = labels.to(dtype=torch.long)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid], labels[valid]


def boxes_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4, 2))
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    return torch.stack(
        [
            torch.stack([x1, y1], dim=1),
            torch.stack([x2, y1], dim=1),
            torch.stack([x2, y2], dim=1),
            torch.stack([x1, y2], dim=1),
        ],
        dim=1,
    )


def corners_to_boxes(corners: torch.Tensor) -> torch.Tensor:
    if corners.numel() == 0:
        return corners.new_zeros((0, 4))
    mins = corners.min(dim=1).values
    maxs = corners.max(dim=1).values
    return torch.stack([mins[:, 0], mins[:, 1], maxs[:, 0], maxs[:, 1]], dim=1)


class Compose:
    def __init__(self, transforms: list[Any]) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict[str, Any], sample_provider: Any | None = None) -> dict[str, Any]:
        # MixUp and Mosaic need extra samples, so we pass a provider only to
        # transforms that explicitly declare that requirement.
        for transform in self.transforms:
            if getattr(transform, "requires_sample_provider", False):
                sample = transform(sample, sample_provider=sample_provider)
            else:
                sample = transform(sample)
        return sample


class RandomFlip:
    def __init__(self, p_horizontal: float = 0.5, p_vertical: float = 0.0) -> None:
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample = clone_sample(sample)
        # image keeps shape [C, H, W]; only pixel order and box coordinates change.
        image = sample["image"]
        boxes = sample["boxes"]
        labels = sample["labels"]
        height, width = image_size(image)

        if random.random() < self.p_horizontal:
            image = torch.flip(image, dims=[2])
            if boxes.numel() > 0:
                x1 = width - boxes[:, 2]
                x2 = width - boxes[:, 0]
                boxes = torch.stack([x1, boxes[:, 1], x2, boxes[:, 3]], dim=1)

        if random.random() < self.p_vertical:
            image = torch.flip(image, dims=[1])
            if boxes.numel() > 0:
                y1 = height - boxes[:, 3]
                y2 = height - boxes[:, 1]
                boxes = torch.stack([boxes[:, 0], y1, boxes[:, 2], y2], dim=1)

        boxes, labels = sanitize_boxes(boxes, labels, height, width)
        sample["image"] = image
        sample["boxes"] = boxes
        sample["labels"] = labels
        return sample


class ColorJitterTransform:
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        p: float = 0.8,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def _factor(self, strength: float, center: float = 1.0) -> float:
        return random.uniform(max(0.0, center - strength), center + strength)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample = clone_sample(sample)
        if random.random() >= self.p:
            return sample

        # Color jitter changes pixel values only, so image shape stays [C, H, W]
        # and boxes / labels shapes stay [N, 4] and [N].
        image = sample["image"].to(dtype=torch.float32) / 255.0
        fn_ids = [0, 1, 2, 3]
        random.shuffle(fn_ids)
        for fn_id in fn_ids:
            if fn_id == 0 and self.brightness > 0:
                image = F.adjust_brightness(image, self._factor(self.brightness))
            elif fn_id == 1 and self.contrast > 0:
                image = F.adjust_contrast(image, self._factor(self.contrast))
            elif fn_id == 2 and self.saturation > 0:
                image = F.adjust_saturation(image, self._factor(self.saturation))
            elif fn_id == 3 and self.hue > 0:
                image = F.adjust_hue(image, random.uniform(-self.hue, self.hue))

        sample["image"] = (image.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8)
        return sample


class RandomAffine:
    def __init__(
        self,
        degrees: float = 10.0,
        translate: tuple[float, float] = (0.1, 0.1),
        scale: tuple[float, float] = (0.9, 1.1),
        shear: tuple[float, float] = (-5.0, 5.0),
        p: float = 0.5,
    ) -> None:
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample = clone_sample(sample)
        if random.random() >= self.p:
            return sample

        # image: [C, H, W]
        # boxes: [N, 4]
        image = sample["image"]
        boxes = sample["boxes"]
        labels = sample["labels"]
        height, width = image_size(image)

        angle = random.uniform(-self.degrees, self.degrees)
        max_dx = self.translate[0] * width
        max_dy = self.translate[1] * height
        translate = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
        scale = random.uniform(self.scale[0], self.scale[1])
        shear_x = random.uniform(self.shear[0], self.shear[1])
        shear_y = random.uniform(self.shear[0], self.shear[1])

        image = F.affine(
            image,
            angle=angle,
            translate=[int(round(translate[0])), int(round(translate[1]))],
            scale=scale,
            shear=[shear_x, shear_y],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )

        if boxes.numel() > 0:
            # Apply the same geometric transform to box corners, then rebuild
            # each box from the transformed corner coordinates.
            center = torch.tensor([width * 0.5, height * 0.5], dtype=torch.float32)
            corners = boxes_to_corners(boxes).reshape(-1, 2)
            corners = (corners - center) * scale
            theta = torch.deg2rad(torch.tensor(angle, dtype=torch.float32))
            rotation = torch.tensor(
                [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]],
                dtype=torch.float32,
            )
            corners = corners @ rotation.T
            corners[:, 0] += torch.tan(torch.deg2rad(torch.tensor(shear_x, dtype=torch.float32))) * corners[:, 1]
            corners[:, 1] += torch.tan(torch.deg2rad(torch.tensor(shear_y, dtype=torch.float32))) * corners[:, 0]
            corners = corners + center + torch.tensor([translate[0], translate[1]], dtype=torch.float32)
            boxes = corners_to_boxes(corners.reshape(-1, 4, 2))

        boxes, labels = sanitize_boxes(boxes, labels, height, width)
        sample["image"] = image
        sample["boxes"] = boxes
        sample["labels"] = labels
        return sample


class RandomCrop:
    def __init__(self, min_scale: float = 0.6, p: float = 0.5) -> None:
        self.min_scale = min_scale
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample = clone_sample(sample)
        if random.random() >= self.p:
            return sample

        # Crop changes the spatial size, so image becomes [C, crop_h, crop_w].
        # boxes remain [N, 4], but coordinates are shifted into the cropped view.
        image = sample["image"]
        boxes = sample["boxes"]
        labels = sample["labels"]
        height, width = image_size(image)

        crop_h = random.randint(max(1, int(height * self.min_scale)), height)
        crop_w = random.randint(max(1, int(width * self.min_scale)), width)
        top = random.randint(0, height - crop_h)
        left = random.randint(0, width - crop_w)

        image = image[:, top : top + crop_h, left : left + crop_w]
        if boxes.numel() > 0:
            boxes = boxes.clone()
            boxes[:, 0::2] -= left
            boxes[:, 1::2] -= top
        boxes, labels = sanitize_boxes(boxes, labels, crop_h, crop_w)

        sample["image"] = image
        sample["boxes"] = boxes
        sample["labels"] = labels
        return sample


@dataclass
class MixUp:
    alpha: float = 0.4
    p: float = 0.5
    requires_sample_provider: bool = True

    def __call__(self, sample: dict[str, Any], sample_provider: Any | None = None) -> dict[str, Any]:
        sample = clone_sample(sample)
        if random.random() >= self.p or sample_provider is None:
            return sample

        other = clone_sample(sample_provider())
        # MixUp expects both images to share the same spatial size before blending.
        # After resizing if needed, both tensors are [C, H, W].
        image_a = sample["image"].to(dtype=torch.float32)
        image_b = other["image"].to(dtype=torch.float32)
        if image_a.shape[-2:] != image_b.shape[-2:]:
            # Resize the second image to the first image size so the two images
            # can be blended and their boxes remain in the same coordinate system.
            target_hw = list(image_a.shape[-2:])
            image_b = F.resize(image_b, target_hw, interpolation=InterpolationMode.BILINEAR)
            scale_x = image_a.shape[-1] / other["image"].shape[-1]
            scale_y = image_a.shape[-2] / other["image"].shape[-2]
            if other["boxes"].numel() > 0:
                other["boxes"][:, 0::2] *= scale_x
                other["boxes"][:, 1::2] *= scale_y

        lam = random.betavariate(self.alpha, self.alpha)
        mixed = (lam * image_a + (1.0 - lam) * image_b).clamp(0.0, 255.0).to(dtype=torch.uint8)

        sample["image"] = mixed
        # MixUp concatenates targets from both images, so box count becomes N1 + N2.
        sample["boxes"] = torch.cat([sample["boxes"], other["boxes"]], dim=0)
        sample["labels"] = torch.cat([sample["labels"], other["labels"]], dim=0)
        return sample


@dataclass
class Mosaic:
    output_size: tuple[int, int] = (640, 640)
    p: float = 0.5
    requires_sample_provider: bool = True

    def __call__(self, sample: dict[str, Any], sample_provider: Any | None = None) -> dict[str, Any]:
        sample = clone_sample(sample)
        if random.random() >= self.p or sample_provider is None:
            return sample

        samples = [sample] + [clone_sample(sample_provider()) for _ in range(3)]
        out_h, out_w = self.output_size
        # Build one large canvas and place four resized samples into its quadrants.
        # Output image shape is fixed as [3, out_h, out_w].
        canvas = torch.zeros((3, out_h, out_w), dtype=torch.uint8)
        xc = random.randint(int(out_w * 0.3), int(out_w * 0.7))
        yc = random.randint(int(out_h * 0.3), int(out_h * 0.7))
        placements = [
            (0, 0, xc, yc),
            (xc, 0, out_w, yc),
            (0, yc, xc, out_h),
            (xc, yc, out_w, out_h),
        ]

        all_boxes: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        for part, (x1, y1, x2, y2) in zip(samples, placements):
            target_w = max(1, x2 - x1)
            target_h = max(1, y2 - y1)
            img = part["image"].to(dtype=torch.float32)
            src_h, src_w = image_size(img)
            resized = F.resize(img, [target_h, target_w], interpolation=InterpolationMode.BILINEAR).to(dtype=torch.uint8)
            canvas[:, y1:y2, x1:x2] = resized[:, : y2 - y1, : x2 - x1]

            boxes = part["boxes"].clone()
            labels = part["labels"].clone()
            if boxes.numel() > 0:
                # Each sub-image uses its own scale and offset inside the final canvas.
                boxes[:, 0::2] *= target_w / src_w
                boxes[:, 1::2] *= target_h / src_h
                boxes[:, 0::2] += x1
                boxes[:, 1::2] += y1
            boxes, labels = sanitize_boxes(boxes, labels, out_h, out_w)
            all_boxes.append(boxes)
            all_labels.append(labels)

        sample["image"] = canvas
        # Mosaic merges four target sets into one, so the final number of boxes is
        # the sum of valid boxes preserved from all four sub-images.
        sample["boxes"] = torch.cat(all_boxes, dim=0) if all_boxes else torch.zeros((0, 4), dtype=torch.float32)
        sample["labels"] = torch.cat(all_labels, dim=0) if all_labels else torch.zeros((0,), dtype=torch.long)
        return sample


def default_train_augmentation_pipeline() -> Compose:
    # A compact training pipeline that mixes light geometric changes with
    # heavier multi-image augmentations.
    return Compose(
        [
            RandomFlip(p_horizontal=0.5),
            ColorJitterTransform(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
            RandomAffine(degrees=10.0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5.0, 5.0), p=0.5),
            RandomCrop(min_scale=0.6, p=0.3),
            MixUp(alpha=0.4, p=0.3),
            Mosaic(output_size=(640, 640), p=0.3),
        ]
    )
