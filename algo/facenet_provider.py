from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import cv2

try:
    import torch
    import torch.nn.functional as F
    from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
except ImportError:  # pragma: no cover - optional dependency at runtime
    torch = None
    F = None
    InceptionResnetV1 = None
    fixed_image_standardization = None

from .facereco_provider import BodyRecord, FaceRecoProvider, Player


_MODEL: InceptionResnetV1 | None = None
_DEVICE: "torch.device" | None = None
_MODEL_DEVICE_NAME: str | None = None


@dataclass
class FaceNetFaceRecoProvider(FaceRecoProvider):
    """FaceNet-based provider using facenet-pytorch embeddings."""

    model_name: str = "vggface2"
    device: str | None = None
    image_size: int = 160
    batch_size: int = 64
    _runtime_info: dict = field(default_factory=dict, init=False, repr=False)

    def provider_name(self) -> str:
        return "facenet"

    def predict_player(self, image_bgr: np.ndarray, body: BodyRecord) -> Player:
        return self.predict_players([image_bgr], [body])[0]

    def predict_players(self, images_bgr: list[np.ndarray], bodies: list[BodyRecord]) -> list[Player]:
        """Embed every (image, body) pair, running the network on batches.

        A single 160x160 forward pass is launch-overhead bound on a GPU
        (~25 ms on a TITAN V) while a batch of 64 costs ~1 ms per face, so
        callers with many faces should pass them together. Batching changes
        an embedding only by float rounding (~1e-7)."""
        if torch is None or F is None or InceptionResnetV1 is None or fixed_image_standardization is None:
            raise RuntimeError(
                "facenet-pytorch is not installed. Install with: pip install facenet-pytorch"
            )

        players: list[Player] = []
        tensors: list = []
        slots: list[int] = []
        for image_bgr, body in zip(images_bgr, bodies):
            tensor = self._preprocess(image_bgr, body)
            if tensor is None:
                players.append(Player(name="", jersey_number=None, confidence=0.0, internal={"embedding": None}))
                continue
            slots.append(len(players))
            tensors.append(tensor)
            players.append(Player(
                name="",
                jersey_number=None,
                confidence=body.confidence,
                internal={"embedding": None, "provider": self.provider_name()},
            ))

        if tensors:
            model = self._get_model()
            with torch.no_grad():
                for start in range(0, len(tensors), self.batch_size):
                    batch = torch.cat(tensors[start:start + self.batch_size]).to(_DEVICE)
                    embeddings = model(batch).cpu().numpy().astype(np.float32)
                    for offset, embedding in enumerate(embeddings):
                        players[slots[start + offset]].internal["embedding"] = embedding
        return players

    def _preprocess(self, image_bgr: np.ndarray, body: BodyRecord):
        """Crop *body*'s face box out of *image_bgr* and return the model's
        1x3xSxS input tensor, or None when there is no usable face box."""
        face_box = body.narrow_face_bbox or body.face_bbox
        if face_box is None:
            return None

        h, w = image_bgr.shape[:2]
        left, top, right, bottom = face_box.as_px_ints(w, h)
        left = max(0, min(w - 1, left))
        right = max(0, min(w - 1, right))
        top = max(0, min(h - 1, top))
        bottom = max(0, min(h - 1, bottom))
        if right <= left or bottom <= top:
            return None

        crop = image_bgr[top:bottom, left:right]
        if crop.size == 0:
            return None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)
        tensor = F.interpolate(tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return fixed_image_standardization(tensor)

    def _get_model(self) -> InceptionResnetV1:
        global _MODEL, _DEVICE, _MODEL_DEVICE_NAME

        if torch is None or InceptionResnetV1 is None:
            raise RuntimeError(
                "facenet-pytorch is not installed. Install with: pip install facenet-pytorch"
            )

        device_name = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if _MODEL is not None and _MODEL_DEVICE_NAME == device_name:
            return _MODEL

        _DEVICE = torch.device(device_name)
        _MODEL = InceptionResnetV1(pretrained=self.model_name).eval().to(_DEVICE)
        _MODEL_DEVICE_NAME = device_name
        return _MODEL
