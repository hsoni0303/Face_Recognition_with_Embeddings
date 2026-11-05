"""Face detection, embedding, and gallery matching.

Pipeline: MTCNN finds and aligns faces -> FaceNet (InceptionResnetV1,
VGGFace2 weights) maps each face to a 512-d embedding -> embeddings are
L2-normalized and compared to a gallery by cosine similarity.
"""

from pathlib import Path

import numpy as np


class FaceEncoder:
    def __init__(self, device="cpu"):
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        self.device = device

    def encode(self, image):
        """Return (boxes, embeddings) for every face in a PIL image.

        boxes: (n, 4) array of x1, y1, x2, y2; embeddings: (n, 512), unit-norm.
        """
        import torch

        boxes, _ = self.mtcnn.detect(image)
        if boxes is None:
            return np.empty((0, 4)), np.empty((0, 512))
        boxes = np.asarray(boxes, dtype=float)  # some versions return an object array
        faces = self.mtcnn.extract(image, boxes, None).to(self.device)  # aligned 160x160
        with torch.no_grad():
            emb = self.model(faces).cpu().numpy()
        return boxes, emb / np.linalg.norm(emb, axis=1, keepdims=True)


class Gallery:
    """Named reference embeddings, persisted as a single .npz file."""

    def __init__(self, names=(), embeddings=None):
        self.names = list(names)
        self.embeddings = np.empty((0, 512)) if embeddings is None else np.asarray(embeddings)

    def add(self, name, embedding):
        self.names.append(name)
        self.embeddings = np.vstack([self.embeddings, embedding])

    def match(self, embedding, threshold=0.7):
        """Return (name, score) of the closest person, or ("unknown", score)."""
        if not self.names:
            return "unknown", 0.0
        scores = self.embeddings @ embedding  # cosine: all vectors are unit-norm
        best = int(np.argmax(scores))
        name = self.names[best] if scores[best] >= threshold else "unknown"
        return name, float(scores[best])

    def save(self, path):
        np.savez(path, names=np.array(self.names), embeddings=self.embeddings)

    @classmethod
    def load(cls, path):
        if not Path(path).exists():
            return cls()
        data = np.load(path)
        return cls(data["names"].tolist(), data["embeddings"])
