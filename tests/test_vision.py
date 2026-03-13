from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from app.vision.service import VisionService


def test_save_annotated_outputs_file(tmp_path: Path) -> None:
    src = tmp_path / "raw.png"
    out = tmp_path / "ann.png"
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.imwrite(str(src), image)

    svc = VisionService()
    saved = svc.save_annotated(str(src), [(10, 20, 30, 40), (80, 70, 50, 50)], str(out))
    assert Path(saved).exists()
