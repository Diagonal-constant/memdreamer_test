import base64
import cv2
import numpy as np
from .prompts import SYSTEM_VISION


def fmt_ts(sec):
    sec = int(float(sec) + 0.5)  # half-up; round() is half-to-even and labels 44.5/45.5/46.5 as 44/46/46
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_ts(x, default=None):
    """Accepts seconds (int/float/str), 'MM:SS' or 'HH:MM:SS'."""
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        pass
    try:
        nums = [float(p) for p in s.split(":")]
    except ValueError:
        return default
    sec = 0.0
    for n in nums:
        sec = sec * 60 + n
    return sec


class Video:
    def __init__(self, path):
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        self.duration = n / fps
        cap.release()

    def mosaics(self, start, end, grid=(3, 3), max_images=5, tile_width=426, max_fps=1.0, quality=85, max_pixels=1_000_000, label_fn=None):
        """The service accepts <=5 images/request, so frames are packed into grid 'mosaics'.
        Uniformly samples up to cols*rows*max_images frames in [start,end); every tile has its ABSOLUTE video
        timestamp burned into the top-left corner. Returns [{'b64': jpeg, 'times': [abs_sec,...]}]."""
        end = min(end, self.duration)
        span = max(end - start, 0.1)
        cols, rows = int(grid[0]), int(grid[1])
        per = cols * rows
        n = int(max(1, min(per * max_images, round(span * max_fps))))
        step = span / n
        cap = cv2.VideoCapture(self.path)
        tiles = []
        for i in range(n):
            t = start + step * (i + 0.5)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, img = cap.read()
            if ok:
                tiles.append((t, img))
        cap.release()
        if not tiles:
            return []
        h, w = tiles[0][1].shape[:2]
        tw = tile_width
        th = round(h * tw / w)
        sc = min(1.0, (max_pixels / (cols * tw * rows * th)) ** 0.5)
        tw, th = max(32, int(tw * sc) // 2 * 2), max(32, int(th * sc) // 2 * 2)
        fs, k = 0.55 * tw / 426, tw / 426
        out = []
        for i in range(0, len(tiles), per):
            canvas = np.zeros((rows * th, cols * tw, 3), np.uint8)
            group = tiles[i:i + per]
            for j, (rt, img) in enumerate(group):
                r, c = divmod(j, cols)
                tile = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                label = label_fn(i + j, rt) if label_fn else fmt_ts(rt)
                (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
                cv2.rectangle(tile, (0, 0), (lw + 10, lh + base + 8), (0, 0, 0), -1)   # solid backing => always legible
                cv2.putText(tile, label, (4, lh + 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
                canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
            ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                out.append({"b64": base64.b64encode(buf).decode(), "times": [t for t, _ in group]})
        return out


def vision_messages(mos, header, prompt, system=SYSTEM_VISION):
    """System message is always sent: otherwise the service injects its default assistant persona."""
    content = [{"type": "text", "text": header}]
    for k, m in enumerate(mos, 1):
        ts = ", ".join(fmt_ts(t) for t in m["times"])
        content.append({"type": "text", "text": f"Image {k}/{len(mos)}: mosaic of {len(m['times'])} frames, tiles left-to-right then top-to-bottom, timestamps: {ts}"})
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + m["b64"]}})
    content.append({"type": "text", "text": prompt})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]
