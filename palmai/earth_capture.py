"""Capture imagery from Google Earth Pro running on this PC.

How a capture works:
  1. The app sets the "camera" (served to Earth as a NetworkLink with flyToView), so Earth flies top-down to
     the requested place; four small magenta markers appear at known coordinates in the view corners.
  2. When the picture has stopped changing (tiles loaded), the app drives Earth's own Save Image (8192x4320).
  3. The markers' pixel positions in the saved file give the exact georeference; the markers are painted out.

Windows only. It uses simulated mouse/keyboard on the real Earth window, so nothing else should be
using the mouse while a capture runs.
"""
import ctypes
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:  # physical pixels everywhere, matching what ImageGrab returns
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:  # noqa: BLE001 - not on Windows / already set
    pass

from . import capture as plan, config, earth, geo  # noqa: E402

log = logging.getLogger("palmai.earth_capture")

IMAGE_W, IMAGE_H = plan.IMAGE_W, plan.IMAGE_H
TEMPLATE = Path(__file__).parent / "assets" / "save_image.png"
LIVE_DIR = config.DATA_DIR / "live"
SETTLE_MIN_S = 9            # Earth's fly-to takes a few seconds
SETTLE_MAX_S = 40
SAVE_TIMEOUT_S = 300
MAX_MARKER_RESIDUAL_PX = 60  # about 5 m on the ground at 0.09 m/pixel (the limit is halved for 4K images)
# Earth's Save Image resolutions: (width, height) and the menu item's offset below the dropdown (px)
RESOLUTIONS = {"8k": ((8192, 4320), 144), "4k": ((3840, 2160), 121)}


class CaptureError(RuntimeError):
    """Raised with a user-facing message when Earth cannot be driven."""


@dataclass
class LiveCapture:
    name: str
    path: Path
    bounds: tuple[float, float, float, float]
    captured_at: str
    range_m: float
    status_image: str


def _gui():
    """Import the Windows GUI libraries lazily so the rest of the app works without them."""
    try:
        import pyautogui
        import win32api
        import win32con
        import win32gui
        from PIL import ImageGrab
    except ImportError as exc:  # pragma: no cover
        raise CaptureError("Capturing from Google Earth needs Windows with pyautogui and pywin32 installed.") from exc
    pyautogui.FAILSAFE = True   # slam the mouse into a screen corner to abort
    pyautogui.PAUSE = 0.15
    return pyautogui, win32api, win32con, win32gui, ImageGrab


def _earth_window(win32gui) -> int:
    found = []

    def visit(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd).startswith("Google Earth Pro"):
            found.append(hwnd)
    win32gui.EnumWindows(visit, None)
    if not found:
        raise CaptureError("Google Earth Pro is not running. Start it and open the Date Palm AI live link.")
    return found[0]


def _window_titled(win32gui, title: str) -> int | None:
    found = []

    def visit(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) == title:
            found.append(hwnd)
    win32gui.EnumWindows(visit, None)
    return found[0] if found else None


def _activate(hwnd: int) -> None:
    _, win32api, win32con, win32gui, _ = _gui()
    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    if win32gui.GetClassName(hwnd).startswith("Qt") and win32gui.GetWindowText(hwnd).startswith("Google Earth Pro"):
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)   # the layout (and the Save Image toolbar) assumes a maximized Earth
    win32api.keybd_event(0x12, 0, 0, 0)       # a tap of Alt lets us take the foreground
    win32api.keybd_event(0x12, 0, 2, 0)
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception as exc:  # noqa: BLE001
        raise CaptureError("Could not bring Google Earth to the front. Click its window once and try again.") from exc
    time.sleep(0.6)


def accept_earth_prompts(seconds: float = 12) -> int:
    """Answer Earth's "only open files from sources you trust" / "reload this file?" prompts.

    Both are small windows titled "Google Earth". The reload prompt is wide and short (answer Yes); the trust
    prompt is nearly square (answer Open). Returns how many prompts were answered.
    """
    pyautogui, _, _, win32gui, _ = _gui()
    answered, deadline = 0, time.time() + seconds
    while time.time() < deadline:
        crash = _window_titled(win32gui, "Google Earth Crash Detected")   # a stray second Earth start: choose "Continue"
        if crash is not None:
            left, top, right, bottom = win32gui.GetWindowRect(crash)
            _activate(crash)
            pyautogui.click(left + 0.90 * (right - left), top + 0.85 * (bottom - top))
            answered += 1
            time.sleep(1.5)
            continue
        hwnd = _window_titled(win32gui, "Google Earth")
        if hwnd is None:
            if answered:
                time.sleep(1.0)
                if _window_titled(win32gui, "Google Earth") is None:
                    break
            time.sleep(0.5)
            continue
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        w, h = right - left, bottom - top
        _activate(hwnd)
        if w / max(h, 1) > 2.5:         # "Do you want to reload file …?"  -> Yes
            pyautogui.click(left + 0.41 * w, top + 0.80 * h)
        else:                           # "Only open files from sources you trust"  -> Open
            pyautogui.click(left + 0.53 * w, top + 0.90 * h)
        answered += 1
        time.sleep(1.5)
    return answered


def _grab(bbox=None) -> np.ndarray:
    _, _, _, _, ImageGrab = _gui()
    return cv2.cvtColor(np.asarray(ImageGrab.grab(bbox=bbox)), cv2.COLOR_RGB2BGR)


def prepare_earth(lon: float, lat: float, range_m: float, auto_date: bool | None = True, cancelled=lambda: False) -> dict | None:
    """Everything that must be true before capturing: shading/overlays off and (optionally) a well-lit date chosen."""
    ensure_capture_settings()
    if auto_date is None:               # keep whatever date Earth is on
        return None
    if not auto_date:
        _close_slider()                 # closing Earth's time slider returns it to the newest imagery
        return None
    return choose_imagery_date(lon, lat, range_m, cancelled, log=lambda m: log.info(m))


def _close_slider() -> None:
    pyautogui, _, _, _, _ = _gui()
    pos = _find_slider()
    if pos is not None:
        pyautogui.click(pos[0] + SLIDER_CLOSE[0], pos[1] + SLIDER_CLOSE[1])
        time.sleep(3)


def _find_save_button(rect) -> tuple[int, int] | None:
    screen = _grab()
    template = cv2.imread(str(TEMPLATE))
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(result)
    if score < 0.8:
        return None
    return loc[0] + template.shape[1] // 2, loc[1] + template.shape[0] // 2


def _wait_until_settled(rect, since: float, cancelled, settle_min: float = SETTLE_MIN_S) -> None:
    """Block until Earth's 3D view stops changing (flight finished, imagery tiles loaded)."""
    left, top, right, bottom = rect
    region = (left + 330, top + 120, right - 30, bottom - 60)
    previous, calm = None, 0
    while time.time() - since < SETTLE_MAX_S:
        if cancelled():
            raise CaptureError("Cancelled.")
        time.sleep(1.2)
        frame = cv2.cvtColor(cv2.resize(_grab(region), (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.int16)
        if previous is not None and time.time() - since >= settle_min:
            diff = float(np.abs(frame - previous).mean())
            calm = calm + 1 if diff < 1.5 else 0
            if calm >= 2:
                return
        previous = frame
    log.warning("Earth's view was still changing after %ss; capturing anyway", SETTLE_MAX_S)


def _save_image(path: Path, cancelled, res: str = "8k") -> None:
    pyautogui, _, _, win32gui, _ = _gui()
    hwnd = _earth_window(win32gui)
    _activate(hwnd)
    rect = win32gui.GetWindowRect(hwnd)
    button = _find_save_button(rect)
    if button is None:
        pyautogui.hotkey("ctrl", "alt", "s")      # shows Earth's "Save Image" toolbar
        time.sleep(2)
        button = _find_save_button(rect)
    if button is None:
        raise CaptureError("Could not find Earth's “Save Image…” button. Open it once by hand (File → Save → Save Image) "
                           "and make sure the Earth window is not covered, then try again.")
    # Always (re)select the resolution: Earth forgets the setting. The dropdown sits 168 px left of the button.
    pyautogui.click(button[0] - 168, button[1])
    time.sleep(0.8)
    pyautogui.click(button[0] - 184, button[1] + RESOLUTIONS[res][1])
    time.sleep(0.8)
    pyautogui.click(*button)
    deadline = time.time() + 15
    dialog = None
    while time.time() < deadline and dialog is None:
        time.sleep(0.5)
        dialog = _window_titled(win32gui, "Save As")
    if dialog is None:
        raise CaptureError("Earth's Save As window did not open.")
    _activate(dialog)
    pyautogui.hotkey("ctrl", "a")
    pyautogui.write(str(path), interval=0.005)
    pyautogui.press("enter")
    # Wait for the file to appear and stop growing.
    start, last_size, stable = time.time(), -1, 0
    while time.time() - start < SAVE_TIMEOUT_S:
        if cancelled():
            raise CaptureError("Cancelled.")
        time.sleep(1.5)
        size = path.stat().st_size if path.exists() else -1
        busy = _window_titled(win32gui, "Saving Image") is not None
        stable = stable + 1 if size > 0 and size == last_size and not busy else 0
        last_size = size
        if stable >= 3:
            return
    raise CaptureError("Earth took too long to save the image.")


def _detect_markers(image: np.ndarray) -> dict[str, tuple[float, float]]:
    b, g, r = (image[:, :, i].astype(np.int16) for i in range(3))
    mask = ((r > 190) & (b > 190) & (g < 110)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = [(stats[i, cv2.CC_STAT_AREA], centroids[i]) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 150]
    blobs = sorted(blobs, key=lambda t: -t[0])[:12]
    h, w = image.shape[:2]
    found: dict[str, tuple[float, float]] = {}
    for area, (x, y) in blobs:
        name = ("N" if y < h / 2 else "S") + ("W" if x < w / 2 else "E")
        name = {"NW": "NW", "NE": "NE", "SW": "SW", "SE": "SE"}[name[0] + name[1]]
        if name not in found:
            found[name] = (float(x), float(y))
    return found


def georeference(image: np.ndarray, expected: list[tuple[str, float, float]]) -> geo.GeoImage:
    """Exact bounds of a capture from the pixel positions of the four corner markers."""
    seen = _detect_markers(image)
    if len(seen) < 4:
        raise CaptureError(f"Only {len(seen)} of 4 corner markers were visible in the capture "
                           "(Earth probably had not finished flying there).")
    h, w = image.shape[:2]
    lon = np.array([e[1] for e in expected])
    lat = np.array([e[2] for e in expected])
    px = np.array([seen[e[0]][0] for e in expected])
    py = np.array([seen[e[0]][1] for e in expected])
    kx, bx = np.polyfit(lon, px, 1)       # x = kx*lon + bx
    ky, by = np.polyfit(lat, py, 1)       # y = ky*lat + by (ky < 0)
    residual = max(np.abs(kx * lon + bx - px).max(), np.abs(ky * lat + by - py).max())
    ratio = abs(kx / ky) / np.cos(np.radians(lat.mean()))
    # Earth's camera is a perspective one, so marker pixels shift a little with terrain height (~30 px = ~2 m).
    if residual > MAX_MARKER_RESIDUAL_PX * (w / IMAGE_W) or not 0.9 < ratio < 1.1:
        raise CaptureError("The markers in the capture did not line up (Earth's view was not top-down or not settled).")
    west, east = (0 - bx) / kx, (w - bx) / kx
    north, south = (0 - by) / ky, (h - by) / ky
    return geo.GeoImage(west, south, east, north, w, h)


# --- choosing a well-lit imagery date -----------------------------------------------------------
HIST_TEMPLATE = Path(__file__).parent / "assets" / "hist_slider.png"
# Offsets (px) from the matched icon strip of Earth's Historical Imagery slider to its buttons.
SLIDER_ZOOM_OUT = (31, 7)
SLIDER_PREV = (31, 36)
SLIDER_NEXT = (302, 36)
SLIDER_CLOSE = (325, 7)
MAX_DATE_STEPS = 14          # how many imagery dates (newest first) are examined
GOOD_MEAN, GOOD_STD = 70, 35  # a view is "well lit" when its mean grey and contrast are at least this


def _find_slider() -> tuple[int, int] | None:
    """Locate the slider's icon strip; only the (white) icon pixels are compared, never the imagery behind them."""
    pyautogui = _gui()[0]
    pyautogui.moveTo(560, 118)                      # the slider fades out unless the mouse is over its panel
    time.sleep(0.8)
    pyautogui.moveTo(565, 122)
    time.sleep(0.4)
    x0, y0 = 392, 78                                # the slider only ever sits at the top-left of the 3D view
    hsv = cv2.cvtColor(_grab((x0, y0, x0 + 40, y0 + 120)), cv2.COLOR_BGR2HSV)
    blue = ((hsv[:, :, 0] > 95) & (hsv[:, :, 0] < 125) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 120)).astype(np.uint8)   # the clock icon
    count, _, stats, centres = cv2.connectedComponentsWithStats(blue)
    blobs = [i for i in range(1, count) if 50 <= stats[i, cv2.CC_STAT_AREA] <= 400 and abs(stats[i, 2] - stats[i, 3]) <= 4]
    if not blobs:
        return None
    best = max(blobs, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return int(centres[best][0]) + x0 - 9, int(centres[best][1]) + y0 - 8


def _checked(box: np.ndarray) -> bool:
    return int((box < 120).sum()) > 8


def ensure_capture_settings() -> None:
    """Switch off everything in Earth that darkens or covers the imagery, every time:
    Sun shading, Atmosphere haze, Grid, Overview map, Scale legend (View menu) and the Title / Legend /
    Scale / Compass overlays (Save Image → Map Options)."""
    pyautogui, _, _, win32gui, ImageGrab = _gui()
    _activate(_earth_window(win32gui))
    view_rows = {"Grid": 212, "Overview Map": 237, "Scale Legend": 261, "Atmosphere": 286, "Sun": 310}
    for name, y in view_rows.items():
        pyautogui.click(92, 40)                                 # View menu
        time.sleep(0.7)
        menu = cv2.cvtColor(np.asarray(ImageGrab.grab((0, 0, 500, 420))), cv2.COLOR_RGB2GRAY)
        if _checked(menu[y - 9:y + 9, 84:100]):
            pyautogui.click(155, y)
            log.info("Earth setting turned off: %s", name)
            time.sleep(0.5)
        else:
            pyautogui.press("esc")
            time.sleep(0.2)
    button = _find_save_button(None)
    if button is None:
        pyautogui.hotkey("ctrl", "alt", "s")
        time.sleep(2)
        button = _find_save_button(None)
    if button is None:
        return
    dropdown = (button[0] - 325, button[1])                     # "Map Options" sits left of the resolution box
    for y_off in (57, 87, 118, 149):                            # Title, Legend, Scale, Compass rows
        pyautogui.click(*dropdown)
        time.sleep(0.7)
        x0, y0 = dropdown[0] - 63, dropdown[1] - 31
        shot = cv2.cvtColor(np.asarray(ImageGrab.grab((x0, y0, x0 + 520, y0 + 240))), cv2.COLOR_RGB2GRAY)
        if _checked(shot[y_off + 31 - 9:y_off + 31 + 9, 24:40]):
            pyautogui.click(x0 + 100, y0 + y_off + 31)
            time.sleep(0.5)
        else:
            pyautogui.press("esc")
            time.sleep(0.2)


def _ensure_slider() -> tuple[int, int]:
    """Make Earth show its Historical Imagery slider and return its position."""
    pyautogui, _, _, win32gui, _ = _gui()
    _activate(_earth_window(win32gui))
    pos = _find_slider()
    for _ in range(2):                     # the menu item toggles: a second click re-opens a slider that was hidden but ticked
        if pos is not None:
            break
        pyautogui.click(92, 40)            # View menu
        time.sleep(0.8)
        pyautogui.click(155, 334)          # Historical Imagery
        time.sleep(3)
        pos = _find_slider()
    if pos is None:
        raise CaptureError("Could not open Google Earth's Historical Imagery slider (View → Historical Imagery).")
    return pos


def _view_metrics(frame: np.ndarray) -> tuple[float, float]:
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(grey.mean()), float(grey.std())


def _view_frame(rect) -> np.ndarray:
    left, top, right, bottom = rect
    return _grab((left + 640, top + 260, right - 120, bottom - 140))


def _wait_for_tiles(rect, cancelled, seconds: float = 14.0) -> np.ndarray:
    """Wait until the picture stops changing after a date change; returns the final frame."""
    deadline, previous, frame = time.time() + seconds, None, None
    time.sleep(2.5)
    while time.time() < deadline:
        if cancelled():
            raise CaptureError("Cancelled.")
        frame = cv2.cvtColor(cv2.resize(_view_frame(rect), (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.int16)
        if previous is not None and float(np.abs(frame - previous).mean()) < 0.8:
            break
        previous = frame
        time.sleep(1.2)
    return _view_frame(rect)


def choose_imagery_date(lon: float, lat: float, range_m: float, cancelled=lambda: False, log=lambda m: None) -> dict:
    """Pick the newest well-lit imagery date for a place by stepping through Earth's Historical Imagery.

    Flies to the place, then walks back through the available dates (newest first), measuring each view's
    brightness and contrast, and leaves Earth on the newest date that is well lit (or the best one found).
    """
    pyautogui, _, _, win32gui, _ = _gui()
    if time.time() - earth.camera.polled_at > 12:
        raise CaptureError("Google Earth is not linked to the app. Open the live link in Earth first.")
    hwnd = _earth_window(win32gui)
    earth.camera.set(lon, lat, range_m)
    flown = time.time()
    time.sleep(4)
    _activate(hwnd)
    rect = win32gui.GetWindowRect(hwnd)
    _wait_until_settled(rect, flown, cancelled)
    ax, ay = _ensure_slider()
    for _ in range(3):
        pyautogui.moveTo(ax + 200, ay + 25)
        time.sleep(0.4)
        pyautogui.click(ax + SLIDER_ZOOM_OUT[0], ay + SLIDER_ZOOM_OUT[1])    # show the whole time range
        time.sleep(0.4)
    earth.camera.clear()      # the view stays; stop re-flying while the dates are being stepped
    time.sleep(3)

    steps = []
    last_strip = None
    unchanged = 0
    position = 0                      # how many "older" clicks Earth is currently away from the newest date
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(MAX_DATE_STEPS):
        frame = _wait_for_tiles(rect, cancelled) if i else _view_frame(rect)
        mean, std = _view_metrics(frame)
        strip = cv2.cvtColor(_status_strip(rect), cv2.COLOR_BGR2GRAY).astype(np.int16)   # includes "Imagery Date: m/d/yyyy"
        if last_strip is not None and strip.shape == last_strip.shape and float(np.abs(strip - last_strip).mean()) < 0.5:
            unchanged += 1
            position -= 1             # that last click did not move the slider
            log(f"date step {i}: same date as before")
            if unchanged >= 2:
                break
            pyautogui.moveTo(ax + 200, ay + 25)
            time.sleep(0.5)
            pyautogui.click(ax + SLIDER_PREV[0], ay + SLIDER_PREV[1])
            position += 1
            continue
        unchanged = 0
        last_strip = strip
        cv2.imwrite(str(LIVE_DIR / f"datestep_{i:02d}.png"), _status_strip(rect))
        steps.append({"step": i, "mean": round(mean, 1), "std": round(std, 1), "good": mean >= GOOD_MEAN and std >= GOOD_STD,
                      "position": position})
        log(f"date step {i} (position {position}): brightness {mean:.0f}, contrast {std:.0f}")
        if cancelled():
            raise CaptureError("Cancelled.")
        pyautogui.moveTo(ax + 200, ay + 25)             # wake the slider, then click "older"
        time.sleep(0.5)
        pyautogui.click(ax + SLIDER_PREV[0], ay + SLIDER_PREV[1])
        position += 1
    good = [s for s in steps if s["good"]]
    chosen = good[0] if good else max(steps, key=lambda s: s["mean"] * min(s["std"], 60))
    for _ in range(max(0, position - chosen["position"])):  # go forward again to the chosen date
        pyautogui.moveTo(ax + 200, ay + 25)
        time.sleep(0.4)
        pyautogui.click(ax + SLIDER_NEXT[0], ay + SLIDER_NEXT[1])
        time.sleep(1.0)
    _wait_for_tiles(rect, cancelled)
    return {"steps": steps, "chosen": chosen["position"], "well_lit": bool(good), "brightness": chosen["mean"], "contrast": chosen["std"]}


def _paint_out_markers(image: np.ndarray) -> np.ndarray:
    b, g, r = (image[:, :, i].astype(np.int16) for i in range(3))
    mask = ((r > 170) & (b > 170) & (g < 130)).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    return cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA) if mask.any() else image


def _status_strip(rect) -> np.ndarray:
    """The bottom-right of Earth's window, which shows the imagery date, so users can see what was captured."""
    left, top, right, bottom = rect
    return _grab((left + (right - left) // 2, bottom - 36, right - 8, bottom - 12))


def capture_view(lon: float, lat: float, range_m: float, name: str, cancelled=lambda: False, date: str = "",
                 out_dir: Path | None = None, keep_camera: bool = False, settle_min: float = SETTLE_MIN_S,
                 res: str | None = None) -> LiveCapture:
    """Fly Earth to (lon, lat), save a top-down 8K image and georeference it. Blocks for about a minute.

    `out_dir` is where the capture and its manifest go (default: data/live). With `keep_camera` the camera is
    left in place afterwards, which saves several seconds when capturing neighbouring cells in a row; the
    caller must then call earth.camera.clear() when finished.
    """
    pyautogui, _, _, win32gui, _ = _gui()
    if time.time() - earth.camera.polled_at > 12:
        raise CaptureError("Google Earth is not linked to the app. Open the live link in Earth "
                           "(Farm map → Google Earth Pro → Open live link) and keep Earth open.")
    hwnd = _earth_window(win32gui)
    res = res or config.LIVE_RESOLUTION
    (expected_w, expected_h), _ = RESOLUTIONS[res]
    out_dir = out_dir or LIVE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{name}.jpg"
    last_error: Exception | None = None
    for attempt in range(2):
        path = out_dir / f"{name}_try{attempt}.jpg"   # a fresh file name each try, so Windows never asks to replace
        path.unlink(missing_ok=True)
        earth.camera.set(lon, lat, range_m, date)
        flown = time.time()
        time.sleep(4 if not keep_camera else 2)   # let Earth poll once so the farm polygons disappear before it flies
        try:
            _activate(hwnd)
            rect = win32gui.GetWindowRect(hwnd)
            _wait_until_settled(rect, flown, cancelled, settle_min)
            settled = time.time()
            strip = _status_strip(rect)
            expected = earth.camera.marker_positions()
            _save_image(path, cancelled, res)
            image = cv2.imread(str(path))
            if image is None or image.shape[:2] != (expected_h, expected_w):
                shape = None if image is None else f"{image.shape[1]}x{image.shape[0]}"
                raise CaptureError(f"Earth saved a {shape} image; expected {expected_w}x{expected_h}. Set Save Image → Resolution accordingly.")
            gi = georeference(image, expected)
            clean = _paint_out_markers(image)
            brightness = float(cv2.cvtColor(cv2.resize(clean, (512, 270)), cv2.COLOR_BGR2GRAY).mean())
            cv2.imwrite(str(final), clean, [cv2.IMWRITE_JPEG_QUALITY, 95])
            path.unlink(missing_ok=True)
            status_name = f"{name}_status.png"
            cv2.imwrite(str(out_dir / status_name), strip)
            record = LiveCapture(name, final, (gi.west, gi.south, gi.east, gi.north),
                                 datetime.now().isoformat(timespec="seconds"), range_m, status_name)
            _register(record, out_dir, brightness, res)
            log.info("capture %s: fly+settle %.0fs, save+process %.0fs", name, settled - flown, time.time() - settled)
            return record
        except CaptureError as exc:
            last_error = exc
            try:  # keep evidence of what Earth showed, for troubleshooting
                cv2.imwrite(str(LIVE_DIR / "last_failure.jpg"), cv2.resize(_grab(), None, fx=0.5, fy=0.5))
            except Exception:  # noqa: BLE001
                pass
            if "Cancelled" in str(exc) or attempt == 1:
                break
            log.warning("Capture %s failed (%s); retrying once", name, exc)
        finally:
            if not keep_camera:
                earth.camera.clear()
                time.sleep(3)   # let Earth poll the now-empty camera so the markers disappear
    raise last_error  # type: ignore[misc]


def _register(record: LiveCapture, out_dir: Path, brightness: float | None = None, res: str = "8k") -> None:
    manifest = out_dir / "manifest.json"
    entries = {c["name"]: c for c in json.loads(manifest.read_text(encoding="utf-8"))["captures"]} if manifest.exists() else {}
    entries[record.name] = {
        "name": record.name, "file": record.path.name, "bounds": list(record.bounds),
        "width": RESOLUTIONS[res][0][0], "height": RESOLUTIONS[res][0][1],
        "region": "live", "range_m": record.range_m, "captured_at": record.captured_at, "status_image": record.status_image,
        "imagery_date": f"captured {record.captured_at[:10]}", "brightness": None if brightness is None else round(brightness)}
    manifest.write_text(json.dumps({"captures": sorted(entries.values(), key=lambda c: c["name"])}, indent=1), encoding="utf-8")
