"""Debug visualization helpers shared across pipelines."""

from __future__ import annotations

import cv2
import numpy as np


def overlay_mask(
    gray: np.ndarray,
    mask255: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Return a 3-channel image: grayscale base + colored foreground overlay."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    fg = mask255 == 255
    if np.any(fg):
        vis[fg] = (0.6 * vis[fg] + 0.4 * np.array(color)).astype(np.uint8)
    return vis


def get_endpoints_intersections(skel01: np.ndarray, foreground_mask=None):
    """Return lists of (x, y) endpoints and intersections from a 0/1 skeleton."""
    from q9 import transition_count_8nbr

    h, w = skel01.shape
    if foreground_mask is None:
        foreground_mask = np.ones((h, w), dtype=bool)
    endpoints = []
    intersections = []
    for y in range(h):
        for x in range(w):
            if not foreground_mask[y, x] or skel01[y, x] != 1:
                continue
            t = transition_count_8nbr(skel01, y, x)
            if t == 2:
                endpoints.append((x, y))
            elif t in (6, 8):
                intersections.append((x, y))
    return endpoints, intersections


def visualize_feature_points_iso_style(skel01: np.ndarray, endpoints, intersections) -> np.ndarray:
    """ISO-style overlay: endpoints = red circles, intersections = red squares."""
    vis = cv2.cvtColor((skel01.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    red = (0, 0, 255)
    for x, y in endpoints:
        cv2.circle(vis, (x, y), 3, red, 1)
    for x, y in intersections:
        cv2.rectangle(vis, (x - 3, y - 3), (x + 3, y + 3), red, 1)
    return vis
