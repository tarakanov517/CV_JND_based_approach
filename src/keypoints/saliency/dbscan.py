import math
import numpy as np
from sklearn.cluster import DBSCAN

from src.geometry import ViewingGeometry
from src.keypoints.types import ClusterInfo

def extract_saliency_clusters(
    heatmap: np.ndarray,
    geometry: ViewingGeometry,
    threshold: float = 0.5,
    eps: float = 15.0,
    min_samples: int = 10,
    weighted_center: bool = True,
    radius_mode: str = "gyration",
    weighted_radius: bool = True,
    use_saliency_weights: bool = True
) -> list[ClusterInfo]:
    """Разбивает карту внимания на кластеры через DBSCAN и считает их центры, радиусы и веса."""
    if heatmap.max() > 1.0:
        norm_map = heatmap.astype(np.float64) / 255.0
    else:
        norm_map = heatmap.astype(np.float64)

    ys, xs = np.where(norm_map >= threshold)
    if len(xs) == 0:
        return []

    points = np.column_stack([xs, ys])
    weights = norm_map[ys, xs]

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    clusters: list[ClusterInfo] = []
    rad_per_px = geometry.rad_per_px(heatmap.shape[1])

    for k in set(labels):
        if k == -1:
            continue

        mask = (labels == k)
        cluster_pts = points[mask]
        cluster_weights = weights[mask]
        n_pts = len(cluster_pts)

        total_w = float(np.sum(cluster_weights))
        if weighted_center and total_w > 0:
            cx = float(np.sum(cluster_pts[:, 0] * cluster_weights) / total_w)
            cy = float(np.sum(cluster_pts[:, 1] * cluster_weights) / total_w)
        else:
            cx = float(np.mean(cluster_pts[:, 0]))
            cy = float(np.mean(cluster_pts[:, 1]))

        if radius_mode == "gyration":
            diff = cluster_pts - np.array([cx, cy])
            dist_sq = np.sum(diff ** 2, axis=1)
            if weighted_radius and total_w > 0:
                radius_px = float(np.sqrt(np.sum(cluster_weights * dist_sq) / total_w))
            else:
                radius_px = float(np.sqrt(np.mean(dist_sq)))
        elif radius_mode == "disk":
            radius_px = 0.5 * float(math.sqrt(n_pts / math.pi))
        elif radius_mode == "pixel":
            radius_px = 0.5
        else:
            raise ValueError(f"Неизвестный режим радиуса: {radius_mode}")

        radius_px = max(radius_px, 0.5)
        angular_size = 2.0 * radius_px * rad_per_px

        cluster_weight = total_w if use_saliency_weights else float(n_pts)

        clusters.append(
            ClusterInfo(
                center=(cx, cy),
                radius_px=radius_px,
                angular_size_rad=angular_size,
                weight=cluster_weight,
                num_pixels=n_pts,
                points=cluster_pts
            )
        )

    clusters.sort(key=lambda c: c.weight, reverse=True)
    return clusters
