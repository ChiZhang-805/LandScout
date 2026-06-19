from __future__ import annotations

import math
from collections import Counter

import numpy as np
from pydantic import BaseModel, Field

from app.geo.gazetteer import SEED_AREAS, GazetteerEntry
from app.llm.schemas import GovernmentEvent


EARTH_RADIUS_M = 6_371_000
MIN_CLUSTER_GEO_CONFIDENCE = 0.6


class CandidateArea(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    description: str
    source: str = "seed"
    radius_m: float = 5000
    evidence_event_ids: list[str] = Field(default_factory=list)


def seed_candidates() -> list[CandidateArea]:
    return [
        CandidateArea(
            id=f"seed_{idx + 1}",
            name=entry.name,
            lat=entry.lat,
            lon=entry.lon,
            description=entry.description,
            source="seed",
        )
        for idx, entry in enumerate(SEED_AREAS)
    ]


def data_driven_candidates(events: list[GovernmentEvent], radius_m: float = 5000) -> list[CandidateArea]:
    points = [
        (event.lat, event.lon, event.id, event.district)
        for event in events
        if event.evidence and event.lat is not None and event.lon is not None
        and event.geo_confidence >= MIN_CLUSTER_GEO_CONFIDENCE
    ]
    if len(points) < 3:
        return []
    try:
        from sklearn.cluster import DBSCAN
    except Exception:
        return []
    coords = np.radians(np.array([[lat, lon] for lat, lon, _, _ in points], dtype=float))
    clustering = DBSCAN(eps=radius_m / EARTH_RADIUS_M, min_samples=3, metric="haversine").fit(coords)
    candidates: list[CandidateArea] = []
    for label in sorted(set(clustering.labels_)):
        if label < 0:
            continue
        members = [points[idx] for idx, item_label in enumerate(clustering.labels_) if item_label == label]
        lat = sum(item[0] for item in members) / len(members)
        lon = sum(item[1] for item in members) / len(members)
        candidates.append(
            CandidateArea(
                id=f"cluster_{label}",
                name=cluster_candidate_name(label + 1, lat, lon, [item[3] for item in members]),
                lat=lat,
                lon=lon,
                description=f"基于 {len(members)} 条高置信事件点 DBSCAN 聚类生成",
                source="dbscan",
                radius_m=radius_m,
                evidence_event_ids=[item[2] for item in members],
            )
        )
    return candidates


def cluster_candidate_name(cluster_number: int, lat: float, lon: float, districts: list[str | None]) -> str:
    district_counts = Counter(district for district in districts if district)
    if district_counts:
        district, _ = district_counts.most_common(1)[0]
        return f"{district}高置信信号聚类 {cluster_number}"
    nearest_seed = nearest_seed_area(lat, lon)
    if nearest_seed:
        return f"{nearest_seed.name}周边信号聚类 {cluster_number}"
    return f"高置信数据聚类区域 {cluster_number}"


def nearest_seed_area(lat: float, lon: float, max_distance_m: float = 8_000) -> GazetteerEntry | None:
    best_entry: GazetteerEntry | None = None
    best_distance = max_distance_m
    for entry in SEED_AREAS:
        distance = haversine_m(lat, lon, entry.lat, entry.lon)
        if distance <= best_distance:
            best_entry = entry
            best_distance = distance
    return best_entry


def generate_candidates(events: list[GovernmentEvent]) -> list[CandidateArea]:
    candidates = seed_candidates() + data_driven_candidates(events)
    for candidate in candidates:
        candidate.evidence_event_ids = [
            event.id
            for event in events
            if event.evidence
            and event.lat is not None
            and event.lon is not None
            and haversine_m(candidate.lat, candidate.lon, event.lat, event.lon) <= candidate.radius_m
        ]
    return candidates


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
