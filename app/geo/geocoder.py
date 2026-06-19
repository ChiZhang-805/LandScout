from __future__ import annotations

import httpx

from app.core.config import settings
from app.geo.gazetteer import GazetteerEntry, gazetteer_lookup
from app.llm.schemas import GovernmentEvent


SHANGHAI_LAT_RANGE = (30.5, 32.0)
SHANGHAI_LON_RANGE = (120.8, 122.2)


class Geocoder:
    def __init__(self, amap_key: str | None = None) -> None:
        self.amap_key = amap_key if amap_key is not None else settings.amap_key

    def geocode_event(self, event: GovernmentEvent) -> GovernmentEvent:
        query_parts = [event.address, event.district, event.project_name, event.title]
        query = " ".join(part for part in query_parts if part)
        if self.amap_key and query:
            amap = self._geocode_amap(query)
            if amap:
                if is_within_shanghai(amap.lat, amap.lon):
                    event.lat = amap.lat
                    event.lon = amap.lon
                    event.geo_confidence = amap.confidence
                    event.metadata["geocoder"] = "amap"
                    event.metadata["geocode_description"] = amap.description
                    return event
                event.metadata["rejected_geocoder"] = "amap"
                event.metadata["rejected_geocode_description"] = amap.description
        fallback = gazetteer_lookup(query)
        if fallback:
            event.lat = fallback.lat
            event.lon = fallback.lon
            event.geo_confidence = fallback.confidence
            event.metadata["geocoder"] = "shanghai_gazetteer"
            event.metadata["geocode_description"] = fallback.description
        else:
            event.geo_confidence = 0.0
            event.metadata["geocoder"] = "unresolved"
        return event

    def geocode_many(self, events: list[GovernmentEvent]) -> list[GovernmentEvent]:
        return [self.geocode_event(event) for event in events]

    def _geocode_amap(self, query: str) -> GazetteerEntry | None:
        try:
            response = httpx.get(
                "https://restapi.amap.com/v3/geocode/geo",
                params={"key": self.amap_key, "address": f"上海市{query}", "city": "上海"},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return None
        geocodes = payload.get("geocodes") or []
        if not geocodes:
            return None
        first = geocodes[0]
        location = first.get("location", "")
        try:
            lon, lat = [float(item) for item in location.split(",", 1)]
        except Exception:
            return None
        level = first.get("level", "")
        confidence = confidence_for_amap_level(level)
        return GazetteerEntry(
            name=first.get("formatted_address") or query,
            lat=lat,
            lon=lon,
            aliases=[query],
            confidence=confidence,
            description=f"AMap geocode level={level}",
        )


def is_within_shanghai(lat: float, lon: float) -> bool:
    return SHANGHAI_LAT_RANGE[0] <= lat <= SHANGHAI_LAT_RANGE[1] and SHANGHAI_LON_RANGE[0] <= lon <= SHANGHAI_LON_RANGE[1]


def confidence_for_amap_level(level: str) -> float:
    if level in {"门牌号", "兴趣点"}:
        return 0.9
    if level in {"道路", "道路交叉路口"}:
        return 0.78
    if level in {"热点商圈", "村庄"}:
        return 0.65
    if level in {"乡镇", "开发区"}:
        return 0.55
    if level in {"区县", "市", "省", "国家"}:
        return 0.45
    return 0.6
