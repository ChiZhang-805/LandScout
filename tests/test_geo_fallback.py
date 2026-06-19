from app.geo.geocoder import Geocoder, confidence_for_amap_level
from app.geo.gazetteer import GazetteerEntry
from app.llm.schemas import EventType, GovernmentEvent


def test_geo_fallback_uses_seed_area_without_amap_key():
    event = GovernmentEvent(
        source_id="fixture",
        source_url="fixture://x",
        event_type=EventType.INDUSTRIAL_PROJECT,
        title="张江科学城产业项目",
        address="张江科学城",
    )
    geocoded = Geocoder(amap_key="").geocode_event(event)
    assert geocoded.lat is not None
    assert geocoded.lon is not None
    assert geocoded.geo_confidence < 0.9
    assert geocoded.metadata["geocoder"] == "shanghai_gazetteer"


def test_geocoder_rejects_amap_results_outside_shanghai_bounds(monkeypatch):
    event = GovernmentEvent(
        source_id="fixture",
        source_url="fixture://x",
        event_type=EventType.INDUSTRIAL_PROJECT,
        title="张江科学城产业项目",
        address="张江科学城",
    )
    geocoder = Geocoder(amap_key="fake")
    monkeypatch.setattr(
        geocoder,
        "_geocode_amap",
        lambda query: GazetteerEntry(
            name="not shanghai",
            lat=39.9042,
            lon=116.4074,
            aliases=[],
            confidence=0.9,
            description="synthetic outside Shanghai",
        ),
    )

    geocoded = geocoder.geocode_event(event)

    assert geocoded.metadata["rejected_geocoder"] == "amap"
    assert geocoded.metadata["geocoder"] == "shanghai_gazetteer"
    assert geocoded.lat == 31.2077
    assert geocoded.lon == 121.5999


def test_amap_administrative_level_is_low_confidence():
    assert confidence_for_amap_level("门牌号") >= 0.9
    assert confidence_for_amap_level("区县") < 0.6
    assert confidence_for_amap_level("开发区") < 0.6
