from app.sources.registry import SourceConfig, SourceRegistry, load_shanghai_registry


def test_source_registry_loads_required_sources():
    registry = load_shanghai_registry()
    assert len(registry.sources) >= 12
    source = registry.get("sh_fgw_major_projects")
    assert source.access_mode == "http"
    assert ".xlsx" in source.attachment_types
    assert registry.get("sh_planning_resources").access_mode == "http_then_playwright"
    assert len(registry.get("sh_planning_resources").base_urls) >= 3
    assert registry.get("sh_transport_commission").priority < registry.get("sh_housing_tender").priority
    assert len(registry.get("sh_econ_info_commission").base_urls) >= 3
    assert registry.get("sh_lingang_committee").official is True


def test_source_registry_rejects_non_positive_limit():
    registry = load_shanghai_registry()
    for limit in (0, -1):
        try:
            registry.select(limit=limit)
        except ValueError as exc:
            assert "positive integer" in str(exc)
        else:
            raise AssertionError(f"limit={limit} should have been rejected")


def test_source_max_pages_cover_configured_entry_urls():
    registry = load_shanghai_registry()
    for source in registry.sources:
        assert source.max_pages >= len(source.base_urls), source.id


def test_source_registry_rejects_duplicate_source_ids():
    source = SourceConfig(id="duplicate", name="One", base_urls=["https://one.gov.cn/"])
    duplicate = SourceConfig(id="duplicate", name="Two", base_urls=["https://two.gov.cn/"])

    try:
        SourceRegistry([source, duplicate])
    except ValueError as exc:
        assert "Duplicate source ids" in str(exc)
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate source ids should be rejected")
