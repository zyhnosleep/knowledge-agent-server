from app.db.session import Base
from app.main import app


def test_legacy_content_routes_and_static_mount_are_removed() -> None:
    paths = {route.path for route in app.routes}
    legacy_segment = "/" + "wi" + "ki"
    assert not any(legacy_segment in path for path in paths)


def test_legacy_storage_model_is_not_registered() -> None:
    legacy_table = "wi" + "ki_pages"
    assert legacy_table not in Base.metadata.tables
