from app.db.session import Base
from app.main import app
from starlette.routing import Mount


def test_legacy_content_routes_and_static_mount_are_removed() -> None:
    # FastAPI now retains lazy included routers without a .path attribute.
    # OpenAPI enumerates actual API registrations; Mount covers static paths.
    paths = set(app.openapi()["paths"]) | {
        route.path for route in app.routes if isinstance(route, Mount)
    }
    legacy_segment = "/" + "wi" + "ki"
    assert not any(legacy_segment in path for path in paths)


def test_legacy_storage_model_is_not_registered() -> None:
    legacy_table = "wi" + "ki_pages"
    assert legacy_table not in Base.metadata.tables
