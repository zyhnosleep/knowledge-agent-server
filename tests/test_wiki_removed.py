from app.main import app


def test_wiki_routes_and_static_mount_are_removed() -> None:
    paths = {route.path for route in app.routes}
    assert not any("/wiki" in path for path in paths)


def test_wiki_storage_model_is_not_registered() -> None:
    from app.db.session import Base

    assert "wiki_pages" not in Base.metadata.tables
