def test_import_app():
    # This should succeed if routers/services are wired correctly.
    from app.main import app  # change only if your FastAPI app is elsewhere
    assert app is not None
