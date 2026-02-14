from mot_web.app_factory import create_app
from fastapi.testclient import TestClient

def test_health_ok():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_index_ok():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200

def test_video_page_ok():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/video/")
    assert resp.status_code == 200

def test_realtime_page_ok():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/realtime/")
    assert resp.status_code == 200
