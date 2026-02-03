from mot_web.app_factory import create_app

def test_health_ok():
    app = create_app()
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"

def test_index_ok():
    app = create_app()
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200

def test_video_page_ok():
    app = create_app()
    client = app.test_client()
    resp = client.get("/video/")
    assert resp.status_code == 200

def test_realtime_page_ok():
    app = create_app()
    client = app.test_client()
    resp = client.get("/realtime/")
    assert resp.status_code == 200
