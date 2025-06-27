from fastapi.testclient import TestClient

from api.rest_api import app


def test_health_and_predict_endpoints():
    client = TestClient(app)

    resp = client.get('/health')
    assert resp.status_code == 200
    assert resp.json()['status'] == 'ok'

    # mock model predict or use default which returns None
    payload = {'input': {}}
    pred = client.post('/predict', json=payload)
    assert pred.status_code == 200
    assert 'output' in pred.json()
