from fastapi.testclient import TestClient

from api.rest_api import app


def test_health_and_predict_endpoints():
    client = TestClient(app)

    resp = client.get('/health')
    assert resp.status_code == 200
    assert resp.json()['status'] == 'ok'

    # Use a 5x5 board instead of 1x1
    # A 1x1 board has no legal moves (placing a stone would be suicide)
    board_5x5 = [[0] * 5 for _ in range(5)]
    payload = {'input': {'board': board_5x5, 'color': 'black'}}
    pred = client.post('/predict', json=payload)
    assert pred.status_code == 200
    assert pred.json()['output'] is not None
