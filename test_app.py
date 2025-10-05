import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    """Test that the index page loads correctly"""
    rv = client.get('/')
    assert rv.status_code == 200

def test_health_check(client):
    """Test that the health check endpoint works"""
    rv = client.get('/health')
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert json_data['status'] == 'healthy'
    assert json_data['service'] == 'cosmos-explorer'

def test_upload_endpoint(client):
    """Test that the upload endpoint exists"""
    rv = client.post('/upload_file')
    # We expect a 400 or 500 status since no file is provided
    assert rv.status_code in [400, 500]

def test_analyze_endpoint(client):
    """Test that the analyze endpoint exists"""
    rv = client.post('/analyze_data')
    # We expect a 400 or 500 status since no data is provided
    assert rv.status_code in [400, 500]

def test_train_endpoint(client):
    """Test that the train endpoint exists"""
    rv = client.post('/train_model')
    # We expect a 400 or 500 status since no files may be uploaded
    assert rv.status_code in [200, 400, 500]

if __name__ == '__main__':
    pytest.main([__file__])