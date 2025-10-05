import os

class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-for-cosmos-explorer'
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'
    
    # Model settings
    MODEL_EPOCHS = int(os.environ.get('MODEL_EPOCHS', 50))
    MODEL_BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', 32))
    MODEL_LEARNING_RATE = float(os.environ.get('MODEL_LEARNING_RATE', 0.001))
    
    # Feature columns for the model
    FEATURE_COLUMNS = ['period', 'planet_radius', 'star_teff', 'star_radius', 'impact']
    
    # Google Cloud settings
    GOOGLE_CLOUD_PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
    
class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False
    
class TestingConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}