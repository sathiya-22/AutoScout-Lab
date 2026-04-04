from flask import Flask

def create_app():
    """
    Creates and configures the Flask application instance.
    """
    app = Flask(__name__)

    # Load configuration from config.py
    # This can be extended to load environment-specific configs
    app.config.from_pyfile('../../config.py', silent=True)

    # Basic error handling placeholder
    @app.errorhandler(404)
    def page_not_found(e):
        return {"error": "Resource not found", "message": str(e)}, 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return {"error": "Internal server error", "message": str(e)}, 500

    # Import and register blueprints
    # Blueprints should be defined in src/api/routes.py
    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    # Example of a simple root route for health check
    @app.route('/')
    def health_check():
        return "Grounded Token Data Generator API is running!", 200

    return app