# run.py

from app import create_app

app = create_app("app.config.ProductionConfig")

if __name__ == "__main__":
    # In prod, you’d run via Gunicorn: `gunicorn run:app`
    app.run(host="0.0.0.0", port=5001)
