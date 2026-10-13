from pathlib import Path
import io

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.skipif(not Path("artifacts/quick/models.pkl").exists(), reason="Modèles non entraînés")
def test_app_original_injection_and_replay():
    app = AppTest.from_file(str(Path("app.py").resolve()), default_timeout=60).run()
    assert not app.exception
    assert len(app.get("download_button")) == 2
    app.slider[0].set_value(200).run()
    assert not app.exception
    if Path("artifacts/full/models.pkl").exists():
        app.sidebar.selectbox[0].select("full").run()
        assert not app.exception
    app.sidebar.selectbox[1].select("Injection test").run()
    assert not app.exception
    app.sidebar.selectbox[1].select("Import CSV").run()
    assert not app.exception


@pytest.mark.skipif(not Path("artifacts/quick/models.pkl").exists(), reason="Modèles non entraînés")
def test_app_valid_and_invalid_import(monkeypatch):
    from streamlit.delta_generator import DeltaGenerator

    csv = pd.read_csv("artifacts/quick/series.csv").iloc[:200].to_csv(index=False).encode()
    monkeypatch.setattr(DeltaGenerator, "file_uploader", lambda *a, **k: io.BytesIO(csv))
    app = AppTest.from_file(str(Path("app.py").resolve()), default_timeout=60).run()
    app.sidebar.selectbox[1].select("Import CSV").run()
    app.checkbox[0].check().run()
    assert not app.exception
    assert len(app.get("download_button")) == 2
    monkeypatch.setattr(DeltaGenerator, "file_uploader", lambda *a, **k: io.BytesIO(b"bad\n1\n"))
    app.run()
    assert not app.exception
    assert "Colonnes requises" in app.error[0].value
