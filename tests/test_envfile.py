import os

from tinywatch_grpo.collection.envfile import load_dotenv


def test_load_dotenv_does_not_override_existing(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("TEACHER_MODEL=from-file\nTEACHER_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setenv("TEACHER_MODEL", "already-set")
    monkeypatch.delenv("TEACHER_API_KEY", raising=False)
    assert load_dotenv(path) == path
    assert os.environ["TEACHER_MODEL"] == "already-set"
    assert os.environ["TEACHER_API_KEY"] == "file-key"
