PYTHON ?= python
ifeq ($(OS),Windows_NT)
	VENV_PY := .venv/Scripts/python.exe
else
	VENV_PY := .venv/bin/python
endif
export PYTHONPATH := src
export OMP_NUM_THREADS := 1
export MPLCONFIGDIR := .mplconfig

.PHONY: setup profile design-rep select-k train evaluate test lint format api ui run docker-build docker-up clean

setup:
	$(PYTHON) -m venv .venv
	$(VENV_PY) -m pip install -r requirements-dev.txt

profile:
	$(VENV_PY) scripts/profile_data.py

design-rep:
	$(VENV_PY) scripts/design_representation.py

select-k:
	$(VENV_PY) scripts/select_k.py

train:
	$(VENV_PY) scripts/train.py

evaluate:
	$(VENV_PY) scripts/evaluate.py

test:
	$(VENV_PY) -m pytest

lint:
	$(VENV_PY) -m ruff check src tests scripts app
	$(VENV_PY) -m black --check src tests scripts app
	$(VENV_PY) -m mypy src

format:
	$(VENV_PY) -m ruff check --fix src tests scripts app
	$(VENV_PY) -m black src tests scripts app

api:
	$(VENV_PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

ui:
	$(VENV_PY) -m streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

run:
	$(VENV_PY) scripts/serve.py

docker-build:
	docker build -t netflix-segmentation .

docker-up:
	docker compose up --build

clean:
	$(VENV_PY) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('.pytest_cache'), pathlib.Path('htmlcov'), pathlib.Path('.mypy_cache')]]"
