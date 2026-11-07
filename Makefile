# Воспроизводимые команды (Linux/macOS; на Windows команды те же).

PYTHON ?= python
EVENTS ?= data/events.csv
ARTIFACTS ?= artifacts

.PHONY: install train evaluate test serve recommend clean

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

train:
	$(PYTHON) -m src.cli train --events $(EVENTS) --artifacts $(ARTIFACTS)

evaluate:
	$(PYTHON) -m src.cli evaluate --artifacts $(ARTIFACTS)

test:
	$(PYTHON) -m pytest -q

serve:
	$(PYTHON) -m src.cli serve --artifacts $(ARTIFACTS) --port 8000

recommend:
	$(PYTHON) -m src.cli recommend --artifacts $(ARTIFACTS) --visitor-id 257597

clean:
	rm -rf artifacts .pytest_cache
