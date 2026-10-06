.PHONY: install install-web install-dev download-model generate checks test server

install:            ## полная установка (локальная генерация)
	pip install -r requirements.txt

install-web:        ## только веб-сервер без torch (Render Free)
	pip install -r requirements-web.txt

install-dev:        ## разработка: веб + тесты
	pip install -r requirements-dev.txt

download-model:     ## предзагрузка весов sd-turbo в кэш Hugging Face
	python scripts/download_model.py

generate:           ## прогон по заданию (5 промптов, seed 116)
	python -m src.generate --prompts prompts.txt --out outputs/

checks:             ## проверки результатов прогона
	python -m src.checks

test:               ## unit-тесты
	pytest -q

server:             ## запуск одностраничного сайта
	uvicorn src.server:app --host 0.0.0.0 --port 8000
