LOCAL_TAG=$(shell date +"%Y-%m-%d-%H-%M")
# running in shell so that the date command is executed at make time only
LOCAL_IMAGE_NAME:=web-service:${LOCAL_TAG}

test:
	pytest ../tests/

quality_checks:
	isort .
	black .
	pylint --recursive=y .

build: quality_checks test
	docker build -t ${LOCAL_IMAGE_NAME} .

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash integration-test/run.sh

publish: build integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash ../scripts/publish.sh

setup:
	pipenv install --dev
	pre-commit install
