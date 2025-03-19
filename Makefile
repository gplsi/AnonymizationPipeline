deploy:
	docker compose -f docker-compose-demo.yml up --build -d

undeploy:
	docker compose -f docker-compose-demo.yml down

stop:
	docker compose -f docker-compose-demo.yml stop

do-example:
	docker run --rm \
		-v ./example_input:/home/anonym/input \
		-v ./models:/home/anonym/models \
		-v ./output:/home/anonym/output \
		-v ./truecaser:/home/anonym/truecaser \
		anonymization \
		-m models/roberta_model_for_anonimization \
		-t huggingface \
		-i input/plain_text/frases.txt \
		-f plain \
		-a label \
		-o output/output_test.jsonl

CONFIG_FILE ?= default
do-config:
	docker run --rm \
		-v ./input:/home/anonym/input \
		-v ./models:/home/anonym/models \
		-v ./output:/home/anonym/output \
		-v ./truecaser:/home/anonym/truecaser \
		-v ./config:/home/anonym/config \
		anonymization \
		-c "config/$(CONFIG_FILE).txt"

INPUT_FILE ?= .gitignore
do-file:
	docker run --rm \
		-v ./input:/home/anonym/input \
		-v ./models:/home/anonym/models \
		-v ./output:/home/anonym/output \
		-v ./truecaser:/home/anonym/truecaser \
		-v ./config:/home/anonym/config \
		anonymization \
		-c "config/$(CONFIG_FILE).txt" \
		-i "input/$(INPUT_FILE)" \
		-o "output/$(INPUT_FILE).jsonl"

LABEL_VERSION ?= v1

run-local:
	python pipeline.py \
		-c "config/$(CONFIG_FILE).txt" \
		--truecaser "" \
		-i "input/$(INPUT_FILE)" \
		-o "output/$(INPUT_FILE).jsonl" \
		--labels "input/$(INPUT_FILE).$(LABEL_VERSION).labels" \
		--store_original \
		--aggregate_output \
		-a intelligent