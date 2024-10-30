$(shell touch .env)
include .env
export $(shell sed 's/=.*//' .env)


core-build:
	docker compose build llm_tools-core


weaviate-start:
	docker compose up -d weaviate