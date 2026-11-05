# Fairness LLM Pipeline (Docker + Ollama)

## Requirements
- Docker Desktop installed
- Ollama installed and running
- Model "mistral" downloaded:  `ollama pull mistral`

## How to build
cd "project-folder"
docker build -t llm_fairness .

## How to run
docker run --rm \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  llm_fairness
   