# Create python env
```bash
python -m venv .venv
source .venv/bin/activate
pip install poetry==1.8.4
poetry install 
```

# Clean up phoenix SQlite db 
```bash
rm -rf ~/.phoenix/
```

# Run phoenix server
```bash
python -m phoenix.server.main serve
```

# Run opentelemetry collector
```bash
cd infra/otlp_collector
docker run -p 4201:4201 -p 4200:4200 -v /home/user/abehsu/devfest_llmops/infra/otlp_collector/collector.yml:/etc/otel-collector-config.yaml -v /home/user/.config/gcloud/application_default_credentials_genie.json:/tmp/application_default_credentials.json -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/application_default_credentials.json otel/opentelemetry-collector-contrib --config=/etc/otel-collector-config.yaml
```



# Run uvicorn server
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 9000
```
swagger page: http://localhost:9001/docs