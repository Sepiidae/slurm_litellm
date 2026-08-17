curl http://nodegpu040:12434/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy-key" \
  -d '{
    "model": "flux.1-dev",
    "prompt": "A cute white cat",
    "n": 1,
    "size": "512x512",
    "response_format": "b64_json"
  }'
