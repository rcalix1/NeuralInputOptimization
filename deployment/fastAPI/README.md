## fastAPI examples 

* link

## Usage

* conda create -n nioenv python=3.10 -y
* conda activate nioenv

* uvicorn app:app --host 127.0.0.1 --port 9000 --reload

* curl -X GET http://localhost:9000/metrics -s -H "Content-Type: application/json" | jq
* curl -X POST http://localhost:9000/optimize -s -H "Content-Type: application/json" -d '{"target_strength": 30}' | jq
