## Fast API

* 2026
* July 8, 2026 meeting

## fastAPI examples 

* I needed to downgrade to numpy version less than 2
* pip install "numpy<2"

## Try for env 

* conda create -n nioenv python=3.10 numpy=1.26 -y
* conda activate nioenv
* pip install torch fastapi joblib  # or whatever else you need
* pip install "scikit-learn==1.6.1"
* pip install uvicorn
* a few others
  

## Usage if JSON

* conda create -n nioenv python=3.10 -y
* conda activate nioenv

* uvicorn app:app --host 127.0.0.1 --port 9000 --reload

* curl -X GET http://localhost:9000/metrics -s -H "Content-Type: application/json" | jq
* curl -X POST http://localhost:9000/optimize -s -H "Content-Type: application/json" -d '{"target_strength": 30}' | jq
* curl -X POST http://localhost:9000/NIOoptimize -s -H "Content-Type: application/json" -d '{"tgt": 127, "hmt": 1770, "prod_rt": 9010, "fta": 2320, "coke_rt": 382}' | jq

## Usage if returning a plain text variable

* conda create -n nioenv python=3.10 -y
* conda activate nioenv

* uvicorn app:app --host 127.0.0.1 --port 9000 --reload

* curl -X GET http://localhost:9000/metrics -s -H "Content-Type: application/json" | jq
* curl -X POST http://localhost:9000/optimize -s -H "Content-Type: application/json" -d '{"target_strength": 30}' | jq
* curl -X POST http://localhost:9000/NIOoptimize -s -H "Content-Type: application/json" -d '{"tgt": 127, "hmt": 1770, "prod_rt": 9010, "fta": 2320, "coke_rt": 382}'
  
* curl -X POST http://localhost:9000/NIOoptimize -s -H "Content-Type: application/json" -d '{"tgt": 127, "hmt": 1770, "prod_rt": 9010, "fta": 2320, "coke_rt": 382, "i_h2i_rate": 0, "i_pci_rate": 175, "i_ngi_rate": 0, "i_o2_volfract": 30, "i_h2_temp": 300, "i_hbtemp": 1480, "i_wind_rt": 195,  "xmin_i_h2i_rate": 0,"xmin_i_pci_rate": 0,"xmin_i_ngi_rate": 0,"xmin_i_o2_volfract": 21,"xmin_i_h2_temp": 300, "xmin_i_hbtemp": 1200, "xmin_i_wind_rt": 150 }' 









## Contact

* Ricardo A. Calix, Ph.D.
* rcalix@rcalix.com

  
