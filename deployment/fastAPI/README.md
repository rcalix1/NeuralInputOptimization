## fastAPI examples 

* I needed to downgrade to numpy version less than 2
* pip install "numpy<2"


## Usage

* conda create -n nioenv python=3.10 -y
* conda activate nioenv

* uvicorn app:app --host 127.0.0.1 --port 9000 --reload

* curl -X GET http://localhost:9000/metrics -s -H "Content-Type: application/json" | jq
* curl -X POST http://localhost:9000/optimize -s -H "Content-Type: application/json" -d '{"target_strength": 30}' | jq


## Potentially needed libraries

* annotated-doc==0.0.4
* annotated-types==0.7.0
* anyio==4.12.0
* click==8.3.1
* exceptiongroup==1.3.1
* fastapi==0.124.0
* filelock==3.20.0
* fsspec==2025.12.0
* h11==0.16.0
* idna==3.11
* Jinja2==3.1.6
* MarkupSafe==3.0.3
* mpmath==1.3.0
* networkx==3.4.2
  
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.3.20
nvidia-nvtx-cu12==12.8.90


* pydantic==2.12.5
* pydantic_core==2.41.5
* starlette==0.50.0
* sympy==1.14.0
* torch==2.9.1
* triton==3.5.1
* typing-inspection==0.4.2
* typing_extensions==4.15.0
* uvicorn==0.38.0
