import os
import secrets
import string
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import httpx

# --- Configuration ---
# Point this to your Central LiteLLM Admin Server
CENTRAL_LITELLM_URL = os.getenv("CENTRAL_LITELLM_URL", "http://localhost:4000").rstrip("/")
CENTRAL_LITELLM_MASTER_KEY = os.getenv("CENTRAL_LITELLM_MASTER_KEY", "sk-1234")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("registration-service")

app = FastAPI(
    title="LiteLLM Instance Registration Service",
    description="Registers local LiteLLM instances into a centralized LiteLLM Proxy."
)

# --- Schemas ---
class RegistrationRequest(BaseModel):
    username: str = Field(..., description="Username registering the local instance")
    host_ip: str = Field(..., alias="host/ip", description="Host/IP address of local LiteLLM proxy")
    port: int = Field(..., description="Port of local LiteLLM proxy")
    key: str = Field(..., description="API key (sk-...) for local LiteLLM proxy")
    model_name: Optional[str] = Field("local-litellm-backend", description="Model name alias on central proxy")

    class Config:
        populate_by_name = True

class RegistrationResponse(BaseModel):
    status: str
    central_user_id: str
    central_api_key: str
    registered_endpoint: str
    model_alias: str


# --- Helper Functions ---
def generate_random_id(prefix: str = "user_", length: int = 8) -> str:
    """Generates a randomized string identifier."""
    random_str = ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(length))
    return f"{prefix}{random_str}"


# --- Endpoint ---
@app.post(
    "/register",
    response_model=RegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a local LiteLLM with Central LiteLLM"
)
async def register_instance(payload: RegistrationRequest):
    """
    1. Generates a randomized user on Central LiteLLM.
    2. Generates a secret virtual key for the central user.
    3. Registers the remote host/IP + port backend into Central LiteLLM.
    4. Configures model route permissions.
    """
    random_user_id = generate_random_id(prefix="user_")
    local_api_base = f"http://{payload.host_ip}:{payload.port}/v1"
    
    auth_headers = {
        "Authorization": f"Bearer {CENTRAL_LITELLM_MASTER_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Step 1: Create a randomized user in Central LiteLLM
        user_data = {
            "user_id": random_user_id,
            "user_email": f"{random_user_id}@registration.local",
            "user_role": "internal_user"
        }
        
        logger.info(f"Creating user {random_user_id} on Central LiteLLM...")
        user_res = await client.post(
            f"{CENTRAL_LITELLM_URL}/user/new",
            json=user_data,
            headers=auth_headers
        )
        if user_res.status_code not in (200, 201):
            logger.error(f"Failed to create user: {user_res.text}")
            raise HTTPException(
                status_code=user_res.status_code,
                detail=f"Failed to create user on central LiteLLM: {user_res.text}"
            )

        # Step 2: Generate a Secret Key for that central user
        key_data = {
            "user_id": random_user_id,
            "key_alias": f"key-{random_user_id}",
            "models": [payload.model_name]  # Grant access to the target model
        }
        
        logger.info(f"Generating key for user {random_user_id}...")
        key_res = await client.post(
            f"{CENTRAL_LITELLM_URL}/key/generate",
            json=key_data,
            headers=auth_headers
        )
        if key_res.status_code not in (200, 201):
            logger.error(f"Failed to generate key: {key_res.text}")
            raise HTTPException(
                status_code=key_res.status_code,
                detail=f"Failed to generate API key on central LiteLLM: {key_res.text}"
            )
            
        central_api_key = key_res.json().get("key")

        # Step 3: Register local Host/IP and Port in Central LiteLLM as a Model Backend
        model_payload = {
            "model_name": payload.model_name,
            "litellm_params": {
                "model": "openai/custom-backend",  # standard proxy wrapper
                "api_base": local_api_base,
                "api_key": payload.key
            },
            "model_info": {
                "user_id": random_user_id,
                "registered_by": payload.username
            }
        }

        logger.info(f"Registering backend model route '{payload.model_name}' -> {local_api_base}")
        model_res = await client.post(
            f"{CENTRAL_LITELLM_URL}/model/new",
            json=model_payload,
            headers=auth_headers
        )
        if model_res.status_code not in (200, 201):
            logger.error(f"Failed to register backend model: {model_res.text}")
            raise HTTPException(
                status_code=model_res.status_code,
                detail=f"Failed to register model backend on central LiteLLM: {model_res.text}"
            )

    return RegistrationResponse(
        status="success",
        central_user_id=random_user_id,
        central_api_key=central_api_key,
        registered_endpoint=local_api_base,
        model_alias=payload.model_name
    )

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    # Listens on Port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
