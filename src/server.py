import os
import secrets
import string
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import httpx

# --- Configuration ---
CENTRAL_LITELLM_URL = os.getenv("CENTRAL_LITELLM_URL", "http://localhost:4000").rstrip("/")
CENTRAL_LITELLM_MASTER_KEY = os.getenv("CENTRAL_LITELLM_MASTER_KEY", "sk-1234")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("registration-service")

app = FastAPI(
    title="LiteLLM Isolated Team Registration Service",
    description="Registers local LiteLLM backends into dedicated Central Teams with isolated wildcard permissions."
)

# --- Schemas ---
class RegistrationRequest(BaseModel):
    username: str = Field(..., description="Username registering the local instance")
    host_ip: str = Field(..., alias="host/ip", description="Host/IP address of local LiteLLM proxy")
    port: int = Field(..., description="Port of local LiteLLM proxy")
    key: str = Field(..., description="API key (sk-...) for local LiteLLM proxy")
    model_alias: Optional[str] = Field("*", description="Model wildcard alias (default '*')")

    class Config:
        populate_by_name = True

class RegistrationResponse(BaseModel):
    status: str
    central_user_id: str
    team_id: str
    central_api_key: str
    registered_endpoint: str
    access_scope: str


def generate_random_id(prefix: str = "user_", length: int = 8) -> str:
    return f"{prefix}{''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(length))}"


@app.post(
    "/register",
    response_model=RegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a personal backend into an isolated Central Team"
)
async def register_instance(payload: RegistrationRequest):
    random_user_id = generate_random_id(prefix="user_")
    team_id = f"team-{random_user_id}"
    local_api_base = f"http://{payload.host_ip}:{payload.port}/v1"
    
    auth_headers = {
        "Authorization": f"Bearer {CENTRAL_LITELLM_MASTER_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        # STEP 1: Create a randomized User on Central LiteLLM
        logger.info(f"Creating user '{random_user_id}' on Central LiteLLM...")
        user_res = await client.post(
            f"{CENTRAL_LITELLM_URL}/user/new",
            json={
                "user_id": random_user_id,
                "user_email": f"{random_user_id}@registration.local",
                "user_role": "internal_user"
            },
            headers=auth_headers
        )
        if user_res.status_code not in (200, 201):
            raise HTTPException(status_code=user_res.status_code, detail=f"User creation failed: {user_res.text}")

        # STEP 2: Create a Dedicated Isolated Team for this specific user
        logger.info(f"Creating isolated team '{team_id}' for user '{random_user_id}'...")
        team_res = await client.post(
            f"{CENTRAL_LITELLM_URL}/team/new",
            json={
                "team_id": team_id,
                "team_alias": f"Dedicated Cluster - {payload.username}",
                "members_with_roles": [{"role": "admin", "user_id": random_user_id}]
            },
            headers=auth_headers
        )
        if team_res.status_code not in (200, 201):
            raise HTTPException(status_code=team_res.status_code, detail=f"Team creation failed: {team_res.text}")

        # STEP 3: Register the local Backend in Central LiteLLM, locked to this Team ID
        logger.info(f"Registering backend '*' -> {local_api_base} (Assigned to Team: '{team_id}')")
        model_payload = {
            "model_name": payload.model_alias,  # Wildcard '*' matching
            "litellm_params": {
                "model": "openai/*",            # Pass-through target model name intact
                "api_base": local_api_base,
                "api_key": payload.key
            },
            "model_info": {
                "user_id": random_user_id,
                "registered_by": payload.username,
                "team_id": team_id              # Enforces that ONLY members/keys of this team can route here
            }
        }

        model_res = await client.post(
            f"{CENTRAL_LITELLM_URL}/model/new",
            json=model_payload,
            headers=auth_headers
        )
        if model_res.status_code not in (200, 201):
            logger.error(f"Failed to register model backend: {model_res.text}")
            raise HTTPException(
                status_code=model_res.status_code,
                detail=f"Failed to register model backend on central LiteLLM: {model_res.text}"
            )

        # STEP 4: Generate a Central API Key bound to the Team
        logger.info(f"Generating team-scoped API key for team '{team_id}'...")
        key_res = await client.post(
            f"{CENTRAL_LITELLM_URL}/key/generate",
            json={
                "user_id": random_user_id,
                "team_id": team_id,             # Binds key access strictly to this team's backends
                "key_alias": f"key-{team_id}",
                "models": ["all-proxy-models"]  # Wildcard access within the assigned team scope
            },
            headers=auth_headers
        )
        if key_res.status_code not in (200, 201):
            raise HTTPException(status_code=key_res.status_code, detail=f"Key generation failed: {key_res.text}")
            
        central_api_key = key_res.json().get("key")

    return RegistrationResponse(
        status="success",
        central_user_id=random_user_id,
        team_id=team_id,
        central_api_key=central_api_key,
        registered_endpoint=local_api_base,
        access_scope="Isolated Team Wildcard"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
