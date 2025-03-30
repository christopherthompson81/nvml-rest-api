"""Script to run the NVML REST API server."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("nvml_rest_api.main:app", host="0.0.0.0", port=8000, reload=True)
