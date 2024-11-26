# Install uv
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

ADD uv.lock /app/uv.lock
ADD pyproject.toml /app/pyproject.toml
ADD .python-version /app/.python-version

WORKDIR /app

RUN uv sync --frozen

# Copy the project into the image
ADD . /app

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# RUN pip install streamlit

# Sync the project into a new environment, using the frozen lockfile

ENTRYPOINT ["uv", "run", "/app/run.py"]
