# Use Ubuntu with pre-installed geospatial packages
FROM --platform=linux/amd64 ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install everything in one layer to avoid GDAL compilation issues
RUN apt-get update && apt-get install -y \
    # Python
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    # Geospatial packages from Ubuntu repos (pre-compiled)
    python3-gdal \
    python3-rasterio \
    python3-geopandas \
    python3-shapely \
    python3-fiona \
    python3-pyproj \
    python3-rtree \
    # Core scientific computing
    python3-numpy \
    python3-pandas \
    python3-matplotlib \
    python3-pil \
    # System dependencies
    gdal-bin \
    libgdal-dev \
    # For healthcheck
    curl \
    procps \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Create app user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy requirements first for better caching
COPY --chown=app:app requirements.txt /home/app/

# Install only the web framework packages (geospatial already installed via apt)
RUN pip install --user \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    streamlit==1.28.1 \
    requests==2.31.0 \
    lxml==4.9.3

# Add user pip binaries to PATH
ENV PATH="/home/app/.local/bin:$PATH"

# Copy application files
COPY --chown=app:app backend.py /home/app/
COPY --chown=app:app frontend.py /home/app/
COPY --chown=app:app run.py /home/app/
COPY --chown=app:app data/ /app/data/

# Create data directory
USER root
RUN mkdir -p /app/data && chown -R app:app /app/data

# Test that geospatial packages work
USER app
RUN python -c "import rasterio; import geopandas; import numpy; print('✅ All geospatial packages imported successfully')"

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use the simple Python runner instead of supervisord
CMD ["python", "run.py"]