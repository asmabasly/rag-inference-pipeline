# Use the official Python image from the Docker Hub
FROM python:3.10

# Set environment variable to prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Install Poetry
RUN python -m pip install --upgrade pip && \
    pip install poetry

# Set the working directory
WORKDIR /app

# Copy the poetry files to the container
COPY pyproject.toml poetry.lock /app/

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Copy the rest of the application code
COPY . /app

# Set the entry point to run your main script
ENTRYPOINT ["python", "main.py"]
