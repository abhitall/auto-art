# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# This step will highlight if Flask is missing from requirements.txt during build.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY auto_art ./auto_art
COPY setup.py .
# Add any other necessary files like README.md if it's part of the package build
# COPY README.md .

# Install the local package (auto_art)
# This makes imports like 'from auto_art.core...' work.
RUN pip install .

# Make port 5000 available to the world outside this container (Flask default port)
EXPOSE 5000

# Set the Flask app environment variable
# This tells Flask where to find the application instance.
ENV FLASK_APP=auto_art.api.app:app

# Define the command to run the application
# Using "flask run" which is suitable for development.
# For production, a more robust WSGI server like Gunicorn would be used.
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
