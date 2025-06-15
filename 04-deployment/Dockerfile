FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Copy your requirements file and your script into the image
COPY requirements.txt .
COPY starter.py .

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Default command to run your script with parameters for May 2023
CMD ["python", "starter.py", "--year", "2023", "--month", "5"]
