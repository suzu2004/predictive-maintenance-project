FROM python:3.11-slim

WORKDIR /project

COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY outputs ./outputs

RUN chmod +x ./app/start.sh

WORKDIR /project/app

EXPOSE 8501
EXPOSE 8000
ENV PM_API_URL=http://localhost:8000

CMD ["./start.sh"]
