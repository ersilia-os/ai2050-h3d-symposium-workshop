FROM python:3.10.7-bullseye

WORKDIR .

COPY . .

RUN python -m pip install --upgrade pip
RUN python -m pip install rdkit
RUN python -m pip install streamlit
RUN python -m pip install git+https://github.com/ersilia-os/ersilia-client.git

EXPOSE 8501
CMD ["streamlit", "run", "app/app.py"]