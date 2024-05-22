FROM python:3.10.7-bullseye

WORKDIR .

COPY . .

ENV OPENAI_API_KEY=placeholder_text
ENV NVIDIA_API_KEY=placeholder_text
ENV CHEMSPACE_API_KEY=placeholder_text

RUN python -m pip install --upgrade pip
RUN python -m pip install rdkit
RUN python -m pip install streamlit
RUN python -m pip install git+https://github.com/ersilia-os/ersilia-client.git
RUN python -m pip install python-dotenv
RUN python -m pip install eosce
RUN python -m pip install lolP
RUN python -m pip install openai
RUN python -m pip install scikit-learn
RUN python -m pip install altair
RUN python -m pip install umap-learn

EXPOSE 8501
CMD ["streamlit", "run", "app/app.py"]
