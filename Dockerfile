FROM python:3.7

RUN mkdir -p /stock-app

WORKDIR /stock-app

COPY ./requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./ ./

EXPOSE 80 443

CMD [ "python3", "app.py" ]

#CMD [ "app.py" ]
#gunicorn index:server --bind 0.0.0.0:8050 --timeout 120 --workers 3