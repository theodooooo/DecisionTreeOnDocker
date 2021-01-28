FROM python:3
ADD arbre_classification.py /
RUN pip install pandas
RUN pip install sklearn

COPY breast-cancer_data.csv ./breast-cancer_data.csv
CMD [ "python", "./arbre_classification.py" ]

