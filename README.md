MLbusiness

Реализация rest api на базе flask 

- dataset: https://www.kaggle.com/vikasukani/diabetes-data-set
- Данные о предсказании наличия диабета у пациентов. 
- model: AdaBoostClassifier

Как использовать:
1. С помощью Model_creation.ipynb обучить модель на данных diabetes-dataset.csv и сохранить в ada_model.dill
2. Используя run_server.py запустить сервер в терминале командой "python run_server.py" либо через IDE.
3. Запустив сервер, с помощью Prediction.ipynb можно обращаться к api