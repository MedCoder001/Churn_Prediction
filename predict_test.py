import requests

url = 'http://localhost:9696/predict'

customer_mail = 'xyz@gmail.com'

customer = {
    "gender" : "female",
    "SeniorCitizen" : 0,
    "Partner" : "yes",
    "Dependents" : "no",
    "tenure" : 12,
    "PhoneService" : "no",
    "MultipleLines" : "no_phone_service",
    "InternetService" : "dsl",
    "OnlineSecurity" : "no",
    "OnlineBackup" : "yes",
    "DeviceProtection" : "no",
    "TechSupport" : "no",
    "StreamingTV" : "no",
    "StreamingMovies" : "no",
    "Contract" : "month-to-month",
    "PaperlessBilling" : "yes",
    "PaymentMethod" : "electronic_check",
    "MonthlyCharges" : 19.7,
    "TotalCharges" : 202.25
}

response = requests.post(url, json=customer).json()
print(response)

if response['churn'] == True:
    print('send promo email to %s' % customer_mail)
else:
    print('do not send promo email to %s' % customer_mail)