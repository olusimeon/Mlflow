# Importing necessary Libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib as plt
import mlflow
import mlflow.sklearn
import warnings


# function for evaluation metrics 
def eval_metrics(y_test, predictions):
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return rmse, mae, r2
    
# Splitting into train and test for calculating the accuracy
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)


# train model
def train(df):
  # Splitting into train and test for calculating the accuracy
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
  
  # Standardization technique is used
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  
  X=df.drop("dependet_variable", axis=1)
  y=df["dependet_variable"].astype(int)


  #training of model with logistic regression
  logmodel = LogisticRegression(fit_intercept = False, C = 1e12, solver='liblinear')
  logmodel.fit(X_train, y_train)
  predictions = logmodel.predict(X_test)
  
  # plotting confusion matrixs 
  confusion_matrix = pd.crosstab(y_test, predictions)
  sns.heatmap(confusion_matrix, annot=True)
  
  print(classification_report(y_test, predictions))
  print(confusion_matrix)
  print(accuracy_score(y_test, predictions))
  
  #track experiment with mlflow 
  with mlflow.start_run(experiment_id= experiment id):
    # evaluation metrics 
    (rmse, mae, r2) = eval_metrics(y_test, predictions)
    
    # logistic regression 
    logmodel = LogisticRegression(fit_intercept = False, solver='liblinear')
    logmodel.fit(X_train, y_train)
  
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(logmodel,"model")

train(X_test)

