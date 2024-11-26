import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def load_and_process_data(file_path):
    data = pd.read_csv(file_path, header=None)
    data.columns = ['var', 'skew', 'curt', 'entr', 'auth']
    return data

def visualize_data(data):
    print(data.head())
    print(data.info())
    sns.pairplot(data, hue='auth')
    plt.show()

def balance_data(data):
    target_count = data.auth.value_counts()
    nb_to_delete = target_count[0] - target_count[1]
    data = data.sample(frac=1, random_state=42).sort_values(by='auth')
    data = data[nb_to_delete:]
    print(data['auth'].value_counts())
    return data

def plot_target_distribution(data):
    plt.figure(figsize=(8, 6))
    plt.title('Distribution of Target', size=18)
    sns.countplot(x=data['auth'])
    target_count = data.auth.value_counts()
    plt.annotate(text=str(target_count[0]), xy=(-0.04, 10 + target_count[0]), size=14)
    plt.annotate(text=str(target_count[1]), xy=(0.96, 10 + target_count[1]), size=14)
    plt.ylim(0, 900)
    plt.show()

def train_model(data):
    x = data.loc[:, data.columns != 'auth']
    y = data.loc[:, data.columns == 'auth']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    scalar = StandardScaler()
    scalar.fit(x_train)
    x_train = scalar.transform(x_train)
    x_test = scalar.transform(x_test)

    clf = LogisticRegression(solver='lbfgs', random_state=42)
    clf.fit(x_train, y_train.values.ravel())

    y_pred = np.array(clf.predict(x_test))
    conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred),
                            columns=["Pred.Negative", "Pred.Positive"],
                            index=['Act.Negative', "Act.Positive"])
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = round((tn + tp) / (tn + fp + fn + tp), 4)

    print(conf_mat)
    print(f'\n Accuracy = {round(100 * accuracy, 2)}%')
    return clf, scalar

def predict_new_data(model, scalar, new_data):
    new_data = np.array(new_data, ndmin=2)
    new_data = scalar.transform(new_data)
    print(f'Prediction:  Class{model.predict(new_data)[0]}')
    print(f'Probability [0/1]:  {model.predict_proba(new_data)[0]}')
