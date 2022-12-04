from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label, LabelSet
import pandas as pd
from pandas import DataFrame
import numpy as np
from pathlib import Path


def show_all_col_data(df: DataFrame, head_count: int=15):
    # Use set_option to change option, option_context to change just in with: block
    # pd.set_option('display.max_columns', None)
    with pd.option_context('display.max_columns', None):
        print(df.head(15))


def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    # Baseline just means predict everything in a single category and see how that shakes out?
    baseline = {}
    baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    # Test data set results
    results = {}
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    # Train data set results
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)
    train_fpr, train_tpr, _ = roc_curve(train_labels, train_probs)

    fg = figure(title="ROC curves", width=600, height=500, tools="pan, reset, save")
    # plt.figure(figsize = (8, 6))
    # plt.rcParams['font.size'] = 16
    
    # Plot both curves
    fg.line(base_fpr, base_tpr, line_width=1.5, legend_label='baseline', line_color="green")
    fg.line(model_fpr, model_tpr, line_width=1.5, legend_label='model', line_color="blue")
    fg.line(train_fpr, train_tpr, line_width=1.5, legend_label='training', line_color="red")
    # labels = LabelSet(x="False Positive Rate", y="True Positive Rate")
    fg.xaxis.axis_label = "False Positive Rate" 
    fg.yaxis.axis_label = "True Positive Rate"
    # plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    # plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    # plt.legend();
    # plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    import pdb; pdb.set_trace()
    show(fg)


def drop_missing_data(df: DataFrame, th: float) -> DataFrame:
    colnames = df.columns
    miss_colnames = []
    for colname in colnames:
        miss_pct = df[colname].isnull().sum()/len(df[colname])
        if miss_pct > th:
            miss_colnames.append(colname)
    return df.drop(columns = miss_colnames)

cur_path = Path(".")
all_paths = list(map(lambda p : p.as_posix(), list(cur_path.glob("2012.csv"))))
df = pd.concat(map(pd.read_csv, all_paths))

# Some data cleaning. This is recommended by the notebook and I am following that 
# for the sake of keeping things fast right now. Edit or adjust if you want.
# There appear to be three labels, so you could avoid the lumping step to make things 
# a bit more complicated if you want
df['_RFHLTH'] = df['_RFHLTH'].replace({2: 0})
df = df.loc[df['_RFHLTH'].isin([0, 1])].copy()
df = df.rename(columns = {'_RFHLTH': 'label'})
df['label'].value_counts()

# Get only numeric columns; others cannot be interpreted without transformation
df = df.select_dtypes('number')

# Drop columns that seem to be aliases for the good/poor health label
df = df.drop(columns = ['POORHLTH', 'PHYSHLTH', 'GENHLTH', 'PAINACT2', 
                        'QLMENTL2', 'QLSTRES2', 'QLHLTH2', 'HLTHPLN1', 'MENTHLTH'])

# Drop columns that are more than 50% missing data
df = drop_missing_data(df, 0.5)
# A quick data check to see if things look okay
show_all_col_data(df)

labels = np.array(df.pop('label'))
train_df, test_df, train_labels, test_labels = train_test_split(df, labels, stratify=labels, test_size=0.4)
# Filling missing with a mean (probably not great practice)
train_df = train_df.fillna(train_df.mean())
test_df = test_df.fillna(test_df.mean())

# Here is the adaboost ensemble. You specify what the model will be
classifier = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=80
)
classifier.fit(train_df, train_labels)

import pdb; pdb.set_trace()
# Get categorical predictions on test/valid data, as well as probabalistic
test_predicts = classifier.predict(test_df)
test_outputs = classifier.predict_proba(test_df)
test_probs = test_outputs[:,1]
rslts = confusion_matrix(test_labels, test_predicts)

# Get the data for training data as well, see how performance and training performance relate
train_predicts = classifier.predict(train_df)
train_outputs = classifier.predict_proba(train_df)
train_probs = train_outputs[:,1]

# The roc curve needs probabilities so it can see how results would change
# at different sensetivities
roc_value = roc_auc_score(test_labels, test_probs)
evaluate_model(test_predicts, test_probs, train_predicts, train_probs)


