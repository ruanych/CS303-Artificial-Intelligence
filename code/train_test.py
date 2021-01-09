import argparse
import joblib
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_file_path', type=str, default='train.json')
    parser.add_argument('-i', '--test_file_path', type=str, default='testdataexample')
    args = parser.parse_args()

    data_json = json.load(open(args.train_file_path, 'r'))

    text_data = [i['data'] for i in data_json]
    text_label = [i['label'] for i in data_json]

    text_estimator = Pipeline(
        [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('linear_svm', LinearSVC()), ])
    param_grid = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'linear_svm__penalty': ('l1', 'l2'),
                  'linear_svm__loss': ('hinge', 'squared_hinge'),
                  'linear_svm__C': (1e-3, 1e-2, 1e-1, 1, 2),
                  }
    text_svm = GridSearchCV(text_estimator, param_grid, refit=True, n_jobs=-1)
    result = text_svm.fit(text_data, text_label)

    test_data = open(args.test_file_path, 'r').read()
    words = json.loads(test_data)
 
    test_results = result.predict(words)
    out = open('output.txt', 'w')

    for re in test_results:
        out.write(str(re)+"\n")
    out.close()
