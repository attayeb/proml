"""
"""
import copy
import shap
import pandas as pd


from sklearn.metrics import (matthews_corrcoef, 
                             balanced_accuracy_score, 
                             accuracy_score, 
                             make_scorer, 
                             classification_report)
from sklearn.model_selection import KFold
import numpy as np


from sklearn.metrics import f1_score, precision_score, recall_score

from upsetplot import from_memberships

from upsetplot import UpSet
import matplotlib.pyplot as plt

def scale_df(df, scaler):
    """"""
    ret = pd.DataFrame(scaler.fit_transform(df))
    ret.index = df.index
    ret.columns = df.columns
    return ret


class TwoStepsCv:
    def __init__(self, classifier, X, y, sample=1000):
        self.model = copy.deepcopy(classifier)
        self.X = X
        self.y = y
        self.sample=sample
        self.scorer = {
            'matthews_corrcoef': make_scorer(matthews_corrcoef),
            'accuracy': make_scorer(accuracy_score),
            'balanced_accuracy': make_scorer(balanced_accuracy_score)}
        
        #self._fit()
        #self.cv = cross_validate(model, self.X, self.y, cv=5, scoring=self.scorer)
        self.score1 = self.cv()
        self.shap(sample=self.sample)
        self.score2 = self.cv2(self.feature_imporatance[:5])
        self.cv_sf_effect = None
        
        
    def _fit(self):
        model = copy.deepcopy(self.model)
        model.fit(self.X, self.y)
        self.model = model
    
    def __repr__(self):
        return "CV-MODEL-CV"       
    
    def shap(self, sample):
        self._fit()
        model = copy.deepcopy(self.model)
        model.fit(self.X, self.y)
        explainer = shap.Explainer(model.predict, 
                                   self.X.sample(sample))
        self.shap_values = explainer(self.X.sample(sample))
        
        feature_names = self.X.columns
        vals = np.abs(self.shap_values.values).mean(0)
        feature_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                  columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'],
                               ascending=False, inplace=True)
        self.feature_imporatance = feature_importance['col_name'].values
        shap.summary_plot(self.shap_values, plot_type="violin")
        
    
    def cv(self):
        print("--- Calculate Cross validation ---")
        kf = KFold(n_splits=6)
        score = []
        for t, v in kf.split(self.X, self.y):
            X_ = self.X.iloc[t]
            y_ = np.array(self.y)[t]
            yv = np.array(self.y)[v]
            Xv = np.array(self.X)[v]
            model = copy.deepcopy(self.model)
            model.fit(X_, y_)
            yp = model.predict(Xv)
            score.append({
                "mcc": matthews_corrcoef(yp, yv),
                "accuracy": accuracy_score(yp, yv),
                "f1score": f1_score(yp, yv),
                "balanced_accuracy": balanced_accuracy_score(yp, yv),
                "precision": precision_score(yp, yv),
                "recall": recall_score(yp, yv),
                "classification_report": classification_report(yp, yv)

            })
        return score
    
    def cv2(self, features):
        print("--- Calculate Cross validation ---")
        kf = KFold(n_splits=10)
        score = []
        for t, v in kf.split(self.X, self.y):
            X_ = self.X.iloc[t]
            X_ = X_.loc[:,features]
            y_ = np.array(self.y)[t]
            yv = np.array(self.y)[v]
            Xv = np.array(self.X.loc[:,features])[v]
            model = copy.deepcopy(self.model)
            model.fit(X_, y_)
            yp = model.predict(Xv)
            score.append({
                "mcc": matthews_corrcoef(yp, yv),
                "accuracy": accuracy_score(yp, yv),
                "f1score": f1_score(yp, yv),
                "balanced_accuracy": balanced_accuracy_score(yp, yv),
                "precision": precision_score(yp, yv),
                "recall": recall_score(yp, yv),
                "classification_report": classification_report(yp, yv)

            })
        return score
    
    def cv_selectedfeatures_effect(self):
        res = []
        for f in range(2,20):
            res.extend({"id": id_,
                         "number of features":f, 
                         "features": "|".join(self.feature_imporatance[:f]), 
                         "mcc":x['mcc'], "acc": x['accuracy'], 
                         "precision":x['precision'], "recall":x['recall'], 
                         "f1score":x['f1score'], "balanced_accuracy":x['balanced_accuracy']} for id_, x in enumerate(self.cv2(self.feature_imporatance[:f])))
        self.cv_sf_effect = res


class Proml():
    def __init__(self, X, Y, sample=1000):
        self.components = []
        self.X = X
        self.Y = Y
        self.sample=sample
    
    def add(self, classifier, title):
        _ret = TwoStepsCv(classifier, self.X, self.Y, sample=self.sample)
        _ret.cv_selectedfeatures_effect()
        self.components.append({"title":title, "model":_ret})

    def plot(self, number_of_features=5, threshold=0, Q={}, colors = [], metric="mcc"):
        
        for component in self.components:
            model=component['model']
            title=component['title']
            _ = copy.deepcopy(model.cv_sf_effect)
            for i in range(len(_)):
                _[i]['features'] = "."+title+"|"+_[i]['features']
            
            try:
                total = total + _
            except NameError:
                total= _
                
        
        __df = pd.DataFrame(total)
        _df_ = from_memberships(__df.features.apply(lambda x: [Q.get(i, i) for i in x.split("|")]), __df)
        mdf = __df.groupby("features").mean().reset_index()
        
        mdf = mdf[mdf['number of features'] < number_of_features]
        mdf = mdf[mdf[metric] > threshold]
        _df_ = from_memberships(mdf.features.apply(lambda x: [Q.get(i, i) for i in x.split("|")]), mdf)
        
        #_df_ = _df_.reorder_levels([22, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
        #                       11, 12, 13, 14, 15, 16, 17, 18, 20, 21], axis=0)
        
        to_plot = _df_[_df_['number of features'] < number_of_features].sort_values(metric)
        u = UpSet(to_plot, intersection_plot_elements=0, sort_by=None,
                  totals_plot_elements=1, element_size=22, 
                  sort_categories_by=None)
        u.add_catplot(value=metric, kind="strip", elements=8, s=10)

        fig = plt.figure()
        if colors == []:
            colors = iter(['red', 'blue', 'green', 'brown', 'black', 'magenta'])
        else:
            colors = iter(colors)
        for component in self.components:
            try:
                u.style_subsets(present=["."+component['title']], facecolor=next(colors), label=component['title'])
            except:
                pass
        

        u.plot(fig)
        fig.axes[2].remove()
        return fig
        