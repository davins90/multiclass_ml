import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import *
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_auc_score

#########################################

def features_importance(model, holdout_data, features):
    shap_values = model.get_feature_importance(holdout_data, type='ShapValues')
    expected_value = shap_values[0,-1]
    shap_values = shap_values[:,:-1]
    return shap.summary_plot(shap_values, holdout_data, feature_names=features, plot_type='bar')

def auc_score(y_trues, y_preds):
    for i, y_pred in enumerate(y_preds):
        y_true = y_trues[i]
        auc = roc_auc_score(y_true, y_pred)
    return auc

def adversarial_validation(x_val,x_test,drop_columns,category_columns):
    """
    
    """
    adv_val = x_val.copy()
    adv_test = x_test.copy()
    adv_val['target'] = 1
    adv_test['target'] = 0
    adv = adv_val.append(adv_test).sample(frac=1)
    adv = adv.drop(columns=drop_columns)
    adv = adv.fillna(0)
    adv = adv.drop_duplicates()
    categorical = category_columns
    adv = adv.astype(str)
    for i in adv:
        if i in categorical:
            pass
        else:
            adv[i] = adv[i].astype(float)
    y = adv['target'].values
    X = adv.drop(columns=['target']).values
    categorical_features_indices = np.where(adv.dtypes != np.float64)[0]
    adv_X_train, adv_X_test, adv_y_train, adv_y_test = train_test_split(X, y , test_size = 0.30 , random_state = 1)
    train_data = Pool(data=adv_X_train,label=adv_y_train,cat_features=categorical_features_indices)
    test_data = Pool(data=adv_X_test,label=adv_y_test,cat_features=categorical_features_indices)
    params = {'iterations': 1000,'eval_metric': 'AUC','od_type': 'Iter','od_wait': 50}
    model = CatBoostClassifier(**params)
    _ = model.fit(train_data, eval_set=test_data, plot=False, verbose=False)
    auc = auc_score([test_data.get_label()],[model.predict_proba(test_data)[:,1]])
    print('AUC Score: ',auc)
    if auc < 0.6:
        ris = 1
        ris_str = 'NO CONCEPT DRIFT: Train and Test set are statistical similar'
        return ris, ris_str
    else:
        ris = 0
        ris_str = 'YES CONCENPT DRIFT- Check features importance: Dev and Test set are not statistical similar'
        features_importance(model, test_data, adv.columns)
        return ris, ris_str

def model_preparation(df,classe_nsp):
    """
    
    """
    classe_nsp = str(classe_nsp)
    df1 = df.copy()
    if classe_nsp == '1.0':
        df1['NSP'] = df['NSP'].mask((df['NSP'] == '2.0') | (df['NSP'] == '3.0'),'0.0')
    elif classe_nsp == '2.0':
        df1['NSP'] = df['NSP'].mask((df['NSP'] == '1.0') | (df['NSP'] == '3.0'),'0.0')
    else:
        df1['NSP'] = df['NSP'].mask((df['NSP'] == '2.0') | (df['NSP'] == '1.0'),'0.0')
    
    ## Bilanciamento classi
    df_under = df1[df1['NSP']==df1['NSP'].value_counts().index[1]]
    df_over = df1[df1['NSP']==df1['NSP'].value_counts().index[0]]
    df_over = df_over.sample(frac=df_under.shape[0]/df_over.shape[0],random_state=0)
    df2 = df_over.append(df_under).sample(frac=1)
    
    ## Definizione label
    if classe_nsp == '2.0':
        df2['NSP'] = df2['NSP'].mask(df2['NSP']=='2.0','1.0')
    elif classe_nsp == '3.0':
        df2['NSP'] = df2['NSP'].mask(df2['NSP']=='3.0','1.0')
    else:
        pass
    
    return df2, df

def data_splitting(df):
    """
    
    """
    X = df.drop(columns=['NSP'])
    y = df[['NSP']]
    print(X.shape)
    
    ##
    if X.shape[0] > 800:
        test_size_init = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = test_size_init, random_state = 1)
        parameters = {'model__depth':[3,4],'model__iterations':[500,1000]}
    else:
        test_size_init = 0.3
        X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = test_size_init, random_state = 1)
        parameters = {'model__depth':[2,3],'model__iterations':[100,200]}
    
    ##
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns
    
    ##
    adversarial_drop = []
    ris, ris_str = adversarial_validation(X_train,X_test,adversarial_drop,categorical_cols)
    if ris == 1:
        print('Adversarial validation check:', ris_str)
    else:
        print('Adversarial validation check:', ris_str)
        test_size_updated = test_size_init + 0.05
        X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = test_size_init, random_state = 1)
        ris, ris_str = adversarial_validation(X_train,X_test,adversarial_drop,categorical_cols)
        if ris == 0:
            print("Manually check dataframe")
            pass
    
    return X_train, X_test, y_train, y_test, numerical_cols, categorical_cols, parameters

def model_training(X_train, y_train, numerical_cols, categorical_cols,parameters):
    """
    
    """
    
    ##
    numerical_preprocessor = Pipeline(
                                     ("scaler", MinMaxScaler())])
    # numerical_preprocessor = Pipeline(steps=[("imputer", IterativeImputer(ExtraTreesRegressor(n_estimators=5,random_state=1,verbose=0),random_state=1,verbose=0,add_indicator=True)),
    #                                  ("scaler", MinMaxScaler())])
    categorical_preprocessor = Pipeline(steps=[("imputer", SimpleImputer(strategy='constant', fill_value='missing',verbose=0,add_indicator=True)),
                                           ("label_enc", OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[("numerical_preprocessor", numerical_preprocessor, numerical_cols),
                                               ("categorical_preprocessor", categorical_preprocessor, categorical_cols)])
    pipe_model = CatBoostClassifier(iterations=500, loss_function='Logloss', eval_metric='Accuracy',verbose=False,early_stopping_rounds=25,depth=3,random_seed=1)
    
    ##
    model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('model', pipe_model)])
    ##
    model.fit(X_train,y_train)
    model_grid = GridSearchCV(model,parameters,cv=5,scoring='accuracy',verbose=0,return_train_score=True).fit(X_train,y_train)
    print('GridSearchCV results...')
    print("Mean Train Scores: \n{}\n".format(model_grid.cv_results_['mean_train_score']))
    print("Mean CV Scores: \n{}\n".format(model_grid.cv_results_['mean_test_score']))
    print("Best Parameters: \n{}\n".format(model_grid.best_params_))
    
    return model_grid

def model_evaluation(model,X_test,y_test):
    """
    
    """
    print('Test results...')
    y_test_predict_grid = model.predict(X_test)    
    print("Model Test Accuracy:", metrics.accuracy_score(y_test, y_test_predict_grid))
    print('--------------------------------------------------')
    print('Model Test Confusion Matrix')
    cm = confusion_matrix(y_test,y_test_predict_grid,normalize='pred') 
    cmd = ConfusionMatrixDisplay(cm,display_labels=['No','Yes'])
    cmd.plot()
    
    ##
    feature_importances = model.best_estimator_.named_steps['model'].get_feature_importance()
    feature_names = X_test.columns
    lista = []
    for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    #         print('{}: {}'.format(name, score))
            lista.append(name)
    print('First ten features by importances:')
    print(lista[0:10])
    
    return model

def model_serving(df,model,class_nsp,X_test):
    """
    
    """
    class_nsp = str(class_nsp)
    df['proba_classe_'+class_nsp] = 0
    df['proba_classe_'+class_nsp] = model.predict_proba(df[X_test.columns])[:,1]
    
    joblib_file = "model_classe_{}.pkl".format(class_nsp)
    joblib.dump(model, joblib_file)

    return df