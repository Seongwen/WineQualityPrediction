# 와인의 화학적 성분을 통한 와인의 품질 예측

사용한 데이터는 포르투갈 와인인 비노베르데 중 레드와인들을 다룬 데이터 셋입니다. 
  

   
데이터 출처 : https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009 </br>
분석코드 : https://www.kaggle.com/midouazerty/redwine-data-for-beginners/notebook) </br>

## 목적
- 와인의 화학적 성분 분석을 통해 와인의 품질을 예측 해본다.
- 코드의 세밀한 분석을 통해 분류 모델의 다양한 기법을 직접 탐구해보며 머신러닝 활용 능력 습득

## 기간
2021.03.04 - 2021.03.08 (4days)

## 팀구성
본인 외 3인 

## 역활
코드분석

## 사용기술
- Python
- logistic regression, knn, XGBOOST, decision tree, Naive Bayes, SVM, Random Forest Classifier

## 분석
### 데이터 분석
```데이터의 shape이 (1596, 12)로 1596개의 적은 데이터 셋.```
   
입력변수 Input variables:
1. 고정 산도 (결합산)
2. 휘발성 산도
3. 구연산
4. 잔류 설탕 (잔당)
5. 염화물
6. 유리 이산화황
7. 총 이산화황
8. 밀도
9. pH
10. 황산염
11. 알코올

</br>

출력변수 Output variable:
1. 품질 (0에서 10)
(실제 데이터에선 3에서 8까지 분포)

</br>

### 시각화
> #### 각 컬럼 구간별 히스토그램
>   - 히스토그램을 통해 각 화학성분이 어떻게 분포되어 있나 확인
>     - 균등하지 않게 분포. 대부분 유사한 성분을 보이고 있음
>     - 와인의 맛은 사람이 느끼는 주관적인 것이며, 데이터 역시 소믈리에의 개인적인 견해가 들어갔을 가능성이 높아 분류 정확도가 높지 않은 것으로 이해

<img src="https://user-images.githubusercontent.com/97740175/173321982-e8c2e5e5-2924-45fb-999e-6f61640e62b8.png" width="60%" height="60%">

------------------

> #### Quality점수 별 파이그래프
>  - 데이터 설명에 Quality는 0부터 10이라고했지만, 실제 데이터는 3부터 8까지의 값만 존재
>  - 그래프를 통해 와인 품질별로 데이터가 균등하게 분포되어 있나 확인
>    - 대부분 5등급 근처 중위권에 분포.
<img src="https://user-images.githubusercontent.com/97740175/173322993-dffbd9c9-acf7-4df7-b571-20b95e31425a.png" width="40%" height="40%">

-------------------

> #### 상관도 분석
>  - 화학적 성분과 와인 품질에 미치는 영향을  상관도와 산점도를통해 분석
>  - 황산염(sulphates), 알코올(alcohol)이 높을수록, 휘발성 산도(volatile acidity)가 낮을수록 와인 품질에 긍정적인 영향을 미침
>    - 산도와 황산도가 와인 품질에 가장 밀접한 영향을 미침
 <img src="https://user-images.githubusercontent.com/97740175/173339894-402bb702-81fb-478c-bd83-4f54701eb917.png" width="100%" height="100%">

</br>

### 모델링
- quality를 타겟 데이터로 분리하고 나머지를 x에 저장   
- GridSearch를 통해 적합한 파라미터를 탐색하여 튜닝

> #### logistic regression
> ```
> 로지스틱 회귀로는 Accuracy --> 56.875 로 결과값이 낮아 실효성이 낮게 나옴.
>```

> #### KNN
>```python
>knn=KNeighborsClassifier(n_neighbors=5,p=2)
>knn.fit(x_train,y_train)
>print(knn.score(x_train,y_train))
>print(knn.score(x_test,y_test))
>```
>0.6551724137931034   
>0.484375   
>  * ##### 튜닝
>```python
>params = {'n_neighbors':range(23, 55), 'p':[1, 2], 'weights':['distance', 'uniform']}
>knn = KNeighborsClassifier()
>grid = GridSearchCV(knn, params, n_jobs=-1)
>grid.fit(x_train, y_train)
>print(grid.best_params_)
>```
>{'n_neighbors': 47, 'p': 1, 'weights': 'distance'}
>```python
>k_range = list(range(1,55))
>scores=[]
>for k in k_range:
>    knn = KNeighborsClassifier(n_neighbors=k, p=1,weights='distance')
>    knn.fit(x_train,y_train)
>    y_pred = knn.predict(x)
>    scores.append(metrics.accuracy_score(y,y_pred))
>```
><img src="https://user-images.githubusercontent.com/97740175/173326703-e3f87106-27f3-42ad-96a2-925505fea539.png" width="50%" height="50%">
>
>```python
>train_acc = []
>test_acc = []
>
>for n in range(1,50):
>    clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=n)
>    clf.fit(x_train, y_train)
>    prediction = clf.predict(x_test)
>    train_acc.append(clf.score(x_train, y_train))
>    test_acc.append((prediction==y_test).mean())
>```
><img src="https://user-images.githubusercontent.com/97740175/173326983-41552774-8fca-4190-bef2-087c522186cc.png" width="50%" height="50%">
>




> #### XGBOOST 
>```python
>xgb_clf = xgb.XGBRFClassifier()
>xgb_clf = xgb_clf.fit(x_train,y_train)
>print(xgb_clf.score(x_train,y_train))
>print(xgb_clf.score(x_test,y_test))
>```
>0.640282131661442    
>0.625
>
>  * ##### 튜닝
>```python
>params = {'max_depth':range(9, 20, 2), 'learning_rate':[0.1], 'n_estimators':range(290, 310, 5)}
>xgb_clf = xgb.XGBRFClassifier()
>grid = GridSearchCV(xgb_clf, params, n_jobs=-1)
>grid.fit(x_train, y_train)
>print(grid.best_params_)
>print(grid.score(x_train,y_train))
>print(grid.score(x_test,y_test))
>```
>{'learning_rate': 0.1, 'max_depth': 13, 'n_estimators': 305}    
>0.8832288401253918    
>0.675
>

> #### decision tree
>```python
>model1 = DecisionTreeClassifier(random_state=1)
>model1.fit(x_train, y_train)
>accuracy5=model1.score(x_test,y_test)
>print(accuracy5*100,'%')
>```
>63.4375 %
>
>  * ##### 튜닝
>```python
>params = {'criterion':["gini"] , 'max_depth' : [15], 'min_samples_leaf':range(6, 15), 'min_samples_split':[3]}
>model1 = DecisionTreeClassifier()
>grid = GridSearchCV(model1, params, n_jobs=-1)
>grid.fit(x_train, y_train)
>print(grid.best_params_)
>print(grid.score(x_test,y_test))
>```
>{'criterion': 'gini', 'max_depth': 15, 'min_samples_leaf': 14, 'min_samples_split': 3}   
>0.60625   
>

> #### Naive Bayes
> - GaussianNB, BernoulliNB, CategoricalNB, ComplementNB, MultinomialNB 중 GaussianNB이 가장 높게 나옴
>```python
>nvclass = GaussianNB()
>nvclass.fit(x_train,y_train)
>y_pr=nvclass.predict(x_test)
>accuracy4=nvclass.score(x_test,y_test)
>print(accuracy4*100,'%')
>```
>59.375 %
>

> #### SVM
>```python
>#Import svm model
>from sklearn import svm
>#Create a svm Classifier
>clf = svm.SVC(kernel='linear') # Linear Kernel
>#Train the model using the training sets
>clf.fit(x_train, y_train)
>#Predict the response for test dataset
>y_pred = clf.predict(x_test)
>#Score/Accuracy
>print("Accuracy --> ", clf.score(x_test, y_test)*100)
>```
>Accuracy -->  61.875
>  * ##### 튜닝
>```python
>params = {'gamma': [0.001, 0.01],
>          'C': [1, 10, 100]}
>clf = svm.SVC(kernel='linear')
>grid = GridSearchCV(clf, params, n_jobs=-1)
>grid.fit(x_train, y_train)
>print(grid.best_params_)
>print(grid.score(x_test,y_test))
>```
>{'C': 100, 'gamma': 0.001}   
>0.621875
>

> #### Random Forest Classifier
>```python
>ran_class=RandomForestClassifier(random_state = 2)
>ran_class.fit(x_train,y_train)
>accuracy3=ran_class.score(x_test,y_test)
>print(accuracy3*100,'%')
>```
>71.875 %  
>  * ##### 튜닝
>    * 튜닝했을때, 개선되지 않았음.
>```python
>params = {'criterion':["entropy"] , 'min_samples_leaf':[2], 'min_samples_split':[6]}
>ran_class=RandomForestClassifier(random_state = 2)
>grid = GridSearchCV(ran_class, params, n_jobs=-1)
>grid.fit(x_train, y_train)
>print(grid.best_params_)
>print(grid.score(x_test,y_test))
>```
>{'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 6}   
>0.703125


#### Pipeline으로 최적화 모델 탐색
```python
pipe = Pipeline([('scaler', MinMaxScaler()), ('clf',KNeighborsClassifier())])

grid = GridSearchCV(pipe, params, n_jobs= -1)
grid.fit(x_train, y_train)

print(f"best model is {grid.best_estimator_}")
print(f"best score is {grid.score(x_test, y_test)}")
```

best model is Pipeline(steps=[('scaler', RobustScaler()),   
                ('clf',   
                 KNeighborsClassifier(n_neighbors=41, p=1,   
                                      weights='distance'))])   
best score is 0.715625   

  

## 결론
- 스케일러 사용시 결과가 더 좋지 않았다. 스케일러를 쓰지 않고, 튜닝을 하지않은 randomforest모델만 70% 이상의 정확도를 보였고, 다른 모델들은 낮은 수치를 보였다
- 데이터가 많지않아서 결과가 좋지 않게나왔을수있다는 피드백을 받음.
-  원래도 적었던 데이터가 튜닝을 통해 더 적은 양으로 학습 됐을수가 있음

## 회고
- 코드 분석을 위한 짧은 기간 프로젝트여서 코드 이해와 분석이외의 다른 작업을 더 해보지 못
- 가이드로 잡았던 코드에서 분류로 진행했기에 시간상의 문제로 회귀모델을 추가로 깊게 못해봄
