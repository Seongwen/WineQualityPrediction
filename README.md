# 와인의 화학적 성분을 통한 와인의 품질 예측
<p>
사용한 데이터는 포르투갈 와인인 비노베르데 중 레드와인들을 다룬 데이터 셋입니다. 
  
</p>
<hr>
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

### 시각화
- #### 각 컬럼 구간별 히스토그램
<img src="https://user-images.githubusercontent.com/97740175/173321982-e8c2e5e5-2924-45fb-999e-6f61640e62b8.png" width="60%" height="60%">

- #### Quality를 hue(색조)로 지정해서 나타낸 컬럼별 산점도 관계분포도
<img src="https://user-images.githubusercontent.com/97740175/173322659-c4b0d4c3-9e92-4b15-bf71-bc157a1747c6.png" width="80%" height="80%">

- #### Quality점수 별 파이그래프
<img src="https://user-images.githubusercontent.com/97740175/173322993-dffbd9c9-acf7-4df7-b571-20b95e31425a.png" width="50%" height="50%">


- #### 상관도 분석
- correlation : 황산염(sulphates), 알코올(alcohol)이 높을수록, 휘발성 산도(volatile acidity)가 낮을수록 와인 품질에 긍정적인 영향을 미침

### 모델링
- quality를 타겟 데이터로 분리하고 나머지를 x에 저장
- #### logistic regression
- #### KNN
- #### XGBOOST 
- #### decision tree
- #### Naive Bayes
- #### SVM
- #### Random Forest Classifier

#### Pipeline으로 최적화 모델 탐색

## 트러블

## 회고/ 느낀점
* 빨강
  * 녹색
    * 파랑
    * 
