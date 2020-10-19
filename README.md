YBIGTA CONFERENCE Spring 2020

# XAI : Explainable AI

#### Demo URL

   [http://hamacojr.asuscomm.com/](http://hamacojr.asuscomm.com/)
   Sorry:( We currently shut down the server due to maintanence cost.

#### Getting Started

1. Install Docker

2. run `docker build -t xai:latest .`

3. run `docker run --name xai -d -p 80:80 xai:latest`

---



1. #### What is XAI?

   Explainable AI \- Clarifying the opaque nature of BlackBox model to become clear and understandable(i.e. WhiteBox model).

   

   설명 가능한 인공지능 - 머신러닝 모델의 동작 과정 내부의 블랙박스로 인한 복잡성을 해소하기 위한 시도    

   

2. #### Why XAI?

   Gain insights and confidence by understanding the learning process of Artificial Intelligence system;  
   Further improve the decision making procedure.

   

   인공지능의 학습 과정을 이해함으로써 얻을 수 있는 인사이트와 신뢰를 통해 인공지능을 통한 의사결정 과정 개선    

   

3. #### Methods in XAI

   - ##### PDP	Partial Dependence Plots

     A **partial dependence plot** visualizes whether the relationship between the target and a feature is linear, monotonic or more complex.

     It shows the marginal effect one or two features have on the predicted outcome of a machine learning model.

     

     모델의 학습 편향을 예측해서 모델의 각 피쳐와 예측 결과 간의 선형 관계를 시각화

     이를 통해 해당 피처와 예측 결과 간의 상관관계를 이해할 수 있음

     

   - ##### LIME    Local Interpretable Model-agnostic Explanations

     **LIME** is a local surrogate model, where surrogate models are trained to approximate the predictions of the underlying black box model.

     It tests what happens to the predictions when you give variations of your data into the machine learning model.

     

     데이터의 어떤 영역을 집중해서 분석했고 어떤 영역을 분류 근거로 사용했는지 알려주는 기법

     입력 데이터에 부분적으로 변화를 주어서 텍스트/이미지의 하이라이트를 찾는 기법

     

   - ##### SHAP    SHapley Additive exPlanations

     Unlike other methodologies that assume the independecy between features, **SHAP** considers the correlation between features.

     Instead of testing how individual feature affects the prediction, it computes the contribution of each feature to the prediction.

     

     기존의 피쳐 중요도 계산법은 피쳐 간의 독립성을 가정하고 있는 반면, 피쳐 간의 의존성을 고려하여 피쳐 중요도를 계산하고 그 결과를 시각화하는 기법

     

   - ##### Filter Visualization

     Throughout the forward pass of CNN model, **Filter Visualization** provides intuitive visualization that shows how each pixel data changes through each layer and filter.

     As the image is passed by the layers, It is likely that the pixels become more sparse and the resolution gets lower. It also provides how each filter act differently within the layer.

     These intuitions can be utilized to further improve the model performance.

     

     CNN 모델의 forward pass를 시각화함으로써 데이터가 각 레이어와 필터를 통과하며 어떻게 변형되는지 직관적으로 확인할 수 있음

     레이어를 지남에 따라 픽셀이 성기고 해상도가 떨어지는 경향이 있으며, 한 레이어 내에서도 필터에 따라 데이터를 인식하는 영역과 정도가 다름. 이렇게 확인한 특성을 모델 개선에 활용할 수 있음.

     

   - ##### LRP    Layer-wise Relevance Propagation

     'Relevance' in **LRP** indicates the contribution of each feature(pixel) to the final output. 

     LRP back tracks the deep learning model by back propagation, and computes the relevence up until the first layer.

     With the relevance values of each layer, we can retrieve the total contribution of each feature(pixel) to the final output.

     

     Relevance란 각 feature(pixel) 값이 최종 아웃풋에 기여하는 정도를 뜻한다. LRP는 딥러닝 모델을 backpropagation을 통해 역추적하여 각 레이어의 relevance값을 구하다 보면, 첫 인풋 레이어에 도달하고, 이때 각 feature(pixel)의 기여도를 구할 수 있다.

     

     

4. #### Datasets and Models

   - ##### Machine Learning

     - **XGBoost**

       Boston Housing Dataset (keras) : 미국 보스턴 지역의 지역별 지표와 해당 구역의 주택 가격의 중간값

   - ##### Deep Learning

     - **CNN**

       MNIST Dataset (keras) : 28*28 픽셀의 크기의 손글씨로 쓴 0부터 9까지의 숫자와 실제 값

     - **LSTM** 

       IMDB Dataset (keras) : IMDB 사이트에서 가져온 영화 리뷰(문장)와 리뷰의 감정(긍정/부정) 라벨



5. #### 웹사이트 사용법

   - ##### Vision / NLP / Machine Learning 선택

     - **Vision**
       - 분석할 숫자 그리기 (Input the number to recognize.)
       - SHAP / LRP / LIME / Filter Visualization 중 선택
         - SHAP: 예측에 영향을 미치는 피처 중요도를 수치화&시각화  
         Visualizes the contribution of each feature to the prediction.
         - LRP: 예측 결과와 예측에 중요한 역할을 한 픽셀들의 히트맵  
         Creates a heatmap that distincts important pixels.
         - LIME: 인풋 이미지에서 중요도가 높은 부분 하이라이트/마스킹 한 결과  
         Highlights the pixels that contributed most to the prediction.
         - Filter Visualization: Layer 0/1/2/3 선택, 각 레이어를 통과한 이후의 데이터의 픽셀 값 결과 확인 가능  
         Visualizes how each pixel changes throughout the CNN layers 0~3.

     - **NLP**
       - 분석할 텍스트 입력 (Input the text to predict sensitivity.)
       - SHAP / LIME 중 선택
         - SHAP: 예측에 영향을 미치는 피처 중요도를 수치화&시각화  
         Visualizes the contribution of each feature to the prediction.
         - LIME: 긍정/부정 예측 결과와 예측에 중요한 역할을 한 단어들 하이라이트한 입력 텍스트 출력  
         Highlights the words that contributed most to the sensitivity prediction.
     - **Machine Learning**
       - PDP
         - 12개의 변수 중 중요도를 확인할 변수 선택 후 결과 그래프 확인  
         Choose a feature to test the relationship with the target.



### References

[Interpretable Machine Learning by Christoph Molnar](https://christophm.github.io/interpretable-ml-book/)  
[XAI 설명 가능한 인공지능, 인공지능을 해부하다 by 안재현](http://www.kyobobook.co.kr/product/detailViewKor.laf?mallGb=KOR&ejkGb=KOR&barcode=9791158392000&orderClick=JAK)

------

Soyoung Kang강소영 (엔지니어링 15기)	Jinwoo Kaek백진우 (사이언스 15기)	Jooeun Kim김주은 (사이언스 16기)	Jungho Jung정정호 (사이언스 16기)	Seokju Jo조석주 (사이언스 16기)
