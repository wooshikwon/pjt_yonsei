Heart Disease UCI 데이터셋
데이터셋 설명
이 데이터셋은 미국 국립 당뇨병, 소화기 및 신장 질환 연구소(National Institute of Diabetes and Digestive and Kidney Diseases)에서 제공하는 데이터입니다. 다양한 의학적 측정치를 기반으로 환자의 심장병 발병 여부를 예측하는 데 사용되며, 이진 분류(Binary Classification) 문제의 대표적인 예제입니다. 이 설명서는 머신러닝 연구자들이 주로 사용하는 14개의 주요 속성에 대해 기술합니다.

컬럼 설명
age: 환자의 나이 (세)

sex: 환자의 성별

    1    Male
    0    Female

cp: 가슴 통증 유형 (Chest pain type)

    0    Typical Angina (전형적 협심증)
    1    Atypical Angina (비전형적 협심증)
    2    Non-anginal Pain (비협심증 통증)
    3    Asymptomatic (무증상)

trestbps: 안정 시 혈압 (mm Hg, 병원 도착 시 측정)

chol: 혈청 콜레스테롤 (mg/dl)

fbs: 공복 혈당 > 120 mg/dl 여부

    1    True
    0    False

restecg: 안정 시 심전도 결과

    0    Normal (정상)
    1    ST-T wave abnormality (ST-T파 이상)
    2    Left ventricular hypertrophy (좌심실 비대)

thalach: 최대 심박수

exang: 운동 유발성 협심증 여부 (Exercise induced angina)

    1    Yes
    0    No

oldpeak: 운동 대비 휴식 시 ST 분절 하강 정도

slope: 운동 시 ST 분절의 기울기

    0    Upsloping (상승)
    1    Flat (평탄)
    2    Downsloping (하강)

ca: 형광 투시경으로 확인된 주요 혈관의 수 (0-3)

thal: 탈라세미아 (Thalassemia)

    0    NULL (결측치에 해당할 수 있음)
    1    Normal (정상)
    2    Fixed defect (고정 결손)
    3    Reversible defect (가역 결손)

target: 심장병 발병 여부 (예측 대상 속성)

    0    No Disease (심장병 아님)
    1    Disease (심장병 맞음)