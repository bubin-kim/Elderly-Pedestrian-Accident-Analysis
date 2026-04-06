# ── 기본 라이브러리 ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')  # 불필요한 경고 메시지 숨김

# ── 시각화 ───────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ── 전처리 & 모델 선택 ───────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# ── 머신러닝 모델 ────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ── 평가지표 ─────────────────────────────────────────────────────────────────
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── 다중공선성 분석 (VIF) ─────────────────────────────────────────────────────
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# ── 시드(Seed) 고정 ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 한글 깨짐 방지

# ── 데이터 불러오기 ───────────────────────────────────────────────────────────
DATA = 'A_elderly_pedestrian_subset_2026-04-01_v1_EPDO_model1.csv'
df = pd.read_csv(DATA)
df.head()

# ── 데이터 타입 & 결측치 확인 ─────────────────────────────────────────────────
print('=== 결측치 현황 ===')
missing = df.isnull().sum()
print(missing[missing > 0])
print()

# ── 수치형/범주형 컬럼 파악 ───────────────────────────────────────────────────
# 문자열(object/str) 타입 컬럼 = 범주형 변수
cat_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
print(f'범주형(문자열) 컬럼 ({len(cat_cols)}개): {cat_cols}')

# ── 타겟 변수(epdo) 기술통계 확인 ────────────────────────────────────────────
# epdo = Equivalent Property Damage Only (사고 심각도 지수)
# 사망·중상·경상 사고를 동일한 기준으로 환산한 값으로 클수록 사고가 심각함
print('=== 타겟 변수(epdo) 기술통계 ===')
print(df['epdo'].describe())

print(f'\nepdo 분포: {df["epdo"].value_counts().sort_index().to_dict()}')

 #── 불필요한 컬럼 제거 ───────────────────────────────────────────────

cols_to_drop = [
    'acc_no',          # 사고 고유 ID: 예측에 무의미
    'acc_ym',          # acc_ymd에 포함된 중복 정보
    'acc_ymd',         # 날짜 원본 (연/월/시간 파생변수로 대체됨)
    'acc_tme',         # 시간 원본 (accident_hour로 대체됨)
    'bjd_cd',          # 법정동 코드: 범주 수 너무 많아 노이즈 유발
    'acc_typ_cd',      # acc_typ_label의 코드 버전 (중복)
    'acc_grd_cd',      # acc_grd_label의 코드 버전 (중복)
    'law_vio_cd',      # law_vio_label의 코드 버전 (중복)
    'rd_typ_cd',       # rd_typ_label의 코드 버전 (중복)
    'day_cd',          # day_label의 코드 버전 (중복)
    'acc_typ_map_cd',  # 사고 유형 맵핑 코드 (acc_typ_label 중복)
    # ── Leakage 위험 컬럼 (타겟 epdo를 구성하는 원재료) ──────────────────────
    # epdo = f(death_cnt, seri_cnt, sltwd_cnt, wnd_cnt) 로 계산되므로
    # 이 4개 컬럼을 입력으로 넣으면 답을 미리 알려주는 것과 같습니다
    'death_cnt',       # 사망자 수 → epdo 계산 직접 사용 (Leakage!)
    'seri_cnt',        # 중상자 수 → epdo 계산 직접 사용 (Leakage!)
    'sltwd_cnt',       # 경상자 수 → epdo 계산 직접 사용 (Leakage!)
    'wnd_cnt',         # 부상신고자 수 → epdo 계산 직접 사용 (Leakage!)
    'dobj_cnt',        # 물적 피해 건수 → epdo와 직접 연관 (Leakage!)
    'severity_binary', # epdo 기반으로 파생된 이진 변수 (Leakage!)
]

df_clean = df.drop(columns=cols_to_drop)
print(f'컬럼 제거 후 크기: {df_clean.shape}')
print(f'남은 컬럼: {df_clean.columns.tolist()}')

# ── 피처(X)와 타겟(y) 분리 ──────────────────────────────────────────
TARGET = 'epdo'                         # 예측할 목표 변수
X = df_clean.drop(columns=[TARGET])     # 입력 피처
y = df_clean[TARGET]                    # 예측 타겟

print(f'X(피처) 크기: {X.shape}')
print(f'y(타겟) 크기: {y.shape}')

# ── Train / Test 분리 ─────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 전체의 20%를 테스트용으로 분리
    random_state=SEED,  
)

print(f'훈련 세트: {X_train.shape[0]}개 샘플')
print(f'테스트 세트: {X_test.shape[0]}개 샘플')
print(f'훈련 비율: {X_train.shape[0]/len(X)*100:.1f}% | 테스트 비율: {X_test.shape[0]/len(X)*100:.1f}%')

# ── 범주형 변수 인코딩  ───────────────────────────────
# 문자열 범주형 변수를 숫자로 변환합니다.

# 범주형(문자열) 컬럼 식별
cat_cols = X_train.select_dtypes(include=['object', 'str']).columns.tolist()
print(f'인코딩 대상 범주형 컬럼 ({len(cat_cols)}개): {cat_cols}')

# LabelEncoder 딕셔너리로 각 컬럼의 인코더를 저장 
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    
    # ① train 데이터로만 fit (범주 목록 학습)
    le.fit(X_train[col])
    
    # ② train & test 각각에 transform 적용
    X_train[col] = le.transform(X_train[col])
    
    # test에 train에 없던 새로운 범주가 있을 경우 처리
    # (unseen label → 가장 빈번한 클래스의 인덱스로 대체)
    X_test[col] = X_test[col].map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    
    label_encoders[col] = le  # 인코더 저장
    print(f'  {col}: {list(le.classes_)} → {list(range(len(le.classes_)))}')

# ── 결측치 처리 ───────────────────────────────────────────────────────

imputer = SimpleImputer(strategy='median')

# ① train으로만 fit (중앙값 계산)
X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

# ② test에는 transform만 적용
X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# 결측치 처리 확인
print(f'처리 후 결측치 (train): {X_train.isnull().sum().sum()}개')
print(f'처리 후 결측치 (test): {X_test.isnull().sum().sum()}개')
print(f'\n 결측치 처리 완료 (중앙값 대체)')
print(f' 최종 피처 수: {X_train.shape[1]}개')
print(f' 최종 피처 목록: {X_train.columns.tolist()}')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Random Forest (랜덤 포레스트) 란?
#   여러 개의 결정 트리(Decision Tree)를 만들어 각각의 예측값을 평균내는 방법.
#   개별 트리들이 서로 다른 무작위 샘플/피처로 학습하여 과적합(Overfitting)에 강합니다.
#
# GridSearchCV (그리드 서치)란?
#   사용자가 지정한 하이퍼파라미터 조합을 모두 시도하고,
#   교차검증(Cross-Validation)으로 가장 좋은 조합을 찾아줍니다.
#   → 수동으로 파라미터를 바꿔가며 시험하는 수고를 자동화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print(' Random Forest GridSearchCV 시작...')

# ── 탐색할 하이퍼파라미터 그리드 정의 ─────────────────────────────────────────
rf_param_grid = {
    'n_estimators': [100, 200],      # 생성할 트리의 수 (많을수록 안정적이나 느림)
    'max_depth': [None, 10, 20],     # 트리의 최대 깊이 (None=제한 없음)
    'min_samples_split': [2, 5],     # 노드 분기에 필요한 최소 샘플 수
    'min_samples_leaf': [1, 2],      # 리프 노드의 최소 샘플 수
}

# ── GridSearchCV 설정 ─────────────────────────────────────────────────────────
rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=SEED),  # 기본 모델
    param_grid=rf_param_grid,       # 탐색할 파라미터 조합
    cv=5,                           # 5-Fold 교차검증 (데이터를 5등분해 번갈아 검증)
    scoring='neg_mean_squared_error', # MSE 기반 평가 (음수: 클수록 좋음)
    n_jobs=-1,                      # 가능한 모든 CPU 코어 사용
    verbose=1                        # 진행 상황 출력
)

# ── 학습 (train 데이터로만!) ─────────────────────────────────────────────────
rf_grid.fit(X_train, y_train)

# ── 최적 하이퍼파라미터 출력 ──────────────────────────────────────────────────
print(f'\n Random Forest 최적 파라미터: {rf_grid.best_params_}')

# 최적 모델 저장
best_rf = rf_grid.best_estimator_

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  XGBoost (익스트림 그래디언트 부스팅)란?
#   이전 트리의 오류를 다음 트리가 수정하는 방식으로 순차적으로 학습합니다.
#   Random Forest와 달리 "틀린 부분에 집중"하여 점진적으로 성능을 개선합니다.
#   속도가 빠르고 과적합 방지 기능(regularization)이 내장되어 있어 경진대회에서 인기!
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print('⚡ XGBoost GridSearchCV 시작...')

# ── 탐색할 하이퍼파라미터 그리드 정의 ─────────────────────────────────────────
xgb_param_grid = {
    'n_estimators': [100, 200],      # 부스팅 라운드 수 (트리의 수)
    'max_depth': [3, 5, 7],          # 트리의 최대 깊이 (작을수록 단순한 모델)
    'learning_rate': [0.05, 0.1],    # 학습률: 각 트리의 기여도 (작을수록 보수적)
    'subsample': [0.8, 1.0],         # 각 트리 학습 시 사용할 샘플 비율
}

# ── GridSearchCV 설정 ─────────────────────────────────────────────────────────
xgb_grid = GridSearchCV(
    estimator=XGBRegressor(
        random_state=SEED,
        verbosity=0,         # XGBoost 내부 로그 숨김
        eval_metric='rmse'   # 평가지표
    ),
    param_grid=xgb_param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# ── 학습 ─────────────────────────────────────────────────────────────────────
xgb_grid.fit(X_train, y_train)

print(f'\n XGBoost 최적 파라미터: {xgb_grid.best_params_}')

# 최적 모델 저장
best_xgb = xgb_grid.best_estimator_

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ● RMSE : 예측값과 실제값의 차이(오차)를 제곱해 평균낸 뒤 루트를 씌운 값
#    → 큰 오차에 더 큰 패널티. 낮을수록 좋음. 타겟 변수와 같은 단위(epdo 단위)
#  ● MAE : 오차의 절댓값 평균
#    → RMSE보다 이상치 영향 적음. 낮을수록 좋음.
#  ● R² : 모델이 타겟 분산의 몇 %를 설명하는지 (1.0 = 완벽한 예측)
#    → 1에 가까울수록 좋음. 0이면 평균으로만 예측하는 것과 동일.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_model(model, X_tr, y_tr, X_te, y_te, model_name):
    """모델의 Train/Test 성능 지표를 계산하고 출력합니다."""
    # 예측값 생성
    y_pred_train = model.predict(X_tr)
    y_pred_test  = model.predict(X_te)
    
    # 평가지표 계산
    metrics = {
        'Train RMSE': np.sqrt(mean_squared_error(y_tr, y_pred_train)),
        'Test  RMSE': np.sqrt(mean_squared_error(y_te, y_pred_test)),
        'Train MAE':  mean_absolute_error(y_tr, y_pred_train),
        'Test  MAE':  mean_absolute_error(y_te, y_pred_test),
        'Train R²':   r2_score(y_tr, y_pred_train),
        'Test  R²':   r2_score(y_te, y_pred_test),
    }
    
    print(f'\n{'='*40}')
    print(f'  {model_name} 평가 결과')
    print(f'{'='*40}')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}')
    
    # Train-Test 성능 차이가 크면 과적합 의심
    rmse_gap = metrics['Train RMSE'] - metrics['Test  RMSE']
    r2_gap   = metrics['Train R²']   - metrics['Test  R²']
    print(f'\n  [진단] RMSE 차이(Train-Test): {rmse_gap:.4f}')
    print(f'  [진단] R²   차이(Train-Test): {r2_gap:.4f}')
    if r2_gap > 0.1:
        print('과적합')
    else:
        print('과적합 없음')
    
    return metrics, y_pred_test

# ── Random Forest 평가 ────────────────────────────────────────────────────────
rf_metrics, rf_pred = evaluate_model(best_rf, X_train, y_train, X_test, y_test, 'Random Forest')

# ── XGBoost 평가 ─────────────────────────────────────────────────────────────
xgb_metrics, xgb_pred = evaluate_model(best_xgb, X_train, y_train, X_test, y_test, 'XGBoost')

# ── 두 모델 성능 비교 테이블 ─────────────────────────────────────────────────
comparison_df = pd.DataFrame({
    'Random Forest': rf_metrics,
    'XGBoost': xgb_metrics
}).T

print('\n 모델 성능 비교 요약')
print(comparison_df.round(4).to_string())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 상관관계(Correlation) 분석이란?
#   두 변수가 얼마나 함께 움직이는지를 -1 ~ +1 사이 값으로 나타냅니다.
#   +1 = 완전 양의 상관, -1 = 완전 음의 상관, 0 = 상관 없음
#   Pearson 상관계수가 가장 일반적이며, 선형 관계를 측정합니다.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 분석을 위한 데이터 재구성 (train 데이터 기준)
train_full = X_train.copy()
train_full['epdo'] = y_train.values

# ── epdo와의 상관관계 계산 ────────────────────────────────────────────────────
corr_with_target = train_full.corr(numeric_only=True)['epdo'].drop('epdo').sort_values(key=abs, ascending=False)

print('epdo와의 상관계수 (절댓값 기준 정렬)')
print(corr_with_target.round(4).to_string())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 다중공선성(Multicollinearity) & VIF(분산팽창인수)란?
#   여러 피처들이 서로 높은 상관관계를 가질 때 다중공선성 문제가 발생합니다.
#   이는 특히 선형 모델에서 계수 추정을 불안정하게 만들고,
#   피처 중요도 해석을 어렵게 합니다.
#
#   VIF(Variance Inflation Factor):
#     해당 피처를 나머지 피처들로 회귀했을 때 설명되는 정도를 측정
#     - VIF < 5: 다중공선성 낮음 (문제 없음)
#     - VIF 5~10: 다중공선성 중간 (주의)
#     - VIF > 10: 다중공선성 높음 (제거 고려)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# VIF 계산 (상수항 추가)
X_vif = add_constant(X_train)

vif_df = pd.DataFrame()
vif_df['Feature'] = X_vif.columns
vif_df['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

# 상수항 제거 및 정렬
vif_df = vif_df[vif_df['Feature'] != 'const'].sort_values('VIF', ascending=False).reset_index(drop=True)

print(' VIF(분산팽창인수) 분석 결과')
print(vif_df.to_string(index=False))

high_vif = vif_df[vif_df['VIF'] > 10]
print(f'\n  VIF > 10 (다중공선성 높음): {high_vif["Feature"].tolist()}')
print('   → 트리 계열 모델(RF, XGBoost)은 다중공선성에 비교적 강건하나,')
print('     선형 모델 사용 시 해당 변수 제거 또는 PCA 고려 필요')

# ── 최종 분석 요약 ────────────────────────────────────────────────────────────
print('=' * 60)
print('   최종 분석 요약 보고서')
print('=' * 60)
print(f'  데이터: {df.shape[0]}개 사고, {X_train.shape[1]}개 피처 사용')
print(f'  Train: {X_train.shape[0]}개 | Test: {X_test.shape[0]}개 (Seed={SEED})')
print()
print('  [Random Forest 최적 파라미터]')
for k, v in rf_grid.best_params_.items():
    print(f'    {k}: {v}')
print(f'  Test RMSE={rf_metrics["Test  RMSE"]:.4f} | MAE={rf_metrics["Test  MAE"]:.4f} | R²={rf_metrics["Test  R²"]:.4f}')
print()
print('  [XGBoost 최적 파라미터]')
for k, v in xgb_grid.best_params_.items():
    print(f'    {k}: {v}')
print(f'  Test RMSE={xgb_metrics["Test  RMSE"]:.4f} | MAE={xgb_metrics["Test  MAE"]:.4f} | R²={xgb_metrics["Test  R²"]:.4f}')
print()

# 최고 모델 판별 (R² 기준)
best_model_name = 'Random Forest' if rf_metrics['Test  R²'] >= xgb_metrics['Test  R²'] else 'XGBoost'
print(f'   더 나은 모델 (Test R² 기준): {best_model_name}')
print()
print('  [VIF 다중공선성 경고 피처]')
high_vif_feats = vif_df[vif_df['VIF'] > 10]['Feature'].tolist()
print(f'  {high_vif_feats if high_vif_feats else "없음 (모든 피처 VIF < 10)"}')
print()
print('  [생성된 시각화 파일]')
viz_files = [
    'viz_01_target_distribution.png',
    'viz_02_correlation_heatmap.png',
    'viz_03_correlation_with_target.png',
    'viz_04_vif.png',
    'viz_05_feature_importance.png',
    'viz_06_actual_vs_predicted.png',
    'viz_07_residuals.png',
    'viz_08_model_comparison.png',
]
for f in viz_files:
    print(f'     {f}')
print('=' * 60)
