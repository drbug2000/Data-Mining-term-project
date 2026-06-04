"""model_gyuchan — support-gated warm/cold CTR 모델 (gyuchan 버전).

기존 model/ 패키지를 건드리지 않고 분리해 둔 패키지. 공통 평가
(model.predictor.evaluate_task_a / evaluate_task_b_ndcg)를 그대로 사용한다.
"""

from models.m04_gated.config import GateConfig
from models.m04_gated.gated_ctr import GatedCTRModel
