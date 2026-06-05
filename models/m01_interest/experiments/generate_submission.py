"""
generate_submission.py — Task B 최종 제출 파일 생성

SOTA config (τ=0.1) 기준으로 test set 전체에 대해 Top-3 AdID를 예측하고
ad_test_answer.csv 를 출력한다.

출력 형식:
    SearchID,AdID 1,AdID 2,AdID 3

실행 (프로젝트 루트에서):
    python models/m01_interest/experiments/generate_submission.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.data.dataset import RecoDataset
from models.m01_interest import ModelConfig, MultiInterestModel
from shared.eval.predictor import score_task_b, train

# ──────────────────────────────────────────────
# SOTA config (τ=0.1)
# ──────────────────────────────────────────────
DATASET_DIR = ROOT / "../datasets"
OUTPUT_PATH = ROOT / "ad_test_answer.csv"
SEED        = 42

CONFIG = ModelConfig(
    k             = 5,
    alpha_search  = 0.01,
    alpha_click   = 0.5,
    alpha_neg     = 0.0,
    temperature   = 0.1,
    gamma         = 0.7,
    gamma_search  = 0.5,
    threshold     = 0.5,
)


def main() -> None:
    np.random.seed(SEED)

    # ── 1. 데이터 로드 ────────────────────────
    print("Loading dataset...")
    t0 = time.time()
    ds = RecoDataset(DATASET_DIR).load()
    cand_embs, cand_ids = ds.all_ad_embs()   # (17518, 384), [17518]
    test_queries = ds.test_ad_queries()       # 214 queries
    print(f"  candidates : {len(cand_ids):,}")
    print(f"  test queries: {len(test_queries):,}")
    print(f"  load: {time.time()-t0:.2f}s")

    # ── 2. 훈련 ──────────────────────────────
    print("\nTraining model...")
    print(f"  Config: {CONFIG.to_dict()}")
    t0 = time.time()
    model = MultiInterestModel(CONFIG)
    train(model, ds.training_stream())
    print(f"  training: {time.time()-t0:.2f}s")

    # ── 3. 전체 후보 스코어링 ─────────────────
    print("\nScoring test queries...")
    t0 = time.time()
    # score_task_b: {search_id -> np.ndarray(n_candidates,)}
    scores_map = score_task_b(model, test_queries, cand_embs)
    print(f"  scoring: {time.time()-t0:.2f}s")

    # ── 4. Top-3 추출 및 CSV 작성 ─────────────
    print(f"\nWriting {OUTPUT_PATH} ...")
    cand_ids_arr = np.array(cand_ids)

    rows_written = 0
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SearchID", "AdID 1", "AdID 2", "AdID 3"])

        for ev in test_queries:
            sid   = ev.search_id
            sc    = scores_map[sid]                         # (17518,)
            top3  = np.argpartition(sc, -3)[-3:]            # 상위 3개 인덱스 (순서 무관)
            top3  = top3[np.argsort(sc[top3])[::-1]]        # 점수 내림차순 정렬
            ad1, ad2, ad3 = cand_ids_arr[top3]
            writer.writerow([sid, int(ad1), int(ad2), int(ad3)])
            rows_written += 1

    print(f"  rows written : {rows_written}")
    print(f"\nDone. Submission saved to:")
    print(f"  {OUTPUT_PATH}")

    # ── 5. 샘플 미리보기 ─────────────────────
    print("\n[Preview - first 5 rows]")
    with open(OUTPUT_PATH, "r") as f:
        for i, line in enumerate(f):
            if i >= 6:
                break
            print(f"  {line.rstrip()}")


if __name__ == "__main__":
    main()
