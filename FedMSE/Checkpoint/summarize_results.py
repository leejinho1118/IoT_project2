import os
import json
import numpy as np
import pandas as pd

# ==========================================
# 1. 경로 및 실험 폴더 설정 (__file__ 기준 절대 경로)
# ==========================================
CHECKPOINT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE_DIR = os.path.join(CHECKPOINT_DIR, "Results", "Update", "10")
OUTPUT_DIR = os.path.join(CHECKPOINT_DIR, "Summarized")

BASE_SUFFIX = "IID-Update_Exp6_scale_100epoch_50client_20rounds_lr1e-05_lamda10_ratio50.0_dataseed1234"

# 보고서용 실험 명칭 : 실제 결과 폴더명 매핑
EXPERIMENTS = {
    "Baseline": BASE_SUFFIX,
    "1. KMEANS3": f"KMEANS3_{BASE_SUFFIX}",
    "2. FT5": f"FT5_{BASE_SUFFIX}",
    "3. DynLam": f"DynLam_{BASE_SUFFIX}",
    "4. KMEANS3_DynLam": f"KMEANS3_DynLam_{BASE_SUFFIX}"
}

NUM_RUNS = 5  # Run_0 ~ Run_4
TARGET_FILE = "FL-IoT_0.5_hybrid_mse_avg_results.json"

# Client-1 ~ Client-10 번호순 정렬 리스트
CLIENTS = [f"NBa-Scen2-Client-{i}" for i in range(1, 11)]
SHORT_CLIENTS = [f"Client-{i}" for i in range(1, 11)]


# ==========================================
# 2. JSON Lines 파싱 및 단일 Run 통계 추출 함수
# ==========================================
def parse_single_run(file_path):
    """단일 Run의 JSON 파일을 읽어 최종 라운드 및 최고 라운드의 지표들을 반환합니다."""
    if not os.path.exists(file_path):
        return None

    rounds_client_aucs = []
    rounds_loss = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            round_key = list(data.keys())[0]
            metrics = data[round_key]

            # Client-1 ~ Client-10 순서대로 AUC 추출
            client_aucs = [metrics[c] for c in CLIENTS if c in metrics]
            if len(client_aucs) == len(CLIENTS):
                rounds_client_aucs.append(client_aucs)
                rounds_loss.append(metrics.get("global_loss", np.nan))

    if not rounds_client_aucs:
        return None

    # shape: (진행된 라운드 수, 10)
    aucs_array = np.array(rounds_client_aucs)
    loss_array = np.array(rounds_loss)
    num_rounds = len(aucs_array)

    # 1) 최종 라운드(Final Round) 데이터 (조기 종료 또는 20라운드 마지막 줄)
    final_client_aucs = aucs_array[-1]
    final_macro_auc = np.mean(final_client_aucs)
    final_loss = loss_array[-1]

    # 2) 최고 성능 라운드(Best Round) 데이터 (10개 클라이언트 평균 AUC가 가장 높은 라운드)
    macro_auc_per_round = np.mean(aucs_array, axis=1)
    best_idx = int(np.argmax(macro_auc_per_round))
    best_client_aucs = aucs_array[best_idx]
    best_macro_auc = macro_auc_per_round[best_idx]
    best_round_num = best_idx + 1

    return {
        "final_client_aucs": final_client_aucs,
        "final_macro_auc": final_macro_auc,
        "final_loss": final_loss,
        "stopped_round": num_rounds,
        "best_client_aucs": best_client_aucs,
        "best_macro_auc": best_macro_auc,
        "best_round": best_round_num,
    }


def df_to_markdown_table(df):
    """외부 tabulate 패키지 의존성 없이 DataFrame을 마크다운 표 문자열로 변환합니다."""
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(val) for val in row.values) + " |")
    return "\n".join([header_line, sep_line] + rows)


# ==========================================
# 3. 메인 집계 및 파일 저장 로직
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"데이터 탐색 경로: {RESULTS_BASE_DIR}")
    print("5개 실험(Run_0 ~ Run_4) 결과를 집계 중입니다...\n")

    overall_rows = []
    client_final_dict = {"Client": SHORT_CLIENTS + ["Average (Macro)"]}
    client_best_dict = {"Client": SHORT_CLIENTS + ["Average (Macro)"]}

    for exp_name, folder_name in EXPERIMENTS.items():
        run_results = []

        for run_idx in range(NUM_RUNS):
            file_path = os.path.join(
                RESULTS_BASE_DIR, folder_name, f"Run_{run_idx}", "AUC", TARGET_FILE
            )
            res = parse_single_run(file_path)
            if res is not None:
                run_results.append(res)
            else:
                print(f"[경고] 파일을 찾을 수 없거나 비어 있습니다: {file_path}")

        if not run_results:
            continue

        # Run_0 ~ Run_4 배열 취합
        final_macros = [r["final_macro_auc"] for r in run_results]
        best_macros = [r["best_macro_auc"] for r in run_results]
        final_losses = [r["final_loss"] for r in run_results]
        stopped_rounds = [r["stopped_round"] for r in run_results]
        best_rounds = [r["best_round"] for r in run_results]

        # shape: (num_valid_runs, 10)
        final_clients_matrix = np.array([r["final_client_aucs"] for r in run_results])
        best_clients_matrix = np.array([r["best_client_aucs"] for r in run_results])

        # [표 1] 전체 요약 행 추가
        overall_rows.append({
            "Experiment": exp_name,
            "Final AUC (Mean ± Std)": f"{np.mean(final_macros):.4f} ± {np.std(final_macros):.4f}",
            "Best AUC (Mean ± Std)": f"{np.mean(best_macros):.4f} ± {np.std(best_macros):.4f}",
            "Final Loss (Mean ± Std)": f"{np.mean(final_losses):.4f} ± {np.std(final_losses):.4f}",
            "Avg Stopped Round": f"{np.mean(stopped_rounds):.1f} (R{min(stopped_rounds)}~R{max(stopped_rounds)})",
            "Avg Best Round": f"{np.mean(best_rounds):.1f}"
        })

        # [표 2] 클라이언트별 Final AUC (Mean ± Std) 열 추가
        final_col = [
            f"{np.mean(final_clients_matrix[:, i]):.4f} ± {np.std(final_clients_matrix[:, i]):.4f}"
            for i in range(len(CLIENTS))
        ]
        final_col.append(f"{np.mean(final_macros):.4f} ± {np.std(final_macros):.4f}")
        client_final_dict[exp_name] = final_col

        # [표 3] 클라이언트별 Best AUC (Mean ± Std) 열 추가
        best_col = [
            f"{np.mean(best_clients_matrix[:, i]):.4f} ± {np.std(best_clients_matrix[:, i]):.4f}"
            for i in range(len(CLIENTS))
        ]
        best_col.append(f"{np.mean(best_macros):.4f} ± {np.std(best_macros):.4f}")
        client_best_dict[exp_name] = best_col

    if not overall_rows:
        print("집계할 데이터가 없습니다. 폴더 경로를 확인해 주세요.")
        return

    # DataFrame 생성
    df_overall = pd.DataFrame(overall_rows)
    df_client_final = pd.DataFrame(client_final_dict)
    df_client_best = pd.DataFrame(client_best_dict)

    # CSV 파일 3종 저장
    path_overall = os.path.join(OUTPUT_DIR, "1_overall_summary.csv")
    path_client_final = os.path.join(OUTPUT_DIR, "2_client_final_auc_summary.csv")
    path_client_best = os.path.join(OUTPUT_DIR, "3_client_best_auc_summary.csv")
    path_md = os.path.join(OUTPUT_DIR, "summary_report.md")

    df_overall.to_csv(path_overall, index=False, encoding="utf-8-sig")
    df_client_final.to_csv(path_client_final, index=False, encoding="utf-8-sig")
    df_client_best.to_csv(path_client_best, index=False, encoding="utf-8-sig")

    # 마크다운 통합 리포트 저장
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("# FedMSE 10-Client Non-IID 실험 최종 요약 리포트 (Run_0 ~ Run_4)\n\n")
        f.write("## 1. 전체 실험 요약 (Overall Summary - Hybrid MSEAvg)\n\n")
        f.write(df_to_markdown_table(df_overall) + "\n\n")
        f.write("## 2. 클라이언트별 최종 라운드 AUC (Final Round Client AUC)\n\n")
        f.write(df_to_markdown_table(df_client_final) + "\n\n")
        f.write("## 3. 클라이언트별 최고 라운드 AUC (Best Round Client AUC)\n\n")
        f.write(df_to_markdown_table(df_client_best) + "\n")

    # 콘솔 출력
    print("=" * 95)
    print(" [1] 전체 실험 요약 (Overall Summary - Run_0 ~ Run_4)")
    print("=" * 95)
    print(df_overall.to_string(index=False))

    print("\n" + "=" * 95)
    print(" [2] 클라이언트별 최종 라운드 AUC (Final Round AUC - Mean ± Std)")
    print("=" * 95)
    print(df_client_final.to_string(index=False))

    print("\n" + "=" * 95)
    print(" [3] 클라이언트별 최고 성능 라운드 AUC (Best Round AUC - Mean ± Std)")
    print("=" * 95)
    print(df_client_best.to_string(index=False))
    print("=" * 95)
    print(f"\n[완료] 모든 요약 파일이 '{OUTPUT_DIR}' 폴더에 저장되었습니다!")


if __name__ == "__main__":
    main()