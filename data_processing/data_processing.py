from datasets import load_dataset, Dataset, concatenate_datasets
import re
import json
from typing import Dict, Any, List

SEED = 3407

# ==========
# 設定
# ==========
ALF_DATASET = "u-10bei/sft_alfworld_trajectory_dataset_v5"
DB_DATASET  = "u-10bei/dbbench_sft_dataset_react_v4"

# アップロード先（自分のHFアカウント/データセット名）
OUT_DATASET_ID = "***/***"

# DBBenchを「SQL 1行のみ」に正規化するか（推奨: True）
NORMALIZE_DBBENCH_TO_SQL_ONLY = True

# ALFWorldを「アクション1行のみ」に正規化する（必須）
NORMALIZE_ALFWORLD_TO_ACTION_ONLY = True


# ==========
# ユーティリティ
# ==========
def extract_action_from_assistant_text(text: str) -> str:
    """
    ALFWorldの assistant content から action を抽出して「アクション文字列だけ」にする。
    想定パターン:
      - "Act: go to drawer 3"
      - "Action: go to drawer 3"
      - "Think: ...\nAct: ..."
    """
    if not isinstance(text, str):
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # 1) Act:/Action: 行を優先して探す
    for ln in lines:
        m = re.match(r"^(Act|Action)\s*:\s*(.+)$", ln, flags=re.IGNORECASE)
        if m:
            return m.group(2).strip()

    # 2) どうしても無ければ、最後の行を採用（保険）
    #    ただし Think: だけで終わるような場合は空にする
    last = lines[-1] if lines else ""
    if re.match(r"^Think\s*:", last, flags=re.IGNORECASE):
        return ""
    # "go to ..." 等が入っていそうならそのまま返す
    return last.strip()


def normalize_alf_messages(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    ALFWorld: assistant発話をすべて「アクション1行」に置換。
    """
    msgs = example.get("messages", [])
    new_msgs = []
    for m in msgs:
        if m.get("role") != "assistant":
            new_msgs.append(m)
            continue
        action = extract_action_from_assistant_text(str(m.get("content", "")))
        new_msgs.append({"role": "assistant", "content": action})
    example["messages"] = new_msgs
    return example


def extract_sql_from_metadata(example: Dict[str, Any]) -> str:
    md = example.get("metadata", {}) or {}
    sql = md.get("sql", "")
    if isinstance(sql, str):
        return sql.strip()
    return ""


def normalize_db_messages_to_sql_only(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    DBBench: messagesを「質問→SQL1行」形式に縮める。
    - user: 元の質問文（テーブル名など含む）
    - assistant: metadata.sql（正解SQL）1行だけ
    """
    msgs = example.get("messages", [])
    # userの最後の質問ターンを拾う（最初の長い指示ではなく、実質問）
    user_turns = [m for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), str)]
    question = user_turns[-1]["content"].strip() if user_turns else ""

    sql = extract_sql_from_metadata(example)
    # SQLが取れない例は後でフィルタするので空でもOK
    example["messages"] = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": sql},
    ]
    return example


def is_db_weak_type(example: Dict[str, Any]) -> bool:
    """
    DBBench弱点判定:
      - aggregation- (MAX/SUM/AVG などをまとめて拾う)
      - counting
      - INSERT / UPDATE (error_recovery / retry も含む)
    """
    md = example.get("metadata", {}) or {}
    t = md.get("type", "") or ""
    if not isinstance(t, str):
        return False
    t_low = t.lower()

    # ここが v6-2 の「2倍複製」ルール
    if "aggregation" in t_low:
        return True
    if "counting" in t_low:
        return True
    if "insert" in t_low:
        return True
    if "update" in t_low:
        return True

    return False


# ==========
# ロード
# ==========
dataset_alf = load_dataset(ALF_DATASET, split="train")
dataset_db  = load_dataset(DB_DATASET,  split="train")

print(f"ALFWorld rows: {len(dataset_alf)}")
print(f"DBBench rows : {len(dataset_db)}")
print("DB columns:", dataset_db.column_names)
print("ALF columns:", dataset_alf.column_names)


# ==========
# 正規化
# ==========
if NORMALIZE_ALFWORLD_TO_ACTION_ONLY:
    dataset_alf = dataset_alf.map(normalize_alf_messages, desc="Normalize ALFWorld to action-only")

if NORMALIZE_DBBENCH_TO_SQL_ONLY:
    dataset_db = dataset_db.map(normalize_db_messages_to_sql_only, desc="Normalize DBBench to SQL-only")

    # SQLが空の行は落とす（metadata.sqlが欠けてる場合など）
    dataset_db = dataset_db.filter(
        lambda ex: isinstance(ex.get("messages", []), list)
                   and len(ex["messages"]) >= 2
                   and ex["messages"][-1].get("role") == "assistant"
                   and isinstance(ex["messages"][-1].get("content"), str)
                   and ex["messages"][-1]["content"].strip() != "",
        desc="Filter DBBench rows with empty SQL",
    )


# ==========
# v6-2: DB弱点を2倍複製
# ==========
db_weak = dataset_db.filter(is_db_weak_type, desc="Select DBBench weak types")
print(f"DB weak rows: {len(db_weak)} / {len(dataset_db)}")

db_weak_2x = concatenate_datasets([db_weak, db_weak])
dataset_db_v6_2 = concatenate_datasets([dataset_db, db_weak_2x])

print(f"DB after upweighting: {len(dataset_db_v6_2)} (added {len(db_weak_2x)})")


# ==========
# 結合 & シャッフル
# ==========
combined = concatenate_datasets([dataset_alf, dataset_db_v6_2]).shuffle(seed=SEED)

print("Combined rows:", len(combined))
print("Sample:", json.dumps(combined[0]["messages"], ensure_ascii=False, indent=2))


# ==========
# 保存 & push
# ==========
combined.save_to_disk("mixed_agent_dataset_v6_v2_local")
combined.push_to_hub(OUT_DATASET_ID)

print("✅ Done:", OUT_DATASET_ID)
