"""Dry Bean final project entry point.

The first milestone only verifies that the approved Kaggle data has the
expected structure. Model training is intentionally added in a later step.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


EXPECTED_ROWS = 13_611
EXPECTED_COLUMNS = 17
EXPECTED_CLASSES = {
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "DERMASON",
    "HOROZ",
    "SEKER",
    "SIRA",
}
DEFAULT_DATA_PATH = Path("data/raw/Dry_Bean_Dataset.arff")
FEATURE_NAMES = (
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
)
FEATURE_LABELS = (
    "面积",
    "周长",
    "长轴长度",
    "短轴长度",
    "长短轴比例",
    "偏心率",
    "凸包面积",
    "等效直径",
    "外接框占比",
    "实心度",
    "圆度",
    "紧致度",
    "形状因子1",
    "形状因子2",
    "形状因子3",
    "形状因子4",
)


@dataclass(frozen=True)
class DataSummary:
    rows: int
    columns: int
    missing_values: int
    class_counts: Counter[str]


def load_arff_summary(path: Path) -> DataSummary:
    """Read the small amount of ARFF structure needed for the data check."""
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    attributes: list[str] = []
    class_counts: Counter[str] = Counter()
    row_count = 0
    missing_values = 0
    in_data = False

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            lowered = line.lower()
            if lowered.startswith("@attribute"):
                attributes.append(line)
                continue
            if lowered == "@data":
                in_data = True
                continue
            if not in_data:
                continue

            values = [value.strip() for value in line.split(",")]
            if len(values) != len(attributes):
                raise ValueError(
                    f"Line {line_number} has {len(values)} values; "
                    f"expected {len(attributes)}."
                )
            row_count += 1
            missing_values += values.count("?")
            class_counts[values[-1]] += 1

    return DataSummary(
        rows=row_count,
        columns=len(attributes),
        missing_values=missing_values,
        class_counts=class_counts,
    )


def check_data(path: Path) -> DataSummary:
    """Validate the downloaded file against the Kaggle data contract."""
    summary = load_arff_summary(path)
    problems: list[str] = []

    if summary.rows != EXPECTED_ROWS:
        problems.append(f"rows={summary.rows}, expected {EXPECTED_ROWS}")
    if summary.columns != EXPECTED_COLUMNS:
        problems.append(
            f"columns={summary.columns}, expected {EXPECTED_COLUMNS}"
        )
    if set(summary.class_counts) != EXPECTED_CLASSES:
        problems.append(
            "classes=" + ", ".join(sorted(summary.class_counts))
        )
    if summary.missing_values != 0:
        problems.append(f"missing_values={summary.missing_values}, expected 0")

    if problems:
        raise ValueError("真实数据检查失败：" + "; ".join(problems))
    return summary


def load_labels(path: Path) -> list[str]:
    """Load only the target column needed by the majority-class baseline."""
    labels: list[str] = []
    in_data = False
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower() == "@data":
                in_data = True
                continue
            if in_data:
                labels.append(line.rsplit(",", maxsplit=1)[-1].strip())
    return labels


def load_dataset(path: Path) -> tuple[list[list[float]], list[str]]:
    """Load the 16 numeric inputs and the class label from the ARFF file."""
    features: list[list[float]] = []
    labels: list[str] = []
    attribute_count = 0
    in_data = False
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@attribute"):
                attribute_count += 1
                continue
            if line.lower() == "@data":
                in_data = True
                continue
            if not in_data:
                continue

            values = [value.strip() for value in line.split(",")]
            if len(values) != attribute_count:
                raise ValueError(
                    f"Line {line_number} has {len(values)} values; "
                    f"expected {attribute_count}."
                )
            try:
                features.append([float(value) for value in values[:-1]])
            except ValueError as error:
                raise ValueError(
                    f"Line {line_number} contains a non-numeric input."
                ) from error
            labels.append(values[-1])
    return features, labels


def majority_label(labels: list[str]) -> str:
    """Return the most frequent label, with a stable alphabetical tie-break."""
    if not labels:
        raise ValueError("没有类别标签，无法建立基线。")
    counts = Counter(labels)
    return min(counts, key=lambda label: (-counts[label], label))


def run_baseline(path: Path) -> tuple[str, int, int, float, float]:
    """Evaluate a majority-class baseline on a fixed stratified holdout."""
    check_data(path)
    labels = load_labels(path)
    train_labels, test_labels = train_test_split(
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    prediction = majority_label(train_labels)
    predictions = [prediction] * len(test_labels)
    accuracy = accuracy_score(test_labels, predictions)
    macro_f1 = f1_score(test_labels, predictions, average="macro", zero_division=0)
    return prediction, len(train_labels), len(test_labels), accuracy, macro_f1


def run_candidate(path: Path) -> tuple[int, int, float, float, int]:
    """Evaluate standardized logistic regression on the fixed holdout."""
    check_data(path)
    features, labels = load_dataset(path)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=42),
    )
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)
    macro_f1 = f1_score(test_y, predictions, average="macro", zero_division=0)
    predicted_class_count = len(set(predictions))
    return len(train_y), len(test_y), accuracy, macro_f1, predicted_class_count


def analyze_candidate_errors(path: Path) -> dict[str, object]:
    """Return reproducible per-class scores and one real holdout error."""
    check_data(path)
    features, labels = load_dataset(path)
    indices = list(range(len(labels)))
    train_i, test_i = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    train_x = [features[index] for index in train_i]
    test_x = [features[index] for index in test_i]
    train_y = [labels[index] for index in train_i]
    test_y = [labels[index] for index in test_i]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=42),
    )
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    report = classification_report(
        test_y,
        predictions,
        labels=sorted(EXPECTED_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    confusions = Counter(
        (actual, predicted)
        for actual, predicted in zip(test_y, predictions)
        if actual != predicted
    )
    most_common_pair, pair_count = confusions.most_common(1)[0]
    actual_label, predicted_label = most_common_pair
    error_position = next(
        position
        for position, (actual, predicted) in enumerate(zip(test_y, predictions))
        if actual == actual_label and predicted == predicted_label
    )
    original_index = test_i[error_position]
    row = test_x[error_position]

    return {
        "errors": sum(confusions.values()),
        "most_common_actual": actual_label,
        "most_common_predicted": predicted_label,
        "most_common_count": pair_count,
        "source_row": original_index + 1,
        "area": row[0],
        "perimeter": row[1],
        "roundness": row[10],
        "per_class_f1": {
            label: float(report[label]["f1-score"])
            for label in sorted(EXPECTED_CLASSES)
        },
    }


def build_random_forest() -> RandomForestClassifier:
    """Create the fixed random-forest candidate used for fair comparison."""
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )


def validate_feature_values(values: list[float]) -> None:
    """Reject incomplete or non-finite input before model prediction."""
    if len(values) != len(FEATURE_NAMES):
        raise ValueError(
            f"单粒预测需要 {len(FEATURE_NAMES)} 个特征值；"
            f"实际收到 {len(values)} 个。"
        )
    for name, value in zip(FEATURE_NAMES, values):
        if not math.isfinite(value):
            raise ValueError(f"{name} 必须是有限数值。")


def collect_feature_values() -> list[float]:
    """Ask for the 16 measurements in the dataset's documented order."""
    print("请输入一粒干豆的 16 项形状特征。输入错误可按 Ctrl+C 取消。")
    values: list[float] = []
    for position, (name, label) in enumerate(
        zip(FEATURE_NAMES, FEATURE_LABELS), start=1
    ):
        raw_value = input(f"{position:02d}. {label}（{name}）: ").strip()
        try:
            values.append(float(raw_value))
        except ValueError as error:
            raise ValueError(
                f"{name} 必须输入数字；实际输入为 {raw_value!r}。"
            ) from error
    validate_feature_values(values)
    return values


def predict_one(path: Path, values: list[float]) -> dict[str, object]:
    """Train the documented candidate on all approved rows and classify one bean."""
    check_data(path)
    validate_feature_values(values)
    features, labels = load_dataset(path)

    outside_ranges: list[str] = []
    for column, (name, value) in enumerate(zip(FEATURE_NAMES, values)):
        column_values = [row[column] for row in features]
        minimum = min(column_values)
        maximum = max(column_values)
        if value < minimum or value > maximum:
            outside_ranges.append(
                f"{name}={value:g} （训练范围 {minimum:g} 至 {maximum:g}）"
            )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=42),
    )
    model.fit(features, labels)
    predicted = str(model.predict([values])[0])
    probabilities = model.predict_proba([values])[0]
    return {
        "prediction": predicted,
        "confidence": float(max(probabilities)),
        "outside_ranges": outside_ranges,
    }


def run_random_forest(path: Path) -> dict[str, object]:
    """Evaluate random forest on exactly the same fixed holdout."""
    check_data(path)
    features, labels = load_dataset(path)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    model = build_random_forest()
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    confusions = Counter(
        (actual, predicted)
        for actual, predicted in zip(test_y, predictions)
        if actual != predicted
    )
    (actual, predicted), pair_count = confusions.most_common(1)[0]
    return {
        "train_rows": len(train_y),
        "test_rows": len(test_y),
        "accuracy": accuracy_score(test_y, predictions),
        "macro_f1": f1_score(
            test_y, predictions, average="macro", zero_division=0
        ),
        "predicted_classes": len(set(predictions)),
        "errors": sum(confusions.values()),
        "most_common_actual": actual,
        "most_common_predicted": predicted,
        "most_common_count": pair_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="干豆品种辅助分类项目")
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="检查获准使用的 Kaggle ARFF 数据文件",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="评估多数类基线",
    )
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="评估标准化加逻辑回归候选模型",
    )
    parser.add_argument(
        "--analyze-errors",
        action="store_true",
        help="显示各类别 F1 和一个真实测试集错误",
    )
    parser.add_argument(
        "--random-forest",
        action="store_true",
        help="在固定测试集上评估300棵树的随机森林",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="根据16项测量值预测一粒干豆的候选品种",
    )
    parser.add_argument(
        "--values",
        type=float,
        nargs=16,
        metavar="VALUE",
        help="按数据字段顺序填写16个特征值；省略时逐项输入",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Dry_Bean_Dataset.arff 的文件路径",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not any(
        (
            args.check_data,
            args.baseline,
            args.candidate,
            args.analyze_errors,
            args.random_forest,
            args.predict,
        )
    ):
        print("请先运行 `python main.py --check-data` 检查真实数据。")
        return 0

    if args.values is not None and not args.predict:
        print("--values 必须和 --predict 一起使用。")
        return 1

    try:
        if args.predict:
            values = args.values if args.values is not None else collect_feature_values()
            prediction = predict_one(args.data, values)
        elif args.random_forest:
            forest = run_random_forest(args.data)
        elif args.analyze_errors:
            analysis = analyze_candidate_errors(args.data)
        elif args.candidate:
            train_rows, test_rows, accuracy, macro_f1, class_count = run_candidate(
                args.data
            )
        elif args.baseline:
            prediction, train_rows, test_rows, accuracy, macro_f1 = run_baseline(
                args.data
            )
        else:
            summary = check_data(args.data)
    except (EOFError, FileNotFoundError, ValueError) as error:
        print(error)
        return 1
    except KeyboardInterrupt:
        print("\n已取消预测。")
        return 130

    if args.predict:
        print("单粒干豆预测完成")
        print("方法：标准化（StandardScaler）+ 逻辑回归（LogisticRegression），使用全部获准数据训练")
        print(f"预测品种：{prediction['prediction']}")
        print(f"模型置信度：{prediction['confidence']:.4f}")
        outside_ranges = prediction["outside_ranges"]
        if outside_ranges:
            print("必须人工复核：输入中有数值超出训练数据范围")
            for warning in outside_ranges:
                print(f"  {warning}")
        else:
            print("范围检查：16项数值都在训练数据范围内")
        print("使用边界：这只是辅助建议，必须由人复核")
        return 0

    if args.random_forest:
        print("随机森林评估完成")
        print("方法：随机森林（RandomForestClassifier，300棵树）")
        print("数据划分：分层抽样，80%训练 / 20%测试，random_state=42")
        print(f"训练集行数：{forest['train_rows']}")
        print(f"测试集行数：{forest['test_rows']}")
        print(f"预测覆盖类别数：{forest['predicted_classes']}")
        print(f"准确率（Accuracy）：{forest['accuracy']:.4f}")
        print(f"宏平均F1（Macro F1）：{forest['macro_f1']:.4f}")
        print(f"测试错误数：{forest['errors']}")
        print(
            "最常见混淆："
            f"{forest['most_common_actual']} -> "
            f"{forest['most_common_predicted']} "
            f"({forest['most_common_count']} rows)"
        )
        return 0

    if args.analyze_errors:
        print("错误分析完成")
        print(f"total 测试错误数：{analysis['errors']}")
        print("各类别F1：")
        for label, score in analysis["per_class_f1"].items():
            print(f"  {label}: {score:.4f}")
        print("最常见混淆：")
        print(
            f"  actual {analysis['most_common_actual']} -> "
            f"predicted {analysis['most_common_predicted']}: "
            f"{analysis['most_common_count']} rows"
        )
        print("一个真实测试集错误：")
        print(f"  原始数据行：{analysis['source_row']}")
        print(f"  真实类别：{analysis['most_common_actual']}")
        print(f"  预测类别：{analysis['most_common_predicted']}")
        print(f"  Area: {analysis['area']:.0f}")
        print(f"  Perimeter: {analysis['perimeter']:.4f}")
        print(f"  roundness: {analysis['roundness']:.6f}")
        return 0

    if args.candidate:
        print("候选模型评估完成")
        print("方法：标准化（StandardScaler）+ 逻辑回归（LogisticRegression）")
        print("数据划分：分层抽样，80%训练 / 20%测试，random_state=42")
        print(f"训练集行数：{train_rows}")
        print(f"测试集行数：{test_rows}")
        print(f"预测覆盖类别数：{class_count}")
        print(f"准确率（Accuracy）：{accuracy:.4f}")
        print(f"宏平均F1（Macro F1）：{macro_f1:.4f}")
        return 0

    if args.baseline:
        print("基线评估完成")
        print("方法：始终预测训练集中数量最多的类别")
        print("数据划分：分层抽样，80%训练 / 20%测试，random_state=42")
        print(f"训练集行数：{train_rows}")
        print(f"测试集行数：{test_rows}")
        print(f"预测品种：{prediction}")
        print(f"准确率（Accuracy）：{accuracy:.4f}")
        print(f"宏平均F1（Macro F1）：{macro_f1:.4f}")
        return 0

    print("真实数据检查通过")
    print(f"行数：{summary.rows}")
    print(f"列数：{summary.columns}")
    print(f"缺失值：{summary.missing_values}")
    print("类别统计：")
    for label, count in sorted(summary.class_counts.items()):
        print(f"  {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



