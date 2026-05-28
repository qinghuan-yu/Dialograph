"""单次生成聊天关系分析报告，不走分块解析。"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib import error
import http.client

sys.path.insert(0, str(Path(__file__).parent))

from analyze import extract_conversation_segments, extract_text_samples  # noqa: E402
from config import load_llm_config  # noqa: E402
from report_output import write_report  # noqa: E402
from run_workflow import (  # noqa: E402
    build_artifacts,
    call_llm_with_retry,
    load_skill_prompt,
    should_retry_with_smaller_prompt,
)


def compact_segments(segments: list[list[dict]], max_messages_per_segment: int = 18) -> str:
    lines = []
    for index, segment in enumerate(segments, start=1):
        lines.append(f"--- 片段 {index} ({segment[0]['time_str'][:10]}) ---")
        if len(segment) <= max_messages_per_segment:
            selected = segment
        else:
            head = max_messages_per_segment // 2
            tail = max_messages_per_segment - head
            selected = segment[:head] + segment[-tail:]

        for message in selected:
            prefix = "【我】" if message["sender"] == "me" else "【对方】"
            lines.append(f"{message['time_str'][11:16]} {prefix} {message['content']}")
    return "\n".join(lines)


def compact_samples(samples: list[dict], max_count: int = 80) -> str:
    selected = samples[:max_count]
    lines = []
    for message in selected:
        prefix = "【我】" if message["sender"] == "me" else "【对方】"
        lines.append(f"{message['time_str']} {prefix} {message['content']}")
    return "\n".join(lines)


def build_single_pass_prompt(name: str, stats_summary: str, segment_text: str, sample_text: str) -> str:
    return f"""请基于以下聊天记录预处理统计报告与代表性片段，直接生成一份完整的关系与行为模式分析报告。

对象：{name}

重要要求：
1. 只输出最终分析报告，不要解释你的工作过程。
2. 报告必须明显接近“法证式分析”，不要泛泛而谈。
3. 严格区分【事实】【推断】【假设】【存疑】。
4. 每个章节都要尽量引用具体片段、时间点或概括性证据。
5. 必须优先寻找反证，避免迎合式结论。
6. 不预设对方性别、性取向、关系目标或恋爱/暧昧前提。
7. 风格参考：冷静、克制、细致、结构完整。
8. 输出为 Markdown。

请严格按以下结构输出：

好的，这是一份基于你提供的聊天记录预处理统计报告与代表性片段，对“我”与“{name}”之间关系与行为模式的分析报告。

**重要声明：** 用 2 到 4 句话说明本报告如何区分可观察事实、合理推断、可能性假设与不能判断的部分，并说明不会读心或做人格审判。

---

### **一、互动概况**

#### **1. 聊天频率变化（可观察事实）**
#### **2. 主动性与话题控制（可观察事实 + 合理推断）**
#### **3. 关系升温/降温节点与重要事件（合理推断）**

### **二、对方的表达风格（基于文本证据）**

### **三、对方的互动模式**

### **四、对方在这段关系中的可能定位（模型评估）**
- 用表格列出：模型 / 支持证据 / 反证 / 可信度
- 至少覆盖：普通熟人、普通朋友、较亲近朋友、工具性/事务性关系、社群/同学/同事式关系、情绪支持或情绪依赖、礼貌维持、低成本社交、亲密或暧昧可能（仅在证据明确支持时评估）

### **五、对方可能的动机逻辑**
- 用表格列出：动机解释 / 支持证据 / 反证 / 可信度 / 验证所需信息

### **六、对方在我面前呈现出的性格倾向**

### **七、我自己的误判风险**

### **八、总结**

#### **1. 最稳妥的结论（可观察事实）**
#### **2. 中等可信的判断（基于事实的合理推断）**
#### **3. 不能下结论的部分**
#### **4. 验证关系定位或互动边界的低风险行动建议**

下面是统计摘要：

{stats_summary}

下面是 15 个代表性对话片段：

{segment_text}

下面是补充文本消息样本：

{sample_text}
"""


def generate_report_with_fallback(llm_config, skill_prompt: str, artifacts: dict) -> str:
    primary_prompt = build_single_pass_prompt(
        artifacts["name"],
        artifacts["stats_summary"],
        compact_segments(
            extract_conversation_segments(artifacts["messages"], n_segments=12),
            max_messages_per_segment=16,
        ),
        compact_samples(extract_text_samples(artifacts["messages"], n_samples=60), max_count=60),
    )

    print(f"单次提示词大小: {len(primary_prompt.encode('utf-8'))} bytes")
    print("正在调用模型生成最终关系分析...")
    try:
        return call_llm_with_retry(
            llm_config,
            skill_prompt,
            primary_prompt,
            timeout=1800,
            max_tokens=4200,
            max_attempts=1,
        )
    except Exception as exc:  # noqa: BLE001
        if not should_retry_with_smaller_prompt(exc):
            raise

        fallback_prompt = build_single_pass_prompt(
            artifacts["name"],
            artifacts["stats_summary"],
            compact_segments(
                extract_conversation_segments(artifacts["messages"], n_segments=8),
                max_messages_per_segment=12,
            ),
            compact_samples(extract_text_samples(artifacts["messages"], n_samples=30), max_count=30),
        )
        print(f"- 主请求失败，切换紧凑回退: {exc}")
        print(f"- 回退提示词大小: {len(fallback_prompt.encode('utf-8'))} bytes")
        return call_llm_with_retry(
            llm_config,
            skill_prompt,
            fallback_prompt,
            timeout=1800,
            max_tokens=3200,
            max_attempts=2,
        )


def main() -> int:
    workspace = Path(__file__).parent.parent
    output_dir = workspace / "analysis"
    output_dir.mkdir(exist_ok=True)
    target = sys.argv[1] if len(sys.argv) > 1 else "黄芷馨"

    llm_config = load_llm_config(workspace)
    if not llm_config.api_key:
        print("LLM_API_KEY 未设置，无法执行单次关系分析。")
        return 1

    talks = sorted((workspace / "talks").glob("*.json"))
    target_file = None
    for talk in talks:
        if target in talk.stem:
            target_file = talk
            break

    if target_file is None:
        print(f"未找到匹配的聊天记录: {target}")
        return 1

    artifacts = build_artifacts(target_file, output_dir, write_supporting_files=True)
    skill_prompt = load_skill_prompt(Path(__file__).parent / "SKILL.md")
    print("=" * 60)
    print("单次聊天关系分析")
    print("=" * 60)
    print(f"对象: {artifacts['name']}")
    print(f"预处理文件: {artifacts['preproc_file']}")

    report = generate_report_with_fallback(llm_config, skill_prompt, artifacts)
    write_report(
        artifacts["analysis_file"],
        artifacts["analysis_meta_file"],
        report,
        source="api",
        script_name="agent/run_single_pass_analysis.py",
        model=llm_config.model,
    )

    print(f"已生成: {artifacts['analysis_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
