"""单次生成聊天关系分析报告，不走分块解析。"""

from __future__ import annotations

import sys
from pathlib import Path


from preview_support import preview_materials, generate_preview
from analysis_types import AnalysisArtifacts
from config import load_llm_config
from report_output import write_report
from privacy_redaction import redact_artifact_call
from analysis_artifacts import build_artifacts, select_chat_files
from report_generation import load_skill_prompt


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
7. 除直接引用原文外，全文使用“对方/此人/该对象/对象昵称”，不要使用“他/她/ta/男方/女方”等带性别预设的代称。
8. 风格参考：冷静、克制、细致、结构完整。
9. 输出为 Markdown。

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


@redact_artifact_call(position=2)
def generate_report_with_fallback(llm_config, skill_prompt: str, artifacts: AnalysisArtifacts) -> str:
    def prompt(compact=False):
        segments, samples = preview_materials(artifacts, compact=compact)
        return build_single_pass_prompt(artifacts['name'], artifacts['stats_summary'], segments, samples)

    primary_prompt = prompt()
    print(f"单次提示词大小: {len(primary_prompt.encode('utf-8'))} bytes")
    print("正在调用模型生成最终关系分析...")
    return generate_preview(llm_config, skill_prompt, primary_prompt, lambda: prompt(compact=True))


def main() -> int:
    workspace = Path(__file__).parent.parent
    output_dir = workspace / "analysis" / "预览"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = sys.argv[1] if len(sys.argv) > 1 else "示例对象"

    llm_config = load_llm_config(workspace)
    if not llm_config.api_key:
        print("LLM_API_KEY 未设置，无法执行单次关系分析。")
        return 1

    talks = select_chat_files(workspace / "talks", [], None)
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
    print("预览：单次聊天关系分析（采样分析，不覆盖正式报告）")
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
        note="预览：仅使用代表性片段，不保证全量证据覆盖，不等同于完整分析报告",
    )

    print(f"已生成: {artifacts['analysis_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
