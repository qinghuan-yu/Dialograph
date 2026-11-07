"""结论处理模块拆分后，规则变化仍须使旧流程缓存失效。"""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'agent'))

from workflow_state import run_identity


class ClaimModuleIdentityTests(unittest.TestCase):
    def test_compilation_and_repair_rule_changes_invalidate_pipeline_cache(self):
        artifacts = {'chat_file': ROOT / 'tests/fixtures/web_smoke.json', 'messages': []}
        config = SimpleNamespace(model='synthetic', api_base='http://invalid.local', thinking=None)
        before = run_identity(artifacts, '合成框架', config, report_system_prompt='合成系统')
        original = Path.read_bytes
        for changed_file in ('claim_compiler.py', 'claim_repair.py', 'claim_review_basis.py'):
            def read_bytes(path):
                if path.name == changed_file:
                    content = original(path) if path.exists() else b''
                    return content + b'\n# changed compilation or repair rule\n'
                return original(path)

            with self.subTest(module=changed_file), patch.object(Path, 'read_bytes', read_bytes):
                after = run_identity(artifacts, '合成框架', config, report_system_prompt='合成系统')
                self.assertEqual(before['input_sha256'], after['input_sha256'])
                self.assertNotEqual(before['pipeline_sha256'], after['pipeline_sha256'])
