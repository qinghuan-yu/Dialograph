"""发布失败时应保留整套旧结果。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
import run_workflow as workflow


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.artifacts = {}
        self.run = {'run_id': 'test-run', 'run_dir': root / 'run'}
        self.run['run_dir'].mkdir()
        for key in ('chunk_dir', 'evidence_dir', 'chunk_manifest_file',
                    'coverage_summary_file', 'evidence_ledger_file', 'evidence_summary_file',
                    'phase_summary_file', 'analysis_file', 'analysis_meta_file'):
            src, dst = root / 'run' / key, root / 'published' / key
            dst.parent.mkdir(exist_ok=True)
            if key.endswith('_dir'):
                src.mkdir()
                dst.mkdir()
                src, dst = src / 'item', dst / 'item'
            src.write_text('new', encoding='utf-8')
            dst.write_text('old', encoding='utf-8')
            self.artifacts[key] = root / 'published' / key
            self.run[key] = root / 'run' / key
        self.artifacts['latest_run_file'] = root / 'published' / 'latest.json'
        self.artifacts['latest_run_file'].write_text('old-pointer', encoding='utf-8')

    def assert_old(self):
        for key, path in self.artifacts.items():
            if path.is_dir():
                path = path / 'item'
            self.assertEqual(path.read_text(encoding='utf-8'),
                             'old-pointer' if key == 'latest_run_file' else 'old', key)

    def test_missing_source_preserves_all_published_outputs(self):
        self.run['analysis_file'].unlink()
        with self.assertRaises(FileNotFoundError):
            workflow.promote_successful_run(self.artifacts, self.run)
        self.assert_old()

    def test_mid_publish_failure_restores_old_files_directories_and_pointer(self):
        import os
        real_replace = os.replace
        for failure_step in range(1, 21):
            with self.subTest(failure_step=failure_step):
                count = 0
                def fail_once(src, dst):
                    nonlocal count
                    count += 1
                    if count == failure_step:
                        raise OSError('simulated disk failure')
                    return real_replace(src, dst)
                with patch('os.replace', side_effect=fail_once), self.assertRaises(OSError):
                    workflow.promote_successful_run(self.artifacts, self.run)
                self.assert_old()

    def test_success_publishes_complete_snapshot_and_pointer(self):
        workflow.promote_successful_run(self.artifacts, self.run)
        for key, path in self.artifacts.items():
            if key == 'latest_run_file':
                self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['run_id'], 'test-run')
            else:
                if path.is_dir():
                    path = path / 'item'
                self.assertEqual(path.read_text(encoding='utf-8'), 'new')
