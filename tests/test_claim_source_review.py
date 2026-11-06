"""来源独立解读与归属不明拦截，只用合成输入。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from test_claim_retrieval import source, QUESTIONS
from claim_fixtures import claim_response, claim_revision
from claim_ledger import compile_claims, _evidence_index, generate_claims, audit_claim_report, render_claim_report
from claim_semantic_review import SemanticReviewError
from report_generation import prepare_final_input


class ClaimSourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.data = source()
        self.prepared = prepare_final_input('合成', '', '', '', [], ledger=self.data, review_questions=QUESTIONS)
        self.index = _evidence_index(self.data)
        self.options = dict(kind='inference', subject='peer', supporting_evidence=['P_79'],
                            counter_review='not_found_in_selected')
        self.compiled = compile_claims(claim_response('这里是仅供报告使用的结论。', **self.options),
                                       self.data, self.prepared, 'analysis')
        self.fact = {'position':79, 'participants':['peer','me'], 'event':'对方表达想了解情况。',
                     'uncertainty':'', 'bindings':[{'participant':'peer','position':79,'quote':'我'},
                                                   {'participant':'me','position':79,'quote':'你'}]}

    def test_source_request_has_no_claim_and_reuses_validated_positions(self):
        from claim_source_review import check_claim_sources
        request = Mock(return_value=json.dumps({'facts':[self.fact]}))
        cache = {}
        with tempfile.TemporaryDirectory() as tmp:
            first = check_claim_sources(self.compiled,self.prepared,self.index,request,Path(tmp),cache)
            second = check_claim_sources(self.compiled,self.prepared,self.index,request,Path(tmp),cache)
        self.assertEqual(request.call_count,1)
        self.assertNotIn('仅供报告使用的结论',request.call_args.args[0])
        self.assertIn('你遇到什么事了？我愿意了解。',request.call_args.args[0])
        self.assertEqual(first,second)

    def test_unknown_referent_blocks_inference_before_general_reviewer(self):
        from claim_source_review import check_claim_sources
        fact = {**self.fact,'participants':['unknown'],'bindings':[],'uncertainty':'对象未能确定。'}
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SemanticReviewError) as caught:
            check_claim_sources(self.compiled,self.prepared,self.index,
                Mock(return_value=json.dumps({'facts':[fact]})),Path(tmp),{})
        self.assertEqual(caught.exception.claim_numbers,{1})

    def test_invalid_or_missing_source_binding_is_rejected(self):
        from claim_source_review import validate_source_facts
        for change in ('quote','position','missing','unknown'):
            with self.subTest(change=change):
                fact = copy.deepcopy(self.fact)
                if change=='quote':
                    fact['bindings'][0]['quote']='原文没有这句话'
                elif change=='position':
                    fact['bindings'][0]['position']=999
                elif change=='missing':
                    fact['bindings'].pop()
                else:
                    fact['participants']=['unknown']
                with self.assertRaises(ValueError):
                    validate_source_facts([fact],{79},self.prepared,self.index)

    def test_unknown_source_can_be_kept_as_explicit_uncertainty(self):
        from claim_source_review import check_claim_sources
        compiled = compile_claims(claim_response('该问题所指对象无法确认。',supporting_evidence=['P_79']),
                                  self.data,self.prepared,'analysis')
        fact = {**self.fact,'participants':['unknown'],'bindings':[],'uncertainty':'对象未能确定。'}
        with tempfile.TemporaryDirectory() as tmp:
            result = check_claim_sources(compiled,self.prepared,self.index,
                Mock(return_value=json.dumps({'facts':[fact]})),Path(tmp),{})
        self.assertEqual(result['facts'],[fact])

    def test_source_failure_enters_one_repair_and_final_record_binds_sources(self):
        fact = {**self.fact,'participants':['unknown'],'bindings':[],'uncertainty':'对象未能确定。'}
        generator = Mock(side_effect=[claim_response('对方对双方关系主动探索。',**self.options),
                                     claim_revision('该问题的具体对象尚不确定。',supporting_evidence=['P_79'])])
        reviewer = Mock(return_value=json.dumps({'reviews':[{'claim_id':'R_1','verdict':'pass','reason':'明确存疑。'}]}))
        source_request = Mock(return_value=json.dumps({'facts':[fact]}))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = generate_claims(self.prepared,self.data,'analysis',generator,attempt_dir=root,
                                     review_request=reviewer,source_request=source_request)
            (root/'input.md').write_text(self.prepared.prompt,encoding='utf-8')
            (root/'report.md').write_text(render_claim_report(result,'合成'),encoding='utf-8')
            (root/'claims.json').write_text(json.dumps(result),encoding='utf-8')
            audit_claim_report(root/'claims.json',root/'report.md',self.data,'合成',root/'input.md','analysis')
            tampered = copy.deepcopy(result)
            del tampered['semantic_review']['source_review']
            (root/'claims.json').write_text(json.dumps(tampered),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'来源独立核查'):
                audit_claim_report(root/'claims.json',root/'report.md',self.data,'合成',root/'input.md','analysis')
        self.assertEqual(generator.call_count,2)
        self.assertEqual(reviewer.call_count,1)
        self.assertEqual(source_request.call_count,1)
        self.assertEqual(result['semantic_review']['source_review']['facts'],[fact])

    def test_invalid_source_quotes_get_one_correction_and_then_stop(self):
        from claim_source_review import check_claim_sources
        invalid = copy.deepcopy(self.fact)
        invalid['bindings'][0]['quote']='伪造原文'
        request = Mock(return_value=json.dumps({'facts':[invalid]}))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(RuntimeError,'两次'):
            check_claim_sources(self.compiled,self.prepared,self.index,request,Path(tmp),{})
        self.assertEqual(request.call_count,2)

    def test_source_correction_repeats_original_positions_and_exact_messages(self):
        from claim_source_review import check_claim_sources
        wrong = copy.deepcopy(self.fact)
        wrong['position']=1
        request = Mock(side_effect=[json.dumps({'facts':[wrong]}),json.dumps({'facts':[self.fact]})])
        with tempfile.TemporaryDirectory() as tmp:
            check_claim_sources(self.compiled,self.prepared,self.index,request,Path(tmp),{})
        self.assertIn('本次原始位置与逐字消息对照',request.call_args.args[0])
        correction = request.call_args.args[0].split('本次原始位置与逐字消息对照',1)[1]
        self.assertIn('"position": 79',correction)
        self.assertIn('你遇到什么事了？我愿意了解。',correction)


if __name__=='__main__':
    unittest.main()
