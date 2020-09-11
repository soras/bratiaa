import pytest

from bratiaa.agree import *
from bratiaa.agree_view import F1AgreementView, extended_iaa_report
from bratiaa.utils import tokenize

AGREE_2_ROOT = 'data/agreement/agree-2'


def test_view_instance_disagreements():
    f1_agreement_view = F1AgreementView(AGREE_2_ROOT)
    expected_disagreements = {
      'esp.train-doc-29.ann':\
        [('ann1', 'LOC', '  18', '  24', "'Madrid'"), 
         ('ann1', 'PER', '1074', '1078', "'Cela'  ")],
      'esp.train-doc-46.ann':\
        [('ann2', 'LOC', ' 42', ' 54', "'Sierra Leona'                 "), 
         ('ann2', 'ORG', '161', '190', "'Fuerzas Revolucionaria Unidas'"), 
         ('ann1', 'ORG', '192', '195', "'FRU'                          "), 
         ('ann2', 'ORG', '408', '426', "'Asociated Press TV'           "), 
         ('ann2', 'ORG', '516', '523', "'Reuters'                      ")]
    }
    covered_docs = set()
    for doc_result in f1_agreement_view.results:
        if doc_result['ann_file'] in expected_disagreements:
            assert doc_result['mismatches'] == expected_disagreements[doc_result['ann_file']]
            covered_docs.add( doc_result['ann_file'] )
    assert covered_docs == set(expected_disagreements.keys())


def test_view_token_disagreements():
    f1_agreement_view = F1AgreementView(AGREE_2_ROOT, token_func=tokenize)
    expected_disagreements = {
      'esp.train-doc-29.ann':\
        [('ann1', 'LOC', '  18', '  25', "'Madrid,'"), 
         ('ann1', 'PER', '1074', '1078', "'Cela'   ")],
      'esp.train-doc-46.ann':\
        [('ann2', 'LOC', ' 42', ' 48', "'Sierra'        "), 
         ('ann2', 'LOC', ' 49', ' 54', "'Leona'         "), 
         ('ann2', 'ORG', '161', '168', "'Fuerzas'       "), 
         ('ann2', 'ORG', '169', '183', "'Revolucionaria'"), 
         ('ann2', 'ORG', '184', '190', "'Unidas'        "), 
         ('ann1', 'ORG', '191', '196', "'(FRU)'         "), 
         ('ann2', 'ORG', '408', '417', "'Asociated'     "), 
         ('ann2', 'ORG', '418', '423', "'Press'         "), 
         ('ann2', 'ORG', '424', '426', "'TV'            "), 
         ('ann2', 'ORG', '516', '524', "'Reuters,'      ")]
    }
    covered_docs = set()
    for doc_result in f1_agreement_view.results:
        if doc_result['ann_file'] in expected_disagreements:
            assert doc_result['mismatches'] == expected_disagreements[doc_result['ann_file']]
            covered_docs.add( doc_result['ann_file'] )
    assert covered_docs == set(expected_disagreements.keys())


def test_instance_extended_iaa_report_smoke():
    f1_agreement = compute_f1_agreement(AGREE_2_ROOT)
    f1_agreement_view = F1AgreementView(AGREE_2_ROOT)
    extended_iaa_report(f1_agreement, f1_agreement_view)


def test_token_extended_iaa_report_smoke():
    f1_agreement = compute_f1_agreement(AGREE_2_ROOT, token_func=tokenize)
    f1_agreement_view = F1AgreementView(AGREE_2_ROOT, token_func=tokenize)
    extended_iaa_report(f1_agreement, f1_agreement_view)

