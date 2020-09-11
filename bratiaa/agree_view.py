#
#  Agreement viewer for the bratiaa module:
#  * Extracts matching & mismatching annotations for viewing;
#  * Extends the reporting function so that all disagreements per document are displayed;
#
#  Developed on bratiaa version:
#    bratiaa   0.1.3+  ( https://github.com/kldtz/bratiaa/tree/b86f5f90d73eec81d795e45e1a2ce80912e4530c )
#

import re
import os, os.path

from itertools import combinations
from functools import partial

import numpy as np
from collections import Counter

from tabulate import tabulate

from bratiaa.agree import input_generator
from bratiaa.utils import read, TokenOverlap
from bratiaa.evaluation import exact_match_instance_evaluation
from bratiaa.evaluation import exact_match_token_evaluation
from bratiaa.evaluation import counter2list


class F1AgreementView:
    '''Uses bratiaa evaluation function to extract exact agreements and disagreements on annotations. 
       Stores agreements and disagreements (pre-formatted annotations) for viewing.
       This is a simplified version of the class bratiaa.agree.F1Agreement.
       TODO: in future, it may be a good idea to make it a subclass of F1Agreement
    '''
    def __init__(self, project_root, input_gen=input_generator, eval_func=exact_match_instance_evaluation, \
                       token_func=None, annotators=None, documents=None):
        input_gen = partial(input_gen, project_root)
        if not (annotators and documents):
            annotators, documents = self._collect_annotators_and_documents(input_gen)
            annotators.sort()
            documents.sort()
        assert len(annotators) > 1, 'At least two annotators are necessary to compute agreement!'
        self._documents = list(documents)
        self._annotators = list(annotators)
        self._eval_func = eval_func    # function used to extract true positives, false positives and false negatives
        self._token_func = token_func  # function used for tokenization
        if token_func is not None and eval_func==exact_match_instance_evaluation:
            # Update evaluation function
            self._eval_func = exact_match_token_evaluation
        # Find all annotation matches and mismatches
        self._all_results = self.find_all_matches_and_mismatches(input_gen)

    def _collect_annotators_and_documents(self, input_gen):
        annotators, documents = set(), []
        for document in input_gen():
            for ann_file in document.ann_files:
                annotators.add(ann_file.annotator_id)
            documents.append(document.doc_id)
        return list(annotators), documents

    def _extract_sorted_annotation_text_tuples(self, ann_set, text_str, annotator=None):
        # Converts Annotations to tuples sorted by offsets
        ann_tuples = []
        for ann in sorted(list(ann_set), key=lambda x:x.offsets):
            label = ann.label
            type  = ann.type
            for (start, end) in ann.offsets:
                text_part = text_str[start:end]
                if annotator is None:
                    ann_tuples.append( (label, start, end, text_part))
                else:
                    ann_tuples.append( (annotator, label, start, end, text_part))
        return ann_tuples

    def _format_text_tuples_for_pretty_printing(self, ann_tuples, skip_str_conversion=[]):
        if len(ann_tuples) == 0:
            return []
        # Find maximum lengths of all fields
        field_max_lens = [0 for f in range(len(ann_tuples[0]))]
        for tuple in ann_tuples:
            for tid, t in enumerate(tuple):
                if isinstance(t, str) and tid not in skip_str_conversion:
                    t = '{!r}'.format(t)
                if len(str(t)) > field_max_lens[tid]:
                    field_max_lens[tid] = len(str(t))  # Update maximum length
        # Format all annotations: pad fields with spaces
        formatted_tuples = []
        for tuple in ann_tuples:
            new_tuple = ()
            for tid, t in enumerate(tuple):
                if isinstance(t, str) and tid not in skip_str_conversion:
                    t = '{!r}'.format(t)
                new_tuple += (('{:'+str(field_max_lens[tid])+'}').format(t),)
            formatted_tuples.append( new_tuple )
        return formatted_tuples

    def find_all_matches_and_mismatches(self, input_gen):
        # Finds all matches / mismatches (or, by other words: agreements & disagreements)
        results = []
        for doc_index, document in enumerate( input_gen() ):
            assert doc_index < len(self._documents), 'Input generator yields more documents than expected!'
            to = None
            text = read(document.txt_path)
            if self._token_func:
                tokens = list(self._token_func(text))
                to = TokenOverlap(text, tokens)
            # TODO: the following solution robustly works for 2 annotators.
            #       in case of more than 2 annotators, it would be very nice 
            #       to further consolidate the results
            for anno_file_1, anno_file_2 in combinations(document.ann_files, 2):
                tp, exp, pred = self._eval_func(anno_file_1.ann_path, anno_file_2.ann_path, tokens=to)
                # Convert exp & pred to fp & fn
                if self._token_func is not None:
                    fp = counter2list(Counter(pred) - Counter(exp))
                    fn = counter2list(Counter(exp) - Counter(pred))
                else:
                    fp = pred.difference(exp)
                    fn = exp.difference(pred)
                doc_result = {}
                # all kinds of metadata
                doc_result['doc_id']    = doc_index
                doc_result['text_file'] = os.path.basename(document.txt_path)
                doc_result['ann_file']  = anno_file_1.ann_path.name
                doc_result['annotator_1'] = anno_file_1.annotator_id
                doc_result['annotator_2'] = anno_file_2.annotator_id
                # true-positive tuples: matching annotations
                tp_tuples = self._extract_sorted_annotation_text_tuples(tp, text)
                tp_tuples = self._format_text_tuples_for_pretty_printing(tp_tuples, skip_str_conversion=[0,1])
                doc_result['matches'] = tp_tuples
                # marked by annotator2, but not by annotator1
                fp_tuples = self._extract_sorted_annotation_text_tuples(fp, text, annotator=anno_file_2.annotator_id)
                # marked by annotator1, but not by annotator2
                fn_tuples = self._extract_sorted_annotation_text_tuples(fn, text, annotator=anno_file_1.annotator_id)
                all_mismatches = sorted(fp_tuples + fn_tuples, key=lambda x:x[2])
                all_mismatches = self._format_text_tuples_for_pretty_printing(all_mismatches, skip_str_conversion=[0,1,2])
                doc_result['mismatches'] = all_mismatches
                results.append( doc_result )
        return results

    @property
    def results(self):
        return self._all_results

    def format_mismatches_tuple_as_str_list(self, mismatches, sep=' | '):
        # Further 
        str_list = []
        last_start = '-1'
        last_end   = '-1'
        for mismatch in mismatches:
            assert len(mismatch) == 5, '{!r}'.format(mismatch)
            start = mismatch[2]
            end   = mismatch[3]
            cur_str = mismatch[0]+sep+mismatch[1]+sep+mismatch[2]+':'+mismatch[3]+sep+mismatch[4]
            if start == last_start or end == last_end:
                str_list.append(cur_str)
            else:
                if len(str_list) > 0:
                    str_list.append('')
                str_list.append(cur_str)
            last_start = start
            last_end   = end
        return str_list

    # For debugging only
    def _print_doc_matches_and_mismatches(self, target_ann_file):
        for result_dict in self._all_results:
            if result_dict['ann_file'] == target_ann_file:
                print( result_dict['ann_file'] )
                print( result_dict['annotator_1'] )
                print( result_dict['annotator_2'] )
                print()
                for m in result_dict['matches']:
                    print(m)
                print()
                for m in result_dict['mismatches']:
                    print(m)
                print()

#
# An extended version of the function bratiaa.agree.iaa_report:
# * in case of agreement per document, outputs mismatching annotations (for inspection);
# * returns results as a list of strings ( raport lines );
#
def extended_iaa_report_str(f1_agreement, biaa_viewer, precision=3, newline='\n'):
    out_str = []
    agreement_type = '* Instance-based F1 agreement'
    if f1_agreement._token_func:
        agreement_type = '* Token-based F1 agreement'
    if biaa_viewer._token_func != f1_agreement._token_func:
        raise Exception('(!) Conflict: token functions for f1_agreement and biaa_viewer are different. Please use the same tokenization functions.')

    out_str.append( '# Inter-Annotator Agreement Report'+newline )
    
    out_str.append( agreement_type )

    out_str.append( newline+'## Project Setup'+newline )
    out_str.append( f'* {len(f1_agreement.annotators)} annotators: {", ".join(f1_agreement.annotators)}' )
    out_str.append( f'* {len(f1_agreement.documents)} agreement documents' )
    out_str.append( f'* {len(f1_agreement.labels)} labels' )

    out_str.append( newline+'## Agreement per Document'+newline )
    doc_stats = np.stack((f1_agreement.documents, *f1_agreement.mean_sd_per_document())).transpose()
    doc_headers = ['Document', 'Mean F1', 'SD F1']
    max_fname = 0
    for doc_stat in doc_stats:
        fname, fscore, sd_dev = doc_stat
        if len(fname) > max_fname:
             max_fname = len(fname)
    for doc_stat in doc_stats:
        fname, fscore, sd_dev = doc_stat
        # Make type conversions for the f-string
        fname  = str(fname)
        fscore = float(fscore)
        sd_dev = float(sd_dev)
        # Print document name, F1-score and SD for F1-score
        out_str.append( f'* `{fname:{max_fname}}`    F1: {fscore:.{precision}f},   SD F1: {sd_dev:.{precision}f}'+newline )
        # Find all annotation mismatches / disagreements for the document
        all_doc_results = []
        for doc_result in biaa_viewer.results:
            if doc_result['ann_file'] == str(fname):
                all_doc_results.append( doc_result )
        assert all_doc_results
        # TODO: a better soultion is required for displaying disagreements of 3 and more annotators
        for doc_result in all_doc_results:
            if len(doc_result['mismatches']) > 0:
                if len(biaa_viewer._annotators) > 2:
                    annotators_pair = '({} vs {})'.format(doc_result["annotator_1"], doc_result["annotator_2"])
                    out_str.append( f'  * Disagreements {annotators_pair}:'+newline )
                else:
                    out_str.append( f'  * Disagreements:'+newline )
                for mismatch_str in biaa_viewer.format_mismatches_tuple_as_str_list(doc_result['mismatches']):
                    out_str.append('            '+mismatch_str )
                out_str.append( newline )
        #out_str.append( newline )

    out_str.append( newline+'## Agreement per Label'+newline )
    label_stats = np.stack((f1_agreement.labels, *f1_agreement.mean_sd_per_label())).transpose()
    label_headers = ['Label', 'Mean F1', 'SD F1']
    out_str.append( tabulate(label_stats, headers=label_headers, tablefmt='github', floatfmt=f'.{precision}f') )

    out_str.append( newline+'## Overall Agreement'+newline )
    avg, stddev = f1_agreement.mean_sd_total()
    out_str.append( f'* Mean F1: {avg:.{precision}f}, SD F1: {stddev:.{precision}f}'+newline )

    return out_str


#
# An extended version of the function bratiaa.agree.iaa_report:
# * in case of agreement per document, outputs mismatching annotations (for inspection);
# * prints out the results;
#
def extended_iaa_report(f1_agreement, biaa_viewer, precision=3, newline='\n'):
    for line in extended_iaa_report_str(f1_agreement, biaa_viewer, precision=precision, newline=newline):
        print(line)

