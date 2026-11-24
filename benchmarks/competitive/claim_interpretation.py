"""Interpret existing paired intervals, without scoring or changing the study.

The margins are those already frozen in sources/PUBLIC_OUTPUT_SPEC.json.
These labels are reporting, not new selection, promotion or training gates.
"""
import math


def interval(value):
    if value is None or len(value)!=2:
        return None
    lo,hi=value
    if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in value) or lo>hi:
        return None
    return lo,hi


def ratio_verdict(ci,margin,measured=True,AA_valid=True):
    """Candidate loss/latency divided by NT210: larger favors NeuroTabular."""
    bounds=interval(ci)
    if not measured or not AA_valid or bounds is None or bounds[0]<=0:
        return 'INSUFFICIENT_EVIDENCE'
    lo,hi=bounds
    if 1-margin<=lo<=hi<=1+margin:
        return 'EQUIVALENT_WITHIN_FROZEN_MARGIN'
    if lo>1+margin:
        return 'NT210_USEFUL_ADVANTAGE'
    if hi<1-margin:
        return 'COMPETITOR_USEFUL_ADVANTAGE'
    return 'INSUFFICIENT_EVIDENCE_FOR_USEFUL_MARGIN'


def balanced_accuracy_verdict(ci,measured=True):
    """Candidate balanced accuracy minus NT210; negative favors NeuroTabular."""
    bounds=interval(ci)
    if not measured or bounds is None:
        return 'INSUFFICIENT_EVIDENCE'
    lo,hi=bounds
    band=.005
    if -band<=lo<=hi<=band:
        return 'EQUIVALENT_WITHIN_FROZEN_MARGIN'
    if hi < -band:
        return 'NT210_USEFUL_ADVANTAGE'
    if lo > band:
        return 'COMPETITOR_USEFUL_ADVANTAGE'
    return 'INSUFFICIENT_EVIDENCE_FOR_USEFUL_MARGIN'


def NLL_difference_verdict(ci,measured=True):
    """Keep the qualified absolute NLL interval; never invent a ratio CI."""
    bounds=interval(ci)
    result=dict(relative_equivalence='NOT_IDENTIFIABLE_FROM_ABSOLUTE_DIFFERENCE_CI',
        primary='INSUFFICIENT_EVIDENCE')
    if not measured or bounds is None:
        return result
    lo,hi=bounds
    if lo>0:
        result['primary']='NT210_LOWER_NLL_CONDITIONAL_INTERVAL'
    elif hi<0:
        result['primary']='COMPETITOR_LOWER_NLL_CONDITIONAL_INTERVAL'
    else:
        result['primary']='INCONCLUSIVE_NLL_CONDITIONAL_INTERVAL'
    return result


def eligibility(record,reference,clock_metric=False):
    if record.get('status')!='MEASURED' or reference.get('status')!='MEASURED':
        return False
    if record.get('track')=='hpo4' and not all(r.get('hpo_complete_search') is True for r in [record,reference]):
        return False
    return not clock_metric or all(r.get('inference_A_A_status')=='VALID' for r in [record,reference])
