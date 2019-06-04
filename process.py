# -*- coding: utf-8 -*-

"""
Process text data
"""

from pythainlp.tokenize import word_tokenize
from pythainlp.ulmfit import pre_rules_th,post_rules_th
from typing import Collection, Callable

__all__ = ['process_thai']

#temporary fix before next pythainlp
from fastai.text import TK_REP, TK_WREP
import re
def replace_rep_nonum(text: str) -> str:
    """
    Replace repetitions at the character level in `text` after the repetition.
    This is done to prevent such case as 'น้อยยยยยยยย' becoming 'น้อ xrep 8 ย';
    instead it will retain the word as 'น้อย xrep 8'
    """
    def _replace_rep(m):
        c, cc = m.groups()
        return f"{c} {TK_REP} "
    re_rep = re.compile(r"(\S)(\1{3,})")
    return re_rep.sub(_replace_rep, text)
pre_rules_th = [pre_rules_th[0]] + [replace_rep_nonum] + pre_rules_th[2:]

def replace_wrep_post_nonum(toks:Collection):
    """Replace reptitive words post tokenization; 
    fastai `replace_wrep` does not work well with Thai."""
    previous_word = None
    rep_count = 0
    res = []
    for current_word in toks+['xxend']:
        if current_word==previous_word: 
            rep_count+=1
        elif (current_word!=previous_word) & (rep_count>0):
            res += [TK_WREP,previous_word]
            rep_count=0
        else:
            res.append(previous_word)
        previous_word=current_word
    return res[1:]

def remove_space(toks:Collection):
    """
    Do not include space for bag-of-word models
    """
    res = []
    for t in toks:
        if t!=' ': res.append(t)
    return res
post_rules_th = post_rules_th + [replace_wrep_post_nonum, remove_space] 

def process_thai(text: str, pre_rules: Collection = pre_rules_th, tok_func: Callable = word_tokenize,
                post_rules: Collection = post_rules_th) -> Collection[str]:
    """
    process Thai text with the same convention as thai2fit as default
    :param str text: text to be cleaned
    :param pre_rules List: rules to apply before tokenization
    :param tok_func Callable: tokenization function
    :param post_rules List: rules to apply after tokenization
    :return: a list of cleaned tokenized texts
    """
    res = text
    for pre in pre_rules: res = pre(res)
    res = word_tokenize(res)
    for post in post_rules: res = post(res)
    return res