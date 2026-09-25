"""NegEx trigger phrases, transcribed from MetaMapLite's ``NegExKeyMap``.

This module is **data, not algorithm**: it supplies the phrase lexicon only.
Scope resolution stays with :mod:`umlsmatch.assertion.negation`, which applies
its own rules (a token window, dependency-driven coordination and clause
bounding). MetaMapLite's own matching -- meta-token merging, longest-phrase-
at-a-position, a 6-token window -- is *not* reproduced here.

Phrase types, as NegEx defines them:

  ``nega``   pre-negation: negates what follows ("no evidence of")
  ``negb``   post-negation: negates what precedes ("was ruled out")
  ``pnega``  pseudo pre-negation: looks like a trigger, is not ("rule her out")
  ``pnegb``  pseudo post-negation ("not ruled out")
  ``conj``   scope terminator ("but", "as a cause of")

:mod:`umlsmatch.assertion.negation` maps these onto its four tuples and unions
them with locally curated phrases that NegEx does not carry; see that module.

-------------------------------------------------------------------------------
Provenance and licence
-------------------------------------------------------------------------------

Derived from MetaMapLite, developed and funded by the U.S. National Library of
Medicine. Courtesy of the U.S. National Library of Medicine.

MetaMapLite is distributed under NLM's open-source BSD licence, which requires
that redistributions of source code retain the Informational Notice below. It is
reproduced here verbatim for that reason; see also the NOTICE file at the root
of this distribution.

Neither the names of the National Library of Medicine, the National Institutes
of Health, nor the names of any of the software developers may be used to
endorse or promote this software. umlsmatch was not developed, funded, endorsed
or reviewed by the NLM, the NIH, or the U.S. Government.

    Informational Notice:

    This software, "MetaMapLite" was developed and funded by the National
    Library of Medicine, part of the National Institutes of Health, and
    agency of the United States Department of Health and Human Services,
    which is making the software available to the public for any
    commercial or non-commercial purpose under the following open-source
    BSD license.

    NOTE: Users of the data distributed with MetaMapLite are
    responsible for compliance with the UMLS Metathesaurus License
    Agreement which requires you to respect the copyrights of the
    constituent vocabularies and to file a brief annual report on your use
    of the UMLS. You also must have activated a UMLS Terminology Services
    (UTS) account.

    LICENSE:

    Government Usage Rights Notice: The U.S. Government retains unlimited,
    royalty-free usage rights to this software, but not ownership, as
    provided by Federal law.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    * Redistributions of source code must retain this Informational Notice.
    * Redistributions in binary form must reproduce this Informational
      Notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of the National Library of Medicine, the National
      Institutes of Health, nor the names of any of the software
      developers may be used to endorse or promote products derived from
      this software without specific prior written permission.  The
      U.S. Government retains an unlimited, royalty-free right to use,
      distribute or modify the software.
    * Please acknowledge NLM as the source of the MetaMapLite software by
      including the phrase "Courtesy of the U.S. National Library of
      Medicine" or "Source: U.S. National Library of Medicine."

    THIS SOFTWARE IS PROVIDED BY THE U.S. GOVERNMENT AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITEDTO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    U.S. GOVERNMENT

    OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

__all__ = ["NEGATION_PHRASE_TYPES"]

NEGATION_PHRASE_TYPES: dict[tuple[str, ...], str] = {
    ("rules", "out"): "nega",
    ("rules", "him", "out", "for"): "nega",
    ("can", "rule", "her", "out", "against"): "nega",
    ("did", "rule", "her", "out"): "nega",
    ("sufficient", "to", "rule", "out", "against"): "nega",
    ("adequate", "to", "rule", "the", "patient", "out", "against"): "nega",
    ("never", "developed"): "nega",
    ("denied",): "nega",
    ("no", "signs", "of"): "nega",
    ("no", "sign", "of"): "nega",
    ("sufficient", "to", "rule", "the", "patient", "out", "for"): "nega",
    ("not", "had"): "nega",
    ("not", "to", "be"): "nega",
    ("can", "rule", "out", "for"): "nega",
    ("no", "findings", "to", "indicate"): "nega",
    ("can", "rule", "out", "against"): "nega",
    ("checked", "for"): "nega",
    ("not",): "nega",
    ("sufficient", "to", "rule", "out"): "nega",
    ("no", "evidence"): "nega",
    ("did", "rule", "him", "out", "against"): "nega",
    ("sufficient", "to", "rule", "him", "out", "against"): "nega",
    ("no", "cause", "of"): "nega",
    ("not", "appear"): "nega",
    ("did", "rule", "him", "out"): "nega",
    ("no", "other", "evidence"): "nega",
    ("did", "rule", "her", "out", "for"): "nega",
    ("sufficient", "to", "rule", "her", "out"): "nega",
    ("sufficient", "to", "rule", "the", "patient", "out", "against"): "nega",
    ("adequate", "to", "rule", "her", "out", "for"): "nega",
    ("unremarkable", "for"): "nega",
    ("no", "new"): "nega",
    ("absence", "of"): "nega",
    ("adequate", "to", "rule", "the", "patient", "out"): "nega",
    ("no", "new", "evidence"): "nega",
    ("to", "exclude"): "nega",
    ("sufficient", "to", "rule", "her", "out", "for"): "nega",
    ("not", "known", "to", "have"): "nega",
    ("no", "findings", "of"): "nega",
    ("not", "feel"): "nega",
    ("can", "rule", "him", "out", "against"): "nega",
    ("sufficient", "to", "rule", "out", "for"): "nega",
    ("adequate", "to", "rule", "out", "for"): "nega",
    ("cannot",): "nega",
    ("denying",): "nega",
    ("not", "exhibit"): "nega",
    ("rules", "out", "for"): "nega",
    ("with", "no"): "nega",
    ("ruled", "out", "against"): "nega",
    ("without", "any", "evidence", "of"): "nega",
    ("no", "radiographic", "evidence", "of"): "nega",
    ("rules", "him", "out"): "nega",
    ("ruled", "the", "patient", "out"): "nega",
    ("rules", "the", "patient", "out"): "nega",
    ("can", "rule", "her", "out"): "nega",
    ("no",): "nega",
    ("no", "mammographic", "evidence", "of"): "nega",
    ("denies",): "nega",
    ("rules", "the", "patient", "out", "for"): "nega",
    ("did", "rule", "the", "patient", "out"): "nega",
    ("did", "rule", "out", "against"): "nega",
    ("no", "suggestion", "of"): "nega",
    ("patient", "was", "not"): "nega",
    ("adequate", "to", "rule", "the", "patient", "out", "for"): "nega",
    ("no", "abnormal"): "nega",
    ("negative", "for"): "nega",
    ("ruled", "him", "out", "for"): "nega",
    ("did", "rule", "out"): "nega",
    ("evaluate", "for"): "nega",
    ("fails", "to", "reveal"): "nega",
    ("ruled", "her", "out"): "nega",
    ("cannot", "see"): "nega",
    ("did", "rule", "her", "out", "against"): "nega",
    ("not", "demonstrate"): "nega",
    ("sufficient", "to", "rule", "him", "out"): "nega",
    ("ruled", "him", "out"): "nega",
    ("can", "rule", "the", "patient", "out", "against"): "nega",
    ("no", "suspicious"): "nega",
    ("can", "rule", "him", "out"): "nega",
    ("can", "rule", "her", "out", "for"): "nega",
    ("rules", "her", "out"): "nega",
    ("declined",): "nega",
    ("ruled", "out", "for"): "nega",
    ("can", "rule", "the", "patient", "out"): "nega",
    ("not", "associated", "with"): "nega",
    ("resolved",): "nega",
    ("adequate", "to", "rule", "out"): "nega",
    ("not", "complain", "of"): "nega",
    ("without", "indication", "of"): "nega",
    ("without",): "nega",
    ("not", "reveal"): "nega",
    ("without", "evidence"): "nega",
    ("ruled", "the", "patient", "out", "for"): "nega",
    ("sufficient", "to", "rule", "her", "out", "against"): "nega",
    ("no", "evidence", "to", "suggest"): "nega",
    ("test", "for"): "nega",
    ("declines",): "nega",
    ("sufficient", "to", "rule", "the", "patient", "out"): "nega",
    ("ruled", "out"): "nega",
    ("did", "rule", "him", "out", "for"): "nega",
    ("can", "rule", "the", "patient", "out", "for"): "nega",
    ("rules", "her", "out", "for"): "nega",
    ("never", "had"): "nega",
    ("not", "have"): "nega",
    ("did", "rule", "the", "patient", "out", "against"): "nega",
    ("not", "know", "of"): "nega",
    ("without", "sign", "of"): "nega",
    ("adequate", "to", "rule", "her", "out"): "nega",
    ("no", "significant"): "nega",
    ("can", "rule", "him", "out", "for"): "nega",
    ("deny",): "nega",
    ("not", "appreciate"): "nega",
    ("ruled", "her", "out", "for"): "nega",
    ("ruled", "the", "patient", "out", "against"): "nega",
    ("rather", "than"): "nega",
    ("did", "rule", "the", "patient", "out", "for"): "nega",
    ("no", "complaints", "of"): "nega",
    ("adequate", "to", "rule", "him", "out"): "nega",
    ("free", "of"): "nega",
    ("can", "rule", "out"): "nega",
    ("ruled", "him", "out", "against"): "nega",
    ("ruled", "her", "out", "against"): "nega",
    ("did", "rule", "out", "for"): "nega",
    ("adequate", "to", "rule", "him", "out", "for"): "nega",
    ("not", "see"): "nega",
    ("sufficient", "to", "rule", "him", "out", "for"): "nega",
    ("unlikely",): "negb",
    ("has", "been", "ruled", "out"): "negb",
    ("free",): "negb",
    ("was", "ruled", "out"): "negb",
    ("have", "been", "ruled", "out"): "negb",
    ("are", "ruled", "out"): "negb",
    ("is", "ruled", "out"): "negb",
    ("ought", "to", "be", "ruled", "out", "for"): "pnega",
    ("rule", "the", "patient", "out"): "pnega",
    ("rule", "her", "out"): "pnega",
    ("might", "be", "ruled", "out", "for"): "pnega",
    ("could", "be", "ruled", "out", "for"): "pnega",
    ("rule", "the", "patient", "out", "for"): "pnega",
    ("will", "be", "ruled", "out", "for"): "pnega",
    ("rule", "him", "out", "for"): "pnega",
    ("what", "must", "be", "ruled", "out", "is"): "pnega",
    ("may", "be", "ruled", "out", "for"): "pnega",
    ("r/o",): "pnega",
    ("ro",): "pnega",
    ("can", "be", "ruled", "out", "for"): "pnega",
    ("should", "be", "ruled", "out", "for"): "pnega",
    ("rule", "him", "out"): "pnega",
    ("rule", "out"): "pnega",
    ("rule", "her", "out", "for"): "pnega",
    ("rule", "out", "for"): "pnega",
    ("be", "ruled", "out", "for"): "pnega",
    ("must", "be", "ruled", "out", "for"): "pnega",
    ("is", "to", "be", "ruled", "out", "for"): "pnega",
    ("be", "ruled", "out"): "pnegb",
    ("not", "ruled", "out"): "pnegb",
    ("being", "ruled", "out"): "pnegb",
    ("will", "be", "ruled", "out"): "pnegb",
    ("did", "not", "rule", "out"): "pnegb",
    ("might", "be", "ruled", "out"): "pnegb",
    ("ought", "to", "be", "ruled", "out"): "pnegb",
    ("may", "be", "ruled", "out"): "pnegb",
    ("not", "been", "ruled", "out"): "pnegb",
    ("should", "be", "ruled", "out"): "pnegb",
    ("can", "be", "ruled", "out"): "pnegb",
    ("could", "be", "ruled", "out"): "pnegb",
    ("must", "be", "ruled", "out"): "pnegb",
    ("is", "to", "be", "ruled", "out"): "pnegb",
    ("although",): "conj",
    ("apart", "from"): "conj",
    ("as", "a", "cause", "for"): "conj",
    ("as", "a", "cause", "of"): "conj",
    ("as", "a", "etiology", "for"): "conj",
    ("as", "a", "etiology", "of"): "conj",
    ("as", "a", "reason", "for"): "conj",
    ("as", "a", "reason", "of"): "conj",
    ("as", "a", "secondary", "cause", "for"): "conj",
    ("as", "a", "secondary", "cause", "of"): "conj",
    ("as", "a", "secondary", "etiology", "for"): "conj",
    ("as", "a", "secondary", "etiology", "of"): "conj",
    ("as", "a", "secondary", "origin", "for"): "conj",
    ("as", "a", "secondary", "origin", "of"): "conj",
    ("as", "a", "secondary", "reason", "for"): "conj",
    ("as", "a", "secondary", "reason", "of"): "conj",
    ("as", "a", "secondary", "source", "for"): "conj",
    ("as", "a", "secondary", "source", "of"): "conj",
    ("as", "a", "source", "for"): "conj",
    ("as", "a", "source", "of"): "conj",
    ("as", "an", "cause", "for"): "conj",
    ("as", "an", "cause", "of"): "conj",
    ("as", "an", "etiology", "for"): "conj",
    ("as", "an", "etiology", "of"): "conj",
    ("as", "an", "origin", "for"): "conj",
    ("as", "an", "origin", "of"): "conj",
    ("as", "an", "reason", "for"): "conj",
    ("as", "an", "reason", "of"): "conj",
    ("as", "an", "secondary", "cause", "for"): "conj",
    ("as", "an", "secondary", "cause", "of"): "conj",
    ("as", "an", "secondary", "etiology", "for"): "conj",
    ("as", "an", "secondary", "etiology", "of"): "conj",
    ("as", "an", "secondary", "origin", "for"): "conj",
    ("as", "an", "secondary", "origin", "of"): "conj",
    ("as", "an", "secondary", "reason", "for"): "conj",
    ("as", "an", "secondary", "reason", "of"): "conj",
    ("as", "an", "secondary", "source", "for"): "conj",
    ("as", "an", "secondary", "source", "of"): "conj",
    ("as", "an", "source", "for"): "conj",
    ("as", "an", "source", "of"): "conj",
    ("as", "the", "cause", "for"): "conj",
    ("as", "the", "cause", "of"): "conj",
    ("as", "the", "etiology", "for"): "conj",
    ("as", "the", "etiology", "of"): "conj",
    ("as", "the", "origin", "for"): "conj",
    ("as", "the", "origin", "of"): "conj",
    ("as", "the", "reason", "for"): "conj",
    ("as", "the", "reason", "of"): "conj",
    ("as", "the", "secondary", "cause", "for"): "conj",
    ("as", "the", "secondary", "cause", "of"): "conj",
    ("as", "the", "secondary", "etiology", "for"): "conj",
    ("as", "the", "secondary", "etiology", "of"): "conj",
    ("as", "the", "secondary", "origin", "for"): "conj",
    ("as", "the", "secondary", "origin", "of"): "conj",
    ("as", "the", "secondary", "reason", "for"): "conj",
    ("as", "the", "secondary", "reason", "of"): "conj",
    ("as", "the", "secondary", "source", "for"): "conj",
    ("as", "the", "secondary", "source", "of"): "conj",
    ("as", "the", "source", "for"): "conj",
    ("as", "the", "source", "of"): "conj",
    ("aside", "from"): "conj",
    ("but",): "conj",
    ("cause", "for"): "conj",
    ("cause", "of"): "conj",
    ("causes", "for"): "conj",
    ("causes", "of"): "conj",
    ("etiology", "for"): "conj",
    ("etiology", "of"): "conj",
    ("except",): "conj",
    ("however",): "conj",
    ("nevertheless",): "conj",
    ("origin", "for"): "conj",
    ("origin", "of"): "conj",
    ("origins", "for"): "conj",
    ("origins", "of"): "conj",
    ("other", "possibilities", "of"): "conj",
    ("reason", "for"): "conj",
    ("reason", "of"): "conj",
    ("reasons", "for"): "conj",
    ("reasons", "of"): "conj",
    ("secondary", "to"): "conj",
    ("source", "for"): "conj",
    ("source", "of"): "conj",
    ("sources", "for"): "conj",
    ("sources", "of"): "conj",
    ("still",): "conj",
    ("though",): "conj",
    ("trigger", "event", "for"): "conj",
    ("yet",): "conj",
    ("other", "than"): "conj",
    ("otherwise",): "conj",
    ("then",): "conj",
    ("to", "account", "for"): "conj",
    ("to", "explain"): "conj",
}
