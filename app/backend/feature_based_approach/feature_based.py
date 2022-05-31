import re
import spacy
from fastpunct import FastPunct
from nltk.corpus import stopwords
from spacytextblob.spacytextblob import SpacyTextBlob
from textblob import Word


class FeatureBased:
    def __init__(self, emotional=True, repetitions=True, plural=True, spelling=True,
                 explanation=True, punctuation=True, excessivity=True, past=True):
        self.explanation = explanation
        self.emotional = emotional
        self.repetitions = repetitions
        self.plural = plural
        self.spelling = spelling
        self.punctuation = punctuation
        self.excessivity = excessivity
        self.past = past

        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe('spacytextblob')
        self.text = None
        self.prep_doc = None
        self.fastpunct = FastPunct()
        self.stopwords = set(stopwords.words('english'))

    def predict(self, text, params=None):
        """
        params (dict) :
        {
        'load' : float 
        'rep' : int
        'plur' : int
        'spell' : int
        'punct' : int
        'excess' : int
        'past' : float
        }
        """
        out = {}
        if self.emotional:
            out['loaded language'] = self._emotional_feature(text)
        if self.repetitions:
            out['lexical repetitions'] = self._repetitions_feature(text)
        if self.plural:
            out['plural forms'] = self._plural_feature(text)
        if self.spelling:
            out['spelling mistakes'] = self._spelling_feature(text)
        if self.punctuation:
            out['punctuation mistakes'] = self._punctuation_feature(text)
        if self.excessivity:
            out['excessivity'] = self._excessivity_feature(text)
        if self.past:
            out['past'] = self._past_feature(text)

        if params:
            votes = [out['loaded language'][0] < params['load'],
                     out['lexical repetitions'][0] > params['rep'],
                     out['plural forms'][0] > params['plur'],
                     out['spelling mistakes'][0] > params['spell'],
                     out['punctuation mistakes'][0] > params['punct'],
                     out['excessivity'][0] > params['excess'] or out['excessivity'][1] is True,
                     out['past'][0] > params['past']]
            if any(votes):
                answer = True
            else:
                answer = False

        else:
            tokens = [token.text for token in self.prep_doc]
            ratio = int(len(tokens) * 0.2)
            votes = [out['loaded language'][0] < 0,
                     out['lexical repetitions'][0] > ratio,
                     out['plural forms'][0] > ratio,
                     out['spelling mistakes'][0] > ratio,
                     out['punctuation mistakes'][0] > ratio,
                     out['excessivity'][0] > 5 or out['excessivity'][1] is True,
                     out['past'][0]]
            if any(votes):
                answer = True
            else:
                answer = False

        return answer, out

    def _emotional_feature(self, text):
        if self.text != text:
            self.text = text
            self.prep_doc = self.nlp(text)
        score = self.prep_doc._.polarity
        if self.explanation:
            expl = sum([x[0] for x in self.prep_doc._.assessments], [])
        return (score, expl)

    def _repetitions_feature(self, text):
        if self.text != text:
            self.text = text
            self.prep_doc = self.nlp(text)
        all_words = [token.lemma_ for token in self.prep_doc
                     if token.pos_ not in ['PUNCT', 'SYM']]
        repeated_words = list(set([word.lower() for word in all_words
                                   if all_words.count(word) > 1]).difference(self.stopwords))
        return (len(repeated_words), repeated_words)

    def _plural_feature(self, text):
        if self.text != text:
            self.text = text
            self.prep_doc = self.nlp(text)
        plural_forms = []
        for token in self.prep_doc:
            if token.dep_ in ['nsubj', 'ROOT']:
                if token.tag_ == 'PRP':
                    if token.text.lower() in ['we', 'they', 'us', 'them']:
                        plural_forms.append(token.text)
                elif token.tag_ == 'NNS':
                    plural_forms.append(token.text)
        return (len(plural_forms), plural_forms)

    def _spelling_feature(self, text):
        if self.text != text:
            self.text = text
            self.prep_doc = self.nlp(text)
        incorrect_words = [token.text for token in self.prep_doc
                           if Word(token.text.lower()).spellcheck()[0][0] != token.text.lower()]
        return (len(incorrect_words), incorrect_words)

    def _punctuation_feature(self, text):
        if self.text != text:
            self.text = text
            self.prep_doc = self.nlp(text)
        corrected_text = self.fastpunct.punct(text)
        prep_text = [token.text for token in self.prep_doc]
        prep_cor_text = [token.text for token in self.nlp(corrected_text)]
        if prep_text != prep_cor_text:
            return len(prep_cor_text) - len(prep_text), prep_cor_text
        else:
            return 0, []

    def _excessivity_feature(self, text):
        if self.text != text:
            self.text = text
            self.prep_doc = self.nlp(text)
        excl_mark = self.text.count('!')
        mult_excl_mark, excl_num = False, 0
        res = re.search('!!+', text)
        if res:
            mult_excl_mark = True
            excl_num = len(res.group())
        return excl_mark, mult_excl_mark

    def _past_feature(self, text):
        if self.text != text:
            self.text = text
            self.prep_doc = self.nlp(text)
        past = [x.text for x in self.prep_doc if x.morph.get('Tense') == ['Past']]
        all_tenses = sum([1 for x in self.prep_doc if x.morph.get('Tense') != []])
        try:
            return len(past) / all_tenses, past
        except ZeroDivisionError:
            return 0, []
