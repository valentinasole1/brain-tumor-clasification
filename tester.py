import spacy


def split_into_paragraphs(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    paragraphs = [sent.text for sent in doc.sents]
    return paragraphs


text = '''This is the first paragraph.
It has multiple lines.

This is the second paragraph.
It also has multiple lines.

This is the third paragraph.'''

a = split_into_paragraphs(text)

print(a)
