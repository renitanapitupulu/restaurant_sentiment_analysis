class Sentence:
    def __init__(self,
                 sentence,
                 aspect_terms=[],
                 aspect_categories=[],
                 polarity=''):
        self.sentence = sentence
        self.aspect_terms = aspect_terms
        self.aspect_categories = aspect_categories
        self.polarity = polarity