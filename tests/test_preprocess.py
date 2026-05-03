from medinfer.nlp.preprocess import normalize


def test_curly_quotes_become_straight():
    assert normalize("I can’t breathe, it’s “bad”") == "I can't breathe, it's \"bad\""


def test_line_breaks_and_tabs_become_spaces():
    assert normalize("fever\nand\tcough\r\n") == "fever and cough  "


def test_length_is_preserved():
    # Offsets from extraction must line up with the original text.
    text = "“No fever”\n\tbut I’m coughing\r\n"
    assert len(normalize(text)) == len(text)


def test_offsets_still_point_at_the_same_words():
    text = "I’ve been\nburning up"
    start = text.index("burning up")
    assert normalize(text)[start:start + len("burning up")] == "burning up"


def test_plain_text_unchanged():
    text = "I have a dry cough and a headache."
    assert normalize(text) == text
