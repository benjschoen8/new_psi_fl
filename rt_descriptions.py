"""
Client-written class descriptions for the RT protocol (simulation of what each owner would type).

Each client writes a description of each of ITS OWN classes, in ITS OWN language, knowing nothing
about the other clients. Nothing here is shared: the only thing agreed in advance is the public
anchor list (rt_protocol.TEXT_ANCHORS).

describe(dataset, class_name, lang) -> str
Languages: en, zh, es, ja, fr, de. Unknown datasets / names fall back to "<name>" in a language template.
Override with a JSON file (yaml key rt_descriptions): {"<client_id>": {"<local_id>": "text", ...}, ...}.
"""
DIGIT = {
    "en": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
    "zh": ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"],
    "es": ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"],
    "ja": ["ゼロ", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう"],
    "fr": ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"],
    "de": ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"],
}
DIGIT_T = {"en": "the handwritten digit {}", "zh": "手寫數字{}", "es": "el dígito escrito a mano {}",
           "ja": "手書きの数字の{}", "fr": "le chiffre manuscrit {}", "de": "die handgeschriebene Ziffer {}"}
UPPER_T = {"en": "the handwritten capital letter {}", "zh": "手寫大寫英文字母{}",
           "es": "la letra mayúscula escrita a mano {}", "ja": "手書きの大文字の{}",
           "fr": "la lettre majuscule manuscrite {}", "de": "der handgeschriebene Großbuchstabe {}"}
LOWER_T = {"en": "the handwritten small letter {}", "zh": "手寫小寫英文字母{}",
           "es": "la letra minúscula escrita a mano {}", "ja": "手書きの小文字の{}",
           "fr": "la lettre minuscule manuscrite {}", "de": "der handgeschriebene Kleinbuchstabe {}"}
PHOTO_T = {"en": "a photo of a {}", "zh": "一張{}的照片", "es": "una foto de {}", "ja": "{}の写真",
           "fr": "une photo {}", "de": "ein Foto von {}"}
CIFAR10 = {
    "airplane":   {"zh": "飛機", "es": "un avión", "ja": "飛行機", "fr": "d'un avion", "de": "einem Flugzeug"},
    "automobile": {"zh": "汽車", "es": "un coche", "ja": "自動車", "fr": "d'une voiture", "de": "einem Auto"},
    "bird":       {"zh": "鳥", "es": "un pájaro", "ja": "鳥", "fr": "d'un oiseau", "de": "einem Vogel"},
    "cat":        {"zh": "貓", "es": "un gato", "ja": "猫", "fr": "d'un chat", "de": "einer Katze"},
    "deer":       {"zh": "鹿", "es": "un ciervo", "ja": "鹿", "fr": "d'un cerf", "de": "einem Hirsch"},
    "dog":        {"zh": "狗", "es": "un perro", "ja": "犬", "fr": "d'un chien", "de": "einem Hund"},
    "frog":       {"zh": "青蛙", "es": "una rana", "ja": "カエル", "fr": "d'une grenouille", "de": "einem Frosch"},
    "horse":      {"zh": "馬", "es": "un caballo", "ja": "馬", "fr": "d'un cheval", "de": "einem Pferd"},
    "ship":       {"zh": "船", "es": "un barco", "ja": "船", "fr": "d'un bateau", "de": "einem Schiff"},
    "truck":      {"zh": "卡車", "es": "un camión", "ja": "トラック", "fr": "d'un camion", "de": "einem Lastwagen"},
}
LANGS = list(DIGIT)
# simple keywords (fuzzy union): the plain word a person types for the class in their language
DIGIT_KW = dict(DIGIT, ja=["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"])   # kanji, not readings
CIFAR_KW = {"en": "plane car bird cat deer dog frog horse ship truck",
            "zh": "飛機 汽車 鳥 貓 鹿 狗 青蛙 馬 船 卡車",
            "ja": "飛行機 車 鳥 猫 鹿 犬 カエル 馬 船 トラック",
            "es": "avión coche pájaro gato ciervo perro rana caballo barco camión",
            "fr": "avion voiture oiseau chat cerf chien grenouille cheval bateau camion",
            "de": "Flugzeug Auto Vogel Katze Hirsch Hund Frosch Pferd Schiff Lastwagen"}


# English-only: two writers per class, keywords that match for sure (MCC 1.0 in calibration):
# writer en0 types the plain word, en1 the capitalised word ("cat" / "Cat"); single letters as is.
EN_WORDS = {"airplane": "airplane", "automobile": "car", "bird": "bird", "cat": "cat", "deer": "deer",
            "dog": "dog", "frog": "frog", "horse": "horse", "ship": "ship", "truck": "truck"}


# test-only synonym writers (tests.fuzzy_topk): 'syn0' / 'syn1' / 'syn2' name a class in different words
SYNONYMS = {"airplane": ["airplane", "plane", "aircraft"], "automobile": ["car", "automobile", "auto"],
            "bird": ["bird", "birds", "songbird"], "cat": ["cat", "kitty", "feline"],
            "deer": ["deer", "stag", "deer"], "dog": ["dog", "puppy", "canine"],
            "frog": ["frog", "toad", "frogs"], "horse": ["horse", "pony", "stallion"],
            "ship": ["ship", "boat", "vessel"], "truck": ["truck", "lorry", "pickup"]}


def keyword(dataset, name, lang):
    """The bare word a client types for its label (fuzzy union): 'three', 'tres', '三', 'gato', 'g'.
    lang 'en0' / 'en1': English writers, at most two keywords per class: the plain word and the
    capitalised word ('car' / 'Car', 'three' / 'Three'); letters as is ('A' and 'a' stay different)."""
    name = str(name)
    if lang[:3] == "syn" and lang[3:].isdigit():
        if name.isdigit() and len(name) == 1:
            return DIGIT["en"][int(name)]
        return SYNONYMS[name][int(lang[3:]) % 3] if name in SYNONYMS else name
    if lang[:2] == "en" and lang[2:].isdigit():
        word = (DIGIT["en"][int(name)] if name.isdigit() and len(name) == 1 else EN_WORDS.get(name, name))
        if len(word) == 1:                                        # a letter: case is the class
            return word
        return word.capitalize() if int(lang[2:]) % 2 else word
    if name.isdigit() and len(name) == 1:
        return DIGIT_KW[lang][int(name)]
    if len(name) == 1 and name.isalpha():                         # Latin letters are written as is
        return name
    if name in CIFAR10:
        return CIFAR_KW[lang].split()[list(CIFAR10).index(name)]
    return name
    if name in CIFAR10:
        if lang == "en":
            return name
        return CIFAR10[name][lang] if lang in ("zh", "ja") else NOUN[lang].split()[list(CIFAR10).index(name)]
    return name


def describe(dataset, name, lang):
    name = str(name)
    if name.isdigit() and len(name) == 1:                         # MNIST / USPS / EMNIST digits
        return DIGIT_T[lang].format(DIGIT[lang][int(name)])
    if len(name) == 1 and name.isalpha():                         # EMNIST letters
        return (UPPER_T if name.isupper() else LOWER_T)[lang].format(name)
    if name in CIFAR10:
        return PHOTO_T[lang].format(name if lang == "en" else CIFAR10[name][lang])
    return PHOTO_T[lang].format(name)                             # unknown: name inside a template


if __name__ == "__main__":
    for lang in LANGS:
        print(lang, "|", describe("MNIST", "3", lang), "|", describe("EMNIST", "g", lang), "|",
              describe("CIFAR10", "truck", lang))
