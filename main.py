import spacy
import pymorphy3
import glob
import pandas as pd
import csv
import os
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

directory_path = r"C:\Users\trumovi\PycharmProjects\lex_prime\corpora"

file_paths = glob.glob(os.path.join(directory_path, "*.csv"))

collocations_dict = {}
with open("dict.csv", newline='', encoding='utf-8') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  for row in reader:
    word1 = row[0]
    word2 = row[1]
    if word1 in collocations_dict:
      collocations_dict[word1].append(word2)
    else:
      collocations_dict[word1] = [word2]


collocations_dict_corpora = {}
for file in file_paths:
  with open(file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    data = []
    for row in reader:
        word1 = row["lex_1"]
        word2 = row["lex_2"]
        frequency = row["w1w2"]
        try:
            frequency = float(frequency)
            if pd.notna(frequency):
                data.append((word1, word2, frequency))
        except (ValueError, TypeError):
            print(f"Пропущено: {word1} + {word2} (некорректное значение frequency: {frequency})")
    data.sort(key=lambda x: x[2], reverse=True)
    for word1, word2, frequency in data:
        if word1 not in collocations_dict_corpora:
            collocations_dict_corpora[word1] = {}
        collocations_dict_corpora[word1][word2] = frequency

word_vectors = KeyedVectors.load("model/model.model")
nlp = spacy.load("ru_core_news_sm")
morph = pymorphy3.MorphAnalyzer()

def lemmatize_noun(noun):
    parsed = morph.parse(noun)[0]
    if 'NOUN' in parsed.tag:
        return parsed.normal_form
    return noun

def find_collocation_errors(text):
    doc = nlp(text)
    errors = []
    for token in doc:
        if token.pos_ == "VERB":
            verb_text = token.text
            verb_lemma = token.lemma_
            dependencies = []
            for child in token.children:
                if child.pos_ in ["NOUN"]:
                    noun_lemma = lemmatize_noun(child.text)
                    has_adp = any(adp.pos_ == "ADP" for adp in child.children)
                    if not has_adp:
                        dependencies.append((child.text, noun_lemma))
            for dep, dep_lemma in dependencies:
                verb_in_dict = verb_lemma in collocations_dict
                dep_in_dict = dep_lemma in collocations_dict

                # Проверяем, что одно слово есть в словаре, а другого нет
                if (verb_in_dict or dep_in_dict) and not (verb_in_dict and dep_in_dict):
                    # Если одно из слов есть в словаре, проверяем сочетание
                    is_valid = (
                            dep_lemma in collocations_dict.get(verb_lemma, [])
                            or verb_lemma in collocations_dict.get(dep_lemma, [])
                    )
                    if not is_valid:
                        errors.append({(verb_lemma.lower(), dep_lemma.lower()), (verb_text.lower(), dep.lower())})
    return errors


def is_collocation_valid(word1, word2, threshold=0.25):
    try:
        vec1 = word_vectors[word1].reshape(1, -1)
        vec2 = word_vectors[word2].reshape(1, -1)
        similarity = cosine_similarity(vec1, vec2)[0][0]
        return similarity >= threshold, similarity
    except KeyError:
        return False, 0.0


def find_collocation_errors_vec(text):
    target_words = {
        'оплатить', 'влияние', 'решение', 'обязанность', 'совет',
        'вопрос', 'обсуждать', 'ошибка', 'мнение', 'интерес'
    }
    doc = nlp(text.lower())
    errors = []
    for token in doc:
        if token.pos_ == "VERB":
            verb = token.text
            verb_lemma = token.lemma_.replace('ё', 'е')
            dependencies = []
            for child in token.children:
                if child.pos_ in ["NOUN"]:
                    noun_lemma = lemmatize_noun(child.text).replace('ё', 'е')
                    has_adp = any(adp.pos_ == "ADP" for adp in child.children)
                    if not has_adp:
                        dependencies.append({child.text, noun_lemma})

            for dep, dep_lemma in dependencies:
                if verb_lemma in target_words or dep_lemma in target_words:
                    is_valid, similarity = is_collocation_valid(verb_lemma, dep_lemma)
                    if not is_valid:
                        errors.append(((verb_lemma, dep_lemma), (verb, dep), similarity))
    return errors

def suggest_corrections(verb, dep):
    suggestions = []
    if verb in collocations_dict_corpora:
        verb_suggestions = collocations_dict_corpora[verb]
        sorted_verb_suggestions = sorted(verb_suggestions.items(), key=lambda x: x[1], reverse=True)[:7]
        for word, freq in sorted_verb_suggestions:
            parsed_word = morph.parse(word)[0]
            if 'NOUN' in parsed_word.tag:
                acc_word = parsed_word.inflect({'accs'}).word
            else:
                acc_word = word
            suggestions.extend([(verb, acc_word, freq)])
    if dep in collocations_dict_corpora:
        dep_suggestions = collocations_dict_corpora[dep]
        sorted_dep_suggestions = sorted(dep_suggestions.items(), key=lambda x: x[1], reverse=True)[:7]
        for word, freq in sorted_dep_suggestions:
            parsed_word = morph.parse(word)[0]
            if 'NOUN' in parsed_word.tag:
                acc_word = parsed_word.inflect({'accs'}).word
            else:
                acc_word = word
            suggestions.extend([(acc_word, dep, freq)])
    unique_suggestions = []
    seen = set()
    for suggestion in suggestions:
        key = (suggestion[0], suggestion[1])
        if key not in seen:
            seen.add(key)
            unique_suggestions.append(suggestion)
    unique_suggestions.sort(key=lambda x: x[2], reverse=True)
    return unique_suggestions


def check_collocation_frequency(verb, dep):
    if verb in collocations_dict_corpora and dep in collocations_dict_corpora[verb]:
        freq = collocations_dict_corpora[verb][dep]
        return f"Сочетание встречается в корпусе {int(freq)} раз"
    if dep in collocations_dict_corpora and verb in collocations_dict_corpora[dep]:
        freq = collocations_dict_corpora[dep][verb]
        return f"Сочетание встречается в корпусе {int(freq)} раз"
    return f"Сочетание не найдено в корпусе"