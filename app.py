from flask import Flask, render_template, request, jsonify
from main_og import find_collocation_errors, find_collocation_errors_vec, suggest_corrections, check_collocation_frequency

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/check', methods=['POST'])
def check_text():
    data = request.get_json()
    text = data.get('text', '')
    method = data.get('method', 'dictionary')

    if not text.strip():
        return jsonify({'error': 'Пожалуйста, введите текст для проверки'})

    try:
        if method == 'dictionary':
            errors = find_collocation_errors(text)
        else:
            errors = find_collocation_errors_vec(text)

        results = []
        for (verb_lemma, dep_lemma), (verb, dep) in errors:
            corrections = suggest_corrections(verb_lemma, dep_lemma)

            all_corrections = [f"{corr[0]} {corr[1]} (частотность в корпусе: {corr[2]})" for corr in corrections]

            frequency = check_collocation_frequency(verb_lemma, dep_lemma)

            results.append({
                'error': f"{verb} {dep}",
                'corrections': all_corrections,
                'type': method,
                'frequency': frequency
            })

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': f"Произошла ошибка: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=True)