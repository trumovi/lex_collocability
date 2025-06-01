# Lex Collocability

**Lex Collocability** — это веб-приложение на Flask для анализа лексической сочетаемости слов.

---

## 🚀 Быстрый старт

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/trumovi/lex_collocability.git
cd lex_collocability
```

### 2. Установите зависимости

Рекомендуется использовать виртуальное окружение:

```bash
python -m venv venv
source venv/bin/activate  # Для Windows: venv\Scripts\activate
```

Установите необходимые библиотеки:

```bash
pip install -r requirements.txt
```

### 3. Загрузите модель

Скачайте модель из [Google Drive](https://drive.google.com/drive/folders/1qkkoyDacKiVXj9Sey8D161wWhPHJwTmv?usp=sharing).

После загрузки переместите папку `model/` в корень проекта:

```bash
# Пример для Linux/macOS
mv /путь/к/скачанной/папке/model ./model
```

Убедитесь, что структура проекта выглядит следующим образом:

```
lex_collocability/
├── app.py
├── model/
│   └── [файлы модели]
├── requirements.txt
└── README.md
└── main.py
```

### 4. Запустите приложение

```bash
python app.py
```

После запуска откройте в браузере адрес [http://127.0.0.1:5000](http://127.0.0.1:5000) для доступа к веб-интерфейсу.

---

### Альтернатива: Скачивание ZIP-архива

Если вы не хотите использовать Git, вы можете скачать проект в виде ZIP-файла:

1. Перейдите на страницу репозитория: [https://github.com/trumovi/lex_collocability](https://github.com/trumovi/lex_collocability)
2. Нажмите зелёную кнопку **Code** и выберите **Download ZIP**.
3. Распакуйте архив в удобную директорию.
4. Далее следуйте инструкциям, начиная с пункта **2. Установите зависимости**.
