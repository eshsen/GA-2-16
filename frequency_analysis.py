import nltk
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pymorphy3
import argparse
import os


# Загрузка данных
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

DEFAULT_ENCODING = 'utf-8'


def setup_argparse():
    """
    Настраивает парсер аргументов командной строки.

    Возвращает
    ---------
    argparse.ArgumentParser
        Объект парсера с заданными параметрами:
        - filename: путь к текстовому файлу для анализа
        - top_n: количество топ слов для отображения (по умолчанию 10)
        - encoding: кодировка файла (по умолчанию utf-8)
        - max_len: максимальная длина текста в символах (опционально)
        - save_path: базовый путь для сохранения графиков (опционально)
    """
    parser = argparse.ArgumentParser(
        description='Анализ частоты слов в текстовом файле')
    parser.add_argument('filename', help='Путь к текстовому файлу для анализа')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Количество топ слов (по умолчанию 10)')
    parser.add_argument('--encoding', default=DEFAULT_ENCODING,
                        help='Кодировка файла (по умолчанию utf-8)')
    parser.add_argument('--max_len', type=int, default=None,
                        help='Максимальная длина текста в символах (обрезать текст)')
    parser.add_argument('--save_path', default=None,
                        help='Базовый путь для сохранения графиков. '
                             'Если указан, графики сохраняются в два файла: '
                             'name_abs.png и name_norm.png')
    return parser


def read_text_from_file(filename, encoding=DEFAULT_ENCODING, max_len=None):
    """
    Читает текст из файла.

    Параметры
    ---------
    filename : str
        Путь к текстовому файлу.
    encoding : str, optional
        Кодировка файла (по умолчанию 'utf-8').
    max_len : int или None, optional
        Максимальное количество символов для чтения (по умолчанию None - весь файл).

    Возвращает
    ---------
    str
        Содержимое файла (возможно обрезанное по max_len).

    Исключения
    ---------
    FileNotFoundError
        Если файл не найден.
    Exception
        При неожиданных ошибках при чтении файла.
    """
    try:
        with open(filename, 'r', encoding=encoding) as file:
            text = file.read()
            if max_len is not None:
                text = text[:max_len]
            return text
    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден.")
        raise
    except Exception as e:
        print(f"Неожиданная ошибка при чтении файла: {e}")
        raise


def preprocess_text(text):
    """
    Предобрабатывает текст: токенизация, фильтрация, приведение к нижнему регистру,
    исключение стоп-слов на русском и английском языках.

    Параметры
    ---------
    text : str
        Исходный текст.

    Возвращает
    ---------
    list of str
        Список очищенных и приведенных к нижнему регистру слов без стоп-слов.
    """
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]

    # Исключение стоп-слов (русских и английских)
    stop_words_english = set(stopwords.words('english'))
    stop_words_russian = set(stopwords.words('russian'))
    all_stop_words = stop_words_english.union(stop_words_russian)

    words = [word for word in words if word not in all_stop_words]

    return words


def normalize_words(words):
    """
    Нормализует список слов с помощью морфологического анализатора pymorphy3,
    приводя каждое слово к его нормальной форме.

    Параметры
    ---------
    words : list of str
        Список слов для нормализации.

    Возвращает
    ---------
    list of str
        Список нормализованных слов (лемм).
    """
    morph = pymorphy3.MorphAnalyzer()
    normalized_words = []

    for word in words:
        parsed = morph.parse(word)[0]
        normalized_words.append(parsed.normal_form)

    return normalized_words


def calculate_word_frequencies(words):
    """
    Подсчитывает частоту появления каждого слова в списке с помощью nltk.FreqDist.

    Параметры
    ---------
    words : list of str
        Список слов.

    Возвращает
    ---------
    nltk.probability.FreqDist
        Объект с частотами слов.
    """
    return FreqDist(words)


def normalize_frequencies(fdist, total_words):
    """
    Нормализует частоты слов, деля каждое количество на общее число слов.

    Параметры
    ---------
    fdist : nltk.probability.FreqDist
        Частота слов.
    total_words : int
        Общее количество слов.

    Возвращает
    ---------
    dict
        Словарь {слово: нормироанная частота}, где частота — отношение к общему количеству слов.
        Пустой словарь, если total_words == 0.
    """
    if total_words == 0:
        return {}
    return {word: count / total_words for word, count in fdist.items()}


def plot_frequencies(freqs, title, top_n=10, log_scale=False, n=0, save_path=None):
    """
    Строит столбчатую диаграмму с топ-N слов по частоте.

    Параметры
    ---------
    freqs : dict или nltk.probability.FreqDist
        Словарь частот слов.
    title : str
        Заголовок графика.
    top_n : int, optional
        Количество отображаемых слов (по умолчанию 10).
    log_scale : bool, optional
        Флаг для логарифмического масштаба оси Y (по умолчанию False).
    n : int или float, optional
        Смещение для подписей над столбцами, чтоб текст не накладывался (по умолчанию 0).
    save_path : str или None, optional
        Путь для сохранения графика (если None – показ графика).
    """
    if not freqs:
        return

    top_items = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, counts = zip(*top_items)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_items)), counts, color='skyblue', edgecolor='black')
    plt.xticks(range(len(top_items)), words, rotation=45)
    plt.xlabel('Слова', fontsize=12)
    plt.ylabel('Частота' if not log_scale else 'Частота (лог)', fontsize=12)
    plt.title(title, fontsize=14)
    if log_scale:
        plt.yscale('log')

    for i, count in enumerate(counts):
        plt.text(x=i, y=count + n, s=f"{count:.3g}", va='bottom', ha='center')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    """
    Основная функция для выполнения всех задач
    """
    parser = setup_argparse()
    args = parser.parse_args()

    try:
        if not os.path.exists(args.filename):
            print(f"Ошибка: файл '{args.filename}' не найден!")
            return

        text = read_text_from_file(
            args.filename, encoding=args.encoding, max_len=args.max_len)
        words = preprocess_text(text)
        normalized_words = normalize_words(words)
        fdist = calculate_word_frequencies(normalized_words)
        total_words = len(normalized_words)
        save_base = args.save_path

        # Абсолютные частоты
        if save_base:
            save_abs = f"{save_base}_abs.png"
        else:
            save_abs = None
        plot_frequencies(dict(fdist), 'Абсолютные частоты топ-{}'.format(args.top_n),
                         top_n=args.top_n, n=0.15, save_path=save_abs)

        # Нормированные частоты
        norm_freqs = normalize_frequencies(fdist, total_words)
        if save_base:
            save_norm = f"{save_base}_norm.png"
        else:
            save_norm = None
        plot_frequencies(norm_freqs, 'Нормированные частоты топ-{}'.format(args.top_n),
                         top_n=args.top_n, log_scale=True, n=0.00015, save_path=save_norm)

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()

