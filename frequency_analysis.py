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
    Настройка аргументов командной строки
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
    Читает текст из файла
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
    Предобработка текста: токенизация, фильтрация, приведение к нижнему регистру
    Исключение стоп-слов (русских и английских)
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
    Нормализация слов с помощью pymorphy3
    """
    morph = pymorphy3.MorphAnalyzer()
    normalized_words = []

    for word in words:
        parsed = morph.parse(word)[0]
        normalized_words.append(parsed.normal_form)

    return normalized_words


def calculate_word_frequencies(words):
    """
    Подсчет частоты слов (FreqDist)
    """
    return FreqDist(words)


def normalize_frequencies(fdist, total_words):
    """
    Возвращает словарь нормированных частот: freq / total_words
    """
    if total_words == 0:
        return {}
    return {word: count / total_words for word, count in fdist.items()}


def plot_frequencies(freqs, title, top_n=10, log_scale=False, n=0, save_path=None, index_offset=0):
    """
    Построение bar plot топ-N слов с подписанными осями
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
