# GA-2-16: Frequency analysis
Скрипт для анализа и визуализации частотности слов в тексте с удалением стоп-слов и нормализацией. По умолчанию графики выводятся интерактивно, но можно сохранить их в файл.          
Данная задача основана на GA-1-16: https://github.com/corsaczagi/GenAI-1-16-word_frequency_analysis).  
## Установка
1. Создайте виртуальное окружение (рекомендуется):
   - Python 3.8+
   - В активированном окружении выполните:
     ```
      pip install -r requirements.txt
      ```
2. Убедитесь, что у вас установлен matplotlib (предмет зависимости в requirements.txt)
## Использование
Запустите скрипт:
```
python frequancy_analysis.py filename --top_n --encoding --max_len --save_path
```
