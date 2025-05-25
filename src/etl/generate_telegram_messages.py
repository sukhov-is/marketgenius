import pandas as pd
import os
from datetime import datetime
import re

# --- Настройки ---
CSV_FILES = {
    'blogs': 'data/processed/gpt/telegram_blogs.csv',
    'news': 'data/processed/gpt/telegram_news.csv',
}

TEMPLATE = {
    'blogs': (
        '📅 {date}\n\n'
        '📝 *Анализ настроений:*\n{tg_summary}\n\n'
        '📊 Топ оценок:\n{top_companies}\n'
        '---\n'
        'Сообщение сгенерировано нейросетью, достоверность информации не гарантируется.\n'
        'Оценки (-3...+3) выставлены автоматически моделью и не являются инвестиционной рекомендацией.'
    ),
    'news': (
        '📅 {date}\n\n'
        '📰 *Самари новостей:*\n\n{tg_summary}\n\n'
        '📊 Топ оценок:\n{top_companies}'
        '---\n'
        'Сообщение сгенерировано нейросетью, достоверность информации не гарантируется.\n'
        'Оценки (-3...+3) выставлены автоматически моделью и не являются инвестиционной рекомендацией.'
    ),
}

# --- Функции ---
def get_top_companies(row, n=3):
    companies = row.index[2:]
    values = row.values[2:]
    comp_scores = [
        (comp, float(val))
        for comp, val in zip(companies, values)
        if str(val).strip() not in ('', 'nan', 'None') and abs(float(val)) > 0
    ]
    if not comp_scores:
        return 'Нет оценок'
    # Сортируем: сначала MOEX*, потом остальные, внутри — по модулю оценки
    moex = [x for x in comp_scores if x[0].startswith('MOEX')]
    other = [x for x in comp_scores if not x[0].startswith('MOEX')]
    moex.sort(key=lambda x: abs(x[1]), reverse=True)
    other.sort(key=lambda x: abs(x[1]), reverse=True)
    sorted_scores = moex + other
    top = sorted_scores[:n]

    def arrow(val):
        return '🟢' if val > 0 else '🔴'

    tokens = [f'#{c} : {v:+.1f}' for c, v in top]
    lines = [' | '.join(tokens[i:i+3]) for i in range(0, len(tokens), 3)]
    return '\n'.join(lines) + '\n'


def format_summary(summary, bullet='🔹'):
    # Список распространенных сокращений
    abbreviations = r'(?:кв|г|гг|млн|млрд|трлн|тыс|руб|долл|евро|проц|п\.п|т\.д|т\.п|др|см|стр|рис|табл|ул|пр|корп|д|кв|эт|тел|факс|e-mail|email|vs|etc|i\.e|e\.g)'
    
    # Улучшенное регулярное выражение для разбиения на предложения
    # Не разбиваем после:
    # 1. Сокращений (кв., г., млн. и т.д.)
    # 2. Если после знака препинания идет строчная буква или цифра
    # 3. Если это часть названия в кавычках
    pattern = r'''
        (?<![А-Яа-яA-Za-z])  # Не предшествует буква
        ''' + abbreviations + r'''  # Известные сокращения
        \.  # Точка после сокращения
        |  # ИЛИ
        (?<=[.!?])  # После точки, восклицательного или вопросительного знака
        (?!["\'\s]*[а-яa-z0-9])  # Но не перед строчной буквой или цифрой (с учетом кавычек)
        \s+  # Пробел(ы)
        (?=[А-ЯA-Z"\'«])  # Перед заглавной буквой или кавычкой
    '''
    
    # Используем negative split - сначала защищаем сокращения
    temp_summary = summary.strip()
    
    # Защищаем сокращения временной заменой точек
    protected = re.sub(r'\b(' + abbreviations + r')\.', r'\1<DOT>', temp_summary, flags=re.IGNORECASE)
    
    # Разбиваем на предложения
    sentences = re.split(r'(?<=[.!?])\s+(?=[А-ЯA-Z"\'«])', protected)
    
    # Восстанавливаем точки в сокращениях
    sentences = [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]
    
    # Добавляем буллеты и склеиваем
    sentences = [f'{bullet} {s}' for s in sentences]
    return '\n\n'.join(sentences)


def generate_messages(csv_path, mode='blogs', n=3, output_path=None, start_date=None, end_date=None):
    df = pd.read_csv(csv_path)
    # Фильтрация по диапазону дат, если указано
    if start_date:
        df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] <= end_date]
    messages = []
    for _, row in df.iterrows():
        bullet = '🔹' if mode == 'news' else '🔸'
        formatted_summary = format_summary(str(row['tg_summary']), bullet)
        # Приводим дату к формату DD.MM.YYYY
        date_raw = str(row['date'])
        try:
            date_fmt = datetime.strptime(date_raw, '%Y-%m-%d').strftime('%d.%m.%Y')
        except ValueError:
            date_fmt = date_raw

        msg = TEMPLATE[mode].format(
            date=date_fmt,
            tg_summary=formatted_summary,
            top_companies=get_top_companies(row, n=n)
        )
        messages.append((date_fmt, msg))
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, msg in messages:
                f.write(msg + '\n' + ('-'*40) + '\n')
    else:
        for _, msg in messages:
            print(msg)
            print('-'*40)

    return messages

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Генерация сообщений для Telegram-канала из CSV.')
    parser.add_argument('--mode', choices=['blogs', 'news'], default='news', help='Тип файла: blogs или news')
    parser.add_argument('--top', type=int, default=21, help='Сколько компаний выводить в топе')
    parser.add_argument('--output', type=str, default="data/processed/gpt/output/news.txt", help='Путь для сохранения сообщений (txt)')
    parser.add_argument('--start-date', type=str, default=None, help='Начальная дата (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='Конечная дата (YYYY-MM-DD)')
    args = parser.parse_args()

    csv_path = CSV_FILES[args.mode]
    generate_messages(
        csv_path,
        mode=args.mode,
        n=args.top,
        output_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date
    ) 