import pandas as pd

# Загрузка Excel файла
input_file = 'news-lenta-cleaned-nan.xlsx'  # Замените на путь к вашему файлу
output_file = 'news-lenta-cleaned-all.xlsx'  # Замените на путь для сохранения результата

# Чтение данных из Excel в DataFrame
df = pd.read_excel(input_file)

# Удаление строк, где значение в столбце B пустое
df_cleaned = df[df['B'].notna() & (df['B'] != 'Все')]

# Сохранение результата в новый Excel файл
df_cleaned.to_excel(output_file, index=False)

print(f"Обработанный файл сохранён как {output_file}")