
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Настройки
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")

# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================================
print("=" * 60)
print("АНАЛИЗ ДАННЫХ STEAM")
print("=" * 60)

# Конкретное имя файла
FILE_NAME = "steam_games.csv"

if not os.path.exists(FILE_NAME):
    print(f"Ошибка: Файл '{FILE_NAME}' не найден!")
    print("Убедитесь, что файл находится в той же папке.")
    exit()

print(f"Загружаем данные из файла: {FILE_NAME}")

# Загрузка данных
try:
    df = pd.read_csv(FILE_NAME)
except UnicodeDecodeError:
    df = pd.read_csv(FILE_NAME, encoding='latin-1')
except Exception as e:
    print(f"Ошибка загрузки: {e}")
    exit()

print(f"Успешно загружено: {df.shape[0]} строк, {df.shape[1]} столбцов")

# Проверяем наличие нужных столбцов
required_cols = ['appid', 'name', 'release_date', 'english', 'developer', 
                 'publisher', 'platforms', 'required_age', 'genres', 
                 'positive_ratings', 'negative_ratings', 'owners', 'price']

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Внимание! Отсутствуют столбцы: {missing_cols}")
    print("Анализ может быть неполным.")

# Показываем первые строки
print("\nПервые 3 строки:")
print(df.head(3))

# ============================================================================
# 2. ПРЕДОБРАБОТКА ДАННЫХ
# ============================================================================
print("\n" + "=" * 60)
print("ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 60)

# 2.1. Обработка пропусков
print("\nОбработка пропусков...")
if 'english' in df.columns:
    df['english'] = df['english'].fillna(1)
    
if 'required_age' in df.columns:
    df['required_age'] = df['required_age'].fillna(0)
    
if 'price' in df.columns:
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['price'] = df['price'].fillna(0)

# Заполняем текстовые столбцы
for col in ['developer', 'publisher', 'genres', 'platforms']:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# 2.2. Обработка дубликатов
print("\nОбработка дубликатов...")
if 'appid' in df.columns:
    df = df.drop_duplicates(subset=['appid'])
    
df = df.drop_duplicates()
print(f"Данные после очистки: {df.shape[0]} строк")

# 2.3. Обработка даты
print("\nОбработка даты релиза...")
if 'release_date' in df.columns:
    # Извлекаем год из даты
    def extract_year(date_val):
        try:
            if pd.isna(date_val):
                return None
            date_str = str(date_val)
            # Ищем 4 цифры подряд (год)
            import re
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
            if year_match:
                return int(year_match.group())
            return None
        except:
            return None
    
    df['release_year'] = df['release_date'].apply(extract_year)
    
    # Удаляем строки без года
    initial_count = len(df)
    df = df[df['release_year'].notna()]
    df['release_year'] = df['release_year'].astype(int)
    print(f"Удалено строк без года: {initial_count - len(df)}")

# 2.4. Обработка owners
print("\nОбработка количества владельцев...")
if 'owners' in df.columns:
    def parse_owners(val):
        try:
            if pd.isna(val):
                return 0
            val_str = str(val)
            if '-' in val_str:
                parts = val_str.split('-')
                return (int(parts[0].replace(',', '')) + int(parts[1].replace(',', ''))) // 2
            return int(val_str.replace(',', ''))
        except:
            return 0
    
    df['owners_parsed'] = df['owners'].apply(parse_owners)

print("Предобработка завершена!")


# ============================================================================
# 3. СРЕДНЯЯ СТОИМОСТЬ ИГР
# ============================================================================
print("\n" + "=" * 60)
print("СРЕДНЯЯ СТОИМОСТЬ ИГР")
print("=" * 60)

if 'price' in df.columns:
    avg_price = df['price'].mean()
    avg_paid = df[df['price'] > 0]['price'].mean()
    
    print(f"Средняя стоимость всех игр: ${avg_price:.2f}")
    print(f"Средняя стоимость платных игр: ${avg_paid:.2f}")
    print(f"Бесплатных игр: {len(df[df['price'] == 0])} ({len(df[df['price'] == 0])/len(df)*100:.1f}%)")
    
    # Гистограмма цен
    plt.figure(figsize=(10, 6))
    price_data = df[df['price'] <= 50]['price']  # Показываем до $50
    plt.hist(price_data, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Распределение цен игр (до $50)', fontsize=14)
    plt.xlabel('Цена ($)')
    plt.ylabel('Количество игр')
    plt.axvline(x=avg_price, color='red', linestyle='--', label=f'Среднее: ${avg_price:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================================
# 4. ПОДДЕРЖКА АНГЛИЙСКОГО ЯЗЫКА
# ============================================================================
print("\n" + "=" * 60)
print("ПОДДЕРЖКА АНГЛИЙСКОГО ЯЗЫКА")
print("=" * 60)

if 'english' in df.columns:
    english_counts = df['english'].value_counts()
    labels = ['Нет', 'Да']
    counts = [english_counts.get(0, 0), english_counts.get(1, 0)]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts, color=['#ff6b6b', '#4ecdc4'])
    plt.title('Поддержка английского языка', fontsize=14)
    plt.ylabel('Количество игр')
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                str(count), ha='center')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Поддерживают английский: {english_counts.get(1, 0)} игр")
    print(f"Не поддерживают: {english_counts.get(0, 0)} игр")

# ============================================================================
# 5. БЕСПЛАТНЫЕ VS ПЛАТНЫЕ ИГРЫ
# ============================================================================
print("\n" + "=" * 60)
print("БЕСПЛАТНЫЕ VS ПЛАТНЫЕ ИГРЫ")
print("=" * 60)

if 'price' in df.columns:
    df['is_free'] = df['price'] == 0
    free_counts = df['is_free'].value_counts()
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Столбчатая диаграмма
    ax[0].bar(['Платные', 'Бесплатные'], [free_counts.get(False, 0), free_counts.get(True, 0)], 
             color=['#3498db', '#2ecc71'])
    ax[0].set_title('Количество игр')
    ax[0].set_ylabel('Количество')
    
    # Круговая диаграмма
    ax[1].pie([free_counts.get(False, 0), free_counts.get(True, 0)], 
             labels=['Платные', 'Бесплатные'], autopct='%1.1f%%', 
             colors=['#3498db', '#2ecc71'])
    ax[1].set_title('Доля игр')
    
    plt.suptitle('Бесплатные vs платные игры', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print(f"Бесплатные игры: {free_counts.get(True, 0)}")
    print(f"Платные игры: {free_counts.get(False, 0)}")

# ============================================================================
# 6. ВЫПУСК ИГР ПО ГОДАМ
# ============================================================================
print("\n" + "=" * 60)
print("ВЫПУСК ИГР ПО ГОДАМ")
print("=" * 60)

if 'release_year' in df.columns:
    # Фильтруем корректные годы
    df_years = df[(df['release_year'] >= 2000) & (df['release_year'] <= 2023)]
    games_by_year = df_years['release_year'].value_counts().sort_index()
    
    # Сводная таблица
    yearly_table = pd.DataFrame({
        'Год': games_by_year.index,
        'Количество игр': games_by_year.values


})
    
    print("Топ-10 лет по выпуску игр:")
    print(yearly_table.sort_values('Количество игр', ascending=False).head(10).to_string(index=False))
    
    # Гистограмма
    plt.figure(figsize=(14, 6))
    bars = plt.bar(games_by_year.index.astype(str), games_by_year.values, color='steelblue')
    plt.title('Количество выпущенных игр по годам', fontsize=14)
    plt.xlabel('Год')
    plt.ylabel('Количество игр')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Подписываем значения
    for i, (year, count) in enumerate(zip(games_by_year.index, games_by_year.values)):
        if count > games_by_year.values.max() * 0.1:  # Только крупные значения
            plt.text(i, count + 5, str(count), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 7. ТОП-10 РАЗРАБОТЧИКОВ И ИЗДАТЕЛЕЙ (С 2017)
# ============================================================================
print("\n" + "=" * 60)
print("ТОП-10 РАЗРАБОТЧИКОВ И ИЗДАТЕЛЕЙ (С 2017)")
print("=" * 60)

if 'release_year' in df.columns and 'developer' in df.columns and 'publisher' in df.columns:
    df_2017 = df[df['release_year'] >= 2017]
    
    top_devs = df_2017['developer'].value_counts().head(10)
    top_pubs = df_2017['publisher'].value_counts().head(10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Разработчики
    ax1.barh(top_devs.index, top_devs.values, color='#e74c3c')
    ax1.set_title('Топ-10 разработчиков (с 2017)', fontsize=12)
    ax1.set_xlabel('Количество игр')
    ax1.invert_yaxis()
    
    # Издатели
    ax2.barh(top_pubs.index, top_pubs.values, color='#f39c12')
    ax2.set_title('Топ-10 издателей (с 2017)', fontsize=12)
    ax2.set_xlabel('Количество игр')
    ax2.invert_yaxis()
    
    plt.suptitle('Топ-10 разработчиков и издателей с 2017 года', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("Топ-5 разработчиков:")
    for i, (dev, count) in enumerate(top_devs.head(5).items(), 1):
        print(f"{i}. {dev}: {count} игр")
    
    print("\nТоп-5 издателей:")
    for i, (pub, count) in enumerate(top_pubs.head(5).items(), 1):
        print(f"{i}. {pub}: {count} игр")

# ============================================================================
# 8. ТОП-6 ИГР ПО КОЛИЧЕСТВУ ВЛАДЕЛЬЦЕВ
# ============================================================================
print("\n" + "=" * 60)
print("ТОП-6 ИГР ПО КОЛИЧЕСТВУ ВЛАДЕЛЬЦЕВ")
print("=" * 60)

if 'owners_parsed' in df.columns and 'name' in df.columns:
    top_games = df.nlargest(6, 'owners_parsed')[['name', 'owners_parsed', 'release_year', 'price']]
    
    print("Топ-6 игр по количеству владельцев:")
    for i, row in enumerate(top_games.itertuples(), 1):
        price_str = "Бесплатно" if row.price == 0 else f"${row.price:.2f}"
        print(f"{i}. {row.name}")
        print(f"   Владельцев: {row.owners_parsed:,} | Год: {row.release_year} | Цена: {price_str}")
        print()
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_games['name'], top_games['owners_parsed'])
    plt.title('Топ-6 игр по количеству владельцев', fontsize=14)
    plt.xlabel('Количество владельцев')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# ============================================================================
# 9. ТОП-10 ЖАНРОВ (С 2015)
# ============================================================================
print("\n" + "=" * 60)
print("ТОП-10 ЖАНРОВ (С 2015)")
print("=" * 60)

if 'release_year' in df.columns and 'genres' in df.columns and 'price' in df.columns:
    df_2015 = df[df['release_year'] >= 2015].copy()
    
    # Разделяем жанры
    def split_genres(genre_str):
        if pd.isna(genre_str) or genre_str == 'Unknown':


return []
        return [g.strip() for g in str(genre_str).split(';')]
    
    # Платные игры
    paid = df_2015[df_2015['price'] > 0]
    paid_genres = []
    for genres in paid['genres']:
        paid_genres.extend(split_genres(genres))
    
    # Бесплатные игры
    free = df_2015[df_2015['price'] == 0]
    free_genres = []
    for genres in free['genres']:
        free_genres.extend(split_genres(genres))
    
    from collections import Counter
    top_paid = pd.Series(Counter(paid_genres)).sort_values(ascending=False).head(10)
    top_free = pd.Series(Counter(free_genres)).sort_values(ascending=False).head(10)
    
    # Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.barh(top_paid.index, top_paid.values, color='#2980b9')
    ax1.set_title('Платные игры', fontsize=12)
    ax1.set_xlabel('Количество игр')
    ax1.invert_yaxis()
    
    ax2.barh(top_free.index, top_free.values, color='#27ae60')
    ax2.set_title('Бесплатные игры', fontsize=12)
    ax2.set_xlabel('Количество игр')
    ax2.invert_yaxis()
    
    plt.suptitle('Топ-10 жанров с 2015 года', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("Топ-5 жанров платных игр:")
    for i, (genre, count) in enumerate(top_paid.head(5).items(), 1):
        print(f"{i}. {genre}: {count} игр")
    
    print("\nТоп-5 жанров бесплатных игр:")
    for i, (genre, count) in enumerate(top_free.head(5).items(), 1):
        print(f"{i}. {genre}: {count} игр")

# ============================================================================
# 10. ИГРЫ С ВЫСОКИМ РЕЙТИНГОМ (≥99%)
# ============================================================================
print("\n" + "=" * 60)
print("ИГРЫ С ВЫСОКИМ РЕЙТИНГОМ (≥99%)")
print("=" * 60)

if 'positive_ratings' in df.columns and 'negative_ratings' in df.columns:
    df['total_ratings'] = df['positive_ratings'] + df['negative_ratings']
    df['positive_pct'] = np.where(
        df['total_ratings'] > 0,
        df['positive_ratings'] / df['total_ratings'] * 100,
        0
    )
    
    high_rated = df[
        (df['positive_pct'] >= 99) &
        (df['negative_ratings'] > 0) &
        (df['total_ratings'] >= 10)
    ].sort_values('positive_pct', ascending=False)
    
    print(f"Найдено игр с ≥99% положительных отзывов: {len(high_rated)}")
    
    if len(high_rated) > 0:
        print("\nТоп-5 игр с наивысшим рейтингом:")
        for i, row in enumerate(high_rated.head(5).itertuples(), 1):
            print(f"{i}. {row.name}: {row.positive_pct:.2f}% ({row.total_ratings} отзывов)")
        
        # Визуализация
        top_show = high_rated.head(10)
        plt.figure(figsize=(10, 6))
        bars = plt.barh(top_show['name'], top_show['positive_pct'])
        plt.title('Игры с ≥99% положительных отзывов', fontsize=14)
        plt.xlabel('Процент положительных отзывов (%)')
        plt.xlim(98, 101)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    else:
        print("Игр с высоким рейтингом не найдено.")

# ============================================================================
# 11. РАСПРЕДЕЛЕНИЕ ПО ПЛАТФОРМАМ
# ============================================================================
print("\n" + "=" * 60)
print("РАСПРЕДЕЛЕНИЕ ИГР ПО ПЛАТФОРМАМ")
print("=" * 60)

if 'platforms' in df.columns:
    platform_counts = {}
    
    for platforms in df['platforms']:
        if pd.isna(platforms):
            continue
        for platform in str(platforms).split(';'):
            platform = platform.strip()
            if platform:
                platform_counts[platform] = platform_counts.get(platform, 0) + 1
    
    platform_df = pd.DataFrame({
        'Платформа': list(platform_counts.keys()),
        'Количество': list(platform_counts.values())

}).sort_values('Количество', ascending=False)
    
    print("Поддержка платформ:")
    print(platform_df.head(10).to_string(index=False))
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    top_platforms = platform_df.head(8)
    bars = plt.bar(top_platforms['Платформа'], top_platforms['Количество'])
    plt.title('Топ платформ по количеству игр', fontsize=14)
    plt.xlabel('Платформа')
    plt.ylabel('Количество игр')
    plt.xticks(rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 5, 
                str(int(height)), ha='center')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 12. ФИЛЬТРАЦИЯ ПО УСЛОВИЯМ
# ============================================================================
print("\n" + "=" * 60)
print("ФИЛЬТРАЦИЯ: ВОЗРАСТ 18+, 2019 ГОД, ЖАНР СИМУЛЯТОР")
print("=" * 60)

conditions_met = True
filtered = df.copy()

if 'required_age' in df.columns:
    filtered = filtered[filtered['required_age'] == 18]
else:
    print("Нет данных о возрасте")
    conditions_met = False

if 'release_year' in df.columns:
    filtered = filtered[filtered['release_year'] == 2019]
else:
    print("Нет данных о годе выпуска")
    conditions_met = False

if 'genres' in df.columns:
    filtered = filtered[filtered['genres'].str.contains('Simulation', case=False, na=False)]
else:
    print("Нет данных о жанрах")
    conditions_met = False

if conditions_met:
    print(f"Найдено игр: {len(filtered)}")
    
    if len(filtered) > 0:
        print("\nНайденные игры:")
        for i, row in enumerate(filtered.head(10).itertuples(), 1):
            print(f"{i}. {row.name}")
            print(f"   Разработчик: {row.developer}")
            print(f"   Цена: ${row.price:.2f}" if 'price' in df.columns else "")
            print()
        
        # Сохраняем результат
        filtered.to_csv('filtered_games.csv', index=False)
        print("Результаты сохранены в 'filtered_games.csv'")
    else:
        print("Игр, соответствующих условиям, не найдено.")
else:
    print("Невозможно выполнить фильтрацию из-за отсутствия данных.")

# ============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================
print("\n" + "=" * 60)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 60)

# Сохраняем очищенные данные
df.to_csv('steam_cleaned.csv', index=False)
print("Очищенные данные сохранены в 'steam_cleaned.csv'")

print("\n" + "=" * 60)
print("АНАЛИЗ ЗАВЕРШЕН!")

print("=" * 60)
