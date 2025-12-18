
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('steam_games.csv')

# 1. Предобработка
print("Размер данных до обработки:", df.shape)

# Удаляем дубликаты
df = df.drop_duplicates()

# Заполняем пропуски
df['price'] = df['price'].fillna(0)
df['english'] = df['english'].fillna(1)
df['required_age'] = df['required_age'].fillna(0)

# Создаем столбец с годом релиза
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
df = df.dropna(subset=['release_year'])
df['release_year'] = df['release_year'].astype(int)

print("Размер данных после обработки:", df.shape)

# 2. Средняя стоимость игр
avg_price = df['price'].mean()
print(f"\nСредняя стоимость игр: ${avg_price:.2f}")

# 3. Диаграмма поддержки английского языка
if 'english' in df.columns:
    english_counts = df['english'].value_counts()
    plt.figure(figsize=(8, 5))
    plt.bar(['Нет', 'Да'], english_counts.values)
    plt.title('Поддержка английского языка')
    plt.ylabel('Количество игр')
    plt.show()

# 4. Диаграмма бесплатных/платных игр
df['is_free'] = df['price'] == 0
free_counts = df['is_free'].value_counts()
plt.figure(figsize=(8, 5))
plt.bar(['Платные', 'Бесплатные'], free_counts.values)
plt.title('Бесплатные vs платные игры')
plt.ylabel('Количество игр')
plt.show()

# 5. Игры по годам и гистограмма
games_by_year = df['release_year'].value_counts().sort_index()
print("\nСводная таблица по годам (первые 10 лет):")
print(games_by_year.head(10))

plt.figure(figsize=(10, 6))
plt.bar(games_by_year.index.astype(str), games_by_year.values)
plt.title('Количество игр по годам релиза')
plt.xlabel('Год')
plt.ylabel('Количество игр')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Топ-10 разработчиков и издателей с 2017 года
df_2017 = df[df['release_year'] >= 2017]

if 'developer' in df.columns:
    top_devs = df_2017['developer'].value_counts().head(10)
    print("\nТоп-10 разработчиков с 2017 года:")
    print(top_devs)
    
    plt.figure(figsize=(10, 6))
    top_devs.plot(kind='barh')
    plt.title('Топ-10 разработчиков (с 2017)')
    plt.xlabel('Количество игр')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

if 'publisher' in df.columns:
    top_pubs = df_2017['publisher'].value_counts().head(10)
    print("\nТоп-10 издателей с 2017 года:")
    print(top_pubs)
    
    plt.figure(figsize=(10, 6))
    top_pubs.plot(kind='barh')
    plt.title('Топ-10 издателей (с 2017)')
    plt.xlabel('Количество игр')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# 7. Топ-6 игр по количеству владельцев
if 'owners' in df.columns:
    try:
        # Пробуем преобразовать owners в числа
        df['owners_numeric'] = pd.to_numeric(df['owners'].astype(str).str.replace(',', ''), errors='coerce')
        top_owners = df.nlargest(6, 'owners_numeric')[['name', 'owners']]
        print("\nТоп-6 игр по количеству владельцев:")
        print(top_owners)
    except:
        print("\nНе удалось обработать данные о владельцах")

# 8. Топ-10 жанров с 2015 года
df_2015 = df[df['release_year'] >= 2015]

if 'genres' in df.columns:
    paid_genres = df_2015[df_2015['price'] > 0]['genres'].value_counts().head(10)
    free_genres = df_2015[df_2015['price'] == 0]['genres'].value_counts().head(10)
    
    print("\nТоп-10 жанров платных игр (с 2015):")
    print(paid_genres)
    
    print("\nТоп-10 жанров бесплатных игр (с 2015):")
    print(free_genres)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    paid_genres.plot(kind='barh', ax=ax1)
    ax1.set_title('Платные игры')
    ax1.invert_yaxis()
    
    free_genres.plot(kind='barh', ax=ax2)
    ax2.set_title('Бесплатные игры')
    ax2.invert_yaxis()
    
    plt.suptitle('Топ-10 жанров с 2015 года')
    plt.tight_layout()
    plt.show()


# 9. Игры с ≥99% положительных отзывов
if 'positive_ratings' in df.columns and 'negative_ratings' in df.columns:
    df['total_ratings'] = df['positive_ratings'] + df['negative_ratings']
    df['positive_percentage'] = df['positive_ratings'] / df['total_ratings'] * 100
    
    high_rated = df[(df['positive_percentage'] >= 99) & (df['negative_ratings'] > 0)]
    print(f"\nИгр с ≥99% положительных отзывов: {len(high_rated)}")
    
    if len(high_rated) > 0:
        print("\nПримеры таких игр:")
        print(high_rated[['name', 'positive_percentage']].head(10))

# 10. Распределение игр по платформам
if 'platforms' in df.columns:
    # Разделяем платформы
    platforms_list = []
    for platforms in df['platforms'].dropna():
        for platform in str(platforms).split(';'):
            platforms_list.append(platform.strip())
    
    from collections import Counter
    platform_counts = Counter(platforms_list)
    
    print("\nТоп-5 платформ:")
    for platform, count in platform_counts.most_common(5):
        print(f"{platform}: {count} игр")

# 11. Фильтрация по условиям
print("\nФильтрация: возраст 18+, 2019 год, жанр симулятор")
try:
    filtered = df[
        (df['required_age'] == 18) &
        (df['release_year'] == 2019) &
        (df['genres'].str.contains('Simulation', case=False, na=False))
    ]
    print(f"Найдено игр: {len(filtered)}")
    
    if len(filtered) > 0:
        print("\nНайденные игры:")
        for i, row in filtered[['name', 'developer']].head(10).iterrows():
            print(f"- {row['name']} ({row['developer']})")
except:
    print("Ошибка при фильтрации")

# Сохраняем результаты
df.to_csv('steam_cleaned.csv', index=False)

print("\nОчищенные данные сохранены в 'steam_cleaned.csv'")
