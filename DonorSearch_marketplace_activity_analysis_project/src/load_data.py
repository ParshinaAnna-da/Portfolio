# src/load_data.py
import os
import pandas as pd

def load_and_clean_unit_cost(file_path=None):
    """
    Загружает и очищает справочник себестоимости.
    Если file_path не передан, ищет файлы по умолчанию.
    """
    # 1. Поиск файла (переносим твою логику, но делаем её более строгой)
    file_to_read = None
    
    if file_path and os.path.exists(file_path):
        file_to_read = file_path
    else:
        local_candidates = ['data/raw/unit_cost.xlsx', 'unit_cost.xlsx', 'Цены для партнеров.xlsx']
        for candidate in local_candidates:
            if os.path.exists(candidate):
                file_to_read = candidate
                break

    # Если не нашли по точным именам, ищем в корне или в data/raw/
    if file_to_read is None:
        search_dirs = ['.', 'data/raw']
        for directory in search_dirs:
            if os.path.exists(directory):
                for f in os.listdir(directory):
                    if f.endswith(('.xlsx', '.xls')) and any(w in str(f).lower() for w in ['себес', 'cost', 'ratck']):
                        file_to_read = os.path.join(directory, f)
                        break
            if file_to_read:
                break

    # Если файл так и не найден — выдаем ошибку
    if file_to_read is None:
        raise FileNotFoundError(
            "КРИТИЧЕСКАЯ ОШИБКА: Локальный файл себестоимости закупки не найден.\n"
            "Пожалуйста, положите файл в папку 'data/raw/' под именем 'unit_cost.xlsx'."
        )

    # 2. Чтение и очистка данных
    df = pd.read_excel(file_to_read)
    
    # Приводим заголовки к нижнему регистру и заменяем пробелы
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace('\xa0', ' ', regex=False)   
        .str.lower()
        .str.replace(r'\s+', '_', regex=True)    
        .str.replace('\n', '', regex=False)
    )
    
    # Очищаем колонку 'название'
    if 'название' in df.columns:
        df['название'] = (
            df['название']
            .astype(str)
            .str.replace('\xa0', ' ', regex=False)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )
        
    # Очищаем колонку 'себестоимость'
    if 'себестоимость' in df.columns:
        df['себестоимость'] = df['себестоимость'].astype(str).str.replace(',', '.', regex=False).str.strip()
        df['себестоимость'] = pd.to_numeric(df['себестоимость'], errors='coerce').fillna(0.0).astype('float64')
        
    # Удаляем дубликаты
    df = df.drop_duplicates().reset_index(drop=True)
    
    return df
