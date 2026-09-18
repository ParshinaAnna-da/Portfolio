# src/load_ozon.py
import os
import pandas as pd
import numpy as np

def load_ozon_files(folder_path: str) -> dict:
    """
    Сканирует указанную папку и загружает файлы Ozon,
    разделяя их по типам отчетов.
    """
    result = {'sales': {}, 'stocks': {}, 'unit': {}, 'unmatched': [], 'failed': {}}

    if not os.path.exists(folder_path):
        print(f" Ошибка! Папка '{folder_path}' не найдена.")
        print(f"Текущая рабочая директория: {os.getcwd()}")
        return result

    files = os.listdir(folder_path)
    print(f" Локальная папка найдена! Всего файлов в папке: {len(files)}")

    for file_name in files:
        if not file_name.endswith(('.xlsx', '.xls')):
            continue
        if file_name.startswith(('~$', '._')):
            continue

        full_path = os.path.join(folder_path, file_name)

        try:
            if 'Отчет по товарам' in file_name:
                result['sales'][file_name] = pd.read_excel(full_path)
            elif 'Управление остатками' in file_name:
                result['stocks'][file_name] = pd.read_excel(full_path, sheet_name=None)
            elif 'Юнит-экономика' in file_name:
                result['unit'][file_name] = pd.read_excel(full_path)
            else:
                result['unmatched'].append(file_name)
                print(f"  Файл не распознан ни по одному шаблону: {file_name}")
                continue

            print(f"  Успешно прочитан: {file_name}")

        except Exception as e:
            result['failed'][file_name] = str(e)
            print(f" Не удалось прочитать {file_name}: {e}")

    print("\n Загрузка завершена.")
    return result


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит названия колонок к нижнему регистру и заменяет пробелы."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace('\xa0', ' ', regex=False)   
        .str.lower()
        .str.replace(r'\s+', '_', regex=True)    
    )
    return df


def concat_with_schema_check(dfs_dict: dict, source_label: str = "") -> pd.DataFrame:
    """Проверяет схемы файлов на совпадение и объединяет их в один DataFrame."""
    normalized = {name: normalize_columns(df) for name, df in dfs_dict.items()}

    all_columns = [set(df.columns) for df in normalized.values()]
    reference = all_columns[0] if all_columns else set()
    mismatches = {}
    for name, df in normalized.items():
        diff_missing = reference - set(df.columns)
        diff_extra = set(df.columns) - reference
        if diff_missing or diff_extra:
            mismatches[name] = {'missing': diff_missing, 'extra': diff_extra}

    if mismatches:
        print(f"  [{source_label}] Расхождение схем между файлами:")
        for name, diff in mismatches.items():
            print(f"   - {name}: нет колонок {diff['missing']}, лишние {diff['extra']}")
    else:
        print(f" [{source_label}] Схемы всех файлов совпадают ({len(reference)} колонок).")

    combined = pd.concat(normalized.values(), ignore_index=True)
    print(f"[{source_label}] Объединено файлов: {len(normalized)}, строк итого: {combined.shape[0]}")
    return combined

# Константы для приведения типов данных Ozon
DATE_COLUMNS = [
    'дата_начисления',
    'дата_принятия_заказа_в_обработку_или_оказания_услуги',
]

INT_COLUMNS = [
    'количество'
]

FLOAT_COLUMNS = [
    'за_продажу_или_возврат_до_вычета_комиссий_и_услуг',
    'вознаграждение_ozon',
    'вознаграждение_ozon,_%',
    'сборка_заказа',
    'обработка_отправления_(drop-off/pick-up)_(разбивается_по_товарам_пропорционально_количеству_в_отправлении)',
    'магистраль',
    'последняя_миля_(разбивается_по_товарам_пропорционально_доле_цены_товара_в_сумме_отправления)',
    'обратная_магистраль',
    'обработка_возврата',
    'обработка_отмененного_или_невостребованного_товара_(разбивается_по_товарам_в_отправлении_в_одинаковой_пропорции)',
    'обработка_невыкупленного_товара',
    'логистика',
    'индекс_локализации',
    'среднее_время_доставки,_часы',
    'обратная_логистика',
    'итого,_руб.',
]

STRING_ID_COLUMNS = [
    'номер_отправления_или_идентификатор_услуги',
    'артикул',
    'sku'
]


def _clean_numeric_string_series(s: pd.Series) -> pd.Series:
    """Очищает строку от скрытых символов, процентов и пробелов перед переводом в число."""
    return (
        s.astype(str)
        .str.replace('\xa0', '', regex=False)
        .str.replace(' ', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.replace(',', '.', regex=False)
    )


def clean_types(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Приводит колонки датафрейма Ozon к правильным типам данных (строки, даты, числа)."""
    df = df.copy()

    for col in STRING_ID_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r'\.0$', '', regex=True)

    for col in DATE_COLUMNS:
        if col not in df.columns:
            continue
        before_na = df[col].isna().sum()
        df[col] = pd.to_datetime(df[col], errors='coerce')
        after_na = df[col].isna().sum()
        if verbose and after_na > before_na:
            print(f"  {col}: не распознано как дата {after_na - before_na} значений")

    for col in INT_COLUMNS:
        if col not in df.columns:
            continue
        before_na = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        after_na = df[col].isna().sum()
        if verbose and after_na > before_na:
            print(f"  {col}: не распознано как число {after_na - before_na} значений")

    for col in FLOAT_COLUMNS:
        if col not in df.columns:
            continue
        before_na = df[col].isna().sum()
        if df[col].dtype == 'object':
            df[col] = _clean_numeric_string_series(df[col])
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Float64')
        after_na = df[col].isna().sum()
        if verbose and after_na > before_na:
            print(f"  {col}: не распознано как число {after_na - before_na} значений")

    expected = set(DATE_COLUMNS + INT_COLUMNS + FLOAT_COLUMNS + STRING_ID_COLUMNS)
    missing_in_df = expected - set(df.columns)
    if verbose and missing_in_df:
        print(f"  Колонки, ожидавшиеся, но отсутствующие в датафрейме: {missing_in_df}")

    return df
# Стратегии обработки пропусков
ERROR_IF_MISSING = ['sku', 'артикул']
ZERO_IF_MISSING = [
    'за_продажу_или_возврат_до_вычета_комиссий_и_услуг',
    'вознаграждение_ozon', 'сборка_заказа', 'магистраль', 
    'последняя_миля_(разбивается_по_товарам_пропорционально_доле_цены_товара_в_сумме_отправления)',
    'логистика', 'итого,_руб.'
]
KEEP_NAN = ['индекс_локализации', 'среднее_время_доставки,_часы']

def smart_fill_gaps(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Интеллектуальное заполнение пропусков с проверкой критических полей."""
    df = df.copy()
    
    # 1. Проверяем обязательные поля на критические пропуски
    for col in ERROR_IF_MISSING:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                raise ValueError(f"КРИТИЧЕСКАЯ ОШИБКА: Поле '{col}' содержит {missing_count} пропусков.")

    # 2. Заполняем нулями там, где это экономически оправданно
    for col in ZERO_IF_MISSING:
        if col in df.columns:
            n_filled = df[col].isna().sum()
            df[col] = df[col].fillna(0.0)
            if verbose and n_filled > 0:
                print(f"  {col}: заполнено нулём {n_filled} пропусков")

    # 3. Логируем оставшиеся NaN в метриках, которые нельзя трогать
    for col in KEEP_NAN:
        if col in df.columns and verbose:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                print(f"  {col}: оставлены пропуски (NaN) в количестве {n_nan}")

    return df


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: str, how: str = 'left', validate: str = None) -> pd.DataFrame:
    """Безопасный merge с проверкой длины и кардинальности данных."""
    merged = pd.merge(left, right, on=on, how=how, validate=validate, indicator=True)
    
    if how == 'left':
        assert len(merged) == len(left), f"Ошибка кардинальности! Размер до merge: {len(left)}, после: {len(merged)}"
        
    print(f"Качество слияния по {on}:")
    display(merged['_merge'].value_counts())
    
    return merged.drop(columns=['_merge'])

def aggregate_by_shipment(df: pd.DataFrame, sum_columns: list,
                           group_col: str = 'номер_отправления_или_идентификатор_услуги') -> pd.DataFrame:
    """Агрегирует финансовые показатели по номеру отправления."""
    existing_cols = [c for c in sum_columns if c in df.columns]
    return df.groupby(group_col)[existing_cols].sum()

def build_sales_analytics_ozon(folder_path: str) -> pd.DataFrame:
    """
    Полный автоматический пайплайн (ETL) для данных Ozon:
    Загрузка -> Объединение -> Типизация -> Умное заполнение пропусков -> Очистка дубликатов.
    """
    # 1. Загрузка файлов
    loaded = load_ozon_files(folder_path)
    
    # 2. Объединение по схемам
    df = concat_with_schema_check(loaded['sales'], source_label="Отчёт по товарам")
    
    # 3. Приведение типов
    df = clean_types(df, verbose=False)
    
    # 4. Обработка пропусков 
    df = smart_fill_gaps(df, verbose=False)
    
    # 5. Дедупликация данных 
    df = df.drop_duplicates().reset_index(drop=True)
    
    return df
def load_stocks_sheets(stocks_raw: dict, folder_path: str, verbose: bool = True) -> dict:
    """
    Загружает и обрабатывает листы Excel-файла остатков Ozon.
    Успешно схлопывает мультизаголовки (двухуровневые шапки) в плоские названия.
    """
    assert len(stocks_raw) == 1, (
        f"Ожидался ровно 1 файл остатков, найдено {len(stocks_raw)}: "
        f"{list(stocks_raw.keys())}."
    )

    stocks_file_name = list(stocks_raw.keys())[0]
    full_path = os.path.join(folder_path, stocks_file_name)
    
    if verbose:
        print(f"Файл остатков: {stocks_file_name}")

    sheet_map = {
        'warehouse': 'Товар-склад',
        'cluster': 'Товар-кластер',
        'goods': 'Товары',
        'clusters': 'Кластеры',
    }

    available_sheets = pd.ExcelFile(full_path).sheet_names
    if verbose:
        print(f"Вкладки в файле: {available_sheets}")

    result = {}
    for key, sheet_name in sheet_map.items():
        if sheet_name not in available_sheets:
            if verbose:
                print(f"  Лист '{sheet_name}' не найден в файле — пропущен.")
            continue

        # Читаем мультизаголовок (0 и 1 строки)
        df = pd.read_excel(full_path, sheet_name=sheet_name, header=[0, 1])

        if verbose:
            print(f"\n--- Превью листа '{sheet_name}' до удаления служебных строк ---")
            display(df.head(3))

        # Удаляем служебные строки метаданных Excel, если они остались
        df = df.drop([0, 1], errors='ignore').reset_index(drop=True)

        # Схлопываем двухуровневый заголовок в один
        df.columns = [
            f"{top} ({bottom})" if "Unnamed" not in str(bottom) else top
            for top, bottom in df.columns
        ]

        # Стандартная чистка названий колонок
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.replace('\xa0', ' ', regex=False)
            .str.lower()
            .str.replace(r'\s+', '_', regex=True)
            .str.replace('\n', '', regex=False)
        )

        result[key] = df
        if verbose:
            print(f" '{sheet_name}' -> ключ '{key}', строк: {len(df)}, колонок: {df.shape[1]}")

    return result

# Добавь это в самый конец файла src/load_ozon.py

TEXT_COLUMNS = [
    'артикул', 'название_товара', 'склад', 'кластер', 'зона_размещения',
    'признак_товара', 'ликвидность_(статус)', 'sku',
]

def clean_stock_types(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Автоматически распознает типы колонок в отчете по остаткам Ozon
    и приводит их к правильным типам данных (строки, целые или вещественные числа).
    """
    df = df.copy()

    for col in df.columns:
        col_str = str(col).lower().strip()
        
        is_id_column = ('sku' in col_str or 'артикул' in col_str) and not any(w in col_str for w in ['дней', 'остат', 'штук', 'продаж'])
        is_pure_text = any(keyword in col_str for keyword in ['склад', 'кластер', 'название_товара', 'зона_размещения', 'признак_товара', 'статус'])
        
        if is_id_column or is_pure_text:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r'\.0$', '', regex=True)
            continue
            
        if 'себестоимость' in col_str:
            continue

        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.isna().all():
            continue

        has_fraction = ((converted.dropna() % 1) != 0).any()

        if has_fraction:
            df[col] = converted.astype('Float64')
        else:
            df[col] = converted.astype('Int64')

    if verbose:
        for col in df.columns:
            col_str = str(col).lower().strip()
            is_id_column = ('sku' in col_str or 'артикул' in col_str) and not any(w in col_str for w in ['дней', 'остат', 'штук', 'продаж'])
            is_pure_text = any(keyword in col_str for keyword in ['склад', 'кластер', 'название_товара', 'зона_размещения', 'признак_товара', 'статус'])
            if is_id_column or is_pure_text or 'себестоимость' in col_str:
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]) or str(df[col].dtype).lower().startswith(('int', 'float')):
                n_na = df[col].isna().sum()
                if n_na > 0:
                    print(f"   {col}: {n_na} пропусков после типизации")

    return df


def fill_stock_gaps(df: pd.DataFrame, verbose: bool = True, exclude_cols: list = None) -> pd.DataFrame:
    """
    Умное заполнение пропусков в данных остатков.
    Зануляет только объемы (штуки, кванты), но сохраняет NaN в оборачиваемости (дни).
    """
    import numpy as np
    df = df.copy()
    if exclude_cols is None:
        exclude_cols = []

    # Добавляем метрики оборачиваемости в список исключений
    business_exceptions = ['дней', 'оборачиваемость', 'скорость_продаж']
    exclude_cols_clean = [str(c).lower().strip() for c in exclude_cols] + business_exceptions

    numeric_cols = df.select_dtypes(include=[np.number, 'Int64', 'Float64']).columns
    
    cols_to_fill = []
    for c in numeric_cols:
        c_str = str(c).lower().strip()
        if any(exc in c_str for exc in exclude_cols_clean):
            continue
        cols_to_fill.append(c)

    n_before = df[cols_to_fill].isna().sum().sum()
    df[cols_to_fill] = df[cols_to_fill].fillna(0)
    
    if verbose and n_before > 0:
        print(f"  [Контроль качества] Заполнено нулём: {n_before} пропусков в {len(cols_to_fill)} числовых колонках (остатки/штуки)")

    if verbose:
        for c in df.columns:
            c_str = str(c).lower().strip()
            if any(exc in c_str for exc in exclude_cols_clean):
                if pd.api.types.is_numeric_dtype(df[c]) or str(df[c].dtype).lower().startswith(('int', 'float')):
                    n_na = df[c].isna().sum()
                    if n_na > 0:
                        print(f"  [Контроль качества] {c}: Пропуски ({n_na} значений) сохранены как NaN (бизнес-метрика)")

    return df

# Добавь это в самый конец файла src/load_ozon.py

def find_id_column(df: pd.DataFrame) -> str:
    """Автоматически находит ключевую колонку ID (sku или артикул) в датафрейме."""
    for col in df.columns:
        if 'sku' in str(col).lower():
            return col
    for col in df.columns:
        if 'артикул' in str(col).lower():
            return col
    return None


def check_stock_duplicates(df: pd.DataFrame, key_columns: list, name: str,
                            drop: bool = False, verbose: bool = True) -> pd.DataFrame:
    """
    Проверяет данные остатков на полные и логические дубликаты по бизнес-ключам.
    Предотвращает задвоение финансовых и количественных показателей.
    """
    df = df.copy()
    full_dups = df.duplicated().sum()
    
    if verbose:
        print(f"• {name}: полных дубликатов строк — {full_dups}")
        
    if full_dups > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        if verbose:
            print(f"  -> удалено {full_dups} полных дублей")

    existing_key = [c for c in key_columns if c in df.columns]
    if existing_key:
        key_dup_mask = df.duplicated(subset=existing_key, keep=False)
        key_dup_count = df.duplicated(subset=existing_key).sum()
        
        if verbose:
            print(f"  Дубликатов по ключу {existing_key}: {key_dup_count}")
            
        if key_dup_count > 0:
            if verbose:
                # Оставляем display только для интерактивного режима в Jupyter
                print("  Примеры дубликатов (проверьте кардинальность):")
                try:
                    display(df[key_dup_mask].sort_values(existing_key).head(10))
                except NameError:
                    # Если функция вызвана вне Jupyter, где нет display, просто выводим через print
                    print(df[key_dup_mask].sort_values(existing_key).head(5))
            if drop:
                df = df.drop_duplicates(subset=existing_key, keep='first').reset_index(drop=True)
                if verbose:
                    print(f"  -> удалено {key_dup_count} логических дублей (keep='first')")

    return df

def build_ozon_stocks(stocks_raw: dict, folder_path: str, drop_logical_dups: bool = False) -> dict:
    """
    Полный автоматический конвейер (ETL) для обработки остатков Ozon.
    """
    sheets = load_stocks_sheets(stocks_raw, folder_path)

    for key in sheets:
        print(f"\n=== Типизация и унификация: {key} ===")
        sheets[key] = clean_stock_types(sheets[key], verbose=True)

        # Исключаем колонки ликвидности из зануления
        liquidity_days_cols = [c for c in sheets[key].columns if 'ликвидн' in str(c).lower()]
        sheets[key] = fill_stock_gaps(sheets[key], verbose=True, exclude_cols=liquidity_days_cols)
        
        # --- ДОБАВЛЯЕМ СТРОЧКУ УНИФИКАЦИИ ИМЕН КОЛОНОК ---
        sheets[key] = unify_columns(sheets[key])

    print("\n=== Проверка дубликатов ===")
    
    # 1. Свод по вкладке Товар-склад (Ключ: SKU + Склад)
    if 'warehouse' in sheets:
        sku_col = find_id_column(sheets['warehouse'])
        key_w = [sku_col, 'склад'] if sku_col else ['склад']
        sheets['warehouse'] = check_stock_duplicates(
            sheets['warehouse'], key_columns=key_w, name='Товар-склад',
            drop=drop_logical_dups, verbose=True)

    # 2. Свод по вкладке Товар-кластер (Ключ: SKU + Кластер)
    if 'cluster' in sheets:
        sku_col = find_id_column(sheets['cluster'])
        key_c = [sku_col, 'кластер'] if sku_col else ['кластер']
        sheets['cluster'] = check_stock_duplicates(
            sheets['cluster'], key_columns=key_c, name='Товар-кластер',
            drop=drop_logical_dups, verbose=True)

    # 3. Свод по вкладке Товары (Ключ: SKU)
    if 'goods' in sheets:
        sku_col = find_id_column(sheets['goods'])
        key_g = [sku_col] if sku_col else []
        sheets['goods'] = check_stock_duplicates(
            sheets['goods'], key_columns=key_g, name='Товары',
            drop=drop_logical_dups, verbose=True)

    # 4. Свод по вкладке Кластеры (Ключ: Кластер)
    if 'clusters' in sheets:
        key_cl = ['кластер'] if 'кластер' in sheets['clusters'].columns else []
        sheets['clusters'] = check_stock_duplicates(
            sheets['clusters'], key_columns=key_cl, name='Кластеры',
            drop=drop_logical_dups, verbose=True)

    return sheets

COLUMN_ALIASES = {
    'остатки_на_складах_ozon_(дней_без_продаж)': 'дней_без_продаж',
    'остатки_на_складах_ozon_(среднесуточные_продажи_за_28_дней_штук)': 'среднесуточные_продажи_за_28_дней_штук'
}

def unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Унифицирует сложные и длинные названия колонок Ozon согласно словарю алиасов."""
    return df.rename(columns=COLUMN_ALIASES)

