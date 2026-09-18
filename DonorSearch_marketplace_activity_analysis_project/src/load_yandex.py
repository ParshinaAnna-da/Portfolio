# src/load_yandex.py
import os
import re
import pandas as pd
import numpy as np
import unicodedata

MONTHS_DICT = {
    'января': '01', 'февраля': '02', 'марта': '03', 'апреля': '04',
    'мая': '05', 'июня': '06', 'июля': '07', 'августа': '08',
    'сентября': '09', 'октября': '10', 'ноября': '11', 'декабря': '12'
}

COLUMNS_TO_INT = [
    'номер_заказа', 
    'ваш_номер_заказа', 
    'количество_переданных_в_доставку,_шт.', 
    'доставлено,_шт.'
]

COLUMNS_TO_DATE = [
    'дата_оформления_заказа', 
    'период',
    'дата_передачи_товара_в_доставку', 
    'дата_доставки_товара',
    'дата_упд',
    'дата_товарной_накладной',
    'дата_счёта-фактуры'
]

COLUMNS_TO_FLOAT = [
    'цена_c_ндс_без_учёта_скидок_за_шт.,_₽',
    'ваша_скидка_по_акции_маркетплейса_на_1_шт.,_₽',
    'ваша_скидка_по_бонусам_сберспасибо_(за_шт.)_на_1_шт.,_₽',
    'ваша_скидка_по_баллам_яндекс.плюса_на_1_шт.,_₽',
    'цена_с_ндс_с_учётом_всех_скидок_за_шт.,_₽',
    'стоимость_всех_переданных_в_доставку_штук_с_ндс_без_учёта_скидок,_₽',
    'сумма_всех_скидок_для_переданных_в_доставку_штук,_₽',
    'стоимость_всех_переданных_в_доставку_штук_с_ндс_с_учётом_всех_скидок,_₽'
]

YANDEX_KEEP_NAN = [
    'ваша_скидка_по_акции_маркетплейса_на_1_шт.,_₽',
    'ваша_скидка_по_бонусам_сберспасибо_(за_шт.)_на_1_шт.,_₽',
    'ваша_скидка_по_баллам_яндекс.плюса_на_1_шт.,_₽',
    'сумма_всех_скидок_для_переданных_в_доставку_штук,_₽'
]


def load_yandex_files(folder_path: str) -> dict:
    """Сканирует папку Яндекс.Маркета и считывает структуру файлов в память."""
    result = {'stats': {}, 'orders': {}, 'stocks': {}, 'sales': {}, 'unmatched': []}

    if not os.path.exists(folder_path):
        print(f" Ошибка! Папка '{folder_path}' не найдена.")
        return result

    files = os.listdir(folder_path)
    for file_name in files:
        if not file_name.endswith(('.xlsx', '.xls')) or file_name.startswith(('~$', '._')):
            continue

        full_path = os.path.join(folder_path, file_name)
        try:
            if 'united_statistics_report' in file_name:
                result['stats'][file_name] = pd.read_excel(full_path)
            elif 'united_orders' in file_name:
                result['orders'][file_name] = pd.read_excel(full_path)
            elif 'fulfillment_offer_analytics' in file_name:
                result['stocks'][file_name] = pd.read_excel(full_path)
            elif 'Аналитика продаж' in file_name:
                result['sales'][file_name] = pd.read_excel(full_path)
            else:
                result['unmatched'].append(file_name)
        except Exception as e:
            print(f"  Не удалось прочитать {file_name}: {e}")

    return result


def _clean_and_type_yandex_df(df: pd.DataFrame) -> pd.DataFrame:
    """Внутренняя функция для безопасного приведения типов."""
    df = df.copy()
    
    for col in COLUMNS_TO_INT:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            
    for col in COLUMNS_TO_FLOAT:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Float64')
            
    for col in COLUMNS_TO_DATE:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    return df


def smart_fill_yandex_gaps(df: pd.DataFrame, name: str, verbose: bool = True) -> pd.DataFrame:
    """Умное заполнение пропусков. Зануляет объемы, но сохраняет NaN в скидках."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number, 'Int64', 'Float64']).columns
    
    cols_to_zero = [c for c in numeric_cols if c not in YANDEX_KEEP_NAN]
    
    n_filled = df[cols_to_zero].isna().sum().sum()
    df[cols_to_zero] = df[cols_to_zero].fillna(0)
    
    if verbose and n_filled > 0:
        print(f"  • Таблица '{name}': заполнено нулём {n_filled} технических пропусков (объёмы/цены).")
                
    return df


def parse_yandex_stats_reports(stats_files_dict: dict, folder_path: str, verbose: bool = True) -> dict:
    """
    Парсит файлы статистики Яндекс.Маркета (united_statistics_report).
    """
    shipped_sheets = []
    delivered_sheets = []
    unclaimed_sheets = []
    returned_sheets = []

    for file_name in stats_files_dict.keys():
        path = os.path.join(folder_path, file_name)
        try:
            excel_obj = pd.ExcelFile(path)
            available_sheets = excel_obj.sheet_names
            
            sheet_mapping = {
                'shipped': 'Товары, переданные в доставку',
                'delivered': 'Доставленные товары',
                'unclaimed': 'Невыкупленные товары',
                'returned': 'Возвращенные товары'
            }
            
            df_shipped_raw = pd.read_excel(excel_obj, sheet_name=sheet_mapping['shipped'], header=None) if sheet_mapping['shipped'] in available_sheets else pd.DataFrame()
            df_delivered_raw = pd.read_excel(excel_obj, sheet_name=sheet_mapping['delivered'], header=None) if sheet_mapping['delivered'] in available_sheets else pd.DataFrame()
            df_unclaimed_raw = pd.read_excel(excel_obj, sheet_name=sheet_mapping['unclaimed'], header=None) if sheet_mapping['unclaimed'] in available_sheets else pd.DataFrame()
            df_return_raw = pd.read_excel(excel_obj, sheet_name=sheet_mapping['returned'], header=None) if sheet_mapping['returned'] in available_sheets else pd.DataFrame()
            
            if df_delivered_raw.empty:
                continue

            # Извлечение периода из шапки
            file_period = "Не указан"
            header_text = " ".join(df_delivered_raw.iloc[:10].astype(str).values.flatten()).lower()

            date_match = re.search(r'за период.*?([а-яё]+)\s+(\d{4})', header_text)
            if date_match:
                month_name = date_match.group(1)
                year = date_match.group(2)
                month_num = MONTHS_DICT.get(month_name, "00")
                file_period = f"{year}-{month_num}"

            # Срезание шапки
            raw_dfs = [df_shipped_raw, df_delivered_raw, df_unclaimed_raw, df_return_raw]
            clean_dfs = []
            
            for df_raw in raw_dfs:
                if df_raw.empty or df_raw.shape[0] <= 17:
                    clean_dfs.append(pd.DataFrame())
                    continue
                    
                df_clean = df_raw.iloc[17:].copy()
                df_clean.columns = df_raw.iloc[16]
                
                df_clean.columns = (
                    df_clean.columns
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .str.replace(' ', '_')
                    .str.replace('\xa0', ' ', regex=False)
                )
                df_clean['период'] = file_period
                
                if 'номер_заказа' in df_clean.columns:
                    df_clean = df_clean[~df_clean['номер_заказа'].astype(str).str.contains('Итого', case=False, na=False)]
                    
                clean_dfs.append(df_clean)
                
            if len(clean_dfs) > 0 and not clean_dfs[0].empty: shipped_sheets.append(clean_dfs[0])
            if len(clean_dfs) > 1 and not clean_dfs[1].empty: delivered_sheets.append(clean_dfs[1])
            if len(clean_dfs) > 2 and not clean_dfs[2].empty: unclaimed_sheets.append(clean_dfs[2])
            if len(clean_dfs) > 3 and not clean_dfs[3].empty: returned_sheets.append(clean_dfs[3])
            
        except Exception as e:
            print(f" Ошибка при обработке файла {file_name}: {e}")

    # Консолидация
    result = {}
    sheet_keys = [('shipped', shipped_sheets, 'df_delivered_in_transit'), 
                  ('delivered', delivered_sheets, 'df_delivered'), 
                  ('unclaimed', unclaimed_sheets, 'df_undelivered'), 
                  ('returned', returned_sheets, 'df_returned')]
                  
    for key, sheets_list, df_name in sheet_keys:
        if sheets_list:
            combined_df = pd.concat(sheets_list, ignore_index=True)
            if 'номер_заказа' in combined_df.columns:
                combined_df['номер_заказа'] = combined_df['номер_заказа'].replace(r'^\s*$', None, regex=True)
                combined_df = combined_df[combined_df['номер_заказа'].notna()].reset_index(drop=True)
            
            typed_df = _clean_and_type_yandex_df(combined_df)
            result[key] = smart_fill_yandex_gaps(typed_df, name=df_name, verbose=verbose)
        else:
            result[key] = pd.DataFrame()
        
    return result


def build_yandex_stocks(stocks_files_dict: dict, folder_path: str, verbose: bool = True) -> dict:
    """
    Полный автоматический конвейер (ETL) для обработки сложных остатков Яндекс.Маркета.
    Потоково читает листы -> Срезает индивидуальные шапки Excel -> 
    Умно заполняет пропуски -> Удаляет полные и логические дубликаты.
    """
    result = {
        'warehouses': pd.DataFrame(),
        'clusters': pd.DataFrame(),
        'categories': pd.DataFrame()
    }
    
    if not stocks_files_dict:
        if verbose:
            print(" Файлы остатков Яндекс.Маркета не найдены в исходной папке.")
        return result

    stocks_file_name = list(stocks_files_dict.keys())[0]
    full_path = os.path.join(folder_path, stocks_file_name)
    
    if verbose:
        print(f" [Конвейер] Начинаем потоковый разбор листов файла: {stocks_file_name}")

    try:
        excel_reader = pd.ExcelFile(full_path)
        available_sheets = excel_reader.sheet_names

        # Шаг 1. Чтение листов с правильными сдвигами skiprows
        sheets_config = {
            'Остатки по складам': ('warehouses', 5),
            'Остатки по кластерам': ('clusters', 6),
            'Категории': ('categories', 4)
        }

        for sheet_name, (key, skip) in sheets_config.items():
            if sheet_name in available_sheets:
                df_sheet = pd.read_excel(excel_reader, sheet_name=sheet_name, skiprows=skip)
                
                # Стандартная чистка названий колонок
                df_sheet.columns = (
                    df_sheet.columns
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .str.replace(' ', '_')
                    .str.replace('\xa0', ' ', regex=False)
                )
                result[key] = df_sheet
            else:
                if verbose:
                    print(f"  Внимание: Лист '{sheet_name}' не найден.")

        print("\n Запуск очистки, умного заполнения пропусков и дедупликации остатков Яндекса...")

        # Шаг 2. Обработка, заполнение пропусков и дедупликация по бизнес-правилам
        for key in ['warehouses', 'clusters', 'categories']:
            df = result[key]
            if df.empty:
                continue

            # А. Удаление полных дубликатов строк целиком
            full_dups = df.duplicated().sum()
            if full_dups > 0:
                df = df.drop_duplicates().reset_index(drop=True)
                if verbose:
                    print(f"  • Таблица '{key}': Удалено {full_dups} полных копий строк.")

            # Б. Умное заполнение пропусков по совету ревьюера (не зануляем скорость/оборачиваемость)
            numeric_cols = df.select_dtypes(include=[np.number, 'Int64', 'Float64']).columns
            # Исключаем метрики времени и оборачиваемости из зануления
            keep_nan_stock_cols = ['дней', 'оборачиваемость', 'скорость', 'прогноз']
            cols_to_zero = [c for c in numeric_cols if not any(k in str(c).lower() for k in keep_nan_stock_cols)]
            
            df[cols_to_zero] = df[cols_to_zero].fillna(0)

            # В. Логическая дедупликация по составным ключам Яндекса
            # Ищем колонку идентификатора (sku или артикул)
            real_id_col = None
            for col in df.columns:
                if 'sku' in str(col).lower() or 'артикул' in str(col).lower():
                    real_id_col = col
                    break

            if key == 'warehouses' and 'склад' in df.columns and real_id_col:
                df = df.drop_duplicates(subset=[real_id_col, 'склад'], keep='first').reset_index(drop=True)
            elif key == 'clusters' and 'кластер' in df.columns and real_id_col:
                df = df.drop_duplicates(subset=[real_id_col, 'кластер'], keep='first').reset_index(drop=True)
            elif key == 'categories' and 'категория' in df.columns:
                # В категориях проверяем связку категории и кластера, если они есть
                subset_cols = ['категория']
                if 'кластер' in df.columns:
                    subset_cols.append('кластер')
                df = df.drop_duplicates(subset=subset_cols, keep='first').reset_index(drop=True)

            result[key] = df

    except Exception as e:
        print(f" Ошибка при автоматической обработке остатков Яндекса: {e}")

    return result


MARKET_MONTHS_DICT = {
    'янв': '01', 'фев': '02', 'мар': '03', 'апр': '04',
    'май': '05', 'июн': '06', 'июл': '07', 'авг': '08',
    'сен': '09', 'окт': '10', 'ноя': '11', 'дек': '12',
}

def parse_market_period(file_name: str, default_year: str = "2025", verbose: bool = True) -> str:
    """Извлекает год и месяц из имени файла аналитики продаж Яндекса."""
    file_name_lower = unicodedata.normalize('NFC', file_name.lower())

    year_match = re.search(r'\d{4}', file_name_lower)
    year = year_match.group(0) if year_match else default_year
    if verbose and not year_match:
        print(f"  Год не найден в имени файла '{file_name}', использован год по умолчанию: {default_year}")

    period_text = "Неизвестно"
    for month_prefix, num in MARKET_MONTHS_DICT.items():
        if unicodedata.normalize('NFC', month_prefix) in file_name_lower:
            period_text = f"{year}-{num}"
            break

    if period_text == "Неизвестно" and verbose:
        print(f"   Не удалось определить месяц из имени файла: '{file_name}'")

    return period_text


def load_market_sales(sales_files_dict: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Конвейер сборки 'Аналитики продаж' Яндекс.Маркета.
    Приводит колонки к единому стандарту, обогащает периодами, приводит типы,
    умно обрабатывает пропуски и схлопывает логические дубликаты по SKU внутри месяца.
    """
    all_sales_months = []

    if not sales_files_dict:
        if verbose:
            print(" Файлы аналитики продаж Яндекс.Маркета не найдены.")
        return pd.DataFrame()

    if verbose:
        print(f" Начинаем разбор {len(sales_files_dict)} отчетов аналитики продаж Яндекса из памяти...")

    for file_name, df_raw in sales_files_dict.items():
        if df_raw is None or df_raw.empty:
            continue
            
        df_month = df_raw.copy()

        # 1. Очистка заголовков
        df_month.columns = (
            df_month.columns
            .astype(str)
            .str.strip()
            .str.replace('\xa0', ' ', regex=False)
            .str.lower()
            .str.replace(r'\s+', '_', regex=True)
        )

        period_text = parse_market_period(file_name, verbose=verbose)
        df_month['период'] = period_text
        all_sales_months.append(df_month)

    if not all_sales_months:
        return pd.DataFrame()

    # Склеиваем всё в один исторический датафрейм
    df_sales_analytics = pd.concat(all_sales_months, ignore_index=True)

    # 2. Типизация и умная обработка пропусков
    df_sales_analytics['период'] = pd.to_datetime(df_sales_analytics['период'], format='%Y-%m', errors='coerce')
    
    # Автоматически находим названия только числовых столбцов
    numeric_cols = df_sales_analytics.select_dtypes(include=[np.number, 'Int64', 'Float64']).columns
    
    # Разделяем числовые колонки на безопасные и чувствительные (доли, коэффициенты)
    keep_nan_sales_cols = ['доля', 'процент', 'конверсия', 'коэффициент', 'дрр', 'рейтинг']
    cols_to_zero = [c for c in numeric_cols if not any(k in str(c).lower() for k in keep_nan_sales_cols)]
    
    # Зануляем только объемы и рубли
    df_sales_analytics[cols_to_zero] = df_sales_analytics[cols_to_zero].fillna(0)

    # 3. Удаление полных дубликатов строк
    df_sales_analytics = df_sales_analytics.drop_duplicates().reset_index(drop=True)

    # 4. Логическая дедупликация и агрегация (groupby) по связке ['ваш_sku', 'период']
    sku_col = 'ваш_sku'
    period_col = 'период'

    if sku_col in df_sales_analytics.columns and period_col in df_sales_analytics.columns:
        # Для всех числовых колонок настраиваем правила агрегации
        agg_rules = {}
        for col in df_sales_analytics.columns:
            if col in [sku_col, period_col]:
                continue
            elif col in numeric_cols:
                # Если колонка чувствительная — усредняем, если объёмная — суммируем
                if any(k in str(col).lower() for k in keep_nan_sales_cols):
                    agg_rules[col] = 'mean'
                else:
                    agg_rules[col] = 'sum'
            else:
                agg_rules[col] = 'first'

        # Схлопываем внутримесячные дубликаты товаров
        df_sales_analytics = df_sales_analytics.groupby([sku_col, period_col], as_index=False).agg(agg_rules)
        
    return df_sales_analytics

def parse_yandex_orders_reports(orders_files_dict: dict, folder_path: str, verbose: bool = True) -> dict:
    """
    Парсит файлы заказов Яндекс.Маркета (united_orders).
    Автоматически приводит типы, умно заполняет пропуски и удаляет полные дубликаты.
    """
    transactions_sheets = []
    service_sheets = []

    # Внутренние технические списки колонок
    columns_to_int = [
        'id_бизнес-аккаунта', 'id_магазинов', 'инн', 'номер_заказа',
        'ваш_номер_заказа', 'номер_платежного_поручения', 'номер_платежного_поручения.1',
        'номер_платежного_поручения.2', 'номер_платежного_поручения.3', 'номер_платежного_поручения.4',
        'номер_платежного_поручения.5', 'номер_платежного_поручения.6', 'номер_платежного_поручения.7',
        'номер_платежного_поручения.8'
    ]

    columns_to_date = [
        'дата_оформления', 'период', 'дата_отгрузки', 'статус_изменен',
        'дата_доставки_заказа', 'дата_платежного_поручения.1', 'дата_реестра_платежей.1',
        'дата_платежного_поручения.2', 'дата_реестра_платежей.2', 'дата_платежного_поручения.3',
        'дата_реестра_платежей.3', 'дата_платежного_поручения.4', 'дата_реестра_платежей.4',
        'дата_платежного_поручения.5', 'дата_реестра_платежей.5', 'дата_платежного_поручения.6',
        'дата_реестра_платежей.6', 'дата_реестра_платежей', 'дата_платежного_поручения',
        'дата_платежного_поручения.7', 'дата_реестра_платежей.7', 'дата_платежного_поручения.8',
        'дата_реестра_платежей.8'
    ]

    if not orders_files_dict:
        return {'transactions': pd.DataFrame(), 'service': pd.DataFrame()}

    for file_name in orders_files_dict.keys():
        path = os.path.join(folder_path, file_name)
        
        # Экстракция периода из имени файла
        date_match = re.findall(r'(\d{2})-(\d{2})-(\d{4})', file_name)
        if date_match and len(date_match) >= 2:
            file_period = f"{date_match[1][2]}-{date_match[1][1]}"
        elif date_match and len(date_match) == 1:
            file_period = f"{date_match[0][2]}-{date_match[0][1]}"
        else:
            file_period = "Не указан"

        try:
            excel_obj = pd.ExcelFile(path)
            available_sheets = excel_obj.sheet_names
            
            if 'Транзакции по заказам и товарам' in available_sheets:
                df_trans = pd.read_excel(excel_obj, sheet_name='Транзакции по заказам и товарам', skiprows=8)
                df_trans.columns = df_trans.columns.astype(str).str.strip().str.lower().str.replace(' ', '_')
                df_trans['период'] = file_period
                transactions_sheets.append(df_trans)
                
            if 'Услуги и маржа по заказам' in available_sheets:
                df_serv = pd.read_excel(excel_obj, sheet_name='Услуги и маржа по заказам', skiprows=6)
                df_serv.columns = df_serv.columns.astype(str).str.strip().str.lower().str.replace(' ', '_')
                df_serv['период'] = file_period
                service_sheets.append(df_serv)

        except Exception as e:
            print(f" Ошибка при чтении файла заказов {file_name}: {e}")

    result = {}
    sheet_configs = [('transactions', transactions_sheets), ('service', service_sheets)]
    
    for key, sheets_list in sheet_configs:
        if sheets_list:
            df = pd.concat(sheets_list, ignore_index=True)
            
            # Зачистка строк 'Итого' и пустых заказов
            if 'номер_заказа' in df.columns:
                df = df[~df['номер_заказа'].astype(str).str.contains('Итого', case=False, na=False)]
                df['номер_заказа'] = df['номер_заказа'].replace(r'^\s*$', None, regex=True)
                df = df[df['номер_заказа'].notna()].reset_index(drop=True)
            
            # 1. Приведение типов данных
            for col in columns_to_int:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            for col in columns_to_date:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

            # 2. Умное заполнение пропусков по совету ревьюера (не зануляем доли и маржинальность)
            numeric_cols = df.select_dtypes(include=[np.number, 'Int64', 'Float64']).columns
            keep_nan_order_keywords = ['маржа', 'доля', 'процент', 'коэффициент', 'индекс']
            cols_to_zero = [c for c in numeric_cols if not any(k in str(c).lower() for k in keep_nan_order_keywords)]
            df[cols_to_zero] = df[cols_to_zero].fillna(0)

            # 3. Удаление полных дубликатов строк целиком
            full_dups = df.duplicated().sum()
            if full_dups > 0:
                df = df.drop_duplicates().reset_index(drop=True)
                if verbose:
                    print(f"  • Витрина '{key}': успешно удалено {full_dups} полных дубликатов строк целиком.")

            result[key] = df
        else:
            result[key] = pd.DataFrame()

    return result
