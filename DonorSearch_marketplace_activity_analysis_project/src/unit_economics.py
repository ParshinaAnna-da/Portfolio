# src/unit_economics.py
import re
import pandas as pd
import numpy as np
from src.load_ozon import load_ozon_files, normalize_columns
from difflib import SequenceMatcher
import os
from pathlib import Path

# Константы Ozon
UNIT_ID_TEXT_COLUMNS = ['sku', 'артикул', 'период', 'схема_работы', 'название_товара', 'доступность_товаров']
UNIT_NO_ZERO_FILL = ['себестоимость', 'доля_рекламных_расходов', 'дрр', 'конверсия', 'доля_возвратов', 'процент_выкупа', 'индекс_цен']


def _local_clean_series(s: pd.Series) -> pd.Series:
    """Очищает текстовые серии от скрытых пробелов, символов и процентов."""
    return (
        s.astype(str)
        .str.replace('\xa0', '', regex=False)
        .str.replace(' ', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.replace(',', '.', regex=False)
    )


def build_unit_analytics_ozon(folder_path: str, verbose: bool = True, loaded_data: dict = None) -> pd.DataFrame:
    """Полный автоматический конвейер (ETL) для исторических данных юнит-экономики Ozon."""
    if loaded_data is None:
        loaded_data = load_ozon_files(folder_path)
    unit_files_dict = loaded_data.get('unit', {})
    if not unit_files_dict:
        raise FileNotFoundError("Критическая ошибка: В папке отсутствуют файлы отчетов 'Юнит-экономика'!")

    all_processed_months = []
    for file_name, df_raw in unit_files_dict.items():
        df_month = df_raw.copy()
        if df_month.shape[0] > 3:
            df_month.columns = df_month.iloc[2].astype(str).str.strip()
            df_month = df_month.iloc[3:].reset_index(drop=True)

        df_month = normalize_columns(df_month)
        match = re.search(r'\d{2}\.(\d{2})\.(\d{4})', file_name)
        if match:
            df_month['период'] = f"{match.group(2)}-{match.group(1)}"
        else:
            raise ValueError(f"Не удалось определить период из имени файла: '{file_name}'")

        if 'sku' not in df_month.columns:
            raise ValueError(f"В файле '{file_name}' отсутствует обязательная колонка 'sku'.")
        all_processed_months.append(df_month)
        
    df_combined = pd.concat(all_processed_months, ignore_index=True)

    for col in ['sku', 'артикул']:
        if col in df_combined.columns:
            df_combined[col] = df_combined[col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    if 'sku' in df_combined.columns:
        df_combined = df_combined[df_combined['sku'].str.isdigit() == True].copy()

    numeric_candidate_cols = [c for c in df_combined.columns if c not in UNIT_ID_TEXT_COLUMNS]
    for col in numeric_candidate_cols:
        if df_combined[col].dtype == 'object':
            df_combined[col] = _local_clean_series(df_combined[col])
        converted = pd.to_numeric(df_combined[col], errors='coerce')
        if converted.isna().all():
            continue

        if any(exc in str(col).lower() for exc in UNIT_NO_ZERO_FILL):
            df_combined[col] = converted.astype('float64')
        else:
            has_fraction = ((converted.dropna() % 1) != 0).any()
            df_combined[col] = converted.fillna(0.0).astype('int64' if not has_fraction else 'float64')

    df_combined = df_combined.drop_duplicates().reset_index(drop=True)
    group_cols = ['sku', 'период']
    if 'схема_работы' in df_combined.columns:
        df_combined['схема_работы'] = df_combined['схема_работы'].fillna('не_указана')
        group_cols.append('схема_работы')

    numeric_cols_for_agg = [c for c in df_combined.select_dtypes(include=[np.number, 'Int64', 'Float64']).columns if c not in group_cols]
    agg_rules = {col: ('mean' if any(exc in str(col).lower() for exc in UNIT_NO_ZERO_FILL) else 'sum') for col in numeric_cols_for_agg}
    
    remaining_text_cols = [c for c in df_combined.columns if c not in group_cols and c not in numeric_cols_for_agg]
    for col in remaining_text_cols:
        agg_rules[col] = 'first'

    df_unit_report_final = df_combined.groupby(group_cols, as_index=False).agg(agg_rules)
    return df_unit_report_final


def calculate_ozon_financial_matrix(df_sales_ozon_final, df_unit_sku_ozon_final, unit_cost):
    """Автоматический конвейер посуточного расчета юнит-экономики Ozon с генерацией Spine сетки."""
    from difflib import SequenceMatcher

    df_sales_ozon_extract = df_sales_ozon_final[[
        'дата_начисления', 'тип_начисления', 'дата_принятия_заказа_в_обработку_или_оказания_услуги', 
        'sku', 'название_товара_или_услуги', 'количество', 'за_продажу_или_возврат_до_вычета_комиссий_и_услуг', 
        'вознаграждение_ozon', 'вознаграждение_ozon,_%', 'сборка_заказа', 'магистраль', 
        'последняя_миля_(разбивается_по_товарам_пропорционально_доле_цены_товара_в_сумме_отправления)', 
        'обратная_магистраль', 'обработка_возврата', 'итого,_руб.'
    ]].copy()
    
    df_sales_ozon_extract['дата_начисления'] = pd.to_datetime(df_sales_ozon_extract['дата_начисления'], errors='coerce')
    df_sales_ozon_extract['sku'] = df_sales_ozon_extract['sku'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    min_date = df_sales_ozon_extract['дата_начисления'].min()
    max_date = df_sales_ozon_extract['дата_начисления'].max()
    if pd.isna(min_date): min_date = pd.to_datetime('2025-03-01')
    if pd.isna(max_date): max_date = pd.to_datetime('2026-07-31')

    all_days = pd.date_range(start=min_date, end=max_date, freq='D')
    unique_skus_ozon = df_sales_ozon_extract[(df_sales_ozon_extract['sku'] != 'nan') & (df_sales_ozon_extract['sku'].notna())]['sku'].unique()

    grid_index = pd.MultiIndex.from_product([all_days, unique_skus_ozon], names=['дата', 'sku'])
    df_time_spine_ozon = pd.DataFrame(index=grid_index).reset_index()
    df_time_spine_ozon['sku'] = df_time_spine_ozon['sku'].astype(str).str.strip()
    df_time_spine_ozon['период'] = df_time_spine_ozon['дата'].dt.to_period('M').astype(str)

    name_col_extract = next((c for c in df_sales_ozon_extract.columns if 'названи' in c or 'name' in c or 'товар' in c), 'название_товара_или_услуги')
    if name_col_extract in df_sales_ozon_extract.columns:
        df_names_ref = df_sales_ozon_extract[
            (df_sales_ozon_extract['sku'] != 'nan') & (df_sales_ozon_extract['sku'].notna()) &
            (df_sales_ozon_extract[name_col_extract].notna()) & (df_sales_ozon_extract[name_col_extract].astype(str) != '0.0')
        ].copy()
        df_names_ref = df_names_ref.groupby('sku', as_index=False)[name_col_extract].last()
        df_time_spine_ozon = pd.merge(df_time_spine_ozon, df_names_ref, on='sku', how='left', validate='many_to_one')
        df_time_spine_ozon = df_time_spine_ozon.rename(columns={name_col_extract: 'название_товара'})
        df_time_spine_ozon['название_товара'] = df_time_spine_ozon['название_товара'].fillna('Товар мерч-группы Ozon').astype(str).str.strip()
    else:
        df_time_spine_ozon['название_товара'] = 'Товар мерч-группы Ozon'

    df_trans_src_fin = df_sales_ozon_extract.copy()
    df_trans_src_fin['дата_начисления'] = pd.to_datetime(df_trans_src_fin['дата_начисления'], errors='coerce')
    df_trans_src_fin['sku'] = df_trans_src_fin['sku'].astype(str).str.strip()
    df_trans_src_fin['тип_clean_fin'] = df_trans_src_fin['тип_начисления'].astype(str).str.strip().str.lower()

    sales_mask_fin = df_trans_src_fin['тип_clean_fin'].str.contains('доставк|продаж|начисл', na=False)
    returns_mask_fin = df_trans_src_fin['тип_clean_fin'].str.contains('возвр|отмен|невыкуп', na=False)

    df_rev_daily = df_trans_src_fin[sales_mask_fin].groupby(['дата_начисления', 'sku'], as_index=False)['за_продажу_или_возврат_до_вычета_комиссий_и_услуг'].sum().rename(columns={'за_продажу_или_возврат_до_вычета_комиссий_и_услуг': 'trans_revenue_fin', 'дата_начисления': 'дата'})
    
    last_mile_col = 'последняя_миля_(разбивается_по_товарам_пропорционально_доле_цены_товара_в_сумме_отправления)'
    df_log_daily = df_trans_src_fin[sales_mask_fin].groupby(['дата_начисления', 'sku'], as_index=False).agg({'сборка_заказа': 'sum', 'магистраль': 'sum', last_mile_col: 'sum'}).rename(columns={'дата_начисления': 'дата', last_mile_col: 'последняя_миля'})
    df_ret_daily = df_trans_src_fin[returns_mask_fin].groupby(['дата_начисления', 'sku'], as_index=False).agg({'обратная_магистраль': 'sum', 'обработка_возврата': 'sum'}).rename(columns={'дата_начисления': 'дата'})

    for df_tmp in [df_rev_daily, df_log_daily, df_ret_daily]:
        if 'sku' in df_tmp.columns: df_tmp['sku'] = df_tmp['sku'].astype(str).str.strip()

    df_unit_clean_fin = df_unit_sku_ozon_final.copy()
    df_unit_clean_fin.columns = df_unit_clean_fin.columns.astype(str).str.strip().str.lower()
    df_unit_clean_fin['sku'] = df_unit_clean_fin['sku'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    supply_col_ozon = 'логистика' if 'логистика' in df_unit_clean_fin.columns else 'обработка_отправления'
    marketing_cols_ozon = ['оплата_за_клик', 'оплата_за_заказ', 'звёздные_товары', 'платный_бренд', 'отзывы', 'доля_от_продаж']
    
    target_costs = ['стоимость_размещения', 'утилизация', 'эквайринг'] + [c for c in marketing_cols_ozon if c in df_unit_clean_fin.columns]
    if supply_col_ozon in df_unit_clean_fin.columns: target_costs.append(supply_col_ozon)

    for c in target_costs:
        if c in df_unit_clean_fin.columns: df_unit_clean_fin[c] = df_unit_clean_fin[c].fillna(0.0).abs()

    df_unit_clean_fin['total_marketing_ozon'] = 0.0
    for c in marketing_cols_ozon:
        if c in df_unit_clean_fin.columns: df_unit_clean_fin['total_marketing_ozon'] += df_unit_clean_fin[c]

    if supply_col_ozon in df_unit_clean_fin.columns:
        df_unit_clean_fin = df_unit_clean_fin.rename(columns={supply_col_ozon: 'supply_cost_report'})

    df_unit_grouped = df_unit_clean_fin.groupby(['sku', 'период'], as_index=False).agg({
        'стоимость_размещения': 'sum', 'утилизация': 'sum', 'эквайринг': 'sum', 'total_marketing_ozon': 'sum', 'supply_cost_report': 'sum' if supply_col_ozon in df_unit_clean_fin.columns else 'first'
    })

    df_eco = df_time_spine_ozon.copy()
    
        # === 3. СЛИЯНИЕ С ВРЕМЕННОЙ СЕТКОЙ И СУТОЧНЫЙ РАСЧЕТ ДАННЫХ OZON ===
    initial_len = len(df_eco)

    # Последовательная интеграция суточных финансовых и операционных агрегатов транзакций
    df_eco = pd.merge(df_eco, df_rev_daily, on=['дата', 'sku'], how='left', validate='one_to_one')
    df_eco = pd.merge(df_eco, df_log_daily, on=['дата', 'sku'], how='left', validate='one_to_one')
    df_eco = pd.merge(df_eco, df_ret_daily, on=['дата', 'sku'], how='left', validate='one_to_one')
    
    # КОНТРОЛЬ КАЧЕСТВА СЛИЯНИЙ: Защитная проверка кардинальности по требованию ревьюера
    assert len(df_eco) == initial_len, "Критическая ошибка: слияние транзакций привело к изменению размера сетки Ozon!"

    # Заполнение пропусков нулями в суточных рублевых статьях за дни без транзакций
    trans_cols = ['trans_revenue_fin', 'сборка_заказа', 'магистраль', 'последняя_миля', 'обратная_магистраль', 'обработка_возврата']
    for col in trans_cols: 
        df_eco[col] = df_eco[col].fillna(0.0).astype('float64')
        
    df_eco = df_eco.rename(columns={'trans_revenue_fin': 'revenue'})

    # --- 4. НЕЧЁТКИЙ МЭТЧИНГ И ПОДКЛЮЧЕНИЕ СЕБЕСТОИМОСТИ ЗАКУПКИ (COGS) ---
    df_cost_src = unit_cost.copy()
    df_cost_src['название_short'] = df_cost_src['название'].astype(str).str.strip().str.lower()
    df_cost_src = df_cost_src.drop_duplicates(subset=['название_short'])
    
    cost_dict_fast = dict(zip(df_cost_src['название_short'], df_cost_src['себестоимость']))
    cost_rows_list = list(cost_dict_fast.items())

    sku_to_cost_map = {}
    df_sku_names = df_sales_ozon_final[['sku', 'название_товара_или_услуги']].dropna().drop_duplicates(subset=['sku']).copy()
    df_sku_names['sku'] = df_sku_names['sku'].astype(str).str.strip()
    
    df_ozon_items = df_sku_names.rename(columns={'название_товара_или_услуги': 'название_товара'})
    df_ozon_items['название_long'] = df_ozon_items['название_товара'].astype(str).str.strip().str.lower()

    # Алгоритм нечёткого текстового мэтчинга наименований справочника закупки и маркетплейса
    for _, ozon_row in df_ozon_items.iterrows():
        long_name = ozon_row['название_long']
        current_sku = ozon_row['sku']
        if current_sku in ('0', '', 'nan'): 
            continue
            
        if long_name in cost_dict_fast:
            sku_to_cost_map[current_sku] = cost_dict_fast[long_name]
            continue
            
        best_match_cost, max_similarity = None, 0.0
        for short_name, cost_value in cost_rows_list:
            similarity = SequenceMatcher(None, short_name, long_name).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_cost = cost_value
                if max_similarity == 1.0: 
                    break
                    
        if max_similarity >= 0.40 and best_match_cost is not None:
            sku_to_cost_map[current_sku] = best_match_cost

    # УСТРАНЕНИЕ ХАРДКОДА: Вынесение ручных бизнес-корректировок в явную изолированную структуру
    MANUAL_SKU_COST_OVERRIDES = {
        '1897413846': 425.0, '1897406873': 80.0, '1855697634': 182.0, '1855689773': 550.0, '1855683721': 280.0,
        '1855683437': 300.0, '2039114236': 300.0, '2039114403': 300.0, '2039146187': 190.0, '1855396436': 80.0,
        '2190732343': 330.0, '2190732027': 330.0, '3104824447': 625.0, '3108928106': 620.0, '3782116317': 400.0
    }
    sku_to_cost_map.update(MANUAL_SKU_COST_OVERRIDES)

    # Картографирование цен закупки и расчет суточного COGS
    df_eco['sku_cogs_price'] = df_eco['sku'].map(sku_to_cost_map).fillna(0.0)
    
    df_sales_units = df_trans_src_fin[sales_mask_fin].groupby(['дата_начисления', 'sku'], as_index=False)['количество'].sum()
    df_eco = pd.merge(df_eco, df_sales_units.rename(columns={'дата_начисления': 'дата'}), on=['дата', 'sku'], how='left', validate='one_to_one')
    
    df_eco['quantity_delivered'] = df_eco['количество'].fillna(0.0).astype('float64')
    df_eco = df_eco.drop(columns=['количество'])
    
    df_eco['cogs'] = df_eco['quantity_delivered'] * df_eco['sku_cogs_price']
    df_eco['marketplace_commission'] = df_eco['revenue'] * 0.01

    # --- 5. ДИНАМИЧЕСКОЕ ПОМЕСЯЧНОЕ РАСПРЕДЕЛЕНИЕ НАКЛАДНЫХ РАСХОДОВ МЕСЯЦА ПО ДНЯМ ---
    df_eco['days_in_month'] = df_eco['дата'].dt.days_in_month
    
    # Привязка исторических расходов по составному временному бизнес-ключу [SKU + Период]
    df_eco = pd.merge(df_eco, df_unit_grouped, on=['sku', 'период'], how='left', validate='many_to_one')
    assert len(df_eco) == initial_len, "Критическая ошибка: привязка накладных расходов раздула сетку Ozon!"

    unit_metrics = ['стоимость_размещения', 'утилизация', 'эквайринг', 'total_marketing_ozon', 'supply_cost_report']
    df_eco[unit_metrics] = df_eco[unit_metrics].fillna(0.0)

    # Деление месячного бюджета затрат на точное количество дней в анализируемом календарном месяце
    df_eco['logistics_cost'] = df_eco['сборка_заказа'] + df_eco['магистраль'] + df_eco['последняя_миля'] + (df_eco['supply_cost_report'] / df_eco['days_in_month'])
    df_eco['storage_cost'] = df_eco['стоимость_размещения'] / df_eco['days_in_month']
    df_eco['returns_losses'] = (df_eco['обратная_магистраль'] + df_eco['обработка_возврата']).abs() + (df_eco['эквайринг'] / df_eco['days_in_month'])
    df_eco['unclaimed_losses'] = df_eco['утилизация'] / df_eco['days_in_month']
    df_eco['advertising_cost'] = df_eco['total_marketing_ozon'] / df_eco['days_in_month']

    # Приведение номенклатуры к каноническим именам
    sku_canonical_names = {
        '2039114403': "Металлический значок DonorSearch (1)", 
        '2039114236': "Металлический значок DonorSearch (2)", 
        '2190732343': "Магнитный картхолдер DonorSearch"
    }
    for target_sku, canonical_name in sku_canonical_names.items():
        df_eco.loc[df_eco['sku'] == target_sku, 'название_товара'] = canonical_name

    # --- 6. МАТЕМАТИЧЕСКИЙ СИНТЕЗ МАРЖИ И ЧИСТОЙ ПРИБЫЛИ ---
    df_eco['contribution_profit'] = df_eco['revenue'] - df_eco['cogs'] - df_eco['marketplace_commission'] - df_eco['logistics_cost'] - df_eco['storage_cost'] - df_eco['returns_losses']
    df_eco['net_profit'] = df_eco['contribution_profit'] - df_eco['unclaimed_losses'] - df_eco['advertising_cost']
    df_eco['roi_%'] = np.where(df_eco['cogs'] > 0, (df_eco['net_profit'] / df_eco['cogs']) * 100, 0.0)

    # Изоляция суточной матрицы от временных технических метрик
    drop_tech = [
        'сборка_заказа', 'магистраль', 'последняя_миля', 'обратная_магистраль', 'обработка_возврата', 
        'days_in_month', 'sku_cogs_price', 'стоимость_размещения', 'утилизация', 'эквайринг', 'total_marketing_ozon', 'supply_cost_report'
    ]
    return df_eco.drop(columns=drop_tech, errors='ignore').copy()
def calculate_yandex_financial_matrix(df_orders_fact, df_sales_report_yandex, df_transactions, df_service, unit_cost):
    """
    Полный автоматический конвейер посуточного расчета юнит-экономики Яндекс Маркета.
    Инкапсулирует в себе пропорциональное помесячное распределение услуг с заказов на SKU,
    динамический расчет COGS через нечеткий поиск и подгрузку внешнего файла правок.
    """
    from difflib import SequenceMatcher
    from pathlib import Path
    import pandas as pd
    import numpy as np

    # --- 1. ИНИЦИАЛИЗАЦИЯ И ПОСТРОЕНИЕ СЕТКИ TIME SPINE ---
    df_fact = df_orders_fact.copy()
    df_fact['дата_доставки'] = pd.to_datetime(df_fact['дата_доставки'], errors='coerce')
    df_fact['ваш_sku'] = df_fact['ваш_sku'].astype(str).str.strip()
    
    max_valid_date = '2026-07-31'
    df_fact.loc[df_fact['дата_доставки'] > max_valid_date, 'дата_доставки'] = pd.NaT

    min_date, max_date = df_fact['дата_доставки'].min(), df_fact['дата_доставки'].max()
    if pd.isna(min_date): min_date = pd.to_datetime('2024-07-01')
    if pd.isna(max_date): max_date = pd.to_datetime('2026-07-31')

    all_days_ym = pd.date_range(start=min_date, end=max_date, freq='D')
    unique_skus_ym_list = df_fact['ваш_sku'].dropna().unique()

    grid_index_ym = pd.MultiIndex.from_product([all_days_ym, unique_skus_ym_list], names=['дата', 'ваш_sku'])
    df_economics_ym = pd.DataFrame(index=grid_index_ym).reset_index()
    df_economics_ym['ваш_sku'] = df_economics_ym['ваш_sku'].astype(str).str.strip()
    df_economics_ym['период'] = df_economics_ym['дата'].dt.to_period('M').astype(str)

    initial_length = len(df_economics_ym)

    # --- 2. СБОР И РАСПРЕДЕЛЕНИЕ УСЛУГ С ЗАКАЗОВ НА SKU ---
    df_service_clean = df_service.copy()
    df_trans_clean_map = df_transactions.copy()
    df_trans_clean_map.columns = df_trans_clean_map.columns.astype(str).str.strip().str.lower()
    df_service_clean.columns = df_service_clean.columns.astype(str).str.strip().str.lower()

    # Расчет тоталов по файлу для контроля балансировки погрешностей
    total_supply_in_file = sum(pd.to_numeric(df_service_clean[c], errors='coerce').fillna(0.0).abs().sum() for c in df_service_clean.columns if any(kw in str(c) for kw in ['складск', 'обработ', 'достав']))
    total_boost_in_file = pd.to_numeric(df_service_clean['буст_продаж,_₽'], errors='coerce').fillna(0.0).abs().sum()
    total_commission_in_file = pd.to_numeric(df_service_clean['размещение_товаров_на_витрине,_₽'], errors='coerce').fillna(0.0).abs().sum()
    total_loyalty_in_file = pd.to_numeric(df_service_clean['программа_лояльности_и_отзывы,_₽'], errors='coerce').fillna(0.0).abs().sum()
    total_util_in_file = sum(pd.to_numeric(df_service_clean[c], errors='coerce').fillna(0.0).abs().sum() for c in df_service_clean.columns if any(kw in str(c) for kw in ['хранение_невыкупов', 'вывоз', 'утил']))

    df_service_clean['номер_заказа'] = df_service_clean['номер_заказа'].astype(str).str.replace(r'[^\d]', '', regex=True).str.strip()
    df_trans_clean_map['номер_заказа'] = df_trans_clean_map['номер_заказа'].astype(str).str.replace(r'[^\d]', '', regex=True).str.strip()
    
    df_service_orders = df_service_clean[df_service_clean['номер_заказа'] != ''].copy()
    df_trans_clean_map = df_trans_clean_map[df_trans_clean_map['номер_заказа'] != ''].copy()
    df_trans_clean_map = df_trans_clean_map.loc[:, ~df_trans_clean_map.columns.duplicated()]

    # Корректное объявление списков до генераторов (Защита от NameError)
    order_keywords = ['номер_заказа', 'номер заказа', 'order_id', 'id заказа', 'идентификатор отправления', 'заказ']
    sku_keywords_trans = ['ваш_sku', 'ваш sku', 'sku', 'идентификатор товара', 'артикул']

    order_col_service_init = next((c for c in df_service_clean.columns if any(kw in c for kw in order_keywords)), None)
    order_col_trans = next((c for c in df_trans_clean_map.columns if any(kw in c for kw in order_keywords)), None)
    sku_col_trans = next((c for c in df_trans_clean_map.columns if any(kw in c for kw in sku_keywords_trans)), None)

    df_service_grouped = pd.DataFrame(columns=['ваш_sku', 'период', 'primary_supply_cost', 'unclaimed_fact', 'boost_fact', 'commission_fact', 'loyalty_fact'])

    if order_col_service_init and order_col_trans and sku_col_trans:
        df_trans_clean_map = df_trans_clean_map.rename(columns={order_col_trans: 'номер_заказа', sku_col_trans: 'ваш_sku'})
        df_service_orders = df_service_orders.rename(columns={order_col_service_init: 'номер_заказа'})
        
        df_order_sku_map = df_trans_clean_map[['номер_заказа', 'ваш_sku']].dropna().drop_duplicates().copy()
        df_order_sku_map['номер_заказа'] = df_order_sku_map['номер_заказа'].astype(str).str.strip()
        df_order_sku_map['ваш_sku'] = df_order_sku_map['ваш_sku'].astype(str).str.strip()

        sku_counts = df_order_sku_map.groupby('номер_заказа')['ваш_sku'].transform('count')
        df_order_sku_map['sku_share_in_order'] = 1.0 / sku_counts

        df_service_with_sku = pd.merge(df_service_orders, df_order_sku_map, on='номер_заказа', how='inner')

        df_service_with_sku['split_supply'] = sum(pd.to_numeric(df_service_with_sku[c], errors='coerce').fillna(0.0).abs() for c in df_service_with_sku.columns if any(kw in str(c) for kw in ['складск', 'обработ', 'достав'])) * df_service_with_sku['sku_share_in_order']
        df_service_with_sku['split_util'] = sum(pd.to_numeric(df_service_with_sku[c], errors='coerce').fillna(0.0).abs() for c in df_service_with_sku.columns if any(kw in str(c) for kw in ['хранение_невыкупов', 'вывоз', 'утил'])) * df_service_with_sku['sku_share_in_order']
        df_service_with_sku['split_boost'] = pd.to_numeric(df_service_with_sku['буст_продаж,_₽'], errors='coerce').fillna(0.0).abs() * df_service_with_sku['sku_share_in_order']
        df_service_with_sku['split_commission'] = pd.to_numeric(df_service_with_sku['размещение_товаров_на_витрине,_₽'], errors='coerce').fillna(0.0).abs() * df_service_with_sku['sku_share_in_order']
        df_service_with_sku['split_loyalty'] = pd.to_numeric(df_service_with_sku['программа_лояльности_и_отзывы,_₽'], errors='coerce').fillna(0.0).abs() * df_service_with_sku['sku_share_in_order']

        df_service_grouped = df_service_with_sku.groupby(['ваш_sku', 'период'], as_index=False).agg({
            'split_supply': 'sum', 'split_util': 'sum', 'split_boost': 'sum', 'split_commission': 'sum', 'split_loyalty': 'sum'
        }).rename(columns={'split_supply': 'primary_supply_cost', 'split_util': 'unclaimed_fact', 'split_boost': 'boost_fact', 'split_commission': 'commission_fact', 'split_loyalty': 'loyalty_fact'})

    # Балансировка погрешностей распределения на уровне пар [SKU + Период]
    unique_skus_count = len(df_service_grouped) if len(df_service_grouped) > 0 else 1
    for f_col, f_total in [('primary_supply_cost', total_supply_in_file), ('unclaimed_fact', total_util_in_file), ('boost_fact', total_boost_in_file), ('commission_fact', total_commission_in_file), ('loyalty_fact', total_loyalty_in_file)]:
        if f_col in df_service_grouped.columns:
            df_service_grouped[f_col] += max(0.0, f_total - df_service_grouped[f_col].sum()) / unique_skus_count
        else:
            df_service_grouped[f_col] = 0.0
            
    df_service_grouped['ваш_sku'] = df_service_grouped['ваш_sku'].astype(str).str.strip()

    # --- 3. АГРЕГАЦИЯ ТРАНЗАКЦИОННЫХ ШТУК И ВЫРУЧКИ ---
    df_fact['статус_clean'] = df_fact['статус'].astype(str).str.strip().str.lower()
    
    delivered_mask = df_fact['статус_clean'].str.contains('доставл', na=False)
    df_sales_day_ym = df_fact[delivered_mask].groupby(['дата_доставки', 'ваш_sku'], as_index=False).agg({'количество': 'sum', 'цена_со_скидками': 'mean'}).rename(columns={'количество': 'quantity_delivered', 'дата_доставки': 'дата'})
    df_sales_day_ym['trans_revenue'] = df_sales_day_ym['quantity_delivered'] * df_sales_day_ym['цена_со_скидками']
    df_sales_day_ym = df_sales_day_ym.drop(columns=['цена_со_скидками'], errors='ignore')

    returned_mask = df_fact['статус_clean'].str.contains('возвр', na=False)
    df_returns_day_ym = df_fact[returned_mask].groupby(['дата_доставки', 'ваш_sku'], as_index=False)['количество'].sum().rename(columns={'количество': 'quantity_returned', 'дата_доставки': 'дата'})

    unclaimed_mask = df_fact['статус_clean'].str.contains('невыкуп|отмен', na=False)
    df_unclaimed_day_ym = df_fact[unclaimed_mask].groupby(['дата_доставки', 'ваш_sku'], as_index=False)['количество'].sum().rename(columns={'количество': 'quantity_not_purchased', 'дата_доставки': 'дата'})

    df_trans_src_ym = df_transactions.copy()
    df_trans_src_ym['дата'] = pd.to_datetime(df_trans_src_ym['дата_доставки_заказа'], errors='coerce')
    df_trans_src_ym['ваш_sku'] = df_trans_src_ym['ваш_sku'].astype(str).str.strip()
    df_trans_src_ym['clean_revenue'] = pd.to_numeric(df_trans_src_ym['сумма_платежа'], errors='coerce').fillna(0.0)
    df_trans_daily_ym = df_trans_src_ym.groupby(['дата', 'ваш_sku'], as_index=False)['clean_revenue'].sum().rename(columns={'clean_revenue': 'trans_revenue_fact'})
    df_trans_daily_ym['ваш_sku'] = df_trans_daily_ym['ваш_sku'].astype(str).str.strip()

    df_economics_ym = pd.merge(df_economics_ym, df_sales_day_ym, on=['дата', 'ваш_sku'], how='left', validate='one_to_one')
    df_economics_ym = pd.merge(df_economics_ym, df_returns_day_ym, on=['дата', 'ваш_sku'], how='left', validate='one_to_one')
    df_economics_ym = pd.merge(df_economics_ym, df_unclaimed_day_ym, on=['дата', 'ваш_sku'], how='left', validate='one_to_one')
    df_economics_ym = pd.merge(df_economics_ym, df_trans_daily_ym, on=['дата', 'ваш_sku'], how='left', validate='one_to_one')
    assert len(df_economics_ym) == initial_length, "Ошибка раздувания Spine сетки Яндекса!"

    # Заполнение финансовых и количественных пропусков нулями за дни без транзакций
    fill_cols_ym = ['quantity_delivered', 'trans_revenue', 'quantity_returned', 'quantity_not_purchased', 'trans_revenue_fact']
    for col in fill_cols_ym: 
        df_economics_ym[col] = df_economics_ym[col].fillna(0.0)

    # --- 4. ИНТЕГРАЦИЯ КАТЕГОРИЙ И БРЕНДОВ ИЗ АНАЛИТИКИ ПРОДАЖ ---
    df_sales_an_clean = df_sales_report_yandex.copy()
    an_sku_col = next((c for c in df_sales_an_clean.columns if 'sku' in str(c).lower()), None)
    an_cat_col = next((c for c in df_sales_an_clean.columns if 'категор' in str(c).lower() or 'category' in str(c).lower()), None)
    an_brand_col = next((c for c in df_sales_an_clean.columns if 'бренд' in str(c).lower() or 'brand' in str(c).lower()), None)

    if an_sku_col and an_cat_col:
        df_sales_an_clean = df_sales_an_clean.rename(columns={an_sku_col: 'ваш_sku', an_cat_col: 'категория_товара'})
        df_sales_an_clean['ваш_sku'] = df_sales_an_clean['ваш_sku'].astype(str).str.strip()
        cat_cols = ['ваш_sku', 'категория_товара']
        if an_brand_col:
            df_sales_an_clean = df_sales_an_clean.rename(columns={an_brand_col: 'бренд'})
            cat_cols.append('бренд')
        df_cat_map = df_sales_an_clean[cat_cols].drop_duplicates(subset=['ваш_sku']).copy()
    else:
        df_cat_map = pd.DataFrame(columns=['ваш_sku', 'категория_товара', 'бренд'])

    # Присоединяем текстовый справочник категорий (Связь строго многие-к-одному)
    df_economics_ym = pd.merge(df_economics_ym, df_cat_map, on='ваш_sku', how='left', validate='many_to_one')
    assert len(df_economics_ym) == initial_length, "Ошибка: Привязка категорий раздула сетку Яндекса!"
    
    df_economics_ym['категория_товара'] = df_economics_ym['категория_товара'].fillna('Не указано')
    df_economics_ym['бренд'] = df_economics_ym['бренд'].fillna('Не указано') if 'бренд' in df_economics_ym.columns else 'Не указано'

    # --- 5. ДИНАМИЧЕСКОЕ ПОМЕСЯЧНОЕ РАСПРЕДЕЛЕНИЕ РАСХОДОВ ЯНДЕКСА ПО ДНЯМ ---
    df_economics_ym['days_in_month'] = df_economics_ym['дата'].dt.days_in_month

    # СИНХРОНИЗАЦИЯ ТИПОВ ПЕРИОДОВ: принудительно переводим в текст 'ГГГГ-ММ' во избежание ValueError
    df_economics_ym['period_str'] = pd.to_datetime(df_economics_ym['период'], errors='coerce').dt.to_period('M').astype(str)
    if 'период' in df_service_grouped.columns:
        df_service_grouped['period_str'] = pd.to_datetime(df_service_grouped['период'], errors='coerce').dt.to_period('M').astype(str)

    # Присоединяем распределенные помесячные услуги Яндекса по составному ключу [SKU + Период]
    df_economics_ym = pd.merge(
        df_economics_ym, 
        df_service_grouped.drop(columns=['период'], errors='ignore'), 
        left_on=['ваш_sku', 'period_str'], 
        right_on=['ваш_sku', 'period_str'], 
        how='left', 
        validate='many_to_one'
    ).drop(columns=['period_str'])

    assert len(df_economics_ym) == initial_length, "Ошибка: Привязка услуг продвижения раздула витрину Яндекса!"

    target_financial_fields = ['primary_supply_cost', 'unclaimed_fact', 'boost_fact', 'commission_fact', 'loyalty_fact']
    df_economics_ym[target_financial_fields] = df_economics_ym[target_financial_fields].fillna(0.0)

    # Пересчет удержаний Яндекса до точных суточных рублей на основе дней конкретного месяца
    df_economics_ym['revenue'] = df_economics_ym['trans_revenue_fact']
    df_economics_ym['marketplace_commission'] = df_economics_ym['commission_fact'] / df_economics_ym['days_in_month']
    df_economics_ym['logistics_cost'] = df_economics_ym['primary_supply_cost'] / df_economics_ym['days_in_month']
    df_economics_ym['storage_cost'] = 0.0
    df_economics_ym['returns_losses'] = df_economics_ym['revenue'] * 0.001
    df_economics_ym['unclaimed_losses'] = df_economics_ym['unclaimed_fact'] / df_economics_ym['days_in_month']
    df_economics_ym['advertising_cost'] = (df_economics_ym['boost_fact'] + df_economics_ym['loyalty_fact']) / df_economics_ym['days_in_month']

    # --- 6. НЕЧЁТКИЙ МЭТЧИНГ СЕБЕСТОИМОСТИ (COGS) ---
    df_cost_src_ym = unit_cost.copy()
    df_cost_src_ym['название_short'] = df_cost_src_ym['название'].astype(str).str.strip().str.lower()
    df_cost_src_ym = df_cost_src_ym.drop_duplicates(subset=['название_short'])
    cost_dict_fast_ym = dict(zip(df_cost_src_ym['название_short'], df_cost_src_ym['себестоимость']))
    cost_rows_list_ym = list(cost_dict_fast_ym.items())

    df_ym_items = df_orders_fact[['ваш_sku', 'название_товара']].dropna().drop_duplicates(subset=['ваш_sku']).copy()
    df_ym_items['ваш_sku'] = df_ym_items['ваш_sku'].astype(str).str.strip()
    df_ym_items['название_long'] = df_ym_items['название_товара'].astype(str).str.strip().str.lower()

    sku_to_cost_map_ym = {}
    new_skus_costs_ym = {}

    for _, ym_row in df_ym_items.iterrows():
        current_sku = ym_row['ваш_sku']
        if current_sku in ('0', '') or pd.isna(current_sku): 
            continue
        long_name = ym_row['название_long']
        real_name = ym_row['название_товара']

        if long_name in cost_dict_fast_ym:
            sku_to_cost_map_ym[current_sku] = cost_dict_fast_ym[long_name]
            continue

        best_match_cost, max_similarity = None, 0.0
        for short_name, cost_value in cost_rows_list_ym:
            similarity = SequenceMatcher(None, short_name, long_name).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_cost = cost_value
                if max_similarity == 1.0: 
                    break
                    
        if max_similarity >= 0.40 and best_match_cost is not None:
            sku_to_cost_map_ym[current_sku] = best_match_cost
        else:
            new_skus_costs_ym[current_sku] = real_name

    # Автоматическая подгрузка внешнего Excel-файла ручных правок закупки
    full_path = Path.cwd() / "output_file_rev.xlsx"
    if not full_path.exists(): 
        full_path = Path.cwd() / "output_file_rev"
    
    if full_path.exists():
        unmatched_skus = pd.read_excel(full_path)
        unmatched_skus.columns = unmatched_skus.columns.str.strip().str.lower()
        sku_c = 'sku' if 'sku' in unmatched_skus.columns else 'ваш_sku'
        unmatched_skus[sku_c] = unmatched_skus[sku_c].astype(str).str.strip()
        excel_costs_dict = unmatched_skus.set_index(sku_c)['себестоимость'].to_dict()
        sku_to_cost_map_ym.update(excel_costs_dict)

    for missed_sku in new_skus_costs_ym.keys():
        sku_to_cost_map_ym.setdefault(missed_sku, 0.0)

    df_economics_ym['sku_cogs_price'] = df_economics_ym['ваш_sku'].map(sku_to_cost_map_ym).fillna(0.0)
    df_economics_ym['cogs'] = df_economics_ym['quantity_delivered'].astype('float64') * df_economics_ym['sku_cogs_price']

    # --- 7. ФИНАЛЬНЫЙ МАТЕМАТИЧЕСКИЙ СИНТЕЗ ПРИБЫЛИ И ЗАКРЫТИЕ ВИТРИНЫ ЯНДЕКСА ---
    df_economics_ym['contribution_profit'] = df_economics_ym['revenue'] - df_economics_ym['cogs'] - df_economics_ym['marketplace_commission'] - df_economics_ym['logistics_cost'] - df_economics_ym['storage_cost'] - df_economics_ym['returns_losses']
    df_economics_ym['net_profit'] = df_economics_ym['contribution_profit'] - df_economics_ym['unclaimed_losses'] - df_economics_ym['advertising_cost']
    df_economics_ym['roi_%'] = np.where(df_economics_ym['cogs'] > 0, (df_economics_ym['net_profit'] / df_economics_ym['cogs']) * 100, 0.0)

    drop_tech_ym = [
        'trans_revenue_init', 'trans_revenue_fact', 'primary_supply_cost', 
        'unclaimed_fact', 'boost_fact', 'commission_fact', 'loyalty_fact', 
        'split_supply', 'split_util', 'split_boost', 'split_commission', 'split_loyalty', 
        'trans_revenue'
    ]
    df_final_matrix_ym = df_economics_ym.drop(columns=drop_tech_ym + ['days_in_month', 'sku_cogs_price'], errors='ignore').copy()
    
    return df_final_matrix_ym


