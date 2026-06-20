import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════

# Columns with high cardinality
HIGH_CARD_COLS = ['P_emaildomain', 'R_emaildomain', 'DeviceInfo', 'id_30', 'id_31', 'id_33']

# Binary columns  
BINARY_M_COLS = ['M1','M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']

BINARY_ID_COLS = ['id_12','id_16', 'id_35','id_27','id_28','id_29','id_36','id_37','id_38','DeviceType']

# Columns between 3 and 10 values
MID_CARD_COLS = ['M4', 'ProductCD', 'card4', 'card6']

MID_ID_COLS = ['id_15', 'id_23', 'id_25', 'id_34']

# ═══════════════════════════════════════════════════════════════
#  UTILITARIES FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def simplify_os(val):
    val = str(val).lower()
    if 'android' in val: return 'Android'
    if 'ios'     in val: return 'iOS'
    if 'mac'     in val: return 'Mac'
    if 'windows' in val: return 'Windows'
    if 'linux'   in val: return 'Linux'
    return 'Other'

def simplify_browser(val):
    val = str(val).lower()
    if 'chrome'  in val: return 'Chrome'
    if 'firefox' in val: return 'Firefox'
    if 'safari'  in val: return 'Safari'
    if 'edge'    in val: return 'Edge'
    if 'samsung' in val: return 'Samsung'
    return 'Other'

def simplify_email(val):
    val = str(val).lower()
    if 'gmail' in val: return 'google'
    if 'yahoo' in val: return 'yahoo'
    if 'hotmail' in val or 'outlook' in val or 'live' in val: return 'microsoft'
    if 'icloud' in val or 'me.com' in val: return 'apple'
    return 'Other'

def simplify_device_info(val):
    if pd.isna(val): return 'Missing'
    
    val = str(val).lower()
    
    if 'samsung' in val: return 'Samsung'
    if 'sm-' in val: return 'Samsung' # Beaucoup de SM- sont des Samsung
    if 'ios' in val or 'apple' in val or 'iphone' in val: return 'Apple'
    if 'windows' in val: return 'Windows'
    if 'mac' in val: return 'Mac'
    if 'lg' in val: return 'LG'
    if 'xt' in val or 'moto' in val: return 'Motorola'
    if 'blade' in val or 'zte' in val: return 'ZTE'
    if 'huawei' in val: return 'Huawei'
    
    return 'Other'

def clean_binary_cols(df, col_name, mapping_dict):
    # .replace() est parfait ici
    df[col_name] = df[col_name].replace(mapping_dict)
    
    # On force le type en numérique, et on met -1 pour tout ce qui reste (les NaN ou valeurs non mappées)
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(-1)
    
    return df

# ═══════════════════════════════════════════════════════════════
#  PREPROCESSING df_transac
# ═══════════════════════════════════════════════════════════════

def preprocess_transac(df_transac: pd.DataFrame) -> pd.DataFrame:
    df = df_transac.copy()

    # 1. Remove constant values
    df = df.drop(columns=['V107', 'D7'], errors='ignore')
    

    # 2. Binary variables M: T→1, F→0, NaN→0
    for col in BINARY_M_COLS:
        if col in df.columns:
            df[col] = df[col].replace({'T': 1, 'F': 0}).fillna(-1).astype(int)

    # 3. one-hot encoding
    for col in MID_CARD_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('Missing')
            df = pd.get_dummies(df, columns=[col], prefix=col, dtype=int)

    # 4. P_emailDomain and R_emailDomain

    if 'R_emaildomain' in df.columns and 'P_emaildomain' in df.columns:
        df['is_recipient_same'] = df['R_emaildomain'].isna().astype(int)

        df['R_emaildomain'] = df['R_emaildomain'].fillna(df['P_emaildomain'])

        df['P_emaildomain'] = df['P_emaildomain'].apply(simplify_email)
        df['R_emaildomain'] = df['R_emaildomain'].apply(simplify_email)


    for col in ['P_emaildomain', 'R_emaildomain']:
        if col in df.columns:
            df[col] = df[col].fillna('Other')
            df = pd.get_dummies(df, columns=[col], prefix=col, dtype=int)

    return df



# ═══════════════════════════════════════════════════════════════
#  PREPROCESSING df_id
# ═══════════════════════════════════════════════════════════════

def preprocess_id(df_id: pd.DataFrame) -> pd.DataFrame:
    df = df_id.copy()

    #binary categories
    mapping_identity = {'Found': 1, 'NotFound': 0}
    mapping_status = {'Found': 1, 'New': 0}
    mapping_tf = {'T': 1, 'F': 0}
    mapping_device = {'desktop': 0, 'mobile' : 1}

    if 'id_28' in df.columns:
        df = clean_binary_cols(df, 'id_28', mapping_status)
    
    for col in ['id_12', 'id_35', 'id_23','id_27','id_29']:
        if col in df.columns:
            df = clean_binary_cols(df, col, mapping_identity)

    for col in ['id_16','id_36', 'id_37', 'id_38']:
        if col in df.columns:
            df = clean_binary_cols(df, col, mapping_tf)

    if 'DeviceType' in df.columns:
        df = clean_binary_cols(df,'DeviceType', mapping_device)


    # 1. id_33 → width / height / pixels

    if 'id_33' in df.columns:
        df['id_33'] = df['id_33'].fillna('0x0')
        res_split = df['id_33'].str.split('x', expand=True)
        df['screen_width']  = pd.to_numeric(res_split[0], errors='coerce').fillna(0).astype(int)
        df['screen_height'] = pd.to_numeric(res_split[1], errors='coerce').fillna(0).astype(int)
        df['screen_pixels'] = df['screen_width'] * df['screen_height']
        df = df.drop(columns=['id_33'])

    # 2. id_30 / id_31 : simplification OS and navigator
    if 'id_30' in df.columns:
        df['os_family'] = df['id_30'].apply(simplify_os)
        df = df.drop(columns=['id_30'])
    if 'id_31' in df.columns:
        df['browser_family'] = df['id_31'].apply(simplify_browser)
        df = df.drop(columns=['id_31'])

    # DeviceInfo: simplification

    if 'DeviceInfo' in df.columns:
        df['DeviceInfo_brand'] = df['DeviceInfo'].apply(simplify_device_info)
        df.drop(columns=['DeviceInfo'], inplace=True)

    # One hot encoding
    for col in MID_ID_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('Missing')
            df = pd.get_dummies(df, columns=[col], prefix=col, dtype=int)


    cols_to_encode = ['P_emaildomain', 'R_emaildomain','os_family', 'browser_family', 'DeviceInfo_brand']
    for col in cols_to_encode:
        if col in df.columns:
            df[col] = df[col].fillna('Missing')
            df = pd.get_dummies(df, columns=[col], prefix=col, dtype=int)
        

    return df


# ═══════════════════════════════════════════════════════════════
#  PIPELINE COMPLET
# ═══════════════════════════════════════════════════════════════

def preprocessing(df_transac: pd.DataFrame, df_id: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet de prétraitement.

    Paramètres
    ----------
    df_transac : DataFrame brut des transactions
    df_id      : DataFrame brut des identités
    y          : Series cible (isFraud) — obligatoire sur le train set
    encoder    : TargetEncoder déjà fitté — obligatoire sur le test set

    Retourne
    --------
    df_combined : DataFrame prêt pour XGBoost
    encoder     : TargetEncoder fitté (à passer lors du preprocessing test)

    Exemple d'utilisation
    ---------------------
        # ── TRAIN ──
        df_train, enc = preprocessing(df_transac_train, df_id_train,
                                      y=df_transac_train['isFraud'])
        X_train = df_train.drop(columns=['TransactionID', 'isFraud'])
        y_train = df_train['isFraud']

        # ── TEST ──
        df_test, _ = preprocessing(df_transac_test, df_id_test, encoder=enc)
        X_test = df_test.drop(columns=['TransactionID'])
    """

    # ── 1. Nettoyage individuel ──────────────────────────────
    df_t = preprocess_transac(df_transac)
    df_i = preprocess_id(df_id)

    # ── 2. Merge (left join : toutes les transactions) ───────
    df_combined = df_t.merge(df_i, on='TransactionID', how='left')



    print(f" Final shape: {df_combined.shape}")
    print(f"  Residuals NaN (float) : {df_combined.isnull().sum().sum()}")

    return df_combined