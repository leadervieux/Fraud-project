import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import prince
from sklearn.preprocessing import StandardScaler


def distribution(df, column):
    cols = [col for col in df.columns if df[col].nunique() > 1 and col.startswith(column) and col[1:].isdigit()]
    df_plot = df[cols + ['isFraud']].copy()


    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    # loop to create boxplots for each C_ variable against isFraud
    for i, col in enumerate(cols):

        sns.boxplot(
            data=df_plot, 
            x='isFraud', 
            y=col, 
            ax=axes[i], 
            palette='Set2',
            showfliers=False
        )
        axes[i].set_title(f'Distribution de {col} vs isFraud')
        axes[i].set_xlabel('Fraude (1) / Non-Fraude (0)')
        axes[i].set_ylabel('Valeur du compteur')


    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    return



def distribution_bin(df, prefix):
    cols = [col for col in df.columns if col.startswith(prefix)]
    
    
    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols):

        sns.barplot(
            data=df, 
            x=col, 
            y='isFraud', 
            ax=axes[i], 
            palette='coolwarm',
            errorbar=None 
        )
        
        axes[i].set_title(f'Taux de fraude selon {col}')
        axes[i].set_xlabel(f'Valeur de {col} (0 ou 1)')
        axes[i].set_ylabel('Taux de fraude (moyenne)')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    return

# Exemple d'appel :
# distribution_binaires(df_transac, 'M')



def mat_corr(df, column):
    cols = [col for col in df.columns if df[col].nunique() > 1 and col.startswith(column)]
    print(cols)
    corr_matrix = df[cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix of C_ Variables")
    plt.show()
    return

def PCA_func(df, column, n_components):
    cols = [col for col in df.columns if df[col].nunique() > 1 and col.startswith(column) and df[col].nunique() > 1 and not col.endswith('_isMissing')]
    df_pca = df[cols].copy()

    #NaN as median
    df_pca = df_pca.fillna(df_pca.median())

    scaler = StandardScaler()

    df_scaled = scaler.fit_transform(df_pca)
    df_scaled = pd.DataFrame(df_scaled, columns=df_pca.columns, index=df_pca.index)

    pca = prince.PCA(n_components=n_components, random_state=42)
    ca = pca.fit(df_scaled)

    # Coordonnées des individus
    coords = pca.row_coordinates(df_scaled)

    fraud = coords[df['isFraud'] == 1]
    nonfraud = coords[df['isFraud'] == 0]

    plt.figure(figsize=(8,6))

    plt.scatter(
        nonfraud[0],
        nonfraud[1],
        alpha=0.05,
        label='Non fraude'
    )

    plt.scatter(
        fraud[0],
        fraud[1],
        alpha=0.5,
        label='Fraude'
    )

    plt.legend()
    plt.show()

    return pca.column_coordinates_, pca.eigenvalues_summary

def ACM_func_robuste(df, prefix):
    # 1. Sélection
    cols = [col for col in df.columns if col.startswith(prefix)]
    
    # 2. Conversion FORCÉE en chaîne de caractères (le format le plus stable pour l'ACM)
    # On ajoute le nom de la colonne pour éviter que le '0' de M1 soit confondu avec le '0' de M2
    df_m = df[cols].copy()
    for col in df_m.columns:
        df_m[col] = col + "_" + df_m[col].astype(str)
        
    # 3. ACM
    mca = prince.MCA(n_components=2, n_iter=3)
    mca = mca.fit(df_m)
    
    # 4. Affichage avec gestion de la figure
    plt.figure(figsize=(10, 8))
    ax = mca.plot(
        df_m,
        x_component=0,
        y_component=1,
        show_row_labels=False,
        show_column_labels=True, # On affiche les labels pour voir si les points sont là
        show_row_markers=True,
        show_column_markers=True
    )
    plt.title("ACM - Visualisation des variables")
    plt.show()
