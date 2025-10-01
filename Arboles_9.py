import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import pickle
import joblib
import os
from datetime import datetime


warnings.filterwarnings('ignore')

# ==========================================
# PASO 1: ANÁLISIS INICIAL DEL DATASET
# ==========================================

def analizar_estructura_dataset(df):
    """
    Analiza la estructura de tu dataset sin asumir nada
    
    Args:
        df: Tu DataFrame original
    
    Returns:
        dict: Información sobre la estructura de tus datos
    """
    
    print("=" * 60)
    print("PASO 1: ANÁLISIS DE ESTRUCTURA DE TU DATASET")
    print("=" * 60)
    
    # Información básica
    print(f"\nInformación General:")
    print(f"   - Número de filas: {len(df):,}")
    print(f"   - Número de columnas: {len(df.columns)}")
    print(f"   - Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Clasificar columnas por tipo
    categorical_columns = []
    numerical_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            categorical_columns.append(col)
        elif df[col].nunique() < 50 and df[col].dtype in ['int64', 'float64']:
            # Numéricas con pocas categorías (probablemente categóricas codificadas)
            categorical_columns.append(col)
        else:
            numerical_columns.append(col)
    
    print(f"\nDistribución por Tipo:")
    print(f"   - Variables categóricas: {len(categorical_columns)}")
    print(f"   - Variables numéricas: {len(numerical_columns)}")
    
    # Análisis de valores faltantes
    missing_info = df.isnull().sum()
    cols_with_missing = missing_info[missing_info > 0].sort_values(ascending=False)
    
    print(f"\nValores Faltantes:")
    if len(cols_with_missing) > 0:
        print(f"   - Columnas con valores faltantes: {len(cols_with_missing)}")
        print(f"   - Total de valores faltantes: {missing_info.sum():,}")
        
        print(f"\n   Top 10 columnas con más valores faltantes:")
        for col, missing_count in cols_with_missing.head(10).items():
            pct = (missing_count / len(df)) * 100
            print(f"     • {col}: {missing_count:,} ({pct:.1f}%)")
    else:
        print("   - No hay valores faltantes en el dataset")
    
    # Análisis de cardinalidad para categóricas
    print(f"\nAnálisis de Variables Categóricas:")
    if categorical_columns:
        print(f"   Primeras 15 variables categóricas y su cardinalidad:")
        high_cardinality = []
        
        for col in categorical_columns[:15]:
            unique_count = df[col].nunique()
            print(f"     • {col}: {unique_count} valores únicos")
            
            if unique_count > 100:
                high_cardinality.append(col)
        
        if high_cardinality:
            print(f"\nVariables con alta cardinalidad (>100 valores): {len(high_cardinality)}")
            for col in high_cardinality[:5]:
                print(f"     • {col}: {df[col].nunique()} valores")
    
    # Análisis de variables numéricas
    print(f"\nAnálisis de Variables Numéricas:")
    if numerical_columns:
        print(f"   Estadísticas de primeras 10 variables numéricas:")
        for col in numerical_columns[:10]:
            stats = df[col].describe()
            print(f"     • {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, "
                  f"mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    # Detectar posibles outliers en numéricas
    print(f"\nDetección de Outliers (método IQR):")
    outlier_info = {}
    
    for col in numerical_columns[:10]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        
        if len(outliers) > 0:
            outlier_pct = (len(outliers) / len(df)) * 100
            outlier_info[col] = len(outliers)
            if outlier_pct > 1:  # Solo mostrar si hay más del 1% de outliers
                print(f"     • {col}: {len(outliers)} outliers ({outlier_pct:.1f}%)")
    
    # Resumen para siguientes pasos
    results = {
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns,
        'missing_columns': cols_with_missing.to_dict(),
        'high_cardinality_cols': high_cardinality if 'high_cardinality' in locals() else [],
        'outlier_info': outlier_info,
        'total_rows': len(df),
        'total_cols': len(df.columns)
    }
    
    print(f"\nRESUMEN PARA SIGUIENTES PASOS:")
    print(f"   - Dataset listo para preprocesamiento")
    print(f"   - Se expandirá a ~{len(categorical_columns) * 3 + len(numerical_columns)} variables después del encoding")
    print(f"   - Adecuado para descubrimiento de segmentos con árboles")
    
    return results

# Crear visualizaciones básicas del dataset
def visualizar_estructura_basica(df, categorical_cols, numerical_cols):
    """
    Crea visualizaciones básicas de la estructura del dataset
    """
    
    print(f"\nCREANDO VISUALIZACIONES BÁSICAS...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Análisis Básico de Tu Dataset', fontsize=16, fontweight='bold')
    
    # 1. Distribución de tipos de variables
    ax1 = axes[0, 0]
    type_counts = ['Categóricas', 'Numéricas']
    counts = [len(categorical_cols), len(numerical_cols)]
    colors = ['lightblue', 'lightcoral']
    
    bars = ax1.bar(type_counts, counts, color=colors, alpha=0.8)
    ax1.set_title('Distribución de Tipos de Variables')
    ax1.set_ylabel('Número de Variables')
    
    # Añadir valores en las barras
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Valores faltantes por columna (top 15)
    ax2 = axes[0, 1]
    missing_data = df.isnull().sum().sort_values(ascending=False)
    top_missing = missing_data.head(15)
    
    if len(top_missing[top_missing > 0]) > 0:
        missing_to_plot = top_missing[top_missing > 0]
        ax2.barh(range(len(missing_to_plot)), missing_to_plot.values, color='red', alpha=0.6)
        ax2.set_yticks(range(len(missing_to_plot)))
        ax2.set_yticklabels([col[:20] + '...' if len(col) > 20 else col 
                            for col in missing_to_plot.index], fontsize=8)
        ax2.set_xlabel('Número de Valores Faltantes')
        ax2.set_title('Top Variables con Valores Faltantes')
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, 'No hay valores\nfaltantes', 
                ha='center', va='center', transform=ax2.transAxes, 
                fontsize=12, fontweight='bold')
        ax2.set_title('Valores Faltantes')
    
    # 3. Cardinalidad de variables categóricas (top 15)
    ax3 = axes[1, 0]
    if categorical_cols:
        cardinality = {col: df[col].nunique() for col in categorical_cols[:15]}
        sorted_cardinality = dict(sorted(cardinality.items(), key=lambda x: x[1], reverse=True))
        
        cols_to_plot = list(sorted_cardinality.keys())
        values_to_plot = list(sorted_cardinality.values())
        
        ax3.barh(range(len(cols_to_plot)), values_to_plot, color='green', alpha=0.6)
        ax3.set_yticks(range(len(cols_to_plot)))
        ax3.set_yticklabels([col[:20] + '...' if len(col) > 20 else col 
                            for col in cols_to_plot], fontsize=8)
        ax3.set_xlabel('Número de Valores Únicos')
        ax3.set_title('Cardinalidad Variables Categóricas')
        ax3.invert_yaxis()
    
    # 4. Distribución de una variable numérica (ejemplo)
    ax4 = axes[1, 1]
    if numerical_cols:
        sample_col = numerical_cols[0]  # Tomar la primera variable numérica
        ax4.hist(df[sample_col].dropna(), bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax4.set_xlabel(sample_col)
        ax4.set_ylabel('Frecuencia')
        ax4.set_title(f'Distribución de {sample_col[:30]}...' if len(sample_col) > 30 else f'Distribución de {sample_col}')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Visualizaciones básicas creadas")


# ==========================================
# PASO 2: PREPARACIÓN DE DATOS PARA CLUSTERING
# ==========================================

def preparar_datos_para_clustering(df,  max_categories_onehot=20):
    """
    Prepara tu dataset para clustering sin asumir estructura específica
    
    Args:
        df: Tu DataFrame original
        max_categories_onehot: Máximo de categorías para One-Hot encoding
    
    Returns:
        dict: Datos preparados y metadatos
    """
    
    print("=" * 60)
    print("🔧 PASO 2: PREPARACIÓN PARA CLUSTERING")
    print("=" * 60)
    
    # Columnas a eliminar
    columnas_eliminar = [
        'IndInactivo', 'Puntaje Acierta Rango', 'Oficina', 'Rango Edad', 'Antiguedad',
        'Nivel Academico', 'Rango Ingresos', 'Segmento Ingresos vs. Antiguedad',
        'Cuenta de Deposito', 'Cuenta AFC', 'Cheque Cuenta', 'Tarjeta Debito', 'PAP',
        'Cupo Activo', 'Credito Libre Inv sin Grtia', 'Credito de Vivienda', 'CrediMutual',
        'Bancaseguros', 'MasterCardCupo', 'MasterCardSaldo', 'VISACupo', 'VISASaldo',
        'Credisolidario', 'Solidaridad Plan Basico', 'Incrementos Plan Basico', 'Vida',
        'Vida Clasica', 'Hospitalizacion', 'Tranquilidad', 'Recuperacion', 'Seguros Auto',
        'Hogar Mas y Total Home', 'Otras polizas', 'Medicina Integral', 'CEM', 'Salud Oral',
        'ValorVida', 'Habeas Data', 'IntervaloValorVida', 'IntervaloValorPerseverancia',
        'IntervaloValorRenta', 'IntervalonumCantidadProductos', 'CuotaCeroUno',
        'CorrespondenciaVirtualoFisica', 'Orden Antiguedad', 'Orden Rango Ingresos',
        'Orden Rango Edad', 'CuotasMoraBUC', 'Desempleo', 'Alerta_Habito_Pago_Externo',
        'Alerta_Capacidad_Pago_Externo', 'Estrellas', 'tenencia', 'uso', 'EstadoScore',
        'CodDaneCiudadHomologada', 'CiudadHomologada', 'Cantidad_AliviosFinancieros_Coomeva',
        'Valor_Entregado_AliviosFinancieros_Coomeva', 'Cantidad_AliviosFinancieros_Bancoomeva',
        'Cantidad_AliviosFinancieros_MP', 'Cantidad_AliviosFinancieros_Corredor',
        'Valor_Entregado_AliviosFinancieros_Corredor', 'Cantidad_AliviosFinancieros_Fundacion',
        'ValorBeneficios_Financiero_Histo', 'ValorBeneficios_Salud_Histo',
        'ValorBeneficios_Educacion_Histo', 'ValorBeneficios_Coomeva_Histo',
        'ValorBeneficios_FondoSolidar_Histo', 'ValorBeneficios_CoomevaCuotaTAC_Histo',
        'ValorBeneficios_SaludSaludOral_Histo', 'ValorBeneficios_SaludCEM_Histo',
        'ValorBeneficios_FinancieroProdAhorro_Histo', 'ValorBeneficios_FinancieroProdCredito_Histo',
        'ValorBeneficios_FinancieroServicios_Histo', 'Beneficiarios solidaridad',
        'Beneficiarios solidaridad - hijos menores', 'Beneficiarios solidaridad - hijos mayores',
        'Beneficiarios solidaridad - padres', 'Beneficiarios solidaridad - conyugue', 'GrupoFamiliar',
        'TIENE EL DEBITO AUTOMATICO AUTORIZADO', 'AntigÃƒÂ¼edad_Producto_CrediSolidario',
        'AntigÃƒÂ¼edad_Producto_CrediAsociado', 'AntigÃƒÂ¼edad_Producto_Fondo_Social_Vivienda_Credito_Patrimonial',
        'AntigÃƒÂ¼edad_Producto_Autos', 'AntigÃƒÂ¼edad_Producto_Hogar', 'AntigÃƒÂ¼edad_Producto_MP',
        'AntigÃƒÂ¼edad_Producto_SaludOral', 'AntigÃƒÂ¼edad_Producto_CEM', 'AntigÃƒÂ¼edad_Producto_Vida',
        'AntigÃƒÂ¼edad_Producto_VidaClasica', 'AntigÃƒÂ¼edad_Producto_Hospitalizacion',
        'AntigÃƒÂ¼edad_Producto_Tranquilidad', 
        'AntigÃ¼edad_Producto_CrediSolidario', 'AntigÃ¼edad_Producto_CrediAsociado', 'AntigÃ¼edad_Producto_Fondo_Social_Vivienda_Credito_Patrimonial', 
        'AntigÃ¼edad_Producto_Autos', 'AntigÃ¼edad_Producto_Hogar', 'AntigÃ¼edad_Producto_MP', 'AntigÃ¼edad_Producto_SaludOral', 'AntigÃ¼edad_Producto_CEM', 
        'AntigÃ¼edad_Producto_Vida', 'AntigÃ¼edad_Producto_VidaClasica', 'AntigÃ¼edad_Producto_Hospitalizacion', 'AntigÃ¼edad_Producto_Tranquilidad',
        'Tipo_Cliente_Asociado',
        'Indice_RecaudoPromedio_Coomeva_Creditos_Ult3M',
        'Indice_RecaudoPromedio_Coomeva_Seguros_Ult_3Meses',
        'Indice_RecaudoPromedio_Bancoomeva_Ult_3Meses', 'Asociado_RiesgoExclusion',
        'Tipo_Cliente_Bancoomeva', 'Tipo_Cliente_Prepagada', 'Tipo_Cliente_CEM',
        'Tipo_Cliente_Seguros', 'Tipo_Cliente_Adicionales', 'CrediAsociado',
        'Tipo_Cliente_Cooperativa', 'Arquetipo_MP', 'Tipo_Cliente_MP', 'Top100Asociado',
        'Turnos_En_Oficinas_Total_Ult12Meses',
        'Turnos_En_Oficinas_Coomeva_Asesores_Integrales_Ult12Meses',
        'Turnos_En_Oficinas_Bancoomeva_FelicitacionesQuejasReclamos_Ult12Meses',
        'Turnos_En_Oficinas_Bancoomeva_Venta_De_Productos_Ult12Meses',
        'Turnos_En_Oficinas_Bancoomeva_Otros_Servicios_Ult12Meses',
        'Turnos_En_Oficinas_UltimaOficinaVisitada_Ult12Meses',
        'Tiempo_Promedio_De_Espera_En_Oficinas_Ult12Meses',
        'Tiempo_Promedio_De_Atencion_En_Oficinas_Ult12Meses',
        'Tiempo_Promedio_Total_En_Oficinas_Ult12Meses', 'FondoSocialViviendaPatrimonial',
        'CantIntencionRetiroCooperativa', 'CantPeriodosInactivoUlt12meses',
        'CantPeriodosInactivoUlt6meses', 'CantPeriodosInactivoUlt3meses',
        'InactivoMesAnterior', 'TipoIdentificacion', 'Score_Fuga_Hogar',
        'Score_Fuga_DisminucionesPlanBasico', 'Score_Fuga_Autos',
        'Score Rentabilidad del Cliente Solidaridad', 'Score_Compra_IncrementosPlanBasico',
        'Score_Compra_Hogar', 'Score_Compra_Autos', 'cluster', 'val_acierta',
        'val_perseverancia', 'rango_perseverancia', 'Rango_renta', 'Rango_acierta',
        'CantidadEventosRecreacion_Histo', 'CantidadEventosRecreacionSinCosto_Histo', 'CantidadEventosRecreacionConCosto_Histo',
        'ValorEventosRecreacionConCosto_Histo', 'CantidadEventosRecreacion_Ult_3Meses', 'CantidadEventosRecreacionSinCosto_Ult_3Meses',
        'CantidadEventosRecreacionConCosto_Ult_3Meses', 'ValorEventosRecreacionConCosto_Ult_3Meses', 'CantidadEventosEducacion_Histo',
        'CantidadEventosEducacionConCosto_Histo', 'ValorEventosEducacionConCosto_Histo', 'CantidadEventosFundacion_Histo',
        'DebitoAutomatico'
    ]
    
    # Columnas a excluir (ID)
    exclude_columns = ['Documento']
    
    # Crear copia del dataframe y eliminar columnas innecesarias
    work_df = df.copy()
    
    # Eliminar columnas innecesarias si existen
    columnas_existentes_eliminar = [col for col in columnas_eliminar if col in work_df.columns]
    if columnas_existentes_eliminar:
        work_df = work_df.drop(columns=columnas_existentes_eliminar)
        print(f"Columnas eliminadas: {len(columnas_existentes_eliminar)}")
        for col in columnas_existentes_eliminar:
            print(f"   - {col}")
    
    # Excluir columna ID si existe
    columnas_excluir_existentes = [col for col in exclude_columns if col in work_df.columns]
    if columnas_excluir_existentes:
        work_df = work_df.drop(columns=columnas_excluir_existentes)
        print(f"Columnas excluidas (ID): {len(columnas_excluir_existentes)}")
        for col in columnas_excluir_existentes:
            print(f"   - {col}")


    # Convertir columnas específicas a formato numérico
    columnas_numericas_especificas = [
        'Edad',
        'score',
        'numCantidadProductos', 
        'salud_financiera_externa',
        'CantidadEventosRecreacion_Ult_3Meses', 
        'CantidadEventosRecreacionSinCosto_Ult_3Meses', 
        'CantidadEventosRecreacionConCosto_Ult_3Meses', 
        'CantidadEventosEducacion_Histo', 
        'CantidadEventosEducacionConCosto_Histo', 
        'CantidadEventosFundacion_Histo', 
        'Cantidad_AliviosFinancieros_GECC'
    ]
    
    columnas_convertidas = []
    for col in columnas_numericas_especificas:
        if col in work_df.columns:
            try:
                # Convertir a numérico, coercing errores a NaN, luego a int64
                work_df[col] = pd.to_numeric(work_df[col], errors='coerce').astype('Int64')
                columnas_convertidas.append(col)
            except Exception as e:
                print(f"No se pudo convertir {col} a int64: {e}")
    
    if columnas_convertidas:
        print(f"Columnas convertidas a formato int64: {len(columnas_convertidas)}")
        for col in columnas_convertidas:
            print(f"   - {col}")

        
    # Re-clasificar columnas en el dataset de trabajo
    categorical_columns = []
    numerical_columns = []
    
    for col in work_df.columns:
        if work_df[col].dtype == 'object':
            categorical_columns.append(col)
        elif work_df[col].nunique() < 50 and work_df[col].dtype in ['int64', 'float64']:
            categorical_columns.append(col)
        else:
            numerical_columns.append(col)
    
    print(f"\nVariables a procesar:")
    print(f"   - Categóricas: {len(categorical_columns)}")
    print(f"   - Numéricas: {len(numerical_columns)}")
    
    # Separar categóricas por estrategia de encoding
    cat_low_cardinality = []   # One-hot encoding
    cat_high_cardinality = []  # Label encoding
    
    for col in categorical_columns:
        unique_count = work_df[col].nunique()
        if unique_count <= max_categories_onehot:
            cat_low_cardinality.append(col)
        else:
            cat_high_cardinality.append(col)
    
    print(f"\nEstrategia de encoding:")
    print(f"   - One-Hot Encoding: {len(cat_low_cardinality)} variables")
    print(f"   - Label Encoding: {len(cat_high_cardinality)} variables")
    print(f"   - Variables numéricas: {len(numerical_columns)} (normalizar)")
    
    # Crear dataset encodificado
    encoded_data = pd.DataFrame(index=work_df.index)
    label_encoders = {}
    encoding_info = {}
    
    # 1. PROCESAR VARIABLES NUMÉRICAS
    print(f"\nProcesando variables numéricas...")
    
    for col in numerical_columns:
        # Manejar valores faltantes con mediana
        median_value = work_df[col].median()
        encoded_data[col] = work_df[col].fillna(median_value)
        
        # Contar valores faltantes imputados
        missing_count = work_df[col].isnull().sum()
        if missing_count > 0:
            print(f"   • {col}: {missing_count} valores faltantes imputados con mediana ({median_value:.2f})")
    
    # 2. PROCESAR CATEGÓRICAS CON ONE-HOT ENCODING
    print(f"\nProcesando categóricas con One-Hot Encoding...")
    
    for col in cat_low_cardinality:
        # Manejar valores faltantes
        col_data = work_df[col].fillna('Missing')
        
        # Crear dummies
        dummies = pd.get_dummies(col_data, prefix=col, drop_first=True)
        
        # Añadir al dataset encodificado
        for dummy_col in dummies.columns:
            encoded_data[dummy_col] = dummies[dummy_col]
        
        print(f"   • {col}: {work_df[col].nunique()} categorías → {len(dummies.columns)} variables")
        
        # Guardar info
        encoding_info[col] = {
            'type': 'onehot',
            'original_categories': work_df[col].nunique(),
            'encoded_columns': list(dummies.columns)
        }
    
    # 3. PROCESAR CATEGÓRICAS CON LABEL ENCODING
    print(f"\nProcesando categóricas con Label Encoding...")
    
    for col in cat_high_cardinality:
        # Manejar valores faltantes
        col_data = work_df[col].fillna('Missing').astype(str)
        
        # Label encoding
        le = LabelEncoder()
        encoded_values = le.fit_transform(col_data)
        encoded_data[f'{col}_label'] = encoded_values
        
        # Guardar encoder para interpretación posterior
        label_encoders[col] = le
        
        print(f"   • {col}: {work_df[col].nunique()} categorías → 1 variable numérica")
        
        # Guardar info
        encoding_info[col] = {
            'type': 'label',
            'original_categories': work_df[col].nunique(),
            'encoded_columns': [f'{col}_label']
        }
    
    print(f"\nResultado del encoding:")
    print(f"   - Dimensiones originales: {work_df.shape}")
    print(f"   - Dimensiones después del encoding: {encoded_data.shape}")
    print(f"   - Expansión: {work_df.shape[1]} → {encoded_data.shape[1]} variables")
    
    # 4. NORMALIZAR TODOS LOS DATOS
    print(f"\nNormalizando datos para clustering...")
    
    scaler = StandardScaler()
    data_normalized = pd.DataFrame(
        scaler.fit_transform(encoded_data),
        columns=encoded_data.columns,
        index=encoded_data.index
    )
    
    print(f"Normalización completada")
    print(f"   - Media de todas las variables: ~0.00")
    print(f"   - Desviación estándar de todas las variables: ~1.00")
    
    # Verificar que no hay valores faltantes
    missing_after_encoding = data_normalized.isnull().sum().sum()
    print(f"\nVerificación final:")
    print(f"   - Valores faltantes después del procesamiento: {missing_after_encoding}")
    print(f"   - Shape final para clustering: {data_normalized.shape}")
    
    # Preparar resultados
    results = {
        'data_normalized': data_normalized,
        'data_encoded_raw': encoded_data,
        'original_data': work_df,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'encoding_info': encoding_info,
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns,
        'cat_low_cardinality': cat_low_cardinality,
        'cat_high_cardinality': cat_high_cardinality,
        'exclude_columns': exclude_columns
    }
    
    return results

def verificar_preparacion_datos(results):
    """
    Verifica que la preparación de datos fue exitosa
    """
    
    print(f"\nVERIFICACIÓN DE PREPARACIÓN DE DATOS:")
    print(f"   - Datos normalizados listos: {results['data_normalized'].shape}")
    print(f"   - Sin valores faltantes: {results['data_normalized'].isnull().sum().sum() == 0}")
    print(f"   - Rango de valores: [{results['data_normalized'].min().min():.2f}, {results['data_normalized'].max().max():.2f}]")
    
    # Mostrar algunas estadísticas
    print(f"\nEstadísticas de verificación:")
    print(f"   - Media general: {results['data_normalized'].mean().mean():.6f}")
    print(f"   - Std general: {results['data_normalized'].std().mean():.6f}")
    
    return True

'''def visualizar_preparacion_datos(results):
    """
    Visualiza el resultado de la preparación de datos
    """
    
    print(f"\nCREANDO VISUALIZACIONES DE PREPARACIÓN...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Resultado de la Preparación de Datos', fontsize=16, fontweight='bold')
    
    data_norm = results['data_normalized']
    
    # 1. Distribución de medias por variable
    ax1 = axes[0, 0]
    means = data_norm.mean()
    ax1.hist(means, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Media de Variables')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución de Medias\n(Debe estar cerca de 0)')
    ax1.axvline(x=0, color='red', linestyle='--', label='Objetivo: 0')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribución de desviaciones estándar
    ax2 = axes[0, 1]
    stds = data_norm.std()
    #ax2.hist(stds, bins=30, color='green', alpha=0.7, edgecolor='black')
    
    if stds.nunique() == 1:
        # Todos los valores son iguales, crear un gráfico de barras simple
        ax2.bar([stds.iloc[0]], [len(stds)], color='green', alpha=0.7, edgecolor='black', width=0.1)
        ax2.set_xlim(stds.iloc[0] - 0.2, stds.iloc[0] + 0.2)
        ax2.text(stds.iloc[0], len(stds)/2, f'Todas las variables\ntienen std = {stds.iloc[0]:.3f}', 
                ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        # Caso normal con valores diferentes
        try:
            ax2.hist(stds, bins='auto', color='green', alpha=0.7, edgecolor='black')
        except:
            ax2.hist(stds, bins=max(1, stds.nunique()), color='green', alpha=0.7, edgecolor='black')



    ax2.set_xlabel('Desviación Estándar de Variables')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de Desviaciones Estándar\n(Debe estar cerca de 1)')
    ax2.axvline(x=1, color='red', linestyle='--', label='Objetivo: 1')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Expansión de dimensionalidad
    ax3 = axes[1, 0]
    original_cols = len(results['original_data'].columns)
    encoded_cols = len(results['data_normalized'].columns)
    
    bars = ax3.bar(['Original', 'Después Encoding'], [original_cols, encoded_cols], 
                   color=['lightblue', 'lightcoral'], alpha=0.8)
    ax3.set_ylabel('Número de Variables')
    ax3.set_title('Expansión de Dimensionalidad')
    
    # Añadir valores en barras
    for bar, value in zip(bars, [original_cols, encoded_cols]):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Heatmap de correlación (muestra de 20 variables)
    ax4 = axes[1, 1]
    sample_cols = data_norm.columns[:20]  # Tomar muestra de 20 variables
    correlation_matrix = data_norm[sample_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, ax=ax4, cbar_kws={'label': 'Correlación'})
    ax4.set_title('Correlación entre Variables\n(Muestra de 20 variables)')
    ax4.set_xlabel('Variables')
    ax4.set_ylabel('Variables')
    
    plt.tight_layout()
    plt.show()
    
    print("Visualizaciones de preparación creadas")'''


# ==========================================
# PASO 3. EVALUACIÓN MULTI-MODELO DE CLUSTERING
# ==========================================


from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

def evaluacion_modelos_clustering(data_normalized, k_range=(4, 10), tipos_covarianza=['full', 'tied', 'diag']):
    """
    Ejecuta múltiples modelos de clustering con diferentes configuraciones
    Especializado para encontrar los mejores clusters sin restricciones de balance
    
    Args:
        data_normalized: Datos normalizados del Paso 2
        k_range: Rango de número de clusters (inicio, fin)
        tipos_covarianza: Tipos de matriz de covarianza a probar para GMM
    
    Returns:
        dict: Resultados completos de todos los modelos
    """
    
    print("=" * 70)
    print("EVALUACIÓN MULTI-MODELO DE CLUSTERING")
    print("=" * 70)
    
    print(f"Configuración:")
    print(f"   • Rango de clusters: {k_range[0]} a {k_range[1]}")
    print(f"   • Modelos a evaluar: GMM, K-Means, K-Means++, DBSCAN, Jerárquico")
    print(f"   • Dataset: {data_normalized.shape}")
    
    # Reducir dimensiones si es muy grande para mejorar clustering
    if data_normalized.shape[1] > 100:
        print(f"\nAplicando PCA para optimizar clustering...")
        pca = PCA(n_components=60, random_state=42)
        data_for_clustering = pca.fit_transform(data_normalized)
        variance_explained = pca.explained_variance_ratio_.sum()
        print(f"PCA aplicado: 60 componentes, {variance_explained:.1%} varianza")
        
        data_for_clustering = pd.DataFrame(data_for_clustering, 
                                         columns=[f'PC{i+1}' for i in range(60)],
                                         index=data_normalized.index)
        pca_model = pca
    else:
        data_for_clustering = data_normalized
        pca_model = None
        print(f"Sin PCA necesario")
    
    resultados_todos_modelos = {}
    
    # ==========================================
    # 1. GAUSSIAN MIXTURE MODEL
    # ==========================================
    print(f"\n1. Ejecutando Gaussian Mixture Models...")
    
    resultados_gmm = {}
    
    for cov_type in tipos_covarianza:
        print(f"\nTipo de covarianza: {cov_type.upper()}")
        print("-" * 40)
        
        resultados_cov = {}
        
        for k in range(k_range[0], k_range[1] + 1):
            print(f"   k={k}...", end="")
            
            try:
                # Configurar Gaussian Mixture
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov_type,
                    random_state=42,
                    max_iter=200,
                    n_init=5,
                    init_params='kmeans'
                )
                
                # Ajustar modelo
                gmm.fit(data_for_clustering)
                
                # Obtener labels y probabilidades
                labels = gmm.predict(data_for_clustering)
                probabilidades = gmm.predict_proba(data_for_clustering)
                
                # Calcular métricas de clustering
                silhouette = silhouette_score(data_for_clustering, labels)
                calinski = calinski_harabasz_score(data_for_clustering, labels)
                davies_bouldin = davies_bouldin_score(data_for_clustering, labels)
                
                # Calcular métricas específicas de GMM
                bic = gmm.bic(data_for_clustering)
                aic = gmm.aic(data_for_clustering)
                log_likelihood = gmm.score(data_for_clustering)
                
                # Analizar distribución de clusters
                unique, counts = np.unique(labels, return_counts=True)
                distribucion = dict(zip(unique, counts))
                cluster_sizes = counts / len(labels)
                
                min_size = cluster_sizes.min()
                max_size = cluster_sizes.max()
                std_size = cluster_sizes.std()
                
                # Analizar calidad de asignaciones (confianza promedio)
                max_probs = probabilidades.max(axis=1)
                confianza_promedio = max_probs.mean()
                asignaciones_ambiguas = (max_probs < 0.6).sum()
                
                # Score compuesto universal (ajustado para todos los modelos)
                silhouette_norm = (silhouette + 1) / 2
                calinski_norm = min(calinski / 3000, 1)
                davies_norm = max(0, 1 - davies_bouldin / 3)
                confianza_norm = confianza_promedio
                
                # Penalización por desbalance extremo
                # Si el cluster más grande tiene más del 90%, penalizar fuertemente
                balance_penalty = 1.0
                if max_size > 0.9:  # Si hay un cluster con más del 90%
                    balance_penalty = 0.1  # Penalización del 90%
                elif max_size > 0.7:  # Si hay un cluster con más del 70%
                    balance_penalty = 0.5  # Penalización del 50%
                elif max_size > 0.5:  # Si hay un cluster con más del 50%
                    balance_penalty = 0.8  # Penalización del 20%
                
                # Bonus por distribución balanceada
                balance_bonus = 1.0
                if std_size < 0.1:  # Desviación estándar baja = más balance
                    balance_bonus = 1.2
                elif std_size < 0.2:
                    balance_bonus = 1.1
                
                score_gmm = (silhouette_norm * 0.4 + 
                           calinski_norm * 0.2 + 
                           davies_norm * 0.2 + 
                           confianza_norm * 0.2) * balance_penalty * balance_bonus
                
                resultados_cov[k] = {
                    'modelo': gmm,
                    'tipo_modelo': 'GMM',
                    'labels': labels,
                    'probabilidades': probabilidades,
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski,
                    'davies_bouldin': davies_bouldin,
                    'bic': bic,
                    'aic': aic,
                    'log_likelihood': log_likelihood,
                    'distribucion': distribucion,
                    'cluster_sizes': cluster_sizes,
                    'min_size': min_size,
                    'max_size': max_size,
                    'std_size': std_size,
                    'confianza_promedio': confianza_promedio,
                    'asignaciones_ambiguas': asignaciones_ambiguas,
                    'score_gmm': score_gmm,
                    'cov_type': cov_type
                }
                
                print(f"S:{silhouette:.3f}, BIC:{bic:.0f}, Conf:{confianza_promedio:.3f}, Score:{score_gmm:.3f}")
                
            except Exception as e:
                print(f"Error: {str(e)[:30]}...")
                continue
        
        if resultados_cov:
            resultados_gmm[cov_type] = resultados_cov
            print(f"{len(resultados_cov)} modelos exitosos")
    
    if resultados_gmm:
        resultados_todos_modelos['gmm'] = resultados_gmm
    
    # ==========================================
    # 2. K-MEANS Y K-MEANS++
    # ==========================================
    print(f"\n2. Ejecutando K-Means y K-Means++...")
    
    for init_method, method_name in [('k-means++', 'K-Means++'), ('random', 'K-Means')]:
        print(f"\nMétodo: {method_name}")
        print("-" * 40)
        
        resultados_kmeans = {}
        
        for k in range(k_range[0], k_range[1] + 1):
            print(f"   k={k}...", end="")
            
            try:
                # Configurar K-Means
                kmeans = KMeans(
                    n_clusters=k,
                    init=init_method,
                    n_init=10,
                    max_iter=300,
                    random_state=42
                )
                
                # Ajustar modelo
                kmeans.fit(data_for_clustering)
                labels = kmeans.labels_
                
                # Calcular métricas
                silhouette = silhouette_score(data_for_clustering, labels)
                calinski = calinski_harabasz_score(data_for_clustering, labels)
                davies_bouldin = davies_bouldin_score(data_for_clustering, labels)
                inertia = kmeans.inertia_
                
                # Analizar distribución
                unique, counts = np.unique(labels, return_counts=True)
                distribucion = dict(zip(unique, counts))
                cluster_sizes = counts / len(labels)
                
                min_size = cluster_sizes.min()
                max_size = cluster_sizes.max()
                std_size = cluster_sizes.std()
                
                # Calcular distancias a centroides para confianza
                distances = kmeans.transform(data_for_clustering)
                min_distances = distances.min(axis=1)
                confianza_promedio = 1 - (min_distances / min_distances.max()).mean()
                
                # Score compuesto
                silhouette_norm = (silhouette + 1) / 2
                calinski_norm = min(calinski / 3000, 1)
                davies_norm = max(0, 1 - davies_bouldin / 3)
                confianza_norm = confianza_promedio
                
                # Penalización por desbalance extremo
                balance_penalty = 1.0
                if max_size > 0.9:
                    balance_penalty = 0.1
                elif max_size > 0.7:
                    balance_penalty = 0.5
                elif max_size > 0.5:
                    balance_penalty = 0.8
                
                # Bonus por distribución balanceada
                balance_bonus = 1.0
                if std_size < 0.1:
                    balance_bonus = 1.2
                elif std_size < 0.2:
                    balance_bonus = 1.1
                
                score_gmm = (silhouette_norm * 0.4 + 
                           calinski_norm * 0.2 + 
                           davies_norm * 0.2 + 
                           confianza_norm * 0.2) * balance_penalty * balance_bonus
                
                resultados_kmeans[k] = {
                    'modelo': kmeans,
                    'tipo_modelo': method_name,
                    'labels': labels,
                    'probabilidades': None,  # K-Means no tiene probabilidades
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski,
                    'davies_bouldin': davies_bouldin,
                    'inertia': inertia,
                    'bic': None,
                    'aic': None,
                    'log_likelihood': None,
                    'distribucion': distribucion,
                    'cluster_sizes': cluster_sizes,
                    'min_size': min_size,
                    'max_size': max_size,
                    'std_size': std_size,
                    'confianza_promedio': confianza_promedio,
                    'asignaciones_ambiguas': 0,
                    'score_gmm': score_gmm,
                    'cov_type': None
                }
                
                print(f"S:{silhouette:.3f}, Inertia:{inertia:.0f}, Score:{score_gmm:.3f}")
                
            except Exception as e:
                print(f"Error: {str(e)[:30]}...")
                continue
        
        if resultados_kmeans:
            resultados_todos_modelos[init_method] = {'default': resultados_kmeans}
            print(f"{len(resultados_kmeans)} modelos exitosos")
    
    # ==========================================
    # 3. DBSCAN
    # ==========================================
    print(f"\n3. Ejecutando DBSCAN...")
    print("-" * 40)
    
    # Para DBSCAN, necesitamos determinar eps óptimo
    # Usamos el método del codo con k-nearest neighbors
    n_neighbors = 5
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(data_for_clustering)
    distances, indices = neigh.kneighbors(data_for_clustering)
    distances = np.sort(distances[:, n_neighbors-1], axis=0)
    
    # Probar diferentes valores de eps basados en percentiles más altos para evitar demasiado ruido
    eps_values = np.percentile(distances, [50, 60, 70, 75, 80, 85, 90, 95])
    
    resultados_dbscan = {}
    
    for i, eps in enumerate(eps_values):
        print(f"   eps={eps:.3f}...", end="")
        
        try:
            # Configurar DBSCAN
            dbscan = DBSCAN(
                eps=eps,
                min_samples=5,
                metric='euclidean',
                n_jobs=-1
            )
            
            # Ajustar modelo
            labels = dbscan.fit_predict(data_for_clustering)
            
            # Verificar si hay clusters válidos (excluyendo ruido -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters < 2:
                print(f"Sólo {n_clusters} clusters encontrados")
                continue
            
            # Calcular métricas (excluyendo puntos de ruido)
            mask = labels != -1
            if mask.sum() < len(labels) * 0.5:  # Si más del 50% es ruido, saltar
                print(f"Demasiado ruido ({(~mask).sum()/len(labels)*100:.1f}%)")
                continue
            
            silhouette = silhouette_score(data_for_clustering[mask], labels[mask])
            calinski = calinski_harabasz_score(data_for_clustering[mask], labels[mask])
            davies_bouldin = davies_bouldin_score(data_for_clustering[mask], labels[mask])
            
            # Analizar distribución
            unique, counts = np.unique(labels, return_counts=True)
            distribucion = dict(zip(unique, counts))
            
            # Calcular tamaños excluyendo ruido
            valid_labels = labels[labels != -1]
            unique_valid, counts_valid = np.unique(valid_labels, return_counts=True)
            cluster_sizes = counts_valid / len(valid_labels)
            
            min_size = cluster_sizes.min() if len(cluster_sizes) > 0 else 0
            max_size = cluster_sizes.max() if len(cluster_sizes) > 0 else 0
            std_size = cluster_sizes.std() if len(cluster_sizes) > 0 else 0
            
            # Confianza basada en proporción de puntos no-ruido
            confianza_promedio = mask.sum() / len(labels)
            
            # Score compuesto con penalización por desbalance
            silhouette_norm = (silhouette + 1) / 2
            calinski_norm = min(calinski / 3000, 1)
            davies_norm = max(0, 1 - davies_bouldin / 3)
            confianza_norm = confianza_promedio
            
            # Penalización por desbalance extremo
            balance_penalty = 1.0
            if max_size > 0.9:
                balance_penalty = 0.1
            elif max_size > 0.7:
                balance_penalty = 0.5
            elif max_size > 0.5:
                balance_penalty = 0.8
            
            # Bonus por distribución balanceada
            balance_bonus = 1.0
            if std_size < 0.1:
                balance_bonus = 1.2
            elif std_size < 0.2:
                balance_bonus = 1.1
            
            # Penalización adicional por demasiado ruido
            noise_penalty = 1.0
            noise_ratio = (labels == -1).sum() / len(labels)
            if noise_ratio > 0.3:  # Más del 30% es ruido
                noise_penalty = 0.5
            elif noise_ratio > 0.2:  # Más del 20% es ruido
                noise_penalty = 0.8
            
            score_gmm = (silhouette_norm * 0.4 + 
                       calinski_norm * 0.2 + 
                       davies_norm * 0.2 + 
                       confianza_norm * 0.2) * balance_penalty * balance_bonus * noise_penalty
            
            resultados_dbscan[f'eps_{eps:.3f}'] = {
                'modelo': dbscan,
                'tipo_modelo': 'DBSCAN',
                'labels': labels,
                'probabilidades': None,
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies_bouldin,
                'eps': eps,
                'n_clusters': n_clusters,
                'n_noise': (labels == -1).sum(),
                'bic': None,
                'aic': None,
                'log_likelihood': None,
                'distribucion': distribucion,
                'cluster_sizes': cluster_sizes,
                'min_size': min_size,
                'max_size': max_size,
                'std_size': std_size,
                'confianza_promedio': confianza_promedio,
                'asignaciones_ambiguas': 0,
                'score_gmm': score_gmm,
                'cov_type': None
            }
            
            print(f"Clusters:{n_clusters}, S:{silhouette:.3f}, Score:{score_gmm:.3f}")
            
        except Exception as e:
            print(f"Error: {str(e)[:30]}...")
            continue
    
    if resultados_dbscan:
        resultados_todos_modelos['dbscan'] = {'default': resultados_dbscan}
        print(f"{len(resultados_dbscan)} modelos exitosos")
    
    # ==========================================
    # 4. CLUSTERING JERÁRQUICO
    # ==========================================
    print(f"\n4. Ejecutando Clustering Jerárquico...")
    
    linkage_methods = ['ward', 'complete', 'average']
    
    for linkage_method in linkage_methods:
        print(f"\nMétodo de enlace: {linkage_method.upper()}")
        print("-" * 40)
        
        resultados_hierarchical = {}
        
        for k in range(k_range[0], k_range[1] + 1):
            print(f"   k={k}...", end="")
            
            try:
                # Configurar Clustering Jerárquico
                hierarchical = AgglomerativeClustering(
                    n_clusters=k,
                    linkage=linkage_method,
                    metric='euclidean' if linkage_method == 'ward' else 'euclidean'
                )
                
                # Ajustar modelo
                labels = hierarchical.fit_predict(data_for_clustering)
                
                # Calcular métricas
                silhouette = silhouette_score(data_for_clustering, labels)
                calinski = calinski_harabasz_score(data_for_clustering, labels)
                davies_bouldin = davies_bouldin_score(data_for_clustering, labels)
                
                # Analizar distribución
                unique, counts = np.unique(labels, return_counts=True)
                distribucion = dict(zip(unique, counts))
                cluster_sizes = counts / len(labels)
                
                min_size = cluster_sizes.min()
                max_size = cluster_sizes.max()
                std_size = cluster_sizes.std()
                
                # Confianza basada en cohesión intra-cluster
                confianza_promedio = 0.7  # Valor base para jerárquico
                
                # Score compuesto con penalización por desbalance
                silhouette_norm = (silhouette + 1) / 2
                calinski_norm = min(calinski / 3000, 1)
                davies_norm = max(0, 1 - davies_bouldin / 3)
                confianza_norm = confianza_promedio
                
                # Penalización por desbalance extremo
                balance_penalty = 1.0
                if max_size > 0.9:
                    balance_penalty = 0.1
                elif max_size > 0.7:
                    balance_penalty = 0.5
                elif max_size > 0.5:
                    balance_penalty = 0.8
                
                # Bonus por distribución balanceada
                balance_bonus = 1.0
                if std_size < 0.1:
                    balance_bonus = 1.2
                elif std_size < 0.2:
                    balance_bonus = 1.1
                
                score_gmm = (silhouette_norm * 0.4 + 
                           calinski_norm * 0.2 + 
                           davies_norm * 0.2 + 
                           confianza_norm * 0.2) * balance_penalty * balance_bonus
                
                resultados_hierarchical[k] = {
                    'modelo': hierarchical,
                    'tipo_modelo': f'Hierarchical-{linkage_method}',
                    'labels': labels,
                    'probabilidades': None,
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski,
                    'davies_bouldin': davies_bouldin,
                    'linkage': linkage_method,
                    'bic': None,
                    'aic': None,
                    'log_likelihood': None,
                    'distribucion': distribucion,
                    'cluster_sizes': cluster_sizes,
                    'min_size': min_size,
                    'max_size': max_size,
                    'std_size': std_size,
                    'confianza_promedio': confianza_promedio,
                    'asignaciones_ambiguas': 0,
                    'score_gmm': score_gmm,
                    'cov_type': None
                }
                
                print(f"S:{silhouette:.3f}, Cal:{calinski:.0f}, Score:{score_gmm:.3f}")
                
            except Exception as e:
                print(f"Error: {str(e)[:30]}...")
                continue
        
        if resultados_hierarchical:
            resultados_todos_modelos[f'hierarchical_{linkage_method}'] = {'default': resultados_hierarchical}
            print(f"{len(resultados_hierarchical)} modelos exitosos")
    
    return {
        'resultados': resultados_todos_modelos,
        'data_used': data_for_clustering,
        'pca_model': pca_model,
        'data_original': data_normalized
    }

def analizar_resultados(resultados_clustering):
    """
    Analiza los resultados de todos los modelos de clustering y selecciona el mejor
    """
    
    print(f"\nANÁLISIS DE RESULTADOS MULTI-MODELO")
    print("=" * 60)
    
    if not resultados_clustering['resultados']:
        print("No hay resultados para analizar")
        return None
    
    # Recopilar todos los resultados en una lista plana
    todos_resultados = []
    
    for tipo_modelo, resultados_tipo in resultados_clustering['resultados'].items():
        print(f"\nAnálisis modelo {tipo_modelo.upper()}:")
        print("-" * 30)
        
        for subtipo, resultados_subtipo in resultados_tipo.items():
            if not resultados_subtipo:
                print(f"No hay resultados para {subtipo}")
                continue
            
            print(f"Resultados {subtipo}:")
            
            for k, resultado in resultados_subtipo.items():
                # Agregar información del modelo
                resultado['modelo_tipo_completo'] = f"{tipo_modelo}_{subtipo}"
                resultado['k_param'] = k
                todos_resultados.append(resultado)
                
                # Mostrar resumen
                print(f"      {k}: S={resultado['silhouette']:.3f}, Score={resultado['score_gmm']:.3f}")
                
                # Mostrar distribución para algunos modelos
                if 'distribucion' in resultado:
                    distribucion = resultado['distribucion']
                    if -1 in distribucion:  # DBSCAN con ruido
                        n_noise = distribucion[-1]
                        total = sum(distribucion.values())
                        print(f"             Ruido: {n_noise}/{total} ({n_noise/total*100:.1f}%)")
    
    # Encontrar el mejor modelo global
    print(f"\nSELECCIÓN DEL MEJOR MODELO GLOBAL:")
    print("-" * 40)
    
    # Ordenar por score_gmm
    todos_resultados_sorted = sorted(todos_resultados, key=lambda x: x['score_gmm'], reverse=True)
    
    # Mostrar top 5 modelos
    print("\nTop 5 mejores modelos:")
    for i, resultado in enumerate(todos_resultados_sorted[:5]):
        tipo = resultado.get('tipo_modelo', resultado.get('cov_type', 'N/A'))
        k_info = resultado.get('k_param', 'N/A')
        
        print(f"   {i+1}. {resultado['modelo_tipo_completo']} (k={k_info}):")
        print(f"      Score: {resultado['score_gmm']:.4f}")
        print(f"      Silhouette: {resultado['silhouette']:.4f}")
        print(f"      Calinski: {resultado['calinski_harabasz']:.1f}")
        print(f"      Davies-Bouldin: {resultado['davies_bouldin']:.3f}")
    
    # Seleccionar el mejor
    mejor_resultado = todos_resultados_sorted[0]
    
    print(f"\nMEJOR MODELO SELECCIONADO:")
    print(f"   • Modelo: {mejor_resultado['modelo_tipo_completo']}")
    print(f"   • Parámetros: k={mejor_resultado.get('k_param', 'N/A')}")
    print(f"   • Score: {mejor_resultado['score_gmm']:.4f}")
    print(f"   • Silhouette: {mejor_resultado['silhouette']:.4f}")
    
    if mejor_resultado.get('bic') is not None:
        print(f"   • BIC: {mejor_resultado['bic']:.1f}")
        print(f"   • AIC: {mejor_resultado['aic']:.1f}")
    
    if mejor_resultado.get('confianza_promedio') is not None:
        print(f"   • Confianza promedio: {mejor_resultado['confianza_promedio']:.3f}")
    
    print(f"\nDistribución del mejor modelo:")
    distribucion = mejor_resultado['distribucion']
    for cluster_id, count in distribucion.items():
        if cluster_id == -1:
            print(f"   • Ruido: {count:,} elementos")
        else:
            pct = (count / sum([v for k,v in distribucion.items() if k != -1])) * 100
            print(f"   • Cluster {cluster_id}: {count:,} elementos ({pct:.1f}%)")
    
    # Preparar retorno compatible con el código original
    if 'gmm' in mejor_resultado['modelo_tipo_completo']:
        mejor_cov_type = mejor_resultado.get('cov_type', 'full')
        mejor_k = mejor_resultado.get('k_param')
    else:
        mejor_cov_type = mejor_resultado['modelo_tipo_completo']
        mejor_k = mejor_resultado.get('k_param', mejor_resultado.get('eps', 'auto'))
    
    return {
        'mejor_modelo': mejor_resultado,
        'mejor_cov_type': mejor_cov_type,
        'mejor_k': mejor_k,
        'todos_resultados': todos_resultados_sorted[:10],  # Top 10 resultados
        'todos_modelos': resultados_clustering['resultados'],  # TODOS los modelos evaluados
        'data_used': resultados_clustering['data_used'],
        'pca_model': resultados_clustering['pca_model']
    }


# ==========================================
# INSTRUCCIONES DE USO
# ==========================================

if __name__ == "__main__":
    ejemplo_uso_completo()




# ==========================================
# VISUALIZACIÓN
# ==========================================

def visualizar_resultados(analisis_clustering, resultados_originales=None):
    """
    Crea visualizaciones para el análisis multi-modelo de clustering
    """
    
    print(f"\nCREANDO VISUALIZACIONES MULTI-MODELO...")
    
    if analisis_clustering is None:
        print("No hay resultados para visualizar")
        return
    
    mejor_modelo = analisis_clustering['mejor_modelo']
    labels = mejor_modelo['labels']
    data_used = analisis_clustering['data_used']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Resultados {mejor_modelo["tipo_modelo"]} (k={analisis_clustering["mejor_k"]})', 
                 fontsize=16, fontweight='bold')
    
    # 1. Distribución de clusters
    ax1 = axes[0, 0]
    distribucion = mejor_modelo['distribucion']
    
    # Filtrar ruido si existe (DBSCAN)
    if -1 in distribucion:
        ruido = distribucion.pop(-1)
        cluster_ids = list(distribucion.keys())
        counts = list(distribucion.values())
        # Agregar ruido al final
        cluster_ids.append(-1)
        counts.append(ruido)
    else:
        cluster_ids = list(distribucion.keys())
        counts = list(distribucion.values())
    
    percentages = [(c/sum(counts))*100 for c in counts]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_ids)))
    bars = ax1.bar(range(len(cluster_ids)), counts, color=colors, alpha=0.8)
    
    # Configurar etiquetas
    ax1.set_xticks(range(len(cluster_ids)))
    ax1.set_xticklabels([f'C{cid}' if cid != -1 else 'Ruido' for cid in cluster_ids])
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Número de Elementos')
    ax1.set_title('Distribución de Clusters')
    ax1.grid(True, alpha=0.3)
    
    # Añadir porcentajes
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Comparación de modelos
    ax2 = axes[0, 1]
    
    # Tomar top 10 modelos para comparar
    top_modelos = analisis_clustering['todos_resultados'][:10]
    
    model_names = []
    silhouette_scores = []
    
    for i, modelo in enumerate(top_modelos):
        tipo = modelo['tipo_modelo']
        k = modelo.get('k_param', '')
        if isinstance(k, str) and k.startswith('eps_'):
            name = f"{tipo}\n{k}"
        else:
            name = f"{tipo}\nk={k}"
        
        model_names.append(name)
        silhouette_scores.append(modelo['silhouette'])
    
    y_pos = np.arange(len(model_names))
    bars = ax2.barh(y_pos, silhouette_scores, alpha=0.7)
    
    # Colorear la mejor barra
    bars[0].set_color('green')
    bars[0].set_alpha(0.9)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_names, fontsize=8)
    ax2.set_xlabel('Silhouette Score')
    ax2.set_title('Comparación Top 10 Modelos')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Añadir valores
    for i, (bar, score) in enumerate(zip(bars, silhouette_scores)):
        ax2.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=8)
    
    # 3. Clusters en espacio PCA (2D)
    ax3 = axes[0, 2]
    
    if data_used.shape[1] >= 2:
        if data_used.shape[1] > 2:
            # Usar PCA para visualización 2D
            from sklearn.decomposition import PCA
            pca_viz = PCA(n_components=2, random_state=42)
            data_2d = pca_viz.fit_transform(data_used)
            explained_var = pca_viz.explained_variance_ratio_.sum()
        else:
            data_2d = data_used.iloc[:, :2].values
            explained_var = 1.0
        
        # Manejar ruido en DBSCAN
        if -1 in np.unique(labels):
            # Crear máscara para puntos no-ruido
            mask_no_ruido = labels != -1
            scatter = ax3.scatter(data_2d[mask_no_ruido, 0], data_2d[mask_no_ruido, 1], 
                                 c=labels[mask_no_ruido], cmap='Set3', alpha=0.6, s=30)
            # Plotear ruido en gris
            ax3.scatter(data_2d[~mask_no_ruido, 0], data_2d[~mask_no_ruido, 1], 
                       c='gray', alpha=0.3, s=10, label='Ruido')
        else:
            scatter = ax3.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, 
                                 cmap='Set3', alpha=0.6, s=30)
        
        ax3.set_xlabel(f'PC1' if data_used.shape[1] > 2 else data_used.columns[0])
        ax3.set_ylabel(f'PC2' if data_used.shape[1] > 2 else data_used.columns[1])
        ax3.set_title(f'Clusters en Espacio 2D\n({explained_var:.1%} varianza)')
        plt.colorbar(scatter, ax=ax3)
    else:
        ax3.text(0.5, 0.5, 'Datos insuficientes\npara visualización 2D', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Visualización 2D')
    
    # 4. Métricas por tipo de modelo
    ax4 = axes[1, 0]
    
    # Agrupar resultados por tipo de modelo
    modelos_por_tipo = {}
    for resultado in analisis_clustering['todos_resultados']:
        tipo = resultado['tipo_modelo']
        if tipo not in modelos_por_tipo:
            modelos_por_tipo[tipo] = []
        modelos_por_tipo[tipo].append(resultado)
    
    # Calcular promedio de score por tipo
    tipos = []
    scores_promedio = []
    scores_std = []
    
    for tipo, resultados in modelos_por_tipo.items():
        tipos.append(tipo)
        scores = [r['score_gmm'] for r in resultados]
        scores_promedio.append(np.mean(scores))
        scores_std.append(np.std(scores))
    
    # Ordenar por score promedio
    indices = np.argsort(scores_promedio)[::-1]
    tipos = [tipos[i] for i in indices]
    scores_promedio = [scores_promedio[i] for i in indices]
    scores_std = [scores_std[i] for i in indices]
    
    x_pos = np.arange(len(tipos))
    bars = ax4.bar(x_pos, scores_promedio, yerr=scores_std, alpha=0.7, capsize=5)
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(tipos, rotation=45, ha='right')
    ax4.set_ylabel('Score Promedio')
    ax4.set_title('Performance por Tipo de Modelo')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Matriz de métricas
    ax5 = axes[1, 1]
    
    # Crear matriz de métricas para top 5 modelos
    metricas = ['Silhouette', 'Calinski', 'Davies-Bouldin', 'Score']
    modelos = []
    matriz_valores = []
    
    for i, modelo in enumerate(analisis_clustering['todos_resultados'][:5]):
        tipo = modelo['tipo_modelo']
        k = modelo.get('k_param', '')
        modelos.append(f"{tipo} ({k})")
        
        valores = [
            modelo['silhouette'],
            min(modelo['calinski_harabasz'] / 3000, 1),  # Normalizar
            max(0, 1 - modelo['davies_bouldin'] / 3),    # Normalizar e invertir
            modelo['score_gmm']
        ]
        matriz_valores.append(valores)
    
    matriz_valores = np.array(matriz_valores).T
    
    im = ax5.imshow(matriz_valores, cmap='YlOrRd', aspect='auto')
    
    # Configurar ticks
    ax5.set_xticks(np.arange(len(modelos)))
    ax5.set_yticks(np.arange(len(metricas)))
    ax5.set_xticklabels(modelos, rotation=45, ha='right', fontsize=8)
    ax5.set_yticklabels(metricas)
    
    # Añadir valores en las celdas
    for i in range(len(metricas)):
        for j in range(len(modelos)):
            text = ax5.text(j, i, f'{matriz_valores[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    ax5.set_title('Comparación de Métricas (Top 5)')
    plt.colorbar(im, ax=ax5)
    
    # 6. Resumen del mejor modelo
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Crear tabla de resumen
    metricas_resumen = [
        ['Métrica', 'Valor'],
        ['Modelo', f"{mejor_modelo['tipo_modelo']}"],
        ['Parámetros', f"{analisis_clustering['mejor_k']}"],
        ['Silhouette Score', f"{mejor_modelo['silhouette']:.4f}"],
        ['Calinski-Harabasz', f"{mejor_modelo['calinski_harabasz']:.1f}"],
        ['Davies-Bouldin', f"{mejor_modelo['davies_bouldin']:.3f}"],
        ['Score Compuesto', f"{mejor_modelo['score_gmm']:.4f}"],
        ['Cluster más pequeño', f"{mejor_modelo['min_size']:.1%}"],
        ['Cluster más grande', f"{mejor_modelo['max_size']:.1%}"]
    ]
    
    # Agregar métricas específicas según el modelo
    if mejor_modelo.get('bic') is not None:
        metricas_resumen.append(['BIC', f"{mejor_modelo['bic']:.1f}"])
        metricas_resumen.append(['AIC', f"{mejor_modelo['aic']:.1f}"])
    
    if mejor_modelo.get('eps') is not None:
        metricas_resumen.append(['Epsilon', f"{mejor_modelo['eps']:.3f}"])
        metricas_resumen.append(['Clusters', f"{mejor_modelo['n_clusters']}"])
        metricas_resumen.append(['Ruido', f"{mejor_modelo['n_noise']}"])
    
    # Crear tabla
    table = ax6.table(cellText=metricas_resumen[1:], 
                     colLabels=metricas_resumen[0],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Colorear header
    for i in range(len(metricas_resumen[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Resumen del Mejor Modelo', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    print("Visualizaciones multi-modelo creadas exitosamente")

# APLICAR SELECCIÓN 
def aplicar_seleccion(analisis_clustering):
    """
    Aplica el modelo seleccionado
    """
    
    print(f"\nAPLICANDO MODELO SELECCIONADO")
    print("-" * 40)
    
    if analisis_clustering is None:
        print("No hay modelo para aplicar")
        return None
    
    mejor_modelo = analisis_clustering['mejor_modelo']
    
    print(f"Modelo aplicado exitosamente:")
    print(f"   • Tipo: {mejor_modelo['tipo_modelo']}")
    print(f"   • Parámetros: {analisis_clustering['mejor_k']}")
    print(f"   • Score: {mejor_modelo['score_gmm']:.4f}")
    print(f"   • Silhouette: {mejor_modelo['silhouette']:.4f}")
    
    # Manejar casos especiales según el tipo de modelo
    if mejor_modelo.get('confianza_promedio') is not None:
        print(f"   • Confianza promedio: {mejor_modelo['confianza_promedio']:.3f}")
    
    if mejor_modelo.get('asignaciones_ambiguas') is not None and mejor_modelo['asignaciones_ambiguas'] > 0:
        print(f"   • Asignaciones ambiguas: {mejor_modelo['asignaciones_ambiguas']:,} de {len(mejor_modelo['labels']):,}")
    
    if mejor_modelo.get('n_noise') is not None and mejor_modelo['n_noise'] > 0:
        print(f"   • Puntos de ruido: {mejor_modelo['n_noise']:,} ({mejor_modelo['n_noise']/len(mejor_modelo['labels'])*100:.1f}%)")
    
    print(f"\nDistribución final:")
    distribucion = mejor_modelo['distribucion']
    
    # Calcular total excluyendo ruido si existe
    total_sin_ruido = sum([count for cid, count in distribucion.items() if cid != -1])
    
    for cluster_id, count in sorted(distribucion.items()):
        if cluster_id == -1:
            print(f"   • Ruido: {count:,} elementos ({count/len(mejor_modelo['labels'])*100:.1f}%)")
        else:
            pct = (count / total_sin_ruido) * 100
            
            # Clasificar tipo de segmento
            if pct > 40:
                tipo = "PRINCIPAL"
            elif pct > 20:
                tipo = "IMPORTANTE"
            elif pct > 10:
                tipo = "SECUNDARIO"
            elif pct > 5:
                tipo = "MEDIO"
            else:
                tipo = "VIP/PREMIUM"
            
            print(f"   • Cluster {cluster_id}: {count:,} elementos ({pct:.1f}%) - {tipo}")
    
    return {
        'labels': mejor_modelo['labels'],
        'modelo': mejor_modelo['modelo'],
        'probabilidades': mejor_modelo.get('probabilidades'),
        'metodo': mejor_modelo['tipo_modelo'].lower().replace(' ', '_'),
        'k': analisis_clustering['mejor_k'],
        'cov_type': mejor_modelo.get('cov_type'),
        'metricas': {
            'silhouette': mejor_modelo['silhouette'],
            'calinski_harabasz': mejor_modelo['calinski_harabasz'],
            'davies_bouldin': mejor_modelo['davies_bouldin'],
            'bic': mejor_modelo.get('bic'),
            'aic': mejor_modelo.get('aic'),
            'confianza_promedio': mejor_modelo.get('confianza_promedio', 0),
            'score_gmm': mejor_modelo['score_gmm']
        },
        'distribucion': mejor_modelo['distribucion'],
        'cluster_sizes': mejor_modelo['cluster_sizes'],
        'data_used': analisis_clustering['data_used'],
        'pca_model': analisis_clustering.get('pca_model'),
        'parametros_especificos': {
            'eps': mejor_modelo.get('eps'),
            'linkage': mejor_modelo.get('linkage'),
            'n_clusters': mejor_modelo.get('n_clusters'),
            'n_noise': mejor_modelo.get('n_noise')
        }
    }


def aplicar_gmm_final(analisis_clustering):
    """
    Aplica el modelo final seleccionado (compatible con nombre original)
    """
    return aplicar_seleccion(analisis_clustering)

# FUNCIÓN PRINCIPAL PARA EJECUTAR TODO
def ejecutar_gmm_completo(data_normalized, k_min=5, k_max=10):
    """
    Ejecuta todo el proceso de evaluación multi-modelo de clustering
    """
    
    print("EJECUTANDO EVALUACIÓN MULTI-MODELO DE CLUSTERING")
    print("=" * 60)
    
    # Paso 1: Ejecutar todos los modelos con diferentes configuraciones
    resultados = evaluacion_modelos_clustering(data_normalized, (k_min, k_max))
    
    # Paso 2: Analizar resultados y seleccionar mejor
    analisis = analizar_resultados(resultados)
    
    if analisis is None:
        print("No se pudo completar el análisis de clustering")
        return None
    
    # Paso 3: Visualizar resultados
    visualizar_resultados(analisis, resultados['resultados'])
    
    # Paso 4: Aplicar modelo final
    clustering_final = aplicar_gmm_final(analisis)
    
    print(f"\nEVALUACIÓN MULTI-MODELO COMPLETADA EXITOSAMENTE")
    print(f"   Mejor modelo: {clustering_final['metodo']}")
    print(f"   Score final: {clustering_final['metricas']['score_gmm']:.4f}")
    
    return clustering_final


# ==========================================
# FUNCIONES ADICIONALES PARA SELECCIÓN MANUAL
# ==========================================

def listar_modelos_disponibles(analisis_clustering):
    """
    Lista todos los modelos disponibles evaluados con sus métricas
    """
    print("MODELOS DISPONIBLES PARA SELECCIÓN MANUAL")
    print("=" * 70)
    
    if analisis_clustering is None or 'todos_modelos' not in analisis_clustering:
        print("No hay modelos disponibles")
        return
    
    # Crear lista de todos los modelos con índice
    modelos_lista = []
    idx = 0
    
    for tipo_modelo, resultados_tipo in analisis_clustering['todos_modelos'].items():
        for subtipo, resultados_subtipo in resultados_tipo.items():
            for k, resultado in resultados_subtipo.items():
                modelos_lista.append({
                    'idx': idx,
                    'tipo': tipo_modelo,
                    'subtipo': subtipo,
                    'k': k,
                    'resultado': resultado
                })
                idx += 1
    
    # Mostrar tabla de modelos
    print(f"\n{'ID':<4} {'Modelo':<25} {'K/Params':<10} {'Silhouette':<12} {'Score':<10} {'Balance':<15}")
    print("-" * 80)
    
    for modelo in modelos_lista:
        resultado = modelo['resultado']
        tipo = resultado.get('tipo_modelo', modelo['tipo'])
        k_param = modelo['k']
        
        # Calcular balance
        max_cluster = resultado['max_size'] * 100
        balance = "DESBALANCEADO" if max_cluster > 90 else "MODERADO" if max_cluster > 70 else "BALANCEADO"
        
        print(f"{modelo['idx']:<4} {tipo:<25} {str(k_param):<10} "
              f"{resultado['silhouette']:<12.4f} {resultado['score_gmm']:<10.4f} "
              f"{balance:<15}")
    
    return modelos_lista

def aplicar_modelo_manual(analisis_clustering, modelo_id=None, tipo_modelo=None, k=None, subtipo='default'):
    """
    Aplica un modelo específico de los evaluados anteriormente
    
    Args:
        analisis_clustering: Resultado de analizar_resultados()
        modelo_id: ID del modelo según listar_modelos_disponibles()
        tipo_modelo: Tipo de modelo ('gmm', 'k-means++', 'hierarchical_ward', etc.)
        k: Número de clusters o parámetros
        subtipo: Para GMM es el tipo de covarianza ('full', 'tied', 'diag')
    
    Returns:
        dict: Resultado de la aplicación del modelo seleccionado
    """
    
    print(f"\nAPLICANDO MODELO MANUAL")
    print("-" * 40)
    
    if analisis_clustering is None or 'todos_modelos' not in analisis_clustering:
        print("No hay modelos disponibles")
        return None
    
    # Si se proporciona modelo_id, usarlo directamente
    if modelo_id is not None:
        modelos_lista = []
        idx = 0
        
        for tipo_m, resultados_tipo in analisis_clustering['todos_modelos'].items():
            for sub, resultados_subtipo in resultados_tipo.items():
                for k_val, resultado in resultados_subtipo.items():
                    if idx == modelo_id:
                        modelo_seleccionado = resultado
                        print(f"Modelo seleccionado por ID: {modelo_id}")
                        print(f"   • Tipo: {resultado['tipo_modelo']}")
                        print(f"   • Parámetros: {k_val}")
                        break
                    idx += 1
                if idx == modelo_id + 1:
                    break
            if idx == modelo_id + 1:
                break
        
        if idx != modelo_id + 1:
            print(f"ID {modelo_id} no encontrado")
            return None
    
    # Si se proporcionan tipo_modelo y k, buscar el modelo
    elif tipo_modelo is not None and k is not None:
        try:
            if tipo_modelo == 'gmm' and subtipo in analisis_clustering['todos_modelos']['gmm']:
                modelo_seleccionado = analisis_clustering['todos_modelos']['gmm'][subtipo][k]
            else:
                modelo_seleccionado = analisis_clustering['todos_modelos'][tipo_modelo][subtipo][k]
            
            print(f"Modelo seleccionado por tipo y parámetros:")
            print(f"   • Tipo: {tipo_modelo}")
            print(f"   • K/Params: {k}")
            print(f"   • Subtipo: {subtipo}")
        except KeyError:
            print(f"No se encontró el modelo {tipo_modelo} con k={k} y subtipo={subtipo}")
            return None
    else:
        print("Debe proporcionar modelo_id o (tipo_modelo y k)")
        return None
    
    # Mostrar métricas del modelo seleccionado
    print(f"\nMétricas del modelo seleccionado:")
    print(f"   • Silhouette: {modelo_seleccionado['silhouette']:.4f}")
    print(f"   • Calinski-Harabasz: {modelo_seleccionado['calinski_harabasz']:.1f}")
    print(f"   • Davies-Bouldin: {modelo_seleccionado['davies_bouldin']:.3f}")
    print(f"   • Score: {modelo_seleccionado['score_gmm']:.4f}")
    
    if modelo_seleccionado.get('bic') is not None:
        print(f"   • BIC: {modelo_seleccionado['bic']:.1f}")
        print(f"   • AIC: {modelo_seleccionado['aic']:.1f}")
    
    print(f"\nDistribución de clusters:")
    distribucion = modelo_seleccionado['distribucion']
    total_sin_ruido = sum([count for cid, count in distribucion.items() if cid != -1])
    
    for cluster_id, count in sorted(distribucion.items()):
        if cluster_id == -1:
            print(f"   • Ruido: {count:,} elementos ({count/sum(distribucion.values())*100:.1f}%)")
        else:
            pct = (count / total_sin_ruido) * 100
            print(f"   • Cluster {cluster_id}: {count:,} elementos ({pct:.1f}%)")
    
    # Retornar en el mismo formato que aplicar_seleccion
    return {
        'labels': modelo_seleccionado['labels'],
        'modelo': modelo_seleccionado['modelo'],
        'probabilidades': modelo_seleccionado.get('probabilidades'),
        'metodo': modelo_seleccionado['tipo_modelo'].lower().replace(' ', '_'),
        'k': k if k is not None else modelo_seleccionado.get('k_param'),
        'cov_type': modelo_seleccionado.get('cov_type'),
        'metricas': {
            'silhouette': modelo_seleccionado['silhouette'],
            'calinski_harabasz': modelo_seleccionado['calinski_harabasz'],
            'davies_bouldin': modelo_seleccionado['davies_bouldin'],
            'bic': modelo_seleccionado.get('bic'),
            'aic': modelo_seleccionado.get('aic'),
            'confianza_promedio': modelo_seleccionado.get('confianza_promedio', 0),
            'score_gmm': modelo_seleccionado['score_gmm']
        },
        'distribucion': modelo_seleccionado['distribucion'],
        'cluster_sizes': modelo_seleccionado['cluster_sizes'],
        'data_used': analisis_clustering['data_used'],
        'pca_model': analisis_clustering.get('pca_model'),
        'parametros_especificos': {
            'eps': modelo_seleccionado.get('eps'),
            'linkage': modelo_seleccionado.get('linkage'),
            'n_clusters': modelo_seleccionado.get('n_clusters'),
            'n_noise': modelo_seleccionado.get('n_noise')
        }
    }

def visualizar_modelo_especifico(modelo_aplicado, titulo_personalizado=None):
    """
    Visualiza los resultados de un modelo específico aplicado manualmente
    
    Args:
        modelo_aplicado: Resultado de aplicar_modelo_manual()
        titulo_personalizado: Título personalizado para la visualización
    """
    
    if modelo_aplicado is None:
        print("No hay modelo para visualizar")
        return
    
    labels = modelo_aplicado['labels']
    data_used = modelo_aplicado['data_used']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    titulo = titulo_personalizado or f"Resultados {modelo_aplicado['metodo']} (k={modelo_aplicado['k']})"
    fig.suptitle(titulo, fontsize=16, fontweight='bold')
    
    # 1. Distribución de clusters
    ax1 = axes[0, 0]
    distribucion = modelo_aplicado['distribucion']
    
    # Filtrar ruido si existe
    if -1 in distribucion:
        cluster_ids = [cid for cid in distribucion.keys() if cid != -1] + [-1]
        counts = [distribucion[cid] for cid in cluster_ids[:-1]] + [distribucion[-1]]
    else:
        cluster_ids = list(distribucion.keys())
        counts = list(distribucion.values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_ids)))
    bars = ax1.bar(range(len(cluster_ids)), counts, color=colors, alpha=0.8)
    
    ax1.set_xticks(range(len(cluster_ids)))
    ax1.set_xticklabels([f'C{cid}' if cid != -1 else 'Ruido' for cid in cluster_ids])
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Número de Elementos')
    ax1.set_title('Distribución de Clusters')
    ax1.grid(True, alpha=0.3)
    
    # Añadir porcentajes
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = (count/sum(counts))*100
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Visualización 2D con PCA
    ax2 = axes[0, 1]
    
    if data_used.shape[1] >= 2:
        if data_used.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca_viz = PCA(n_components=2, random_state=42)
            data_2d = pca_viz.fit_transform(data_used)
            explained_var = pca_viz.explained_variance_ratio_.sum()
        else:
            data_2d = data_used.iloc[:, :2].values
            explained_var = 1.0
        
        if -1 in np.unique(labels):
            mask_no_ruido = labels != -1
            scatter = ax2.scatter(data_2d[mask_no_ruido, 0], data_2d[mask_no_ruido, 1], 
                                 c=labels[mask_no_ruido], cmap='Set3', alpha=0.6, s=30)
            ax2.scatter(data_2d[~mask_no_ruido, 0], data_2d[~mask_no_ruido, 1], 
                       c='gray', alpha=0.3, s=10, label='Ruido')
        else:
            scatter = ax2.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, 
                                 cmap='Set3', alpha=0.6, s=30)
        
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title(f'Clusters en Espacio 2D\n({explained_var:.1%} varianza)')
        plt.colorbar(scatter, ax=ax2)
    
    # 3. Métricas del modelo
    ax3 = axes[1, 0]
    
    metricas = modelo_aplicado['metricas']
    metricas_nombres = ['Silhouette', 'Score']
    metricas_valores = [metricas['silhouette'], metricas['score_gmm']]
    
    bars = ax3.bar(metricas_nombres, metricas_valores, alpha=0.7, color=['blue', 'green'])
    ax3.set_ylabel('Valor')
    ax3.set_title('Métricas Principales')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, metricas_valores):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 4. Tabla de resumen
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    resumen = [
        ['Métrica', 'Valor'],
        ['Método', modelo_aplicado['metodo']],
        ['Clusters', str(modelo_aplicado['k'])],
        ['Silhouette', f"{metricas['silhouette']:.4f}"],
        ['Calinski-H', f"{metricas['calinski_harabasz']:.1f}"],
        ['Davies-B', f"{metricas['davies_bouldin']:.3f}"],
        ['Score', f"{metricas['score_gmm']:.4f}"]
    ]
    
    if metricas.get('bic') is not None:
        resumen.append(['BIC', f"{metricas['bic']:.1f}"])
    
    table = ax4.table(cellText=resumen[1:], 
                     colLabels=resumen[0],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 0.8])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.show()
    
    print("Visualización creada exitosamente")

# ==========================================
# PASO 4: ÁRBOL DE INTERPRETACIÓN PARA GMM
# ==========================================

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def crear_arbol_interpretacion(results_paso2, gmm_final, max_depth=6):
    """
    Crea un árbol de decisión para interpretar los clusters encontrados
    Usa los datos originales (sin normalizar) para mejor interpretabilidad
    
    Args:
        results_paso2: Resultados del paso 2 (preparación de datos)
        gmm_final: Resultados del clustering GMM del paso 3
        max_depth: Profundidad máxima del árbol
    
    Returns:
        dict: Árbol entrenado y reglas de interpretación
    """
    
    print("=" * 60)
    print("PASO 4: CREAR ÁRBOL DE INTERPRETACIÓN DE CLUSTERS")
    print("=" * 60)
    
    # Obtener datos originales codificados (sin normalizar) para interpretabilidad
    X_original = results_paso2['data_encoded_raw']  # Datos codificados pero sin normalizar
    y_clusters = gmm_final['labels']  # Labels de clusters
    
    print(f"Datos para el árbol:")
    print(f"   • Features: {X_original.shape[1]} variables")
    print(f"   • Muestras: {X_original.shape[0]:,}")
    print(f"   • Clusters a explicar: {len(np.unique(y_clusters))}")
    
    # Entrenar árbol de decisión
    print(f"\nEntrenando árbol de decisión...")
    
    arbol_interpretacion = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=50,    # Evitar splits muy específicos
        min_samples_leaf=25,     # Hojas con suficientes muestras
        random_state=42,
        criterion='gini'
    )
    
    arbol_interpretacion.fit(X_original, y_clusters)
    
    # Evaluar qué tan bien el árbol explica los clusters
    accuracy = arbol_interpretacion.score(X_original, y_clusters)
    
    # Validación cruzada para robustez
    cv_scores = cross_val_score(arbol_interpretacion, X_original, y_clusters, cv=5)
    
    print(f"Árbol entrenado exitosamente")
    print(f"   • Precisión en datos completos: {accuracy:.3f}")
    print(f"   • Validación cruzada: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    if accuracy > 0.85:
        print(f"Excelente: El árbol explica muy bien los clusters")
    elif accuracy > 0.70:
        print(f"Bueno: El árbol explica bien los clusters")
    elif accuracy > 0.50:
        print(f"Regular: El árbol explica parcialmente los clusters")
    else:
        print(f"Bajo: Los clusters son difíciles de explicar con reglas simples")
    
    # Obtener importancia de características
    feature_importance = pd.DataFrame({
        'variable': X_original.columns,
        'importancia': arbol_interpretacion.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    print(f"\nTOP 15 VARIABLES MÁS IMPORTANTES PARA SEGMENTACIÓN:")
    for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
        print(f"   {i:2d}. {row['variable']}: {row['importancia']:.4f}")
    
    # Extraer reglas textuales
    reglas_texto = export_text(
        arbol_interpretacion,
        feature_names=list(X_original.columns),
        max_depth=max_depth
    )
    
    print(f"\nREGLAS DE SEGMENTACIÓN (Primeras líneas):")
    print("-" * 50)
    lineas_reglas = reglas_texto.split('\n')
    for linea in lineas_reglas[:20]:  # Mostrar primeras 20 líneas
        if linea.strip():
            print(f"   {linea}")
    
    if len(lineas_reglas) > 20:
        print(f"   ... (y {len(lineas_reglas) - 20} líneas más)")
    
    return {
        'arbol': arbol_interpretacion,
        'accuracy': accuracy,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance,
        'reglas_texto': reglas_texto,
        'X_usado': X_original,
        'y_clusters': y_clusters
    }

def entrenar_random_forest_interpretacion(results_paso2, gmm_final, n_estimators=100):
    """
    Entrena un Random Forest para obtener importancias más robustas
    """
    
    print(f"\nENTRENANDO RANDOM FOREST PARA IMPORTANCIAS ROBUSTAS...")
    
    X_original = results_paso2['data_encoded_raw']
    y_clusters = gmm_final['labels']
    
    # Random Forest
    rf_interpretacion = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_interpretacion.fit(X_original, y_clusters)
    
    # Evaluación
    rf_accuracy = rf_interpretacion.score(X_original, y_clusters)
    rf_cv_scores = cross_val_score(rf_interpretacion, X_original, y_clusters, cv=5)
    
    print(f"Random Forest entrenado")
    print(f"   • Precisión: {rf_accuracy:.3f}")
    print(f"   • Validación cruzada: {rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}")
    
    # Importancia de características
    rf_feature_importance = pd.DataFrame({
        'variable': X_original.columns,
        'importancia_rf': rf_interpretacion.feature_importances_
    }).sort_values('importancia_rf', ascending=False)
    
    print(f"\nTOP 10 VARIABLES - RANDOM FOREST:")
    for i, (_, row) in enumerate(rf_feature_importance.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['variable']}: {row['importancia_rf']:.4f}")
    
    return {
        'random_forest': rf_interpretacion,
        'rf_accuracy': rf_accuracy,
        'rf_cv_scores': rf_cv_scores,
        'rf_feature_importance': rf_feature_importance
    }

def interpretar_variables_originales(feature_importance, results_paso2):
    """
    Interpreta las variables importantes en términos de las variables originales
    """
    
    print(f"\nINTERPRETACIÓN EN VARIABLES ORIGINALES:")
    print("=" * 50)
    
    encoding_info = results_paso2['encoding_info']
    
    # Agrupar importancias por variable original
    importancia_original = {}
    
    for _, row in feature_importance.head(20).iterrows():
        var_encoded = row['variable']
        importancia = row['importancia']
        
        # Buscar la variable original
        variable_original = None
        
        # Buscar en variables con one-hot encoding
        for var_orig, info in encoding_info.items():
            if info['type'] == 'onehot':
                if var_encoded in info['encoded_columns']:
                    variable_original = var_orig
                    break
            elif info['type'] == 'label':
                if var_encoded in info['encoded_columns']:
                    variable_original = var_orig
                    break
        
        # Si no se encontró en encoding, es una variable numérica original
        if variable_original is None:
            if var_encoded in results_paso2['numerical_columns']:
                variable_original = var_encoded
        
        # Agrupar importancia
        if variable_original:
            if variable_original not in importancia_original:
                importancia_original[variable_original] = 0
            importancia_original[variable_original] += importancia
    
    # Ordenar por importancia total
    importancia_ordenada = sorted(importancia_original.items(), 
                                 key=lambda x: x[1], reverse=True)
    
    print(f"VARIABLES ORIGINALES MÁS IMPORTANTES:")
    for i, (var_original, importancia_total) in enumerate(importancia_ordenada[:15], 1):
        # Determinar tipo de variable
        if var_original in results_paso2['categorical_columns']:
            tipo = "Categórica"
            if var_original in results_paso2['original_data'].columns:
                cardinalidad = results_paso2['original_data'][var_original].nunique()
                extra_info = f"({cardinalidad} categorías)"
            else:
                extra_info = ""
        else:
            tipo = "Numérica"
            extra_info = ""
        
        print(f"   {i:2d}. {var_original} ({tipo}) {extra_info}: {importancia_total:.4f}")
    
    return importancia_ordenada

def visualizar_arbol_interpretacion(arbol_results, rf_results=None):
    """
    Visualiza el árbol de interpretación y las importancias
    """
    
    print(f"\nCREANDO VISUALIZACIONES DEL ÁRBOL...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Interpretación de Clusters con Árboles de Decisión', 
                 fontsize=16, fontweight='bold')
    
    # 1. Visualización del árbol (simplificada)
    ax1 = axes[0, 0]
    plot_tree(arbol_results['arbol'], 
              max_depth=3,  # Mostrar solo primeros 3 niveles
              feature_names=[name[:15] + '...' if len(name) > 15 else name 
                           for name in arbol_results['X_usado'].columns],
              class_names=[f'Cluster {i}' for i in range(len(np.unique(arbol_results['y_clusters'])))],
              filled=True, 
              rounded=True, 
              fontsize=8,
              ax=ax1)
    ax1.set_title('Árbol de Decisión (3 primeros niveles)')
    
    # 2. Importancia de características - Árbol Individual
    ax2 = axes[0, 1]
    top_features = arbol_results['feature_importance'].head(15)
    
    y_pos = np.arange(len(top_features))
    ax2.barh(y_pos, top_features['importancia'], color='lightblue', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([name[:25] + '...' if len(name) > 25 else name 
                        for name in top_features['variable']], fontsize=8)
    ax2.set_xlabel('Importancia')
    ax2.set_title('Top 15 Variables - Árbol Individual')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Comparación de precisión
    ax3 = axes[1, 0]
    
    modelos = ['Árbol Individual']
    precisiones = [arbol_results['accuracy']]
    cv_means = [arbol_results['cv_scores'].mean()]
    cv_stds = [arbol_results['cv_scores'].std()]
    
    if rf_results:
        modelos.append('Random Forest')
        precisiones.append(rf_results['rf_accuracy'])
        cv_means.append(rf_results['rf_cv_scores'].mean())
        cv_stds.append(rf_results['rf_cv_scores'].std())
    
    x = np.arange(len(modelos))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, precisiones, width, label='Precisión Total', 
                   alpha=0.8, color='skyblue')
    bars2 = ax3.bar(x + width/2, cv_means, width, label='Validación Cruzada', 
                   yerr=cv_stds, alpha=0.8, color='lightcoral', capsize=5)
    
    ax3.set_xlabel('Modelos')
    ax3.set_ylabel('Precisión')
    ax3.set_title('Comparación de Precisión')
    ax3.set_xticks(x)
    ax3.set_xticklabels(modelos)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Añadir valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
    
    # 4. Importancia Random Forest (si existe)
    ax4 = axes[1, 1]
    
    if rf_results:
        top_rf_features = rf_results['rf_feature_importance'].head(15)
        
        y_pos = np.arange(len(top_rf_features))
        ax4.barh(y_pos, top_rf_features['importancia_rf'], color='forestgreen', alpha=0.8)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([name[:25] + '...' if len(name) > 25 else name 
                            for name in top_rf_features['variable']], fontsize=8)
        ax4.set_xlabel('Importancia')
        ax4.set_title('Top 15 Variables - Random Forest')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3, axis='x')
    else:
        ax4.text(0.5, 0.5, 'Random Forest\nno ejecutado', 
                ha='center', va='center', transform=ax4.transAxes, 
                fontsize=12, fontweight='bold')
        ax4.set_title('Random Forest - No Disponible')
    
    plt.tight_layout()
    plt.show()
    
    print("Visualizaciones del árbol creadas")

# ==========================================
# PASO 5: ANÁLISIS DE PERFILES DE CLUSTERS
# ==========================================

def analizar_perfiles_clusters(df_original, gmm_final, results_paso2, verify_exclusions=True):
    """
    Analiza el perfil demográfico y de comportamiento de cada cluster
    usando los datos originales para interpretabilidad
    
    Args:
        df_original: DataFrame original sin transformaciones
        gmm_final: Resultados del clustering GMM del paso 3
        results_paso2: Resultados del paso 2 (para info de encoding)
        verify_exclusions: Si True, verifica que se excluyeron IDs correctamente
    
    Returns:
        dict: Perfiles detallados por cluster
    """
    
    print("=" * 60)
    print("PASO 5: ANÁLISIS DE PERFILES DE CLUSTERS")
    print("=" * 60)
    
    # VERIFICACIÓN DE EXCLUSIONES
    if verify_exclusions:
        print("Verificando que IDs fueron excluidos correctamente...")
        
        excluded_cols = results_paso2.get('exclude_columns', [])
        print(f"Columnas que fueron excluidas: {excluded_cols}")
        
        # Verificar si 'documento' está en las variables que se usaron
        vars_usadas = results_paso2['categorical_columns'] + results_paso2['numerical_columns']
        
        documento_cols = [col for col in vars_usadas if 'documento' in col.lower()]
        if documento_cols:
            print(f"ALERTA: Variables 'documento' encontradas: {documento_cols}")
        else:
            print(f"Variables 'documento' correctamente excluidas")
        
        print()
    
    # Añadir clusters al dataframe original
    df_con_clusters = df_original.copy()
    df_con_clusters['cluster'] = gmm_final['labels']
    
    clusters_unicos = sorted(df_con_clusters['cluster'].unique())
    perfiles_clusters = {}
    
    print(f"Analizando {len(clusters_unicos)} clusters...")
    
    for cluster_id in clusters_unicos:
        print(f"\nANÁLISIS DEL CLUSTER {cluster_id}")
        print("=" * 40)
        
        # Filtrar datos del cluster
        datos_cluster = df_con_clusters[df_con_clusters['cluster'] == cluster_id]
        tamaño_cluster = len(datos_cluster)
        porcentaje_cluster = (tamaño_cluster / len(df_con_clusters)) * 100
        
        print(f"Tamaño: {tamaño_cluster:,} registros ({porcentaje_cluster:.1f}% del total)")
        
        perfil = {
            'cluster_id': cluster_id,
            'tamaño': tamaño_cluster,
            'porcentaje': porcentaje_cluster,
            'perfil_categorico': {},
            'perfil_numerico': {},
            'diferencias_significativas': []
        }
        
        # ANÁLISIS DE VARIABLES CATEGÓRICAS
        print(f"\nPERFIL CATEGÓRICO:")
        
        vars_categoricas = results_paso2['categorical_columns']
        
        for var in vars_categoricas[:15]:  # Analizar las primeras 15
            if var in datos_cluster.columns and datos_cluster[var].notna().sum() > 0:
                
                # Moda del cluster
                moda_cluster = datos_cluster[var].mode()
                if len(moda_cluster) > 0:
                    moda_valor = moda_cluster.iloc[0]
                    moda_freq = (datos_cluster[var] == moda_valor).sum()
                    moda_pct = (moda_freq / len(datos_cluster)) * 100
                    
                    # Comparar con población general
                    moda_general = df_original[var].mode()
                    if len(moda_general) > 0:
                        moda_pct_general = (df_original[var] == moda_general.iloc[0]).mean() * 100
                        diferencia = moda_pct - moda_pct_general
                        
                        # Sólo mostrar diferencias significativas
                        if abs(diferencia) > 10:
                            direccion = "↗" if diferencia > 0 else "↘"
                            print(f"   • {var}: {moda_valor} ({moda_pct:.1f}% vs {moda_pct_general:.1f}% general) {direccion}")
                            
                            perfil['perfil_categorico'][var] = {
                                'moda': moda_valor,
                                'porcentaje_cluster': moda_pct,
                                'porcentaje_general': moda_pct_general,
                                'diferencia': diferencia
                            }
                            
                            if abs(diferencia) > 20:
                                perfil['diferencias_significativas'].append({
                                    'variable': var,
                                    'tipo': 'categorica',
                                    'valor': moda_valor,
                                    'diferencia': diferencia
                                })
        
        # ANÁLISIS DE VARIABLES NUMÉRICAS
        print(f"\nPERFIL NUMÉRICO:")
        
        vars_numericas = results_paso2['numerical_columns']
        
        for var in vars_numericas[:15]:  # Analizar las primeras 15
            if var in datos_cluster.columns and datos_cluster[var].notna().sum() > 0:
                
                # Estadísticas del cluster
                media_cluster = datos_cluster[var].mean()
                mediana_cluster = datos_cluster[var].median()
                std_cluster = datos_cluster[var].std()
                
                # Comparar con población general
                media_general = df_original[var].mean()
                mediana_general = df_original[var].median()
                
                # Calcular diferencia porcentual
                if media_general != 0:
                    diff_pct = ((media_cluster - media_general) / abs(media_general)) * 100
                else:
                    diff_pct = 0
                
                # Solo mostrar diferencias significativas
                if abs(diff_pct) > 15:
                    direccion = "↗" if diff_pct > 0 else "↘"
                    print(f"   • {var}: μ={media_cluster:.2f} (vs {media_general:.2f} general, {diff_pct:+.1f}%) {direccion}")
                    
                    perfil['perfil_numerico'][var] = {
                        'media_cluster': media_cluster,
                        'mediana_cluster': mediana_cluster,
                        'std_cluster': std_cluster,
                        'media_general': media_general,
                        'diferencia_porcentual': diff_pct
                    }
                    
                    if abs(diff_pct) > 30:
                        perfil['diferencias_significativas'].append({
                            'variable': var,
                            'tipo': 'numerica',
                            'diferencia_porcentual': diff_pct
                        })
        
        # RESUMEN DEL CLUSTER
        print(f"\nRESUMEN DEL CLUSTER {cluster_id}:")
        if len(perfil['diferencias_significativas']) > 0:
            print(f"   • {len(perfil['diferencias_significativas'])} características distintivas encontradas")
            
            # Mostrar las 3 diferencias más significativas
            diffs_ordenadas = sorted(perfil['diferencias_significativas'], 
                                   key=lambda x: abs(x.get('diferencia', x.get('diferencia_porcentual', 0))), 
                                   reverse=True)
            
            print(f"   • Top 3 características distintivas:")
            for i, diff in enumerate(diffs_ordenadas[:3], 1):
                if diff['tipo'] == 'categorica':
                    print(f"     {i}. {diff['variable']}: {diff['valor']} ({diff['diferencia']:+.1f}% vs general)")
                else:
                    print(f"     {i}. {diff['variable']}: {diff['diferencia_porcentual']:+.1f}% vs general")
        else:
            print(f"   • Cluster con perfil similar al promedio general")
        
        perfiles_clusters[cluster_id] = perfil
    
    return perfiles_clusters, df_con_clusters

def generar_nombres_clusters(perfiles_clusters):
    """
    Genera nombres descriptivos automáticos para cada cluster
    """
    
    print(f"\nGENERANDO NOMBRES DESCRIPTIVOS PARA CLUSTERS...")
    
    nombres_clusters = {}
    
    for cluster_id, perfil in perfiles_clusters.items():
        
        descriptores = []
        
        # Buscar características distintivas más fuertes
        diffs_significativas = perfil['diferencias_significativas']
        
        if diffs_significativas:
            # Ordenar por magnitud de diferencia
            diffs_ordenadas = sorted(diffs_significativas, 
                                   key=lambda x: abs(x.get('diferencia', x.get('diferencia_porcentual', 0))), 
                                   reverse=True)
            
            # Tomar las 2 características más distintivas para el nombre
            for diff in diffs_ordenadas[:2]:
                var_name = diff['variable'].lower()
                
                # Patrones comunes en nombres de variables
                if 'edad' in var_name or 'age' in var_name:
                    if diff.get('diferencia_porcentual', 0) > 20:
                        descriptores.append("Mayor Edad")
                    elif diff.get('diferencia_porcentual', 0) < -20:
                        descriptores.append("Menor Edad")
                
                elif 'ingreso' in var_name or 'income' in var_name or 'salario' in var_name:
                    if diff.get('diferencia_porcentual', 0) > 20:
                        descriptores.append("Altos Ingresos")
                    elif diff.get('diferencia_porcentual', 0) < -20:
                        descriptores.append("Bajos Ingresos")
                
                elif 'producto' in var_name or 'cuenta' in var_name:
                    if diff.get('diferencia_porcentual', 0) > 20:
                        descriptores.append("Multi-Producto")
                    elif diff.get('diferencia_porcentual', 0) < -20:
                        descriptores.append("Mono-Producto")
                
                elif 'antiguedad' in var_name or 'tenure' in var_name:
                    if diff.get('diferencia_porcentual', 0) > 20:
                        descriptores.append("Alta Antigüedad")
                    elif diff.get('diferencia_porcentual', 0) < -20:
                        descriptores.append("Baja Antigüedad")
                
                elif diff['tipo'] == 'categorica':
                    # Para categóricas, usar el valor más común si es descriptivo
                    valor = str(diff['valor']).title()
                    if len(valor) < 15:  # Solo si el nombre no es muy largo
                        descriptores.append(valor)
        
        # Crear nombre final
        if descriptores:
            nombre = " + ".join(descriptores[:2])  # Máximo 2 descriptores
        else:
            nombre = f"Grupo {cluster_id + 1}"  # Nombre genérico
        
        # Añadir tamaño como contexto
        tamaño_pct = perfil['porcentaje']
        if tamaño_pct > 40:
            contexto = "Mayoritario"
        elif tamaño_pct > 25:
            contexto = "Principal"
        elif tamaño_pct > 15:
            contexto = "Secundario"
        else:
            contexto = "Nicho"
        
        nombre_final = f"{nombre} ({contexto})"
        nombres_clusters[cluster_id] = nombre_final
        
        print(f"   • Cluster {cluster_id}: '{nombre_final}'")
    
    return nombres_clusters

def visualizar_perfiles_clusters(perfiles_clusters, df_con_clusters):
    """
    Crea visualizaciones de los perfiles de clusters
    """
    
    print(f"\nCREANDO VISUALIZACIONES DE PERFILES...")
    
    num_clusters = len(perfiles_clusters)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Perfiles de Clusters Descubiertos', fontsize=16, fontweight='bold')
    
    # 1. Distribución de tamaños de clusters
    ax1 = axes[0, 0]
    cluster_ids = list(perfiles_clusters.keys())
    tamaños = [perfiles_clusters[cid]['tamaño'] for cid in cluster_ids]
    porcentajes = [perfiles_clusters[cid]['porcentaje'] for cid in cluster_ids]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_ids)))
    bars = ax1.bar(cluster_ids, tamaños, color=colors, alpha=0.8)
    
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Número de Registros')
    ax1.set_title('Distribución de Tamaños por Cluster')
    ax1.grid(True, alpha=0.3)
    
    # Añadir porcentajes en las barras
    for bar, pct in zip(bars, porcentajes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Número de características distintivas por cluster
    ax2 = axes[0, 1]
    num_diffs = [len(perfiles_clusters[cid]['diferencias_significativas']) for cid in cluster_ids]
    
    bars2 = ax2.bar(cluster_ids, num_diffs, color='lightcoral', alpha=0.8)
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Número de Características Distintivas')
    ax2.set_title('Características Distintivas por Cluster')
    ax2.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for bar, num in zip(bars2, num_diffs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{num}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Gráfico de pastel de distribución
    ax3 = axes[1, 0]
    ax3.pie(porcentajes, labels=[f'C{cid}\n{pct:.1f}%' for cid, pct in zip(cluster_ids, porcentajes)],
           colors=colors, autopct='', startangle=90)
    ax3.set_title('Distribución Porcentual de Clusters')

    
    plt.tight_layout()
    plt.show()
    
    print("Visualizaciones de perfiles creadas")

# ==========================================
# PASO 6: RESULTADOS FINALES Y EXPORTACIÓN
# ==========================================

from datetime import datetime
import os
import pandas as pd # Se añade la importación de pandas, necesaria para exportar_resultados

def generar_resumen_ejecutivo(perfiles_clusters, nombres_clusters, arbol_results, modelo_resultado):
    """
    Genera un resumen ejecutivo completo de la segmentación
    """
    
    print("=" * 70)
    print("RESUMEN EJECUTIVO - SEGMENTACIÓN NO SUPERVISADA CON ÁRBOLES")
    print("=" * 70)
    
    # Información general
    num_clusters = len(perfiles_clusters)
    total_registros = sum([perfil['tamaño'] for perfil in perfiles_clusters.values()])
    accuracy_interpretacion = arbol_results['accuracy']
    
    print(f"\nINFORMACIÓN GENERAL:")
    print(f"   • Total de registros analizados: {total_registros:,}")
    print(f"   • Número de segmentos descubiertos: {num_clusters}")
    print(f"   • Método de clustering: {modelo_resultado['metodo'].upper()}")
    print(f"   • Interpretabilidad del árbol: {accuracy_interpretacion:.1%}")
    
    # Métricas de calidad del clustering
    metricas = modelo_resultado['metricas']
    print(f"\nCALIDAD DE LA SEGMENTACIÓN:")
    print(f"   • Silhouette Score: {metricas['silhouette']:.3f}")
    
    # === AJUSTE PARA MANEJAR BIC AUSENTE O NONE ===
    if 'bic' in metricas and metricas['bic'] is not None:
        print(f"   • BIC Score: {metricas['bic']:.1f}")
    else:
        print(f"   • BIC Score: No aplica o no disponible para este método")
    # ===============================================
    
    print(f"   • Confianza promedio GMM: {metricas['confianza_promedio']:.3f}")
    
    # Evaluar calidad
    if metricas['silhouette'] > 0.5:
        calidad_silhouette = "Excelente"
    elif metricas['silhouette'] > 0.3:
        calidad_silhouette = "Buena"
    elif metricas['silhouette'] > 0.1:
        calidad_silhouette = "Aceptable"
    else:
        calidad_silhouette = "Baja"
    
    print(f"   • Evaluación general: {calidad_silhouette}")
    
    # Resumen por cluster
    print(f"\nRESUMEN POR SEGMENTO:")
    for cluster_id, perfil in perfiles_clusters.items():
        nombre = nombres_clusters.get(cluster_id, f"Cluster {cluster_id}")
        print(f"\n{nombre}")
        print(f"      • Tamaño: {perfil['tamaño']:,} registros ({perfil['porcentaje']:.1f}%)")
        print(f"      • Características distintivas: {len(perfil['diferencias_significativas'])}")
        
        # Mostrar las 2 características más importantes
        if perfil['diferencias_significativas']:
            diffs_ordenadas = sorted(perfil['diferencias_significativas'], 
                                   key=lambda x: abs(x.get('diferencia', x.get('diferencia_porcentual', 0))), 
                                   reverse=True)[:2]
            
            print(f"      • Características clave:")
            for diff in diffs_ordenadas:
                if diff['tipo'] == 'categorica':
                    print(f"        - {diff['variable']}: {diff['valor']} ({diff['diferencia']:+.1f}% vs promedio)")
                else:
                    print(f"        - {diff['variable']}: {diff['diferencia_porcentual']:+.1f}% vs promedio")
    
    # Variables más importantes para la segmentación
    print(f"\nVARIABLES MÁS IMPORTANTES PARA SEGMENTACIÓN:")
    top_vars = arbol_results['feature_importance'].head(10)
    for i, (_, row) in enumerate(top_vars.iterrows(), 1):
        print(f"   {i:2d}. {row['variable']}: {row['importancia']:.4f}")
    
    print(f"\nSEGMENTACIÓN COMPLETADA EXITOSAMENTE")
    
    return {
        'num_clusters': num_clusters,
        'total_registros': total_registros,
        'accuracy_interpretacion': accuracy_interpretacion,
        'metricas_clustering': metricas,
        'calidad_general': calidad_silhouette
    }

def generar_insights_negocio(perfiles_clusters, nombres_clusters):
    """
    Genera insights y recomendaciones de negocio
    """
    
    print(f"\n" + "=" * 70)
    print("INSIGHTS Y RECOMENDACIONES DE NEGOCIO")
    print("=" * 70)
    
    insights = []
    
    for cluster_id, perfil in perfiles_clusters.items():
        nombre = nombres_clusters.get(cluster_id, f"Cluster {cluster_id}")
        tamaño_pct = perfil['porcentaje']
        
        print(f"\n{nombre.upper()} ({tamaño_pct:.1f}% de clientes)")
        print("-" * 50)
        
        # Análisis del tamaño del segmento
        if tamaño_pct > 40:
            estrategia_tamaño = "SEGMENTO PRINCIPAL - Prioridad máxima"
            recomendacion_tamaño = "Estrategia de retención masiva y cross-selling"
        elif tamaño_pct > 25:
            estrategia_tamaño = "SEGMENTO IMPORTANTE - Alta prioridad"
            recomendacion_tamaño = "Desarrollo de productos específicos y campañas dedicadas"
        elif tamaño_pct > 15:
            estrategia_tamaño = "SEGMENTO SECUNDARIO - Prioridad media"
            recomendacion_tamaño = "Estrategias de nicho y personalización"
        else:
            estrategia_tamaño = "SEGMENTO ESPECIALIZADO - Atención VIP"
            recomendacion_tamaño = "Productos exclusivos y atención personalizada"
        
        print(f"Estrategia por tamaño: {estrategia_tamaño}")
        print(f"Recomendación: {recomendacion_tamaño}")
        
        # Recomendaciones basadas en características distintivas
        print(f"\nRecomendaciones específicas:")
        
        caracteristicas_clave = perfil['diferencias_significativas']
        if caracteristicas_clave:
            for diff in caracteristicas_clave[:3]:  # Top 3 características
                var_name = diff['variable'].lower()
                
                # Recomendaciones específicas por tipo de variable
                if 'edad' in var_name:
                    if diff.get('diferencia_porcentual', 0) > 20:
                        print(f"   • Grupo de mayor edad → Productos de ahorro, seguros, planificación jubilación")
                    else:
                        print(f"   • Grupo más joven → Productos digitales, créditos, inversiones dinámicas")
                
                elif 'ingreso' in var_name:
                    if diff.get('diferencia_porcentual', 0) > 20:
                        print(f"   • Altos ingresos → Productos premium, inversiones, banca privada")
                    else:
                        print(f"   • Ingresos moderados → Productos básicos, facilidades de pago")
                
                elif 'producto' in var_name:
                    if diff.get('diferencia_porcentual', 0) > 20:
                        print(f"   • Multi-producto → Programas de lealtad, productos complementarios")
                    else:
                        print(f"   • Mono-producto → Estrategias de cross-selling agresivas")
                
                elif 'antiguedad' in var_name:
                    if diff.get('diferencia_porcentual', 0) > 20:
                        print(f"   • Alta antigüedad → Programas VIP, beneficios por lealtad")
                    else:
                        print(f"   • Nuevos clientes → Programas de bienvenida, educación financiera")
                
                else:
                    # Recomendación genérica
                    print(f"   • Variable {diff['variable']} distintiva → Personalizar según esta característica")
        else:
            print(f"   • Perfil estándar → Aplicar estrategias generales de la organización")
        
        # Guardar insight estructurado
        insight = {
            'cluster_id': cluster_id,
            'nombre': nombre,
            'tamaño_pct': tamaño_pct,
            'estrategia': estrategia_tamaño,
            'recomendacion': recomendacion_tamaño,
            'caracteristicas_clave': caracteristicas_clave[:3]
        }
        insights.append(insight)
    
    print(f"\n" + "=" * 70)
    print("PRÓXIMOS PASOS RECOMENDADOS:")
    print("=" * 70)
    
    pasos = [
        "1. Validar segmentos con equipos de negocio",
        "2. Desarrollar estrategias de comunicación por segmento",
        "3. Crear campañas de marketing personalizadas",
        "4. Ajustar productos y servicios por segmento",
        "5. Implementar métricas de seguimiento por segmento",
        "6. Establecer objetivos específicos por cluster",
        "7. Programar re-evaluación de segmentos cada 6 meses"
    ]
    
    for paso in pasos:
        print(f"{paso}")
    
    return insights

def exportar_resultados(df_con_clusters, perfiles_clusters, nombres_clusters, arbol_results, output_prefix='segmentacion_gmm'):
    """
    Exporta todos los resultados a archivos
    """
    
    print(f"\nEXPORTANDO RESULTADOS...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 1. Dataset con clusters asignados
        archivo_datos = f"{output_prefix}_datos_con_clusters_{timestamp}.csv"
        df_con_clusters.to_csv(archivo_datos, index=False)
        print(f"Datos con clusters: {archivo_datos}")
        
        # 2. Resumen de perfiles
        perfiles_resumen = []
        for cluster_id, perfil in perfiles_clusters.items():
            resumen = {
                'cluster_id': cluster_id,
                'nombre_cluster': nombres_clusters.get(cluster_id, f"Cluster {cluster_id}"),
                'tamaño': perfil['tamaño'],
                'porcentaje': perfil['porcentaje'],
                'num_caracteristicas_distintivas': len(perfil['diferencias_significativas'])
            }
            
            # Añadir top 3 características distintivas
            diffs_ordenadas = sorted(perfil['diferencias_significativas'], 
                                   key=lambda x: abs(x.get('diferencia', x.get('diferencia_porcentual', 0))), 
                                   reverse=True)[:3]
            
            for i, diff in enumerate(diffs_ordenadas, 1):
                if diff['tipo'] == 'categorica':
                    resumen[f'caracteristica_{i}'] = f"{diff['variable']}: {diff['valor']} ({diff['diferencia']:+.1f}%)"
                else:
                    resumen[f'caracteristica_{i}'] = f"{diff['variable']}: {diff['diferencia_porcentual']:+.1f}%"
            
            perfiles_resumen.append(resumen)
        
        archivo_perfiles = f"{output_prefix}_perfiles_clusters_{timestamp}.csv"
        pd.DataFrame(perfiles_resumen).to_csv(archivo_perfiles, index=False)
        print(f"Perfiles de clusters: {archivo_perfiles}")
        
        # 3. Importancia de variables
        archivo_importancia = f"{output_prefix}_importancia_variables_{timestamp}.csv"
        # Asegúrate de que 'feature_importance' sea un DataFrame
        if not isinstance(arbol_results['feature_importance'], pd.DataFrame):
            # Si no es un DataFrame, intenta convertirlo. Esto es una suposición,
            # lo ideal es que ya venga como DataFrame de la función previa.
            try:
                arbol_results['feature_importance'] = pd.DataFrame(arbol_results['feature_importance'])
            except:
                print("Advertencia: arbol_results['feature_importance'] no es un DataFrame y no se pudo convertir.")
                # Crear un DataFrame vacío para evitar errores
                arbol_results['feature_importance'] = pd.DataFrame(columns=['variable', 'importancia'])
        
        arbol_results['feature_importance'].to_csv(archivo_importancia, index=False)
        print(f"Importancia de variables: {archivo_importancia}")
        
        # 4. Reglas del árbol
        archivo_reglas = f"{output_prefix}_reglas_segmentacion_{timestamp}.txt"
        with open(archivo_reglas, 'w', encoding='utf-8') as f:
            f.write("REGLAS DE SEGMENTACIÓN - ÁRBOL DE DECISIÓN\n")
            f.write("=" * 50 + "\n\n")
            f.write(arbol_results['reglas_texto'])
        print(f"Reglas de segmentación: {archivo_reglas}")
        
        # 5. Estadísticas por cluster
        stats_clusters = []
        for cluster_id, perfil in perfiles_clusters.items():
            # Variables categóricas más importantes
            for var, info in perfil['perfil_categorico'].items():
                stats_clusters.append({
                    'cluster_id': cluster_id,
                    'tipo_variable': 'categorica',
                    'variable': var,
                    'valor': info['moda'],
                    'porcentaje_cluster': info['porcentaje_cluster'],
                    'porcentaje_general': info['porcentaje_general'],
                    'diferencia': info['diferencia']
                })
            
            # Variables numéricas más importantes
            for var, info in perfil['perfil_numerico'].items():
                stats_clusters.append({
                    'cluster_id': cluster_id,
                    'tipo_variable': 'numerica',
                    'variable': var,
                    'valor': info['media_cluster'],
                    'media_general': info['media_general'],
                    'diferencia_porcentual': info['diferencia_porcentual']
                })
        
        archivo_stats = f"{output_prefix}_estadisticas_detalladas_{timestamp}.csv"
        pd.DataFrame(stats_clusters).to_csv(archivo_stats, index=False)
        print(f"Estadísticas detalladas: {archivo_stats}")
        
        print(f"\nTODOS LOS ARCHIVOS EXPORTADOS EXITOSAMENTE")
        print(f"Prefijo: {output_prefix}_{timestamp}")
        
        return {
            'archivo_datos': archivo_datos,
            'archivo_perfiles': archivo_perfiles,
            'archivo_importancia': archivo_importancia,
            'archivo_reglas': archivo_reglas,
            'archivo_stats': archivo_stats
        }
        
    except Exception as e:
        print(f"Error en exportación: {str(e)}")
        print(f"Verifica permisos de escritura en el directorio")
        return None

def crear_reporte_final(df_con_clusters, perfiles_clusters, nombres_clusters, arbol_results, modelo_resultado):
    """
    Crea un reporte final comprehensivo en formato texto
    """
    
    print(f"\nGENERANDO REPORTE FINAL...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    reporte = []
    reporte.append("=" * 80)
    reporte.append("REPORTE DE SEGMENTACIÓN NO SUPERVISADA CON ÁRBOLES DE DECISIÓN")
    reporte.append("=" * 80)
    reporte.append(f"Fecha de generación: {timestamp}")
    reporte.append(f"Total de registros analizados: {len(df_con_clusters):,}")
    reporte.append(f"Número de variables originales: {len(df_con_clusters.columns) - 1}")  # -1 por la columna cluster
    reporte.append(f"Método de clustering: {modelo_resultado['metodo'].upper()}")
    reporte.append(f"Número de clusters: {modelo_resultado['k']}")
    reporte.append("")
    
    # Métricas de calidad
    reporte.append("MÉTRICAS DE CALIDAD DEL CLUSTERING:")
    reporte.append("-" * 40)
    metricas = modelo_resultado['metricas']
    reporte.append(f"• Silhouette Score: {metricas['silhouette']:.4f}")
    
    # === AJUSTE PARA MANEJAR BIC AUSENTE O NONE EN EL REPORTE ===
    if 'bic' in metricas and metricas['bic'] is not None:
        reporte.append(f"• BIC Score: {metricas['bic']:.2f}")
    else:
        reporte.append(f"• BIC Score: No aplica o no disponible para este método")
    # =============================================================
    
    reporte.append(f"• Confianza promedio GMM: {metricas['confianza_promedio']:.4f}")
    reporte.append(f"• Interpretabilidad del árbol: {arbol_results['accuracy']:.1%}")
    reporte.append("")
    
    # Variables más importantes
    reporte.append("VARIABLES MÁS IMPORTANTES PARA SEGMENTACIÓN:")
    reporte.append("-" * 50)
    # Asegúrate de que 'feature_importance' sea un DataFrame
    if not isinstance(arbol_results['feature_importance'], pd.DataFrame):
        try:
            arbol_results['feature_importance'] = pd.DataFrame(arbol_results['feature_importance'])
        except:
            arbol_results['feature_importance'] = pd.DataFrame(columns=['variable', 'importancia'])
            
    for i, (_, row) in enumerate(arbol_results['feature_importance'].head(15).iterrows(), 1):
        reporte.append(f"{i:2d}. {row['variable']}: {row['importancia']:.4f}")
    reporte.append("")
    
    # Perfil detallado por cluster
    reporte.append("PERFILES DETALLADOS POR CLUSTER:")
    reporte.append("=" * 50)
    
    for cluster_id, perfil in perfiles_clusters.items():
        nombre = nombres_clusters.get(cluster_id, f"Cluster {cluster_id}")
        
        reporte.append(f"\nCLUSTER {cluster_id}: {nombre}")
        reporte.append("-" * 60)
        reporte.append(f"Tamaño: {perfil['tamaño']:,} registros ({perfil['porcentaje']:.1f}% del total)")
        reporte.append(f"Características distintivas: {len(perfil['diferencias_significativas'])}")
        
        if perfil['perfil_categorico']:
            reporte.append(f"\nCaracterísticas Categóricas Distintivas:")
            for var, info in list(perfil['perfil_categorico'].items())[:5]:
                reporte.append(f"  • {var}: {info['moda']} ({info['porcentaje_cluster']:.1f}% vs {info['porcentaje_general']:.1f}% general)")
        
        if perfil['perfil_numerico']:
            reporte.append(f"\nCaracterísticas Numéricas Distintivas:")
            for var, info in list(perfil['perfil_numerico'].items())[:5]:
                reporte.append(f"  • {var}: {info['media_cluster']:.2f} ({info['diferencia_porcentual']:+.1f}% vs general)")
        
        # Recomendaciones
        reporte.append(f"\nRecomendaciones de Negocio:")
        if perfil['porcentaje'] > 30:
            reporte.append("  • SEGMENTO PRINCIPAL: Estrategia de retención y cross-selling masivo")
        elif perfil['porcentaje'] > 20:
            reporte.append("  • SEGMENTO IMPORTANTE: Desarrollo de productos específicos")
        elif perfil['porcentaje'] > 10:
            reporte.append("  • SEGMENTO SECUNDARIO: Estrategias de nicho")
        else:
            reporte.append("  • SEGMENTO ESPECIALIZADO: Atención VIP y productos exclusivos")
        
        reporte.append("")
    
    # Próximos pasos
    reporte.append("PRÓXIMOS PASOS RECOMENDADOS:")
    reporte.append("-" * 40)
    pasos = [
        "1. Validar segmentos con equipos de negocio",
        "2. Desarrollar estrategias de comunicación específicas",
        "3. Crear campañas de marketing personalizadas",
        "4. Ajustar productos y servicios por segmento",
        "5. Implementar métricas de seguimiento",
        "6. Re-evaluar segmentación cada 6 meses"
    ]
    
    for paso in pasos:
        reporte.append(f"  {paso}")
    
    reporte.append("")
    reporte.append("=" * 80)
    reporte.append("FIN DEL REPORTE")
    reporte.append("=" * 80)
    
    # Guardar reporte
    archivo_reporte = f"reporte_segmentacion_gmm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    try:
        with open(archivo_reporte, 'w', encoding='utf-8') as f:
            f.write('\n'.join(reporte))
        print(f"Reporte guardado: {archivo_reporte}")
    except Exception as e:
        print(f"No se pudo guardar el reporte: {str(e)}")
        print(f"Contenido del reporte:")
        for linea in reporte:
            print(f"   {linea}")
    
    return reporte

def validar_segmentacion_completa(df_con_clusters, modelo_resultado):
    """
    Validación final de que la segmentación se completó correctamente
    """
    
    print(f"\nVALIDACIÓN FINAL DE LA SEGMENTACIÓN:")
    print("-" * 50)
    
    validaciones = []
    
    # 1. Verificar que todos los registros tienen cluster asignado
    registros_sin_cluster = df_con_clusters['cluster'].isnull().sum()
    if registros_sin_cluster == 0:
        validaciones.append("✅ Todos los registros tienen cluster asignado")
    else:
        validaciones.append(f"❌ {registros_sin_cluster} registros sin cluster asignado")
    
    # 2. Verificar distribución de clusters
    distribucion = df_con_clusters['cluster'].value_counts()
    
    # Manejar caso de clusters vacíos si la función de clustering los permite
    if len(distribucion) < modelo_resultado['k']:
        validaciones.append(f"⚠️ Se esperaban {modelo_resultado['k']} clusters, pero se encontraron {len(distribucion)}. Posibles clusters vacíos.")
        
    clusters_pequenos = (distribucion < len(df_con_clusters) * 0.01).sum()  # Menos del 1%
    
    if clusters_pequenos == 0:
        validaciones.append("✅ Todos los clusters tienen tamaño razonable (>1%)")
    else:
        validaciones.append(f"⚠️ {clusters_pequenos} clusters muy pequeños (<1%)")
    
    # 3. Verificar calidad de clustering
    silhouette = modelo_resultado['metricas']['silhouette']
    if silhouette > 0.3:
        validaciones.append(f"✅ Calidad de clustering aceptable (Silhouette: {silhouette:.3f})")
    else:
        validaciones.append(f"⚠️ Calidad de clustering baja (Silhouette: {silhouette:.3f})")
    
    # 4. Verificar balanceo de clusters
    if len(distribucion) > 1: # Solo calcular si hay más de un cluster
        std_sizes = distribucion.std()
        mean_size = distribucion.mean()
        # Evitar división por cero si mean_size es cero (nunca debería pasar si hay registros)
        balance_ratio = std_sizes / mean_size if mean_size != 0 else float('inf')
        
        if balance_ratio < 1.0:
            validaciones.append(f"✅ Clusters bien balanceados (ratio: {balance_ratio:.2f})")
        else:
            validaciones.append(f"⚠️ Clusters desbalanceados (ratio: {balance_ratio:.2f})")
    else:
        validaciones.append("N/A Balanceo de clusters (solo hay un cluster o ninguno)")

    # Mostrar resultados
    for validacion in validaciones:
        print(f"   {validacion}")
    
    # Resultado final
    problemas = sum(1 for v in validaciones if v.startswith('❌'))
    advertencias = sum(1 for v in validaciones if v.startswith('⚠️'))
    
    print(f"\nRESULTADO DE VALIDACIÓN:")
    if problemas == 0 and advertencias == 0:
        print("SEGMENTACIÓN COMPLETAMENTE EXITOSA")
        resultado = "EXITOSA"
    elif problemas == 0:
        print("SEGMENTACIÓN EXITOSA CON ADVERTENCIAS MENORES")
        resultado = "EXITOSA_CON_ADVERTENCIAS"
    else:
        print("SEGMENTACIÓN CON PROBLEMAS - REVISAR")
        resultado = "CON_PROBLEMAS"
    
    return {
        'resultado': resultado,
        'problemas': problemas,
        'advertencias': advertencias,
        'validaciones': validaciones
    }

# FUNCIÓN PRINCIPAL PARA EJECUTAR TODO EL PASO 6
def ejecutar_paso_final_completo(df_original, modelo_resultado, results_paso2, arbol_results, perfiles_clusters, nombres_clusters):
    """
    Ejecuta todas las funciones del paso final
    """
    
    print("EJECUTANDO PASO FINAL COMPLETO...")
    
    # 1. Resumen ejecutivo
    resumen = generar_resumen_ejecutivo(perfiles_clusters, nombres_clusters, arbol_results, modelo_resultado)
    
    # 2. Insights de negocio
    insights = generar_insights_negocio(perfiles_clusters, nombres_clusters)
    
    # 3. Exportar resultados
    df_con_clusters = df_original.copy()
    
    # Asegúrate de que 'labels' exista en modelo_resultado para la asignación de clusters
    if 'labels' in modelo_resultado:
        df_con_clusters['cluster'] = modelo_resultado['labels']
    else:
        print("Advertencia: modelo_resultado no contiene 'labels'. No se asignará la columna 'cluster' al DataFrame exportado.")
        df_con_clusters['cluster'] = None # Asignar None o un valor por defecto
        
    archivos = exportar_resultados(df_con_clusters, perfiles_clusters, nombres_clusters, arbol_results)
    
    # 4. Crear reporte final
    reporte = crear_reporte_final(df_con_clusters, perfiles_clusters, nombres_clusters, arbol_results, modelo_resultado)
    
    # 5. Validación final
    validacion = validar_segmentacion_completa(df_con_clusters, modelo_resultado)
    
    return {
        'resumen_ejecutivo': resumen,
        'insights_negocio': insights,
        'archivos_exportados': archivos,
        'reporte_final': reporte,
        'validacion_final': validacion,
        'df_final': df_con_clusters
    }



