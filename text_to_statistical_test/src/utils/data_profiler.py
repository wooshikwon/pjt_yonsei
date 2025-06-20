import pandas as pd
import io

def profile_dataframe(df: pd.DataFrame) -> str:
    """
    Pandas 데이터프레임을 분석하여 풍부한 메타데이터 요약을 Markdown 형식으로 생성합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임.

    Returns:
        str: 데이터프레임의 요약 정보가 담긴 Markdown 형식의 문자열.
    """
    buffer = io.StringIO()
    
    buffer.write("### Data Summary\n")
    buffer.write(f"- **Shape**: {df.shape[0]} rows, {df.shape[1]} columns\n")
    buffer.write("\n")
    
    buffer.write("### Column Details\n")
    for col in df.columns:
        col_series = df[col]
        
        # 기본 정보
        buffer.write(f"- **{col}** (`{col_series.dtype}`)\n")
        
        # 데이터 종류 식별
        if pd.api.types.is_numeric_dtype(col_series):
            data_type = "Numeric"
        else:
            data_type = "Categorical"
        buffer.write(f"  - **Type**: {data_type}\n")
        
        # 결측치 정보
        missing_count = col_series.isnull().sum()
        if missing_count > 0:
            missing_pct = (missing_count / len(df)) * 100
            buffer.write(f"  - **Missing**: {missing_count} ({missing_pct:.1f}%)\n")
        else:
            buffer.write(f"  - **Missing**: 0\n")
            
        # 종류별 상세 정보
        if data_type == "Numeric":
            buffer.write(f"  - **Mean**: {col_series.mean():.2f}\n")
            buffer.write(f"  - **Std Dev**: {col_series.std():.2f}\n")
            buffer.write(f"  - **Min | Max**: {col_series.min()} | {col_series.max()}\n")
        else: # Categorical
            unique_vals = col_series.unique()
            num_unique = len(unique_vals)
            buffer.write(f"  - **Unique Values**: {num_unique}\n")
            # 고유값이 10개 이하일 경우만 샘플 표시
            if num_unique <= 10:
                display_vals = [str(v) for v in unique_vals[:5]]
                if num_unique > 5:
                    display_vals.append("...")
                buffer.write(f"  - **Samples**: {', '.join(display_vals)}\n")

        buffer.write("\n")
        
    return buffer.getvalue() 