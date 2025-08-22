def compute_snow_from_tb(df, tb19v_col="brightness_temp_19v", tb37v_col="brightness_temp_37v", threshold=10):
    print("❄️ Starting physical snow estimation...")

    df = df.copy()
    tb19v = df[tb19v_col]
    tb37v = df[tb37v_col]

    df["TB_diff_snow"] = tb19v - tb37v
    df["is_snow_physical"] = (df["TB_diff_snow"] > threshold).astype(int)

    print(f"📊 TB19V - TB37V threshold: {threshold} K")
    print(f"✅ Snow pixels detected: {df['is_snow_physical'].sum()} out of {len(df)}")
    print(f"📈 Mean diff TB19V - TB37V: {df['TB_diff_snow'].mean():.2f} K")

    return df


