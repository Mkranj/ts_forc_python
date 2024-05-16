def create_metrics_df(original_df, forecast_df):
    metrics_df = (forecast_df.set_index("ds")[['yhat']].
    join(original_df.set_index("ds").y)
    )

    metrics_df.dropna(inplace = True)
    return(metrics_df)