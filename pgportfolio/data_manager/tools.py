import pandas as pd

def write_panel_to_dbase(panel: pd.Panel):
    from pathlib import Path
    import logging
    logger = logging.getLogger(f"{__name__}")
    logger.propagate= False

    path_name = f"misc/logs"
    Path(path_name).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(filename=f"{path_name}/{__name__}.log")
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.setLevel(logging.WARNING)
    logger.addHandler(file_handler)

    for item in panel.items:
        df = panel[item].T
        def write_df(df):
            try:
                date = df.index.date[0].strftime("%Y%m%d")
                path_name = f"data_base/data/{date[:4]}/{date}"
                Path(path_name).mkdir(parents=True, exist_ok=True)
                df.to_parquet(fname=f"{path_name}/{item}.parquet.gzip", compression="gzip", engine="fastparquet")
                return 1
            except:
                return 0
        s = df.assign(date=df.index.date).groupby("date").apply(lambda df: write_df(df.iloc[:, :-1]))
        if not (s==1).all():
            logger.warning(f"Writing not successful at dates, {set(s.index[s==0])}")
