import pandas as pd


def date_features(input_df, datetime_column='tms_gmt'):
    """
    Given a datetime column, extracts useful date information
    (minute, hour, dow...)
    """
    df = input_df.copy()

    return (df.set_index(time_column)
              .assign(minute=lambda df: df.index.minute,
                      hour=lambda df: df.index.hour,
                      day=lambda df: df.index.day,
                      dow=lambda df: df.index.dayofweek,
                      month=lambda df: df.index.month,
                      week=lambda df: df.index.week,
                      woy=lambda df: df.index.weekofyear,
                      year=lambda df: df.index.year))
