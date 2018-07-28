import pytz

DEFAULT_TIMEZONE = 'Europe/Paris'
DEFAULT_TMS_GMT_COL = "tms_gmt"


def date_features(input_df, tms_gmt_col=DEFAULT_TMS_GMT_COL):
    """
    Given a datetime column, extracts useful date information
    (minute, hour, dow...)
    """
    df = input_df.copy()
    # TODO: Use getattr instead of hardcoding these variables.
    return (df.set_index(tms_gmt_col)
              .assign(minute=lambda df: df.index.minute,
                      hour=lambda df: df.index.hour,
                      day=lambda df: df.index.day,
                      dow=lambda df: df.index.dayofweek,
                      month=lambda df: df.index.month,
                      week=lambda df: df.index.week,
                      woy=lambda df: df.index.weekofyear,
                      year=lambda df: df.index.year))


def localize_datetime(input_df, timezone=DEFAULT_TIMEZONE,
                      tms_gmt_col=DEFAULT_TMS_GMT_COL):
    """
    Convert datetime column from UTC to another timezone.
    """
    tmz = pytz.timezone(timezone)
    df = input_df.copy()
    return (df.set_index(tms_gmt_col)
              .tz_localize(pytz.utc)  #  UTC time
              .tz_convert(tmz))  # Timezone time
