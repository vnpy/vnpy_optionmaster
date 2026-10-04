"""按上交所日历计算期权剩余交易日。"""
from datetime import datetime, timedelta
import exchange_calendars


ANNUAL_DAYS: int = 240

# Get public holidays data from Shanghai Stock Exchange
cn_calendar: exchange_calendars.ExchangeCalendar = exchange_calendars.get_calendar('XSHG')
holidays: list = [x.to_pydatetime() for x in cn_calendar.precomputed_holidays()]

# Filter future public holidays
start: datetime = datetime.today()
PUBLIC_HOLIDAYS: list[datetime] = [x for x in holidays if x >= start]


def calculate_days_to_expiry(option_expiry: datetime) -> int:
    """从当天零点起向后逐日走到到期日，周末和上交所休市日不计数，返回计数（初值为 1）。"""
    current_dt: datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    days: int = 1

    while current_dt < option_expiry:
        current_dt += timedelta(days=1)

        # Ignore weekends
        if current_dt.weekday() in [5, 6]:
            continue

        # Ignore public holidays
        if current_dt in PUBLIC_HOLIDAYS:
            continue

        days += 1

    return days
